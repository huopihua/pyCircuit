#include "HardwareEmissionPlan.h"

#include "BackendPlanning.h"
#include "Dialect/ACIR/HardwareSourceChecks.h"
#include "FinalCppNames.h"
#include "FinalSourceMap.h"
#include "HardwareEmitCommon.h"
#include "mlir/Parser/Parser.h"
#include "pycircuit/Dialect/ACIR/ACIRDialect.h"
#include "pycircuit/Dialect/ACIR/SourceUnitValidation.h"

#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

using namespace mlir;

namespace acir::compiler {

struct VerifiedHardwareEmissionPlan::FrozenBackendState {
  ac::HardwareSourceCheckPlan sourceChecks;
  std::shared_ptr<const ObservationEmissionPlan> observationPlan;
  std::optional<ac::HardwarePhysicalProjection> physicalProjection;
  std::string physicalProjectionUnavailableReason;
};

VerifiedHardwareEmissionPlan::VerifiedHardwareEmissionPlan(
    std::unique_ptr<MLIRContext> context, OwningOpRef<ModuleOp> snapshot,
    HardwareEmissionPlan &&plan,
    std::unique_ptr<FrozenBackendState> backendState)
    : context(std::move(context)), snapshot(std::move(snapshot)),
      plan(std::move(plan)), backendState(std::move(backendState)) {}
VerifiedHardwareEmissionPlan::VerifiedHardwareEmissionPlan(
    VerifiedHardwareEmissionPlan &&) = default;
VerifiedHardwareEmissionPlan &VerifiedHardwareEmissionPlan::operator=(
    VerifiedHardwareEmissionPlan &&) = default;
VerifiedHardwareEmissionPlan::~VerifiedHardwareEmissionPlan() = default;

const ac::HardwareSourceCheckPlan &BackendPlanningAccess::sourceChecks(
    const VerifiedHardwareEmissionPlan &hardware) {
  return hardware.backendState->sourceChecks;
}
std::shared_ptr<const ObservationEmissionPlan> &
BackendPlanningAccess::observationPlan(
    const VerifiedHardwareEmissionPlan &hardware) {
  return hardware.backendState->observationPlan;
}
const ac::HardwarePhysicalProjection *BackendPlanningAccess::physicalProjection(
    const VerifiedHardwareEmissionPlan &hardware) {
  const auto &projection = hardware.backendState->physicalProjection;
  return projection ? &*projection : nullptr;
}
StringRef BackendPlanningAccess::physicalProjectionUnavailableReason(
    const VerifiedHardwareEmissionPlan &hardware) {
  return hardware.backendState->physicalProjectionUnavailableReason;
}

namespace {

std::string printAttribute(Attribute value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  value.print(stream);
  return text;
}

std::string printType(Type value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  value.print(stream);
  return text;
}

std::string symbolName(Operation *operation) {
  auto name = SymbolTable::getSymbolName(operation);
  return name ? name.getValue().str() : std::string();
}

FailureOr<uint64_t> unsignedInteger(Attribute value, Operation *site,
                                    StringRef description) {
  if (auto integer = dyn_cast<IntegerAttr>(value)) {
    if (integer.getValue().isNegative() ||
        integer.getValue().getActiveBits() > 64)
      return site->emitOpError() << description << " must fit u64";
    return integer.getValue().getZExtValue();
  }
  if (auto integer = dyn_cast<ac::MathIntAttr>(value)) {
    uint64_t result = 0;
    if (integer.getCanonicalValue().starts_with("-") ||
        integer.getCanonicalValue().getAsInteger(10, result))
      return site->emitOpError() << description << " must fit u64";
    return result;
  }
  return site->emitOpError() << description << " must be an integer";
}

FailureOr<std::string> qualifiedName(StringRef raw, Operation *site,
                                     StringRef separator) {
  SmallVector<StringRef> parts;
  raw.split(parts, '.', -1, true);
  std::string result;
  for (StringRef part : parts) {
    if (part.empty())
      return site->emitOpError() << "empty hardware symbol component";
    auto item = legalizeIdentifier(part, [&] { return site->emitOpError(); });
    if (failed(item))
      return failure();
    if (!result.empty())
      result += separator.str();
    result += *item;
  }
  return result;
}

class PlanBuilder {
public:
  PlanBuilder(ModuleOp package, ac::HardwareAnalysis &analysis,
              const ac::HardwareSourceCheckPlan &checkPlan)
      : package(package), analysis(analysis), checkPlan(checkPlan) {
    plan.package = package;
  }

  FailureOr<HardwareEmissionPlan> build() {
    if (!package)
      return failure();
    Attribute rawStage = package->getAttr("ac.stage");
    auto stage = dyn_cast_or_null<StringAttr>(rawStage);
    // Match executable hardware admission: an absent stage is supported by
    // legacy final artifacts, while an explicit non-final or malformed stage
    // cannot opt into emission by changing a diagnostic label.
    if (rawStage && (!stage || stage.getValue() != "final"))
      return package.emitOpError()
             << "hardware emission planning requires ac.stage = final";
    for (Operation &operation : package.getBody()->getOperations())
      if (isa<ac::ModuleOp, ac::ModuleImportOp>(operation)) {
        HardwareEmissionDefinitionId id = plan.definitionAnchors.size();
        definitionIds.try_emplace(&operation, id);
        plan.definitionAnchors.push_back(&operation);
      }

    auto rootBindings = ac::detail::resolveHardwareRootBindings(analysis);
    auto logicalBindings =
        ac::detail::resolveHardwareLogicalRootBindings(analysis);
    if (failed(rootBindings) || failed(logicalBindings))
      return failure();

    auto root = dyn_cast<ac::ModuleOp>(rootBindings->owner);
    auto logicalRoot = dyn_cast<ac::ModuleOp>(logicalBindings->owner);
    if (!root || !logicalRoot)
      return package.emitOpError()
             << "hardware emission entry must resolve to module definitions";

    auto rootIndex = visit(root, *rootBindings);
    if (failed(rootIndex))
      return failure();
    plan.rootModule = *rootIndex;
    const HardwareEmissionModulePlan &rootPlan = plan.modules[*rootIndex];
    plan.rootInterface.publicInputCount = rootPlan.publicInputCount;
    plan.rootInterface.publicOutputCount = rootPlan.publicOutputCount;
    plan.rootInterface.defaultInputs = rootPlan.rootInputDefaults;
    plan.rootInterface.clockInput = rootPlan.clockInput;
    plan.rootInterface.resetInput = rootPlan.resetInput;
    plan.rootInterface.closedSystem = rootPlan.closedSystem;
    for (const HardwareEmissionPortPlan &port : rootPlan.inputs)
      plan.rootInterface.inputNames.push_back(port.name);
    for (const HardwareEmissionPortPlan &port : rootPlan.outputs)
      plan.rootInterface.outputNames.push_back(port.name);
    for (unsigned index = 0; index < rootPlan.publicOutputCount; ++index)
      if (rootPlan.outputs[index].availabilityOutput)
        plan.rootInterface.availabilityOutputs.push_back(
            *rootPlan.outputs[index].availabilityOutput);
    llvm::sort(plan.rootInterface.availabilityOutputs);
    plan.rootInterface.availabilityOutputs.erase(
        std::unique(plan.rootInterface.availabilityOutputs.begin(),
                    plan.rootInterface.availabilityOutputs.end()),
        plan.rootInterface.availabilityOutputs.end());
    rootReachableModuleCount = plan.modules.size();

    // Every non-generic definition emits a concrete scalar wrapper even when
    // it is not reachable from the selected root. Materialize the same closure
    // that HardwareEmitContext::prepare() validates so all backend queries are
    // served by the frozen plan.
    for (ac::ModuleOp definition : analysis.getDefinitions()) {
      if (!definition.getParameters().empty() ||
          !definition.getTypeParameters().empty())
        continue;
      ac::HardwareBindings bindings;
      bindings.owner = definition;
      if (failed(visit(definition, bindings)))
        return failure();
    }

    if (failed(buildEntry(root, logicalRoot)) || failed(buildChecks()) ||
        failed(buildDefinitions()) || failed(buildReachableDefinitions()) ||
        failed(buildPrimitives()) || failed(buildSourceUnits()))
      return failure();
    ac::HardwareBindings packageBindings;
    if (failed(buildQueries(package, packageBindings, packageStaticValues,
                            plan.packageTypes)))
      return failure();
    for (Operation *declaration : plan.nominalDeclarations) {
      Type nominal;
      if (auto structure = dyn_cast<ac::StructOp>(declaration))
        nominal = ac::StructType::get(package.getContext(),
                                      structure.getSymNameAttr());
      else if (auto enumeration = dyn_cast<ac::EnumOp>(declaration))
        nominal = ac::EnumType::get(package.getContext(),
                                    enumeration.getSymNameAttr());
      if (!nominal)
        continue;
      auto width =
          analysis.getPackedWidth(nominal, packageBindings, declaration);
      auto paths =
          analysis.getFieldPaths(nominal, packageBindings, declaration);
      if (failed(width) || failed(paths))
        return failure();
      HardwareEmissionTypeQueryPlan query;
      query.id = nextTypeQueryId++;
      query.diagnosticSource = nominal;
      query.resolved = nominal;
      query.packedWidth = *width;
      for (const ac::FieldPath &path : *paths) {
        llvm::SmallVector<std::string, 2> stored;
        for (StringAttr component : path)
          stored.push_back(component.getValue().str());
        query.fieldPaths.push_back(std::move(stored));
      }
      plan.packageTypes.push_back(std::move(query));
    }
    buildExecutionMetadata();
    if (failed(buildSystemMetadata(root)))
      return failure();
    determineFourStateRequirement();
    return std::move(plan);
  }

private:
  llvm::SmallVector<HardwareEmissionStaticPlan, 0> packageStaticValues;
  // Package-scope queries have no occurrence bindings but still must be frozen
  // for nominal type widths used by target capability checks.
  FailureOr<HardwareEmissionBindings>
  freezeBindings(const ac::HardwareBindings &bindings) {
    auto owner = definitionIds.find(bindings.owner);
    if (owner == definitionIds.end())
      return package.emitOpError()
             << "hardware bindings owner has no stable definition ID";
    HardwareEmissionBindings frozen;
    frozen.owner = owner->second;
    frozen.integers = bindings.integers;
    frozen.types = bindings.types;
    return frozen;
  }
  LogicalResult buildQueries(
      Operation *root, const ac::HardwareBindings &bindings,
      llvm::SmallVectorImpl<HardwareEmissionStaticPlan> &staticValues,
      llvm::SmallVectorImpl<HardwareEmissionTypeQueryPlan> &types,
      bool walkRoot = true) {
    llvm::DenseSet<Attribute> seenAttributes;
    llvm::DenseSet<Type> seenTypes;
    std::function<LogicalResult(Attribute, Operation *)> addAttribute;
    std::function<LogicalResult(Type, Operation *)> addType;
    addAttribute = [&](Attribute attribute, Operation *site) -> LogicalResult {
      if (!attribute || !seenAttributes.insert(attribute).second)
        return success();
      if (auto expression = dyn_cast<ac::StaticExprAttr>(attribute);
          expression && analysis.isStaticEvaluable(expression, bindings)) {
        auto value = analysis.evaluateStatic(expression, bindings, site);
        if (failed(value))
          return failure();
        staticValues.push_back({nextStaticQueryId++, expression, *value});
      }
      LogicalResult status = success();
      attribute.walkImmediateSubElements(
          [&](Attribute child) {
            if (succeeded(status))
              status = addAttribute(child, site);
          },
          [&](Type child) {
            if (succeeded(status))
              status = addType(child, site);
          });
      return status;
    };
    addType = [&](Type type, Operation *site) -> LogicalResult {
      if (!type || !seenTypes.insert(type).second)
        return success();
      if (isa<ac::BitsType, ac::TableType, ac::StructType, ac::EnumType,
              ac::TypeParamType>(type)) {
        HardwareEmissionTypeQueryPlan query;
        query.id = nextTypeQueryId++;
        query.diagnosticSource = type;
        auto formal = dyn_cast<ac::TypeParamType>(type);
        bool hasUnboundFormal = false;
        type.walk([&](Type nested) {
          if (auto parameter = dyn_cast<ac::TypeParamType>(nested))
            hasUnboundFormal |=
                !bindings.types.contains(parameter.getName().getValue());
        });
        if (hasUnboundFormal) {
          query.resolved = type;
          // A symbolic formal is still one complete hardware value.  Preserve
          // its empty field path so generic value-merge lowering can emit the
          // whole-value pass-through without rediscovering type structure.
          if (isa<ac::TypeParamType>(type))
            query.fieldPaths.emplace_back();
          types.push_back(std::move(query));
          LogicalResult status = success();
          type.walkImmediateSubElements(
              [&](Attribute child) {
                if (succeeded(status))
                  status = addAttribute(child, site);
              },
              [&](Type child) {
                if (succeeded(status))
                  status = addType(child, site);
              });
          return status;
        }
        auto resolved = analysis.resolveType(type, bindings, site);
        if (failed(resolved))
          return failure();
        query.resolved = *resolved;
        bool concreteWidth = !isa<ac::TypeParamType>(*resolved);
        if (auto bits = dyn_cast<ac::BitsType>(type))
          concreteWidth &=
              analysis.isStaticEvaluable(bits.getWidth(), bindings);
        if (auto table = dyn_cast<ac::TableType>(type))
          for (Attribute extent : table.getShape())
            concreteWidth &= analysis.isStaticEvaluable(extent, bindings);
        concreteWidth &=
            isa<ac::StructType, ac::EnumType>(type) ||
            analysis.isStaticEvaluable(TypeAttr::get(type), bindings);
        if (concreteWidth) {
          auto width = analysis.getPackedWidth(type, bindings, site);
          if (failed(width))
            return failure();
          query.packedWidth = *width;
          auto paths = analysis.getFieldPaths(type, bindings, site);
          if (failed(paths))
            return failure();
          for (const ac::FieldPath &path : *paths) {
            llvm::SmallVector<std::string, 2> stored;
            for (StringAttr component : path)
              stored.push_back(component.getValue().str());
            query.fieldPaths.push_back(std::move(stored));
          }
        }
        types.push_back(std::move(query));
      }
      LogicalResult status = success();
      type.walkImmediateSubElements(
          [&](Attribute child) {
            if (succeeded(status))
              status = addAttribute(child, site);
          },
          [&](Type child) {
            if (succeeded(status))
              status = addType(child, site);
          });
      return status;
    };
    for (const auto &binding : bindings.integers)
      if (failed(addAttribute(binding.getValue(), root)))
        return failure();
    for (const auto &binding : bindings.types)
      if (failed(addType(binding.getValue(), root)))
        return failure();
    if (!walkRoot)
      return success();
    auto walk = root->walk([&](Operation *operation) {
      for (NamedAttribute attribute : operation->getAttrs())
        if (failed(addAttribute(attribute.getValue(), operation)))
          return WalkResult::interrupt();
      for (Type type : operation->getOperandTypes())
        if (failed(addType(type, operation)))
          return WalkResult::interrupt();
      for (Type type : operation->getResultTypes())
        if (failed(addType(type, operation)))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return success(!walk.wasInterrupted());
  }

  LogicalResult buildTableViews(
      Operation *root, const ac::HardwareBindings &bindings,
      llvm::SmallVectorImpl<HardwareEmissionTableViewPlan> &views) {
    auto walk = root->walk([&](ac::TableViewOp view) {
      auto shape = analysis.getTableShape(view.getInput().getType(), bindings,
                                          view);
      auto source = analysis.getViewSourceOrdinal(view, 0, bindings);
      if (failed(shape) || failed(source))
        return WalkResult::interrupt();
      views.push_back({view, *source, std::move(*shape)});
      return WalkResult::advance();
    });
    return success(!walk.wasInterrupted());
  }

  LogicalResult buildDependencies(
      ac::ModuleOp definition, const ac::HardwareBindings &bindings,
      const llvm::DenseMap<Value, HardwareEmissionValueId> *frozenValueIds,
      llvm::SmallVectorImpl<HardwareEmissionDependencyPlan> &result) {
    llvm::DenseMap<Value, HardwareEmissionValueId> valueIds;
    if (frozenValueIds) {
      valueIds = *frozenValueIds;
    } else {
      HardwareEmissionValueId nextValueId = 0;
      for (BlockArgument argument : definition.getBody().front().getArguments())
        valueIds.try_emplace(argument, nextValueId++);
      definition.walk([&](Operation *operation) {
        if (operation == definition)
          return;
        for (Value result : operation->getResults())
          valueIds.try_emplace(result, nextValueId++);
        for (Region &region : operation->getRegions())
          for (Block &block : region)
            for (BlockArgument argument : block.getArguments())
              valueIds.try_emplace(argument, nextValueId++);
      });
    }
    auto add = [&](Value value) -> LogicalResult {
      auto outputId = valueIds.find(value);
      if (outputId == valueIds.end())
        return definition.emitOpError()
               << "hardware dependency output has no value identity";
      auto paths = analysis.getFieldPaths(value.getType(), bindings, definition);
      if (failed(paths))
        return failure();
      for (const ac::FieldPath &path : *paths) {
        auto dependencies = analysis.getDependencies(value, path, bindings);
        if (failed(dependencies))
          return failure();
        HardwareEmissionDependencyPlan row;
        row.output.id = outputId->second;
        row.output.value = value;
        for (StringAttr component : path)
          row.output.path.push_back(component.getValue().str());
        for (const ac::WireEndpoint &dependency : *dependencies) {
          auto inputId = valueIds.find(dependency.value);
          if (inputId == valueIds.end())
            return definition.emitOpError()
                   << "hardware dependency input has no value identity";
          HardwareEmissionDependencyEndpoint endpoint;
          endpoint.id = inputId->second;
          endpoint.value = dependency.value;
          for (StringAttr component : dependency.path)
            endpoint.path.push_back(component.getValue().str());
          row.inputs.push_back(std::move(endpoint));
        }
        result.push_back(std::move(row));
      }
      return success();
    };
    SmallVector<std::pair<HardwareEmissionValueId, Value>> orderedValues;
    for (const auto &entry : valueIds)
      orderedValues.push_back({entry.second, entry.first});
    llvm::sort(orderedValues, [](const auto &left, const auto &right) {
      return left.first < right.first;
    });
    for (const auto &[id, value] : orderedValues) {
      (void)id;
      if (failed(add(value)))
        return failure();
    }
    return success();
  }

  FailureOr<HardwareEmissionTypePlan>
  concreteType(Type source, const ac::HardwareBindings &bindings,
               Operation *site) {
    auto resolved = analysis.resolveType(source, bindings, site);
    if (failed(resolved))
      return failure();
    Type type = *resolved;
    auto width = analysis.getPackedWidth(type, bindings, site);
    if (failed(width) || *width == 0)
      return failure();

    HardwareEmissionTypePlan result;
    result.resolvedType = type;
    result.packedWidth = *width;
    result.canonicalType = printType(type);
    if (isa<ac::BitsType>(type)) {
      result.kind = HardwareEmissionTypeKind::Bits;
    } else if (auto table = dyn_cast<ac::TableType>(type)) {
      result.kind = HardwareEmissionTypeKind::Table;
      auto shape = analysis.getTableShape(table, bindings, site);
      if (failed(shape))
        return failure();
      result.shape = std::move(*shape);
    } else if (isa<ac::StructType>(type)) {
      result.kind = HardwareEmissionTypeKind::Struct;
    } else if (isa<ac::EnumType>(type)) {
      result.kind = HardwareEmissionTypeKind::Enum;
    } else {
      return site->emitOpError()
             << "hardware emission plan requires a concrete hardware type";
    }

    auto leaves = analysis.getLeaves(type, bindings, site);
    if (failed(leaves))
      return failure();
    for (const ac::PackedLeaf &leaf : *leaves) {
      HardwareEmissionPackedField field;
      for (StringAttr component : leaf.path)
        field.path.push_back(component.getValue().str());
      field.low = leaf.low;
      field.width = leaf.width;
      result.fields.push_back(std::move(field));
    }
    if (result.fields.empty())
      result.fields.push_back({{}, 0, result.packedWidth});
    return result;
  }

  FailureOr<HardwareEmissionModuleKey>
  moduleKey(ac::ModuleOp definition, const ac::HardwareBindings &bindings,
            Operation *site) {
    HardwareEmissionModuleKey key;
    key.definition = definition.getSymName().str();
    for (Attribute raw : definition.getParameters()) {
      auto parameter = dyn_cast<DictionaryAttr>(raw);
      auto name =
          parameter ? parameter.getAs<StringAttr>("name") : StringAttr();
      Attribute value =
          name ? bindings.integers.lookup(name.getValue()) : Attribute();
      if (!name || !value)
        return site->emitOpError()
               << "hardware emission plan has an unresolved integer binding";
      auto integer = dyn_cast<ac::MathIntAttr>(value);
      if (!integer)
        return site->emitOpError()
               << "hardware emission plan integer binding is not canonical";
      key.integers.push_back(
          {name.getValue().str(), integer.getCanonicalValue().str()});
    }
    for (Attribute raw : definition.getTypeParameters()) {
      auto name = dyn_cast<StringAttr>(raw);
      Type type = name ? bindings.types.lookup(name.getValue()) : Type();
      if (!name || !type)
        return site->emitOpError()
               << "hardware emission plan has an unresolved type binding";
      auto concrete = analysis.resolveType(type, bindings, site);
      if (failed(concrete))
        return failure();
      key.types.push_back({name.getValue().str(), printType(*concrete)});
    }
    return key;
  }

  std::string canonicalKey(const HardwareEmissionModuleKey &key) {
    std::string result = key.definition;
    for (const auto &binding : key.integers)
      result += "\nI" + binding.name + "=" + binding.value;
    for (const auto &binding : key.types)
      result += "\nT" + binding.name + "=" + binding.value;
    return result;
  }

  FailureOr<unsigned> visit(ac::ModuleOp definition,
                            const ac::HardwareBindings &bindings) {
    auto key = moduleKey(definition, bindings, definition);
    if (failed(key))
      return failure();
    std::string canonical = canonicalKey(*key);
    auto existing = moduleIndices.find(canonical);
    if (existing != moduleIndices.end()) {
      if (activeModules.contains(existing->second))
        return definition.emitOpError()
               << "hardware emission does not support recursive module "
                  "specializations";
      return existing->second;
    }

    const unsigned moduleIndex = plan.modules.size();
    moduleIndices.try_emplace(canonical, moduleIndex);
    activeModules.insert(moduleIndex);
    llvm::scope_exit leave([&] { activeModules.erase(moduleIndex); });

    HardwareEmissionModulePlan module;
    module.diagnosticAnchor = definition;
    module.key = std::move(*key);
    auto frozenBindings = freezeBindings(bindings);
    if (failed(frozenBindings))
      return failure();
    module.concreteBindings = std::move(*frozenBindings);
    module.sourceOwner = definition.getSourceOwner();
    auto rootKind = definition->getAttrOfType<StringAttr>("ac.root_kind");
    module.closedSystem = rootKind && rootKind.getValue() == "system";
    auto portLayout = ac::getModulePortLayout(definition);
    if (failed(portLayout))
      return failure();
    module.publicInputCount = portLayout->sourceInputCount;
    module.publicOutputCount = portLayout->sourceOutputCount;
    for (Attribute raw : portLayout->rootInputDefaults) {
      auto value = unsignedInteger(raw, definition, "root input default");
      if (failed(value))
        return failure();
      module.rootInputDefaults.push_back(*value);
    }
    if (auto domain =
            definition->getAttrOfType<DictionaryAttr>("ac.domain_inputs")) {
      auto clock = domain.getAs<IntegerAttr>("clock");
      auto reset = domain.getAs<IntegerAttr>("reset");
      if (clock) {
        auto value = unsignedInteger(clock, definition, "clock input");
        if (failed(value))
          return failure();
        module.clockInput = *value;
      }
      if (reset) {
        auto value = unsignedInteger(reset, definition, "reset input");
        if (failed(value))
          return failure();
        module.resetInput = *value;
      }
    }

    plan.modules.push_back(std::move(module));
    auto &stored = plan.modules[moduleIndex];
    auto &body = definition.getBody().front();
    llvm::DenseMap<Value, HardwareEmissionValueId> valueIds;
    auto addValue = [&](Value value,
                        HardwareEmissionValueKind kind) -> LogicalResult {
      if (valueIds.contains(value))
        return success();
      auto type = concreteType(value.getType(), bindings, definition);
      if (failed(type))
        return failure();
      HardwareEmissionValueId id = stored.values.size();
      valueIds.try_emplace(value, id);
      stored.values.push_back({id, value, kind, std::move(*type)});
      return success();
    };
    for (BlockArgument argument : body.getArguments())
      if (failed(addValue(argument, HardwareEmissionValueKind::ModuleInput)))
        return failure();
    definition.walk([&](Operation *operation) {
      if (operation == definition || failed(valueBuildStatus))
        return;
      for (Value result : operation->getResults())
        if (failed(
                addValue(result, HardwareEmissionValueKind::OperationResult))) {
          valueBuildStatus = failure();
          return;
        }
      for (Region &region : operation->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            if (failed(addValue(argument,
                                HardwareEmissionValueKind::RegionArgument))) {
              valueBuildStatus = failure();
              return;
            }
    });
    if (failed(valueBuildStatus))
      return failure();
    valueBuildStatus = success();

    auto inputNames = definition.getInputNames();
    auto outputNames = definition.getOutputNames();
    for (auto [index, argument] : llvm::enumerate(body.getArguments())) {
      stored.inputs.push_back(
          {cast<StringAttr>(inputNames[index]).getValue().str(),
           stored.values[valueIds.lookup(argument)].type,
           valueIds.lookup(argument)});
    }
    auto yield = dyn_cast<ac::YieldOp>(body.getTerminator());
    if (!yield || yield.getValues().size() != outputNames.size())
      return definition.emitOpError()
             << "hardware emission module requires a complete yield";
    for (auto [index, value] : llvm::enumerate(yield.getValues())) {
      auto found = valueIds.find(value);
      if (found == valueIds.end())
        return yield.emitOpError()
               << "hardware emission output has no value identity";
      stored.outputs.push_back(
          {cast<StringAttr>(outputNames[index]).getValue().str(),
           stored.values[found->second].type, found->second, std::nullopt});
    }
    for (auto [index, companion] : llvm::enumerate(portLayout->outputValidity))
      if (companion >= 0)
        stored.outputs[index].availabilityOutput =
            static_cast<unsigned>(companion);

    auto schedule = analysis.analyzeModule(definition, bindings);
    if (failed(schedule))
      return failure();
    llvm::DenseSet<Operation *> temporal(schedule->storageWork.begin(),
                                         schedule->storageWork.end());
    for (Operation *operation : schedule->schedule) {
      HardwareEmissionScheduleStep step;
      step.id = stored.schedule.size();
      step.diagnosticAnchor = operation;
      step.opcode = operation->getName().getStringRef().str();
      step.temporalCut = temporal.contains(operation);
      step.kind = step.temporalCut ? HardwareEmissionStepKind::Temporal
                  : isa<ac::InstanceOp>(operation)
                      ? HardwareEmissionStepKind::Instance
                  : isa<ac::CollectionOp>(operation)
                      ? HardwareEmissionStepKind::Collection
                  : operation->hasTrait<OpTrait::IsTerminator>()
                      ? HardwareEmissionStepKind::Terminator
                      : HardwareEmissionStepKind::Combinational;
      for (Value input : operation->getOperands()) {
        auto found = valueIds.find(input);
        if (found == valueIds.end())
          return operation->emitOpError()
                 << "hardware emission schedule input has no value identity";
        step.inputs.push_back(found->second);
      }
      for (Value output : operation->getResults()) {
        auto found = valueIds.find(output);
        if (found == valueIds.end())
          return operation->emitOpError()
                 << "hardware emission schedule output has no value identity";
        step.outputs.push_back(found->second);
      }
      stored.schedule.push_back(std::move(step));
    }

    // Reserve before recursive visits so adding a specialization cannot
    // invalidate the current module reference.
    llvm::SmallVector<HardwareEmissionChildPlan, 2> children;
    for (Operation &operation : body) {
      if (!isa<ac::InstanceOp, ac::CollectionOp>(operation))
        continue;
      HardwareEmissionChildPlan child;
      child.diagnosticAnchor = &operation;
      child.id = plan.occurrenceAnchors.size();
      plan.occurrenceAnchors.push_back(&operation);
      FailureOr<ac::HardwareBindings> childBindings = failure();
      Operation *callee = nullptr;
      if (auto instance = dyn_cast<ac::InstanceOp>(operation)) {
        child.instanceName = instance.getInstanceName().str();
        child.callee = instance.getCallee().str();
        childBindings = analysis.bindInstance(instance, bindings);
        callee = analysis.resolveCallee(instance);
      } else {
        auto collection = cast<ac::CollectionOp>(operation);
        child.instanceName = collection.getInstanceName().str();
        child.callee = collection.getCallee().str();
        childBindings = analysis.bindInstance(collection, bindings);
        callee = analysis.resolveCallee(collection);
        auto shape =
            analysis.resolveShape(collection.getShape(), bindings, collection);
        if (failed(shape))
          return failure();
        child.shape = std::move(*shape);
      }
      if (failed(childBindings) || !callee)
        return failure();
      auto frozenChildBindings = freezeBindings(*childBindings);
      if (failed(frozenChildBindings))
        return failure();
      child.concreteBindings = std::move(*frozenChildBindings);
      if (auto queue = dyn_cast<ac::InstanceOp>(operation);
          queue && ac::isQueueInstance(queue)) {
        auto rawConfig = ac::getQueueConfiguration(queue);
        auto config = analysis.resolveQueue(queue, bindings);
        if (failed(rawConfig) || failed(config))
          return failure();
        if (!config->tokenCardinality ||
            config->packedWidth % config->tokenCardinality)
          return operation.emitOpError()
                 << "queue token geometry is not evenly packed";
        child.queue = HardwareEmissionQueuePlan{
            config->depth, config->tokenCardinality,
            config->packedWidth / config->tokenCardinality,
            config->availabilityLatency,
            config->readyPolicy == ac::QueueReadyPolicy::LocalOccupancy
                ? HardwareEmissionQueueReadyPolicy::LocalOccupancy
                : HardwareEmissionQueueReadyPolicy::DownstreamPop,
            rawConfig->getAs<ac::StaticExprAttr>("depth"),
            rawConfig->getAs<ac::StaticExprAttr>("availability_latency")};
      }
      child.primitiveKind = analysis.getPrimitiveKind(callee).str();
      auto childDefinition = dyn_cast<ac::ModuleOp>(callee);
      if (child.primitiveKind.empty()) {
        if (!childDefinition)
          return operation.emitOpError()
                 << "ordinary hardware child has no module definition";
        auto index = visit(childDefinition, *childBindings);
        if (failed(index))
          return failure();
        child.kind = HardwareEmissionChildKind::Module;
        child.moduleIndex = *index;
        child.bindings = plan.modules[*index].key;
      } else {
        child.kind = HardwareEmissionChildKind::Primitive;
        if (failed(buildQueries(callee, *childBindings, child.staticValues,
                                child.types)))
          return failure();
        auto import = dyn_cast<ac::ModuleImportOp>(callee);
        if (!import)
          return operation.emitOpError()
                 << "hardware primitive has no import declaration";
        // Use a temporary module-shaped key for a fully normalized primitive
        // binding. Primitive pins and target runtime classes remain target-plan
        // concerns and deliberately do not enter this shared plan.
        child.bindings.definition = import.getSymName().str();
        for (const auto &item : childBindings->integers) {
          auto integer = dyn_cast<ac::MathIntAttr>(item.getValue());
          if (!integer)
            return operation.emitOpError()
                   << "primitive integer binding is not concrete";
          child.bindings.integers.push_back(
              {item.getKey().str(), integer.getCanonicalValue().str()});
        }
        for (const auto &item : childBindings->types)
          child.bindings.types.push_back(
              {item.getKey().str(), printType(item.getValue())});
        llvm::sort(child.bindings.integers,
                   [](const auto &left, const auto &right) {
                     return left.name < right.name;
                   });
        llvm::sort(child.bindings.types,
                   [](const auto &left, const auto &right) {
                     return left.name < right.name;
                   });
      }
      children.push_back(std::move(child));
    }
    plan.modules[moduleIndex].children = std::move(children);
    if (failed(buildQueries(definition, bindings,
                            plan.modules[moduleIndex].staticValues,
                            plan.modules[moduleIndex].types)))
      return failure();
    if (failed(buildTableViews(definition, bindings,
                               plan.modules[moduleIndex].tableViews)))
      return failure();
    if (failed(buildDependencies(definition, bindings, &valueIds,
                                 plan.modules[moduleIndex].dependencies)))
      return failure();
    if (failed(buildObservations(moduleIndex, definition, bindings)))
      return failure();
    return moduleIndex;
  }

  LogicalResult buildEntry(ac::ModuleOp implementation, ac::ModuleOp logical) {
    auto systems = package.getOps<ac::SystemOp>();
    if (!llvm::hasSingleElement(systems))
      return package.emitOpError()
             << "hardware emission requires exactly one ac.system";
    auto system = *systems.begin();
    DictionaryAttr logicalEntry = system.getEntry();
    if (auto boundary =
            system->getAttrOfType<DictionaryAttr>("ac.entry_boundary"))
      logicalEntry = boundary.getAs<DictionaryAttr>("logical_entry");
    if (!logicalEntry)
      return system.emitOpError() << "hardware emission has no logical entry";

    plan.entry.implementationDefinition = implementation.getSymName().str();
    plan.entry.logicalDefinition = logical.getSymName().str();
    plan.entry.sourceOwner = implementation.getSourceOwner();
    for (StringRef kind :
         {StringRef("parameters"), StringRef("type_arguments")})
      for (Attribute value : logicalEntry.getAs<ArrayAttr>(kind))
        plan.entry.logicalArguments.push_back(
            {kind.str(), printAttribute(value)});
    plan.resetSemantics = checkPlan.resetSemantics;
    return success();
  }

  LogicalResult buildChecks() {
    for (const ac::HardwareResolvedSourceCheck &resolved : checkPlan.checks) {
      if (resolved.bindingIndex >= checkPlan.bindings.size())
        return package.emitOpError()
               << "hardware emission check has an invalid binding";
      const auto &binding = checkPlan.bindings[resolved.bindingIndex];
      auto location = emissionMetadataJson(binding.location, binding.expect);
      auto identity = emissionMetadataJson(binding.checkID, binding.expect);
      if (failed(location) || failed(identity))
        return failure();
      plan.checks.push_back(
          {resolved.ordinal, resolved.bindingIndex, resolved.ownerOccurrence,
           binding.kind.getValue().str(),
           binding.message ? binding.message.getValue().str() : std::string(),
           std::move(*location), std::move(*identity)});
    }
    return success();
  }

  LogicalResult buildObservations(unsigned moduleIndex, ac::ModuleOp definition,
                                  const ac::HardwareBindings &bindings) {
    for (auto observe :
         definition.getBody().front().getOps<ac::SourceObserveOp>()) {
      auto identity =
          observe->getAttrOfType<DictionaryAttr>("ac.observation_id");
      auto encoded = emissionMetadataJson(identity, observe);
      if (failed(encoded))
        return failure();
      HardwareEmissionObservationPlan observation;
      observation.id = plan.observations.size();
      observation.moduleIndex = moduleIndex;
      observation.diagnosticAnchor = observe;
      observation.kind = observe.getKind().str();
      observation.identity = std::move(*encoded);
      for (Value value : observe.getValues()) {
        auto width =
            analysis.getPackedWidth(value.getType(), bindings, observe);
        if (failed(width) || *width == 0)
          return failure();
        observation.payloadWidths.push_back(*width);
      }
      auto logical = ac::detail::resolveObservationOwner(analysis, observe,
                                                         bindings, {});
      if (failed(logical))
        return failure();
      for (ac::InstanceOp instance : logical->path)
        observation.logicalSuffix.push_back(instance);
      plan.observations.push_back(std::move(observation));
    }
    return success();
  }

  LogicalResult buildSourceUnits() {
    auto units = ac::collectFinalSourceUnits(
        package, [&] { return package.emitOpError(); });
    if (failed(units))
      return failure();
    llvm::DenseMap<Attribute, unsigned> owners;
    for (const ac::FinalSourceUnitView &unit : *units) {
      auto owner = sourceOwnerComponents(
          unit.owner, [&] { return package.emitOpError(); });
      if (failed(owner))
        return failure();
      auto header = sourcePathName(owner->filePath, ".hpp",
                                   [&] { return package.emitOpError(); });
      if (failed(header))
        return failure();
      owners.try_emplace(unit.owner, plan.sourceUnits.size());
      HardwareEmissionSourceUnitPlan sourceUnit;
      sourceUnit.sourceOwner = unit.owner;
      sourceUnit.cppHeaderPath = std::move(*header);
      for (Operation *declaration : unit.declarations) {
        auto found = definitionIndices.find(declaration);
        if (found == definitionIndices.end())
          continue;
        sourceUnit.definitionIndices.push_back(found->second);
        const auto &names = plan.definitions[found->second].names;
        if (sourceUnit.cppSourcePath.empty()) {
          sourceUnit.cppSourcePath = names.source;
          sourceUnit.verilogPath = names.verilog;
        }
      }
      llvm::DenseSet<unsigned> emitted, active;
      std::function<LogicalResult(unsigned)> schedule =
          [&](unsigned definitionIndex) -> LogicalResult {
        if (emitted.contains(definitionIndex))
          return success();
        if (!active.insert(definitionIndex).second)
          return plan.definitions[definitionIndex]
                     .diagnosticAnchor->emitOpError()
                 << "hardware emission definition dependency is recursive";
        const auto &definition = plan.definitions[definitionIndex];
        for (const HardwareEmissionOccurrenceBinding &child :
             definition.templateChildren) {
          auto found = definitionIndices.find(
              plan.definitionAnchors[child.bindings.owner]);
          if (found != definitionIndices.end() &&
              plan.definitions[found->second].sourceOwner == unit.owner &&
              failed(schedule(found->second)))
            return failure();
        }
        active.erase(definitionIndex);
        emitted.insert(definitionIndex);
        sourceUnit.emissionOrder.push_back(definitionIndex);
        return success();
      };
      for (unsigned definitionIndex : sourceUnit.definitionIndices)
        if (failed(schedule(definitionIndex)))
          return failure();
      llvm::StringSet<> includes;
      for (unsigned definitionIndex : sourceUnit.definitionIndices) {
        const auto &definition = plan.definitions[definitionIndex];
        for (const HardwareEmissionOccurrenceBinding &child :
             definition.templateChildren) {
          auto found = definitionIndices.find(
              plan.definitionAnchors[child.bindings.owner]);
          if (found == definitionIndices.end())
            continue;
          const std::string &header =
              plan.definitions[found->second].names.header;
          if (header != sourceUnit.cppHeaderPath)
            includes.insert(header);
        }
      }
      for (const auto &include : includes)
        sourceUnit.cppIncludes.push_back(include.getKey().str());
      llvm::sort(sourceUnit.cppIncludes);
      plan.sourceUnits.push_back(std::move(sourceUnit));
    }
    for (auto [index, module] : llvm::enumerate(plan.modules)) {
      auto found = owners.find(module.sourceOwner);
      if (found == owners.end())
        return module.diagnosticAnchor->emitOpError()
               << "hardware emission module owner is absent from source units";
      plan.sourceUnits[found->second].moduleIndices.push_back(index);
    }
    auto maps =
        prepareFinalSourceMaps(*units, [&] { return package.emitOpError(); });
    if (failed(maps))
      return failure();
    plan.sourceMaps = std::move(*maps);
    return success();
  }

  LogicalResult buildDefinitions() {
    for (Operation &operation : package.getBody()->getOperations())
      if (isa<ac::StructOp, ac::EnumOp>(operation))
        plan.nominalDeclarations.push_back(&operation);
    for (ac::ModuleOp definition : analysis.getDefinitions()) {
      HardwareEmissionDefinitionPlan item;
      item.id = definitionIds.lookup(definition);
      item.diagnosticAnchor = definition;
      item.symbol = definition.getSymName().str();
      item.sourceOwner = definition.getSourceOwner();
      auto cpp = qualifiedName(item.symbol, definition, "::");
      auto rtl = qualifiedName(item.symbol, definition, "_");
      auto owner = sourceOwnerComponents(
          item.sourceOwner, [&] { return definition.emitOpError(); });
      if (failed(cpp) || failed(rtl) || failed(owner))
        return failure();
      SmallVector<StringRef> pieces;
      StringRef(*cpp).split(pieces, "::");
      item.names.cppQualified = "::" + *cpp;
      item.names.cppClass = pieces.back().str();
      item.names.cppNamespace =
          cpp->substr(0, cpp->size() - pieces.back().size());
      if (item.names.cppNamespace.ends_with("::"))
        item.names.cppNamespace.resize(item.names.cppNamespace.size() - 2);
      item.names.cppFamilyClass = "pyc_family_" + item.names.cppClass;
      item.names.cppFamilyQualified =
          "::" + (item.names.cppNamespace.empty()
                       ? ""
                       : item.names.cppNamespace + "::") +
          item.names.cppFamilyClass;
      item.names.rtl = "ac_" + *rtl;
      auto header = sourcePathName(owner->filePath, ".hpp",
                                   [&] { return definition.emitOpError(); });
      auto source = sourcePathName(owner->filePath, ".cpp",
                                   [&] { return definition.emitOpError(); });
      auto verilog = sourcePathName(owner->filePath, ".v",
                                    [&] { return definition.emitOpError(); });
      if (failed(header) || failed(source) || failed(verilog))
        return failure();
      item.names.header = std::move(*header);
      item.names.source = std::move(*source);
      item.names.verilog = std::move(*verilog);
      ac::HardwareBindings templateBindings;
      templateBindings.owner = definition;
      auto templateSchedule =
          analysis.analyzeModule(definition, templateBindings);
      if (failed(templateSchedule))
        return failure();
      item.templateSchedule.assign(templateSchedule->schedule.begin(),
                                   templateSchedule->schedule.end());
      auto yield =
          dyn_cast<ac::YieldOp>(definition.getBody().front().getTerminator());
      if (!yield)
        return definition.emitOpError()
               << "hardware emission definition requires a complete yield";
      item.outputValues.assign(yield.getValues().begin(),
                               yield.getValues().end());
      for (Operation &operation : definition.getBody().front()) {
        FailureOr<ac::HardwareBindings> child = failure();
        if (auto instance = dyn_cast<ac::InstanceOp>(operation))
          child = analysis.bindInstance(instance, templateBindings);
        else if (auto collection = dyn_cast<ac::CollectionOp>(operation))
          child = analysis.bindInstance(collection, templateBindings);
        else
          continue;
        if (failed(child))
          return failure();
        auto frozenChild = freezeBindings(*child);
        if (failed(frozenChild))
          return failure();
        HardwareEmissionOccurrenceId occurrenceId =
            plan.occurrenceAnchors.size();
        plan.occurrenceAnchors.push_back(&operation);
        HardwareEmissionOccurrenceBinding occurrence{
            occurrenceId,
            &operation,
            operation.getAttrOfType<StringAttr>("instance_name").getValue().str(),
            isa<ac::CollectionOp>(operation),
            std::move(*frozenChild),
            {},
            std::nullopt};
        if (auto queue = dyn_cast<ac::InstanceOp>(operation);
            queue && ac::isQueueInstance(queue)) {
          auto config = ac::getQueueConfiguration(queue);
          if (failed(config))
            return failure();
          StringRef policy =
              config->getAs<StringAttr>("ready_policy").getValue();
          occurrence.queue = HardwareEmissionQueuePlan{
              0, 0, 0, 0,
              policy == "local_occupancy"
                  ? HardwareEmissionQueueReadyPolicy::LocalOccupancy
                  : HardwareEmissionQueueReadyPolicy::DownstreamPop,
              config->getAs<ac::StaticExprAttr>("depth"),
              config->getAs<ac::StaticExprAttr>("availability_latency")};
        }
        if (auto collection = dyn_cast<ac::CollectionOp>(operation);
            collection && definition.getParameters().empty() &&
            definition.getTypeParameters().empty()) {
          auto shape = analysis.resolveShape(collection.getShape(),
                                             templateBindings, collection);
          if (failed(shape))
            return failure();
          occurrence.shape = std::move(*shape);
        }
        item.templateChildren.push_back(std::move(occurrence));
      }
      auto independent = analysis.getIndependentWorkSubtrees(definition);
      if (failed(independent))
        return failure();
      item.independentWorkChildren = std::move(*independent);
      if (definition.getTypeParameters().empty()) {
        llvm::DenseMap<Value, HardwareEmissionValueId> valueIds;
        HardwareEmissionValueId nextValueId = 0;
        for (BlockArgument argument : definition.getBody().front().getArguments())
          valueIds.try_emplace(argument, nextValueId++);
        definition.walk([&](Operation *operation) {
          if (operation == definition)
            return;
          for (Value result : operation->getResults())
            valueIds.try_emplace(result, nextValueId++);
          for (Region &region : operation->getRegions())
            for (Block &block : region)
              for (BlockArgument argument : block.getArguments())
                valueIds.try_emplace(argument, nextValueId++);
        });
        if (failed(buildDependencies(definition, templateBindings, nullptr,
                                     item.dependencies)))
          return failure();
      }
      if (failed(buildQueries(definition, templateBindings, item.staticValues,
                              item.types)))
        return failure();
      definitionIndices.try_emplace(definition, plan.definitions.size());
      plan.definitions.push_back(std::move(item));
    }
    return success();
  }

  LogicalResult buildReachableDefinitions() {
    llvm::DenseSet<unsigned> reached;
    for (unsigned moduleIndex = 0; moduleIndex < rootReachableModuleCount;
         ++moduleIndex) {
      auto found =
          definitionIndices.find(plan.modules[moduleIndex].diagnosticAnchor);
      if (found == definitionIndices.end())
        return plan.modules[moduleIndex].diagnosticAnchor->emitOpError()
               << "reachable module is absent from the definition inventory";
      if (reached.insert(found->second).second)
        plan.reachableDefinitions.push_back(found->second);
    }
    return success();
  }

  LogicalResult buildPrimitives() {
    llvm::StringSet<> standardSources;
    package.walk([&](ac::InstanceOp instance) {
      if (ac::isQueueInstance(instance) &&
          standardSources.insert("include/verilog/fifo.v").second)
        plan.standardVerilogSources.push_back("include/verilog/fifo.v");
    });
    for (ac::ModuleImportOp imported : package.getOps<ac::ModuleImportOp>()) {
      StringRef kind = analysis.getPrimitiveKind(imported);
      if (kind.empty())
        return imported.emitOpError()
               << "unresolved module import in hardware emission plan";
      HardwareEmissionPrimitivePlan primitive;
      primitive.id = definitionIds.lookup(imported);
      primitive.diagnosticAnchor = imported;
      primitive.symbol = imported.getSymName().str();
      primitive.kind = kind.str();
      if (kind != "fifo_local" && kind != "fifo_downstream_pop") {
        primitive.standardSource = "include/verilog/" + kind.str() + ".v";
        if (standardSources.insert(primitive.standardSource).second)
          plan.standardVerilogSources.push_back(primitive.standardSource);
      }
      plan.primitives.push_back(std::move(primitive));
    }
    return success();
  }

  void buildExecutionMetadata() {
    auto strategy =
        package->getAttrOfType<StringAttr>("ac.execution_selection");
    if (!strategy)
      return;
    HardwareEmissionExecutionPlan execution;
    execution.strategy = strategy.getValue().str();
    if (auto protocol =
            package->getAttrOfType<StringAttr>("ac.execution_protocol"))
      execution.protocol = protocol.getValue().str();
    if (auto publication = package->getAttrOfType<StringAttr>(
            "ac.execution_publication_field"))
      execution.publicationField = publication.getValue().str();
    if (auto state =
            package->getAttrOfType<IntegerAttr>("ac.execution_state_output"))
      if (!state.getValue().isNegative() &&
          state.getValue().getActiveBits() <= 64)
        execution.stateOutput = state.getValue().getZExtValue();
    plan.execution = std::move(execution);
  }

  LogicalResult buildSystemMetadata(ac::ModuleOp root) {
    auto samples = ac::getSystemSamples(root);
    if (failed(samples))
      return failure();
    for (Attribute raw : *samples)
      plan.systemSamples.push_back(cast<StringAttr>(raw).getValue().str());
    return success();
  }

  void determineFourStateRequirement() {
    plan.needsFourState = llvm::any_of(plan.systemSamples, [](StringRef row) {
      return row.contains('x') || row.contains('z');
    });
    auto knownNonzeroLiteral = [](Value value) {
      if (auto constant = value.getDefiningOp<ac::BitsConstantOp>()) {
        auto tree = constant.getValue().getTree();
        auto kind = tree.getAs<StringAttr>("kind");
        auto literal = tree.getAs<DictionaryAttr>("value");
        auto number = literal ? literal.getAs<ac::MathIntAttr>("value")
                              : ac::MathIntAttr();
        return kind && kind.getValue() == "literal" && number &&
               number.getCanonicalValue() != "0";
      }
      if (auto constant = value.getDefiningOp<ac::BitsConstant4Op>()) {
        StringRef pattern = constant.getPattern();
        return !pattern.contains('x') && !pattern.contains('z') &&
               pattern.contains('1');
      }
      return false;
    };
    for (const HardwareEmissionModulePlan &module : plan.modules) {
      module.diagnosticAnchor->walk([&](Operation *operation) {
        if (auto constant = dyn_cast<ac::BitsConstant4Op>(operation)) {
          StringRef pattern = constant.getPattern();
          plan.needsFourState |= pattern.contains('x') || pattern.contains('z');
        }
        if (operation->hasAttr("ac.verification_observer"))
          plan.needsFourState = true;
        if (auto compare = dyn_cast<ac::BitsCompareOp>(operation))
          plan.needsFourState |= compare.getPredicate() == "case_eq";
        if (auto binary = dyn_cast<ac::BitsBinaryOp>(operation))
          if ((binary.getOpcode() == "udiv" || binary.getOpcode() == "urem") &&
              !knownNonzeroLiteral(binary.getRhs()))
            plan.needsFourState = true;
        if (isa<ac::TableGetOp>(operation))
          plan.needsFourState = true;
      });
      for (const HardwareEmissionChildPlan &child : module.children)
        if (child.kind == HardwareEmissionChildKind::Primitive &&
            child.primitiveKind != "dff" && child.primitiveKind != "dffe")
          plan.needsFourState = true;
    }
  }

  ModuleOp package;
  ac::HardwareAnalysis &analysis;
  HardwareEmissionPlan plan;
  const ac::HardwareSourceCheckPlan &checkPlan;
  llvm::StringMap<unsigned> moduleIndices;
  llvm::DenseMap<Operation *, unsigned> definitionIndices;
  llvm::DenseMap<Operation *, HardwareEmissionDefinitionId> definitionIds;
  llvm::DenseSet<unsigned> activeModules;
  unsigned rootReachableModuleCount = 0;
  HardwareEmissionStaticQueryId nextStaticQueryId = 0;
  HardwareEmissionTypeQueryId nextTypeQueryId = 0;
  LogicalResult valueBuildStatus = success();
};

} // namespace

namespace {
LogicalResult freezeObservationPlan(
    HardwareEmissionPlan &plan,
    const ac::HardwareSourceCheckPlan &sourceChecks,
    std::shared_ptr<const ObservationEmissionPlan> &observationPlan) {
  HardwareEmitContext context(plan, sourceChecks, observationPlan, nullptr);
  if (failed(context.prepareNativeChecks()))
    return failure();
  plan.targetCapabilitiesValidated = true;
  return success();
}
} // namespace

FailureOr<VerifiedHardwareEmissionPlan>
prepareHardwareEmissionPlan(ModuleOp package) {
  if (!package)
    return failure();
  std::string serialized;
  llvm::raw_string_ostream serializedStream(serialized);
  package.print(serializedStream, OpPrintingFlags().enableDebugInfo());
  serializedStream.flush();
  DialectRegistry dialects;
  dialects.insert<ac::ACIRDialect>();
  auto context = std::make_unique<MLIRContext>(dialects);
  context->loadAllAvailableDialects();
  ScopedDiagnosticHandler forwardDiagnostics(
      context.get(), [&](Diagnostic &diagnostic) {
        std::string text;
        llvm::raw_string_ostream out(text);
        diagnostic.print(out);
        out.flush();
        package.emitError() << text;
        return success();
      });
  OwningOpRef<ModuleOp> snapshot =
      parseSourceString<ModuleOp>(serialized, context.get());
  if (!snapshot)
    return package.emitOpError()
           << "hardware emission could not own a private final-ACIR snapshot";
  ac::HardwareAnalysis analysis(*snapshot);
  if (failed(analysis.verify()))
    return failure();
  auto sourceChecks = analysis.getSourceCheckPlan();
  if (failed(sourceChecks))
    return failure();
  auto plan = PlanBuilder(*snapshot, analysis, *sourceChecks).build();
  if (failed(plan) || failed(verifyHardwareEmissionPlan(*plan)))
    return failure();
  auto backendState =
      std::make_unique<VerifiedHardwareEmissionPlan::FrozenBackendState>();
  backendState->sourceChecks = std::move(*sourceChecks);
  {
    // Physical RTL is an optional target capability. Freeze either its
    // projection or its absence now, while analysis is still open; do not let
    // an expected physical-target rejection invalidate logical emission.
    ScopedDiagnosticHandler captureExpectedPhysicalRejection(
        snapshot->getContext(), [&](Diagnostic &diagnostic) {
          std::string text;
          llvm::raw_string_ostream out(text);
          diagnostic.print(out);
          out.flush();
          if (!backendState->physicalProjectionUnavailableReason.empty())
            backendState->physicalProjectionUnavailableReason += "\n";
          backendState->physicalProjectionUnavailableReason += text;
          return success();
        });
    auto physicalProjection =
        analysis.getPhysicalProjection(backendState->sourceChecks);
    if (succeeded(physicalProjection)) {
      backendState->physicalProjection = std::move(*physicalProjection);
      backendState->physicalProjectionUnavailableReason.clear();
    } else if (backendState->physicalProjectionUnavailableReason.empty()) {
      backendState->physicalProjectionUnavailableReason =
          "physical RTL projection is unavailable for this hardware plan";
    }
  }
  if (failed(freezeObservationPlan(*plan, backendState->sourceChecks,
                                   backendState->observationPlan)))
    return failure();
  if (failed(verifyHardwareEmissionPlan(*plan)))
    return failure();
  return VerifiedHardwareEmissionPlan(std::move(context), std::move(snapshot),
                                      std::move(*plan),
                                      std::move(backendState));
}

LogicalResult verifyHardwareEmissionPlan(const HardwareEmissionPlan &plan) {
  if (!plan.package || plan.modules.empty() ||
      plan.rootModule >= plan.modules.size())
    return failure();
  auto error = [&]() -> InFlightDiagnostic {
    return plan.package->emitOpError();
  };
  const HardwareEmissionModulePlan &root = plan.modules[plan.rootModule];
  if (plan.rootInterface.inputNames.size() != root.inputs.size() ||
      plan.rootInterface.outputNames.size() != root.outputs.size() ||
      plan.rootInterface.publicInputCount != root.publicInputCount ||
      plan.rootInterface.publicOutputCount != root.publicOutputCount ||
      plan.rootInterface.defaultInputs != root.rootInputDefaults ||
      plan.rootInterface.clockInput != root.clockInput ||
      plan.rootInterface.resetInput != root.resetInput ||
      plan.rootInterface.closedSystem != root.closedSystem ||
      !llvm::is_sorted(plan.rootInterface.availabilityOutputs) ||
      llvm::adjacent_find(plan.rootInterface.availabilityOutputs) !=
          plan.rootInterface.availabilityOutputs.end())
    return error() << "hardware emission root interface is inconsistent";
  for (auto [index, port] : llvm::enumerate(root.inputs))
    if (plan.rootInterface.inputNames[index] != port.name)
      return error() << "hardware emission root input inventory is invalid";
  for (auto [index, port] : llvm::enumerate(root.outputs))
    if (plan.rootInterface.outputNames[index] != port.name)
      return error() << "hardware emission root output inventory is invalid";
  auto validQueue = [](const HardwareEmissionQueuePlan &queue,
                       bool concrete) {
    if (!queue.depthExpression || !queue.availabilityLatencyExpression)
      return false;
    if (!concrete)
      return true;
    return queue.depth != 0 && queue.tokenCardinality != 0 &&
           queue.tokenElementWidth != 0;
  };
  llvm::DenseSet<Operation *> definitionAnchorSet;
  for (Operation *anchor : plan.definitionAnchors)
    if (!anchor || !definitionAnchorSet.insert(anchor).second)
      return error()
             << "hardware emission definition identity table is invalid";
  llvm::DenseSet<HardwareEmissionOccurrenceId> occurrenceIds;
  llvm::DenseSet<HardwareEmissionStaticQueryId> staticQueryIds;
  llvm::DenseSet<HardwareEmissionTypeQueryId> typeQueryIds;
  auto recordQueries = [&](ArrayRef<HardwareEmissionStaticPlan> statics,
                           ArrayRef<HardwareEmissionTypeQueryPlan> types) {
    for (const HardwareEmissionStaticPlan &query : statics)
      if (!query.diagnosticSource || !query.value ||
          !staticQueryIds.insert(query.id).second)
        return false;
    for (const HardwareEmissionTypeQueryPlan &query : types)
      if (!query.diagnosticSource || !query.resolved ||
          !typeQueryIds.insert(query.id).second)
        return false;
    return true;
  };
  if (!recordQueries({}, plan.packageTypes))
    return error() << "hardware emission package query identity is invalid";
  if (plan.entry.implementationDefinition !=
          plan.modules[plan.rootModule].key.definition ||
      plan.entry.logicalDefinition.empty() || !plan.entry.sourceOwner)
    return error() << "hardware emission plan has an invalid entry";
  llvm::DenseSet<Operation *> definitions;
  for (const HardwareEmissionDefinitionPlan &definition : plan.definitions) {
    const HardwareEmissionDefinitionNames &names = definition.names;
    if (definition.id >= plan.definitionAnchors.size() ||
        plan.definitionAnchors[definition.id] != definition.diagnosticAnchor ||
        !definition.diagnosticAnchor || definition.symbol.empty() ||
        !definition.sourceOwner ||
        !definitions.insert(definition.diagnosticAnchor).second ||
        names.cppQualified.empty() || names.cppClass.empty() ||
        names.cppFamilyClass.empty() || names.cppFamilyQualified.empty() ||
        names.rtl.empty() || names.header.empty() || names.source.empty() ||
        names.verilog.empty())
      return error()
             << "hardware emission plan has an invalid definition inventory";
    for (const HardwareEmissionOccurrenceBinding &child :
         definition.templateChildren)
      if (child.id >= plan.occurrenceAnchors.size() ||
          plan.occurrenceAnchors[child.id] != child.diagnosticAnchor ||
          !child.diagnosticAnchor || child.instanceName.empty() ||
          child.bindings.owner >= plan.definitionAnchors.size() ||
          (child.queue && !validQueue(*child.queue, false)) ||
          !occurrenceIds.insert(child.id).second)
        return error()
               << "hardware emission plan has an invalid template occurrence";
    for (Operation *operation : definition.templateSchedule)
      if (!operation || operation->getParentOfType<ac::ModuleOp>() !=
                            definition.diagnosticAnchor)
        return error()
               << "hardware emission plan has an invalid template schedule";
    auto module = cast<ac::ModuleOp>(definition.diagnosticAnchor);
    if (definition.outputValues.size() !=
            module.getFunctionType().getNumResults() ||
        llvm::any_of(definition.outputValues, [&](Value value) {
          return !value || value.getParentRegion() != &module.getBody();
        }))
      return error()
             << "hardware emission plan has an invalid definition output";
    if (!recordQueries(definition.staticValues, definition.types))
      return error() << "hardware emission definition query identity is invalid";
    for (Operation *child : definition.independentWorkChildren)
      if (!child || child->getParentOfType<ac::ModuleOp>() !=
                        definition.diagnosticAnchor)
        return error()
               << "hardware emission plan has an invalid work partition";
  }
  llvm::DenseSet<Operation *> primitives;
  llvm::StringSet<> primitiveSymbols;
  for (const HardwareEmissionPrimitivePlan &primitive : plan.primitives)
    if (primitive.id >= plan.definitionAnchors.size() ||
        plan.definitionAnchors[primitive.id] != primitive.diagnosticAnchor ||
        !primitive.diagnosticAnchor || primitive.symbol.empty() ||
        primitive.kind.empty() ||
        !primitives.insert(primitive.diagnosticAnchor).second ||
        !primitiveSymbols.insert(primitive.symbol).second)
      return error()
             << "hardware emission plan has an invalid primitive inventory";
  llvm::StringSet<> standardSources;
  for (StringRef source : plan.standardVerilogSources)
    if (!source.starts_with("include/verilog/") ||
        !standardSources.insert(source).second)
      return error()
             << "hardware emission plan has invalid standard RTL sources";
  llvm::DenseSet<unsigned> ownedModules;
  for (const HardwareEmissionModulePlan &module : plan.modules) {
    if (module.concreteBindings.owner >= plan.definitionAnchors.size() ||
        plan.definitionAnchors[module.concreteBindings.owner] !=
            module.diagnosticAnchor ||
        !module.diagnosticAnchor || module.key.definition.empty() ||
        !module.sourceOwner)
      return error() << "hardware emission plan has an invalid module";
    if (module.publicInputCount > module.inputs.size() ||
        module.publicOutputCount > module.outputs.size() ||
        module.rootInputDefaults.size() !=
            module.inputs.size() - module.publicInputCount)
      return error() << "hardware emission plan has an invalid port boundary";
    if (module.clockInput && *module.clockInput >= module.inputs.size())
      return error() << "hardware emission plan clock input is out of range";
    if (module.resetInput && *module.resetInput >= module.inputs.size())
      return error() << "hardware emission plan reset input is out of range";
    if (module.clockInput && module.resetInput &&
        *module.clockInput == *module.resetInput)
      return error() << "hardware emission clock and reset inputs alias";
    for (auto [valueIndex, value] : llvm::enumerate(module.values))
      if (value.id != valueIndex || !value.source || !value.type.resolvedType ||
          value.type.packedWidth == 0 || value.type.canonicalType.empty())
        return error() << "hardware emission plan has an invalid value";
    auto validValue = [&](HardwareEmissionValueId value) {
      return value < module.values.size();
    };
    llvm::DenseMap<Value, HardwareEmissionValueId> valueIds;
    for (const HardwareEmissionValuePlan &value : module.values)
      if (!valueIds.try_emplace(value.source, value.id).second)
        return error() << "hardware emission plan has duplicate value sources";
    for (const HardwareEmissionPortPlan &port : module.inputs)
      if (port.name.empty() || !validValue(port.value) ||
          !(port.type == module.values[port.value].type))
        return error() << "hardware emission plan has an invalid input port";
    for (const HardwareEmissionPortPlan &port : module.outputs) {
      if (port.name.empty() || !validValue(port.value) ||
          !(port.type == module.values[port.value].type) ||
          (port.availabilityOutput &&
           *port.availabilityOutput >= module.outputs.size()))
        return error() << "hardware emission plan has an invalid output port";
      if (port.availabilityOutput) {
        const HardwareEmissionPortPlan &companion =
            module.outputs[*port.availabilityOutput];
        if (*port.availabilityOutput < module.publicOutputCount ||
            companion.type.packedWidth != 1)
          return error()
                 << "hardware emission plan has an invalid availability port";
      }
    }
    for (auto [stepIndex, step] : llvm::enumerate(module.schedule)) {
      if (step.id != stepIndex || !step.diagnosticAnchor ||
          step.opcode.empty() ||
          step.diagnosticAnchor->getParentOfType<ac::ModuleOp>() !=
              module.diagnosticAnchor ||
          llvm::any_of(step.inputs,
                       [&](auto value) { return !validValue(value); }) ||
          llvm::any_of(step.outputs,
                       [&](auto value) { return !validValue(value); }))
        return error() << "hardware emission plan has an invalid schedule";
    }
    for (const HardwareEmissionDependencyPlan &dependency :
         module.dependencies) {
      auto validEndpoint = [&](const HardwareEmissionDependencyEndpoint &item) {
        auto found = valueIds.find(item.value);
        return item.value && validValue(item.id) && found != valueIds.end() &&
               found->second == item.id;
      };
      if (!validEndpoint(dependency.output) ||
          llvm::any_of(dependency.inputs, [&](const auto &input) {
            return !validEndpoint(input);
          }))
        return error()
               << "hardware emission plan has an invalid dependency endpoint";
    }
    if (!recordQueries(module.staticValues, module.types))
      return error() << "hardware emission module query identity is invalid";
    for (const HardwareEmissionChildPlan &child : module.children) {
      if (child.id >= plan.occurrenceAnchors.size() ||
          plan.occurrenceAnchors[child.id] != child.diagnosticAnchor ||
          !child.diagnosticAnchor || child.instanceName.empty() ||
          child.callee.empty() || !occurrenceIds.insert(child.id).second ||
          !recordQueries(child.staticValues, child.types))
        return error() << "hardware emission plan has an invalid child";
      if (child.kind == HardwareEmissionChildKind::Module) {
        if (!child.moduleIndex || *child.moduleIndex >= plan.modules.size() ||
            !child.primitiveKind.empty() ||
            !(child.bindings == plan.modules[*child.moduleIndex].key))
          return error() << "hardware emission module child is unresolved";
      } else if (child.primitiveKind.empty() || child.moduleIndex ||
                 (child.queue && !validQueue(*child.queue, true))) {
        return error() << "hardware emission primitive child is unresolved";
      }
    }
  }
  llvm::DenseSet<unsigned> ownedDefinitions;
  llvm::DenseSet<unsigned> reachableDefinitions;
  for (unsigned definition : plan.reachableDefinitions)
    if (definition >= plan.definitions.size() ||
        !reachableDefinitions.insert(definition).second)
      return error()
             << "hardware emission reachable definition inventory is invalid";
  if (plan.reachableDefinitions.empty())
    return error() << "hardware emission root closure is empty";
  for (const HardwareEmissionSourceUnitPlan &unit : plan.sourceUnits) {
    if (!unit.sourceOwner || unit.cppHeaderPath.empty())
      return error() << "hardware emission plan has an invalid source unit";
    for (unsigned definition : unit.definitionIndices)
      if (definition >= plan.definitions.size() ||
          !ownedDefinitions.insert(definition).second ||
          plan.definitions[definition].sourceOwner != unit.sourceOwner ||
          plan.definitions[definition].names.header != unit.cppHeaderPath ||
          plan.definitions[definition].names.source != unit.cppSourcePath ||
          plan.definitions[definition].names.verilog != unit.verilogPath)
        return error()
               << "hardware emission definition ownership is inconsistent";
    llvm::DenseSet<unsigned> scheduled;
    for (unsigned definition : unit.emissionOrder)
      if (definition >= plan.definitions.size() ||
          plan.definitions[definition].sourceOwner != unit.sourceOwner ||
          !scheduled.insert(definition).second)
        return error() << "hardware emission definition order is inconsistent";
    if (scheduled.size() != unit.definitionIndices.size() ||
        llvm::any_of(unit.definitionIndices, [&](unsigned definition) {
          return !scheduled.contains(definition);
        }))
      return error() << "hardware emission definition order is incomplete";
    if (!llvm::is_sorted(unit.cppIncludes) ||
        llvm::adjacent_find(unit.cppIncludes) != unit.cppIncludes.end() ||
        llvm::is_contained(unit.cppIncludes, unit.cppHeaderPath))
      return error()
             << "hardware emission C++ include inventory is inconsistent";
    llvm::StringSet<> expectedIncludes;
    for (unsigned definition : unit.definitionIndices)
      for (const HardwareEmissionOccurrenceBinding &child :
           plan.definitions[definition].templateChildren) {
        Operation *anchor = plan.definitionAnchors[child.bindings.owner];
        auto found = llvm::find_if(plan.definitions, [&](const auto &candidate) {
          return candidate.diagnosticAnchor == anchor;
        });
        if (found != plan.definitions.end() &&
            found->names.header != unit.cppHeaderPath)
          expectedIncludes.insert(found->names.header);
      }
    if (expectedIncludes.size() != unit.cppIncludes.size() ||
        llvm::any_of(unit.cppIncludes, [&](StringRef include) {
          return !expectedIncludes.contains(include);
        }))
      return error()
             << "hardware emission C++ include inventory is incomplete";
    for (unsigned module : unit.moduleIndices)
      if (module >= plan.modules.size() ||
          !ownedModules.insert(module).second ||
          plan.modules[module].sourceOwner != unit.sourceOwner)
        return error() << "hardware emission source ownership is inconsistent";
  }
  if (ownedModules.size() != plan.modules.size())
    return error() << "hardware emission plan has an unowned module";
  if (ownedDefinitions.size() != plan.definitions.size())
    return error() << "hardware emission plan has an unowned definition";
  if (plan.sourceMaps.size() != plan.sourceUnits.size())
    return error() << "hardware emission plan has incomplete source maps";
  for (auto [index, sourceMap] : llvm::enumerate(plan.sourceMaps))
    if (sourceMap.sourceOwner != plan.sourceUnits[index].sourceOwner ||
        failed(verifyFinalSourceMapPlan(
            sourceMap, [&] { return plan.package->emitOpError(); })))
      return error() << "hardware emission source-map plan is inconsistent";
  for (auto [index, observation] : llvm::enumerate(plan.observations))
    if (observation.id != index ||
        observation.moduleIndex >= plan.modules.size() ||
        observation.kind.empty() || observation.identity.empty() ||
        llvm::any_of(observation.payloadWidths,
                     [](uint64_t width) { return width == 0; }))
      return error() << "hardware emission plan has an invalid observation";
  if (occurrenceIds.size() != plan.occurrenceAnchors.size())
    return error() << "hardware emission occurrence identity is incomplete";
  for (HardwareEmissionOccurrenceId id = 0;
       id < plan.occurrenceAnchors.size(); ++id)
    if (!occurrenceIds.contains(id))
      return error() << "hardware emission occurrence identity has a gap";
  for (HardwareEmissionStaticQueryId id = 0; id < staticQueryIds.size(); ++id)
    if (!staticQueryIds.contains(id))
      return error() << "hardware emission static query identity has a gap";
  for (HardwareEmissionTypeQueryId id = 0; id < typeQueryIds.size(); ++id)
    if (!typeQueryIds.contains(id))
      return error() << "hardware emission type query identity has a gap";
  return success();
}

} // namespace acir::compiler
