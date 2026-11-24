#include "HardwareEmitCommon.h"
#include "../Dialect/ACIR/ACIRSourceContracts.h"
#include "../Dialect/ACIR/HardwareSourceChecks.h"
#include "FinalCppNames.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <limits>
#include <tuple>
using namespace mlir;
namespace acir::compiler {
std::string jsonString(StringRef value) {
  std::string text;
  llvm::raw_string_ostream out(text);
  out << llvm::json::Value(value);
  return text;
}
std::string displayLiteral(StringRef value) {
  std::string text;
  text.reserve(value.size());
  for (char character : value) {
    switch (character) {
    case '\\':
      text += "\\\\";
      break;
    case '"':
      text += "\\\"";
      break;
    case '\n':
      text += "\\n";
      break;
    case '\r':
      text += "\\r";
      break;
    case '\t':
      text += "\\t";
      break;
    case '%':
      text += "%%";
      break;
    default:
      text += character;
      break;
    }
  }
  return text;
}
namespace {
FailureOr<std::string> identifier(StringRef raw, Operation *site) {
  auto value = legalizeIdentifier(raw, [&] { return site->emitOpError(); });
  if (failed(value))
    return failure();
  return *value;
}
FailureOr<std::string> qualified(StringRef raw, Operation *site,
                                 StringRef separator) {
  SmallVector<StringRef> parts;
  raw.split(parts, '.', -1, true);
  std::string output;
  for (auto part : parts) {
    if (part.empty())
      return site->emitOpError() << "empty hardware symbol component";
    auto item = identifier(part, site);
    if (failed(item))
      return failure();
    if (!output.empty())
      output += separator.str();
    output += *item;
  }
  return output;
}
bool closedStatic(Attribute attribute, const ac::HardwareBindings &bindings,
                  const HardwareEmitContext *context = nullptr,
                  Operation *site = nullptr) {
  if (auto expression = dyn_cast_or_null<ac::StaticExprAttr>(attribute))
    return closedStatic(expression.getTree(), bindings, context, site);
  if (auto dictionary = dyn_cast_or_null<DictionaryAttr>(attribute)) {
    auto kind = dictionary.getAs<StringAttr>("kind");
    if (context && kind && kind.getValue() == "select") {
      auto condition = dictionary.getAs<ac::StaticExprAttr>("condition");
      if (closedStatic(condition, bindings, context, site)) {
        auto value = context->evaluatedStatic(condition, bindings);
        if (succeeded(value))
          if (auto boolean = dyn_cast<BoolAttr>(*value))
            return closedStatic(dictionary.getAs<ac::StaticExprAttr>(
                                    boolean.getValue() ? "yes" : "no"),
                                bindings, context, site);
      }
    }
    if (context && kind && kind.getValue() == "binary") {
      auto opcode = dictionary.getAs<StringAttr>("operator");
      if (opcode &&
          (opcode.getValue() == "and_bool" || opcode.getValue() == "or_bool")) {
        auto lhs = dictionary.getAs<ac::StaticExprAttr>("lhs");
        if (closedStatic(lhs, bindings, context, site)) {
          auto value = context->evaluatedStatic(lhs, bindings);
          if (succeeded(value))
            if (auto boolean = dyn_cast<BoolAttr>(*value))
              if ((opcode.getValue() == "and_bool" && !boolean.getValue()) ||
                  (opcode.getValue() == "or_bool" && boolean.getValue()))
                return true;
        }
      }
    }
    if (kind && kind.getValue() == "reference") {
      auto ref = dictionary.getAs<DictionaryAttr>("ref");
      if (!bindings.owner || !ref || !ref.getAs<FlatSymbolRefAttr>("owner") ||
          ref.getAs<FlatSymbolRefAttr>("owner").getValue() !=
              SymbolTable::getSymbolName(bindings.owner).getValue())
        return false;
      auto name = ref.getAs<StringAttr>("name");
      auto found = name ? bindings.integers.find(name.getValue())
                        : bindings.integers.end();
      return found != bindings.integers.end() &&
             closedStatic(found->second, {}, context, site);
    }
    if (kind && kind.getValue() == "type_width") {
      auto type = dictionary.getAs<TypeAttr>("type");
      if (!type)
        return false;
      Type value = type.getValue();
      if (auto formal = dyn_cast<ac::TypeParamType>(value)) {
        if (!bindings.owner ||
            formal.getOwner().getValue() !=
                SymbolTable::getSymbolName(bindings.owner).getValue())
          return false;
        auto found = bindings.types.find(formal.getName().getValue());
        if (found == bindings.types.end() ||
            isa<ac::TypeParamType>(found->second))
          return false;
        value = found->second;
      }
      if (auto bits = dyn_cast<ac::BitsType>(value))
        return closedStatic(bits.getWidth(), bindings, context, site);
      return isa<ac::StructType, ac::EnumType>(value);
    }
    for (auto field : dictionary)
      if (!closedStatic(field.getValue(), bindings, context, site))
        return false;
  }
  if (auto array = dyn_cast_or_null<ArrayAttr>(attribute))
    for (auto item : array)
      if (!closedStatic(item, bindings, context, site))
        return false;
  return true;
}
StringRef arithmeticHelper(const HardwareEmitContext &context, StringRef opcode,
                           bool rtl) {
  bool floorDiv = opcode == "floordiv";
  if (!rtl)
    return floorDiv ? "pyc_floordiv" : "pyc_mod";
  if (auto *package = context.rtlPackage)
    (floorDiv ? package->floorDiv : package->mod) = true;
  return floorDiv ? "pycircuit_types::pyc_floordiv" : "pycircuit_types::pyc_mod";
}
// True when this node renders as a signed arithmetic helper call, whose result
// is already `logic signed [63:0]`.
bool arithmeticCall(DictionaryAttr tree) {
  auto kind = tree ? tree.getAs<StringAttr>("kind") : StringAttr();
  if (!kind || kind.getValue() != "binary")
    return false;
  auto opcode = tree.getAs<StringAttr>("operator");
  return opcode && (opcode.getValue() == "floordiv" ||
                    opcode.getValue() == "mod");
}
FailureOr<std::string>
staticTree(HardwareEmitContext const &context, DictionaryAttr tree,
           Operation *site, bool rtl,
           const ac::HardwareBindings *bindings = nullptr) {
  auto kind = tree.getAs<StringAttr>("kind");
  if (!kind)
    return site->emitOpError() << "static expression has no kind";
  StringRef tag = kind.getValue();
  ac::HardwareBindings empty;
  auto &scope = bindings ? *bindings : empty;
  if (tag == "literal") {
    auto data = tree.getAs<DictionaryAttr>("value");
    auto value = data ? data.get("value") : Attribute();
    if (auto integer = dyn_cast_or_null<ac::MathIntAttr>(value))
      return integer.getCanonicalValue().str();
    if (auto boolean = dyn_cast_or_null<BoolAttr>(value))
      return std::string(boolean.getValue() ? "1" : "0");
    return site->emitOpError() << "unsupported static literal";
  }
  // A frozen binding already contains the authoritative value for a direct
  // formal reference.  Render that binding below; query-table lookup is for
  // derived closed expressions.
  if (tag != "reference" && closedStatic(tree, scope, &context, site)) {
    auto value = context.evaluatedStatic(
        ac::StaticExprAttr::get(tree.getContext(), tree), scope);
    if (failed(value))
      return failure();
    if (auto integer = dyn_cast<ac::MathIntAttr>(*value))
      return integer.getCanonicalValue().str();
    if (auto boolean = dyn_cast<BoolAttr>(*value))
      return std::string(boolean.getValue() ? "1" : "0");
  }

  if (tag == "reference") {
    auto ref = tree.getAs<DictionaryAttr>("ref");
    auto name = ref ? ref.getAs<StringAttr>("name") : StringAttr();
    if (!name)
      return site->emitOpError() << "static reference has no name";
    if (bindings && bindings->owner && ref.getAs<FlatSymbolRefAttr>("owner") &&
        ref.getAs<FlatSymbolRefAttr>("owner").getValue() ==
            SymbolTable::getSymbolName(bindings->owner).getValue()) {
      auto found = bindings->integers.find(name.getValue());
      if (found != bindings->integers.end()) {
        if (auto integer = dyn_cast<ac::MathIntAttr>(found->second))
          return integer.getCanonicalValue().str();
        if (auto expression = dyn_cast<ac::StaticExprAttr>(found->second))
          return staticTree(context, expression.getTree(), site, rtl, bindings);
      }
    }
    return rtl ? context.rtlFormalName(name.getValue(), site)
               : context.cppFormalName(name.getValue(), site);
  }
  if (tag == "type_width") {
    auto type = tree.getAs<TypeAttr>("type");
    if (!type)
      return site->emitOpError() << "type_width has no type";
    auto payload = type.getValue();
    if (bindings) {
      auto resolved = context.resolvedType(payload, *bindings);
      if (failed(resolved))
        return failure();
      payload = *resolved;
    }
    auto rendered = rtl ? context.rtlType(payload, site)
                        : (bindings ? context.cppType(payload, *bindings, site)
                                    : context.cppType(payload, site));
    if (failed(rendered))
      return failure();
    return rtl ? ("$bits(" + *rendered + ")")
               : ("gfsim::hardware_traits<" + *rendered + ">::width");
  }
  if (tag == "select") {
    auto a = staticTree(context,
                        tree.getAs<ac::StaticExprAttr>("condition").getTree(),
                        site, rtl, bindings);
    auto b =
        staticTree(context, tree.getAs<ac::StaticExprAttr>("yes").getTree(),
                   site, rtl, bindings);
    auto c = staticTree(context, tree.getAs<ac::StaticExprAttr>("no").getTree(),
                        site, rtl, bindings);
    if (failed(a) || failed(b) || failed(c))
      return failure();
    return "((" + *a + ") ? (" + *b + ") : (" + *c + "))";
  }
  auto opcode = tree.getAs<StringAttr>("operator");
  if (!opcode)
    return site->emitOpError() << "static operator is absent";
  auto symbol = llvm::StringSwitch<StringRef>(opcode.getValue())
                    .Case("neg", "-")
                    .Case("invert", "~")
                    .Case("not", "!")
                    .Case("add", "+")
                    .Case("sub", "-")
                    .Case("mul", "*")
                    .Case("floordiv", "/")
                    .Case("mod", "%")
                    .Case("and_bits", "&")
                    .Case("or_bits", "|")
                    .Case("xor_bits", "^")
                    .Case("shl", "<<")
                    .Case("shr", ">>")
                    .Case("eq", "==")
                    .Case("ne", "!=")
                    .Case("lt", "<")
                    .Case("le", "<=")
                    .Case("gt", ">")
                    .Case("ge", ">=")
                    .Case("and_bool", "&&")
                    .Case("or_bool", "||")
                    .Default("");
  if (opcode.getValue() == "to_int")
    symbol = "+";
  if (symbol.empty())
    return site->emitOpError() << "unsupported static operator";
  if (tag == "unary") {
    auto operandTree = tree.getAs<ac::StaticExprAttr>("operand").getTree();
    auto a = staticTree(context, operandTree, site, rtl, bindings);
    if (failed(a))
      return failure();
    if (opcode.getValue() == "to_int")
      return *a;
    auto operand =
        rtl ? (arithmeticCall(operandTree)
                   ? *a
                   : "$signed(64'( " + *a + " ))")
            : "std::int64_t(" + *a + ")";
    return "(" + symbol.str() + "(" + operand + "))";
  }
  if (tag != "binary")
    return site->emitOpError() << "unsupported static expression";
  auto lhsTree = tree.getAs<ac::StaticExprAttr>("lhs").getTree();
  auto rhsTree = tree.getAs<ac::StaticExprAttr>("rhs").getTree();
  auto a = staticTree(context, lhsTree, site, rtl, bindings);
  auto b = staticTree(context, rhsTree, site, rtl, bindings);
  if (failed(a) || failed(b))
    return failure();
  auto signedValue = [&](const std::string &value, DictionaryAttr node) {
    if (!rtl)
      return "std::int64_t(" + value + ")";
    // The helpers already return `logic signed [63:0]`, so the cast is
    // redundant for them. Wrapping the call in a size cast would make that
    // cast's base a function call, which some consumers reject while
    // elaborating the enclosing width.
    if (arithmeticCall(node))
      return value;
    return "$signed(64'(" + value + "))";
  };
  if (opcode.getValue() == "floordiv" || opcode.getValue() == "mod")
    return arithmeticHelper(context, opcode.getValue(), rtl).str() + "(" +
           signedValue(*a, lhsTree) + ", " + signedValue(*b, rhsTree) + ")";
  if (opcode.getValue() == "shl")
    return rtl ? "(" + signedValue(*a, lhsTree) + " <<< " +
                     signedValue(*b, rhsTree) + ")"
               : "(" + signedValue(*a, lhsTree) + " * (std::int64_t{1} << " +
                     signedValue(*b, rhsTree) + "))";
  if (opcode.getValue() == "shr")
    return (rtl ? "(" + signedValue(*a, lhsTree) + " >>> " +
                      signedValue(*b, rhsTree) + ")"
                : "pyc_floordiv(" + signedValue(*a, lhsTree) +
                      ", (std::int64_t{1} << " + signedValue(*b, rhsTree) +
                      "))");
  return "(" + signedValue(*a, lhsTree) + " " + symbol.str() + " " +
         signedValue(*b, rhsTree) + ")";
}

// ---------------------------------------------------------------------------
// Native observation plan.
//
// `ac.observe` is already produced by the source importers; this plan is the
// only thing native emission adds. It turns every *occurrence* of every
// `ac.observe` site into one `gfsim::ObservationDescriptor` and records the
// slot arithmetic the generated `Work()` needs.
//
// Layout. For a module definition D let S(D) be the number of descriptors one
// lane of D contributes and W(D) = S(D) + sum(shape(c) * W(c)) over its child
// occurrences, i.e. the number of descriptors one lane of D contributes
// together with its whole instance subtree. An occurrence O with L lanes owns
// the contiguous block [base, base + L*W(D)). Lane `l` of O stages site `s`
// into `base + l*S(D) + s`, and the i-th child occurrence of D starts at
// `base + L * (S(D) + sum over j < i of shape_j * W(D_j))`. Every occurrence of
// a definition therefore shares one generated code path, and the parent hands
// its children a runtime base; no occurrence identity is baked into a shared
// class.
//
// Descriptor order. `ObservationSlots::Configure` requires the table to be
// strictly increasing in (ownerKey, registrationKey, siteKey, stableOrdinal,
// kind). Keys are assigned in exactly the block order above: `ownerKey`
// advances per (occurrence, lane) and `stableOrdinal` is the slot index, so the
// generated order is already sorted and stays globally unique.
// ---------------------------------------------------------------------------
FailureOr<ac::ModuleOp> observationRoot(HardwareEmitContext &context) {
  auto root =
      context.plan.rootModule < context.plan.modules.size()
          ? dyn_cast_or_null<ac::ModuleOp>(
                context.plan.modules[context.plan.rootModule].diagnosticAnchor)
          : ac::ModuleOp();
  if (!root)
    return context.emitPlanError() << "root module is unresolved";
  return root;
}
FailureOr<ObservationEmissionPlan>
buildObservationPlan(HardwareEmitContext &context) {
  ObservationEmissionPlan plan;
  // Full hardware validation is performed by prepareImpl before emission.
  // Without observations, no descriptor geometry needs to be instantiated.
  if (context.plan.observations.empty())
    return plan;
  auto root = observationRoot(context);
  if (failed(root))
    return failure();
  llvm::DenseMap<Attribute, uint64_t> registrationKeys;
  llvm::DenseMap<Attribute, uint64_t> identityKeys;
  uint64_t nextSiteKey = 0;
  llvm::SetVector<Operation *> reached;
  std::function<LogicalResult(ac::ModuleOp, const ac::HardwareBindings &)>
      collect = [&](ac::ModuleOp definition,
                    const ac::HardwareBindings &bindings) -> LogicalResult {
    if (!reached.insert(definition))
      return success();
    if (failed(ac::detail::verifyModuleObservationIdentities(definition)))
      return failure();
    struct Group {
      ac::SourceObserveOp observe;
      uint64_t registrationRank, ordinal, identityRank, valueCount;
    };
    SmallVector<Group> groups;
    SmallVector<ObservationEmissionSite> sites;
    for (Operation &op : definition.getBody().front()) {
      auto observe = dyn_cast<ac::SourceObserveOp>(op);
      if (!observe)
        continue;
      auto identity = op.getAttrOfType<DictionaryAttr>("ac.observation_id");
      auto key = [&](Attribute value,
                     llvm::DenseMap<Attribute, uint64_t> &table) {
        auto found = table.find(value);
        if (found == table.end())
          found = table.try_emplace(value, table.size()).first;
        return found->second;
      };
      uint64_t registrationKey =
          key(identity.get("registration"), registrationKeys);
      uint64_t identityKey = key(identity, identityKeys);
      groups.push_back({observe, registrationKey,
                        static_cast<uint64_t>(
                            identity.getAs<IntegerAttr>("ordinal").getInt()),
                        identityKey,
                        std::max<std::size_t>(1, observe.getValues().size())});
    }
    llvm::sort(groups, [](const Group &a, const Group &b) {
      return std::tie(a.registrationRank, a.ordinal, a.identityRank) <
             std::tie(b.registrationRank, b.ordinal, b.identityRank);
    });
    for (const auto &group : groups) {
      auto observe = group.observe;
      auto frozen = llvm::find_if(
          context.plan.observations,
          [&](const HardwareEmissionObservationPlan &observation) {
            return observation.diagnosticAnchor == observe;
          });
      if (frozen == context.plan.observations.end())
        return observe.emitOpError()
               << "observation is absent from the frozen emission plan";
      uint64_t siteKey = nextSiteKey++;
      // A site with no hardware value operand still publishes once, which is
      // the same placeholder rule the source importer applies to a
      // literal-only `log`.
      bool gauge = observe.getKind() == "report";
      auto identity = observe->getAttrOfType<DictionaryAttr>("ac.observation_id");
      auto registration =
          emissionMetadataJson(identity.get("registration"), observe);
      auto site = observationSiteMetadataJson(observe);
      auto spec = emissionMetadataJson(observe.getSpec(), observe);
      if (failed(registration) || failed(site) || failed(spec))
        return failure();
      auto report = observe.getSpec().getAs<StringAttr>("name");
      for (uint64_t index = 0; index < group.valueCount; ++index) {
        ObservationEmissionSite emissionSite{
            frozen->id, observe, static_cast<unsigned>(index),
            index < observe.getValues().size(), gauge,
            group.registrationRank, siteKey};
        emissionSite.kind = observe.getKind().str();
        emissionSite.registrationJson = *registration;
        emissionSite.siteJson = *site;
        emissionSite.specJson = *spec;
        emissionSite.reportName = report ? report.getValue().str() : "";
        emissionSite.valueCount = group.valueCount;
        sites.push_back(std::move(emissionSite));
      }
    }
    plan.definitions[definition] = std::move(sites);
    const auto &parent = bindings;
    for (Operation &op : definition.getBody().front()) {
      if (!isa<ac::InstanceOp, ac::CollectionOp>(op))
        continue;
      auto *callee = isa<ac::InstanceOp>(op)
                         ? context.resolveCallee(cast<ac::InstanceOp>(op))
                         : context.resolveCallee(cast<ac::CollectionOp>(op));
      auto child = dyn_cast_or_null<ac::ModuleOp>(callee);
      uint64_t shape = 1;
      if (auto collection = dyn_cast<ac::CollectionOp>(op)) {
        auto extents = context.occurrenceShape(&op, parent);
        if (failed(extents))
          return failure();
        for (uint64_t extent : *extents) {
          if (extent != 0 &&
              shape > std::numeric_limits<uint64_t>::max() / extent)
            return op.emitOpError()
                   << "native observation lane count overflows";
          shape *= extent;
        }
      }
      auto bindings = context.bindOccurrence(&op, parent);
      if (failed(bindings))
        return failure();
      auto planned = context.occurrenceId(&op, parent);
      if (failed(planned))
        return failure();
      auto *site =
          &plan.children[definition].emplace_back(ObservationEmissionChild{
              *planned, &op, child, shape});
      (void)site;
      if (child &&
          failed(collect(child, cast<ac::HardwareBindings>(*bindings))))
        return failure();
    }
    return success();
  };
  auto rootBindings = context.rootBindings();
  if (failed(rootBindings) || failed(collect(*root, *rootBindings)))
    return failure();
  llvm::DenseSet<Operation *> active;
  std::function<FailureOr<uint64_t>(Operation *)> width =
      [&](Operation *definition) -> FailureOr<uint64_t> {
    if (!definition)
      return uint64_t{0};
    if (auto found = plan.subtreeWidth.find(definition);
        found != plan.subtreeWidth.end())
      return found->second;
    if (!active.insert(definition).second)
      return definition->emitOpError()
             << "recursive instantiation is not supported by native "
                "observation emission";
    uint64_t total = plan.definitions[definition].size();
    for (const ObservationEmissionChild &child : plan.children[definition]) {
      auto nested = width(child.definition);
      if (failed(nested))
        return failure();
      if (child.shape != 0 &&
          *nested > std::numeric_limits<uint64_t>::max() / child.shape)
        return definition->emitOpError()
               << "native observation descriptor count overflows";
      uint64_t block = child.shape * *nested;
      if (block > std::numeric_limits<uint64_t>::max() - total)
        return definition->emitOpError()
               << "native observation descriptor count overflows";
      total += block;
    }
    active.erase(definition);
    plan.subtreeWidth[definition] = total;
    return total;
  };
  if (failed(width(*root)))
    return failure();
  uint64_t ownerKey = 0;
  using InstancePath = SmallVector<ObservationEmissionPathEntry>;
  std::function<LogicalResult(ac::ModuleOp, ArrayRef<InstancePath>,
                              const ac::HardwareBindings &)>
      layout = [&](ac::ModuleOp definition, ArrayRef<InstancePath> paths,
                   const ac::HardwareBindings &bindings) -> LogicalResult {
    uint64_t lanes = paths.size();
    auto sites = plan.definitions.find(definition);
    uint64_t units = sites == plan.definitions.end() ? 0 : sites->second.size();
    llvm::DenseMap<Operation *, SmallVector<ac::InstanceOp>> logicalSuffixes;
    if (sites != plan.definitions.end())
      for (const auto &site : sites->second) {
        if (logicalSuffixes.contains(site.diagnosticAnchor))
          continue;
        auto logical = context.observationLogicalSuffix(
            cast<ac::SourceObserveOp>(site.diagnosticAnchor), bindings);
        logicalSuffixes.try_emplace(site.diagnosticAnchor, logical.begin(),
                                    logical.end());
      }
    for (uint64_t lane = 0; lane < lanes; ++lane) {
      if (sites != plan.definitions.end())
        for (const ObservationEmissionSite &site : sites->second) {
          if (plan.table.size() >= std::numeric_limits<std::uint32_t>::max())
            return definition.emitOpError()
                   << "native observation ordinal exceeds u32";
          InstancePath logicalPath(paths[lane]);
          ac::HardwareBindings logicalBindings = bindings;
          for (auto instance :
               logicalSuffixes.find(site.diagnosticAnchor)->second) {
            auto found = context.occurrenceId(instance, logicalBindings);
            if (failed(found))
              return failure();
            logicalPath.push_back({*found, instance, 0,
                                   instance.getInstanceName().str(), false});
            auto child = context.bindOccurrence(instance, logicalBindings);
            if (failed(child))
              return failure();
            logicalBindings = std::move(*child);
          }
          plan.occurrences.push_back(
              {site, plan.table.size(), paths[lane], std::move(logicalPath)});
          plan.table.push_back({plan.table.size(), ownerKey,
                                site.registrationKey, site.siteKey,
                                site.gauge ? 1u : 0u});
          if (!site.gauge)
            ++plan.events;
        }
      ++ownerKey;
    }
    for (const ObservationEmissionChild &child : plan.children[definition]) {
      // Geometry remains definition-local. Actual binding-dependent shape
      // changes cannot reuse the first layout for an observed subtree.
      if (auto collection = dyn_cast<ac::CollectionOp>(child.op);
          collection && plan.subtreeWidth.lookup(child.definition) != 0) {
        auto extents = context.occurrenceShape(collection, bindings);
        if (failed(extents))
          return failure();
        uint64_t shape = 1;
        for (auto extent : *extents) {
          if (extent != 0 &&
              shape > std::numeric_limits<uint64_t>::max() / extent)
            return collection.emitOpError()
                   << "native observation lane count overflows";
          shape *= extent;
        }
        if (shape != child.shape)
          return collection.emitOpError()
                 << "binding-dependent observation geometry is unsupported";
      }
      plan.childBaseUnits.try_emplace(child.id, units);
      uint64_t block = child.shape * plan.subtreeWidth.lookup(child.definition);
      if (block > std::numeric_limits<uint64_t>::max() - units)
        return definition.emitOpError()
               << "native observation descriptor count overflows";
      units += block;
      if (child.definition) {
        SmallVector<InstancePath> childPaths;
        for (const auto &path : paths)
          for (uint64_t lane = 0; lane < child.shape; ++lane) {
            childPaths.push_back(path);
            auto name = context.occurrenceName(child.op, bindings);
            if (failed(name))
              return failure();
            childPaths.back().push_back(
                {child.id, child.op, lane, std::move(*name),
                 context.occurrenceIsCollection(child.op)});
          }
        auto childBindings = context.bindOccurrence(child.op, bindings);
        if (failed(childBindings) ||
            failed(layout(child.definition, childPaths, *childBindings)))
          return failure();
      }
    }
    return success();
  };
  const InstancePath rootPath;
  if (failed(layout(*root, ArrayRef<InstancePath>(rootPath), *rootBindings)))
    return failure();
  for (std::size_t index = 1; index < plan.table.size(); ++index) {
    const auto &previous = plan.table[index - 1];
    const auto &current = plan.table[index];
    if (!(std::tie(previous[1], previous[2], previous[3], previous[0]) <
          std::tie(current[1], current[2], current[3], current[0])))
      return context.emitPlanError()
             << "native observation descriptors are not strictly ordered";
    if (previous[0] == current[0])
      return context.emitPlanError()
             << "native observation stable ordinals are not unique";
  }
  for (const ObservationEmissionOccurrence &occurrence : plan.occurrences) {
    const ObservationEmissionSite &site = occurrence.site;
    auto validJson = [](StringRef text) {
      auto parsed = llvm::json::parse(text);
      bool valid = static_cast<bool>(parsed);
      if (!valid)
        llvm::consumeError(parsed.takeError());
      return valid;
    };
    if (site.valueCount == 0 || site.kind.empty() ||
        !validJson(site.registrationJson) || !validJson(site.siteJson) ||
        !validJson(site.specJson) ||
        llvm::any_of(occurrence.path, [](const auto &entry) {
          return entry.instanceName.empty();
        }) ||
        llvm::any_of(occurrence.logicalOwnerPath, [](const auto &entry) {
          return entry.instanceName.empty();
        }))
      return context.emitPlanError()
             << "native observation publication plan is incomplete";
  }
  return plan;
}
} // namespace
HardwareEmitContext::HardwareEmitContext(
    const HardwareEmissionPlan &plan,
    const ac::HardwareSourceCheckPlan &sourceChecks,
    std::shared_ptr<const ObservationEmissionPlan> &observationPlanCache,
    const ac::HardwarePhysicalProjection *physicalProjection, bool physical)
    : plan(plan), frozenSourceChecks(sourceChecks),
      observationPlanCache(observationPlanCache),
      frozenPhysicalProjection(physicalProjection), physical_(physical) {}
LogicalResult HardwareEmitContext::prepare() { return prepareImpl(false); }
LogicalResult HardwareEmitContext::prepareNativeChecks() {
  if (failed(prepareImpl(true)))
    return failure();
  if (!observationPlanCache) {
    auto observations = buildObservationPlan(*this);
    if (failed(observations))
      return failure();
    observationPlanCache =
        std::make_shared<ObservationEmissionPlan>(std::move(*observations));
  }
  observationPlan_ = observationPlanCache;
  return success();
}
const ac::HardwareSourceCheckPlan &HardwareEmitContext::sourceChecks() const {
  assert(sourceChecks_ && "emission context must be prepared");
  return *sourceChecks_;
}
const ObservationEmissionPlan &HardwareEmitContext::observationPlan() const {
  assert(observationPlan_ && "observation plan must be prepared");
  return *observationPlan_;
}
bool HardwareEmitContext::managesChecks() {
  return !physical_ &&
         (isSystem() || (sourceChecks_ && !sourceChecks_->checks.empty()));
}
bool HardwareEmitContext::resetIfOne() const {
  return sourceChecks().resetSemantics == ac::HardwareResetSemantics::IfOne;
}
std::string HardwareEmitContext::cppReset(StringRef expression) const {
  return resetIfOne() ? "gfsim::interpretReset(" + expression.str() +
                            ", gfsim::ResetSemantics::IfOne)"
                      : expression.str();
}
std::string HardwareEmitContext::rtlReset(StringRef expression) const {
  return resetIfOne() ? "((" + expression.str() + ") === 1'b1)"
                      : expression.str();
}
LogicalResult HardwareEmitContext::prepareImpl(bool allowSourceChecks) {
  sourceChecks_.reset();
  observationPlan_.reset();
  names_.clear();
  definitions_.clear();
  primitiveKinds_.clear();
  if (!plan.package)
    return failure();
  // Capability validation below needs to distinguish primitive storage
  // occurrences before the renderer-facing name tables are populated.  Keep
  // this lookup derived exclusively from the frozen plan, but make it
  // available for the whole preparation phase.
  for (const HardwareEmissionPrimitivePlan &primitive : plan.primitives)
    primitiveKinds_.try_emplace(primitive.diagnosticAnchor, primitive.kind);
  // Building the plan includes full package verification. Retain that result
  // instead of verifying and expanding the same occurrence closure twice.
  const ac::HardwareSourceCheckPlan &checks = frozenSourceChecks;
  // The C++ native observation plan prepares through `prepareNativeChecks()`
  // and then re-closes the one instrumentation op it still cannot emit. This
  // shared gate stays closed for `ac.observe` because the Verilog target has no
  // observation channel yet, and a silently dropped observation is
  // exactly what the fail-closed contract forbids.
  if (!allowSourceChecks && !plan.observations.empty())
    return plan.observations.front().diagnosticAnchor->emitOpError()
           << "hardware instrumentation emission is not implemented";
  if (!allowSourceChecks && !plan.checks.empty())
    return emitPlanError()
           << "hardware instrumentation emission is not implemented";
  auto checkInteger = [&](Attribute value, Operation *site) -> LogicalResult {
    auto integer = dyn_cast<ac::MathIntAttr>(value);
    if (!integer)
      return success();
    llvm::APSInt number(integer.getCanonicalValue());
    if (llvm::APSInt::compareValues(number,
                                    llvm::APSInt("-9223372036854775808")) < 0 ||
        llvm::APSInt::compareValues(number,
                                    llvm::APSInt("9223372036854775807")) > 0)
      return site->emitOpError()
             << "symbolic static arithmetic exceeds the emitted signed 64-bit "
                "domain; closed expressions are evaluated exactly";
    return success();
  };
  std::function<LogicalResult(Attribute, const ac::HardwareBindings &,
                              Operation *, bool)>
      inspect;
  inspect = [&](Attribute attribute, const ac::HardwareBindings &bindings,
                Operation *site, bool symbolic) -> LogicalResult {
    if (auto expression = dyn_cast_or_null<ac::StaticExprAttr>(attribute)) {
      bool demand = symbolic || !isClosedStatic(expression);
      if (!demand)
        return success();
      auto value = evaluatedStatic(expression, bindings);
      if (failed(value) || failed(checkInteger(*value, site)))
        return failure();
      auto tree = expression.getTree();
      auto op = tree.getAs<StringAttr>("operator");
      if (op && (op.getValue() == "shl" || op.getValue() == "shr")) {
        auto count =
            evaluatedStatic(tree.getAs<ac::StaticExprAttr>("rhs"), bindings);
        if (failed(count))
          return failure();
        auto integer = dyn_cast<ac::MathIntAttr>(*count);
        if (!integer || llvm::APSInt::compareValues(
                            llvm::APSInt(integer.getCanonicalValue()),
                            llvm::APSInt("63")) >= 0)
          return site->emitOpError() << "symbolic static shift count must be "
                                        "below 63 in both emitted targets";
      }
      if (op && (op.getValue() == "floordiv" || op.getValue() == "mod")) {
        auto lhs = evaluatedStatic(tree.getAs<ac::StaticExprAttr>("lhs"),
                                   bindings),
             rhs = evaluatedStatic(tree.getAs<ac::StaticExprAttr>("rhs"),
                                   bindings);
        if (failed(lhs) || failed(rhs))
          return failure();
        if (cast<ac::MathIntAttr>(*lhs).getCanonicalValue() ==
                "-9223372036854775808" &&
            cast<ac::MathIntAttr>(*rhs).getCanonicalValue() == "-1")
          return site->emitOpError() << "symbolic static division exceeds the "
                                        "emitted signed 64-bit domain";
      }
      auto kind = tree.getAs<StringAttr>("kind");
      if (kind && kind.getValue() == "select") {
        auto condition = tree.getAs<ac::StaticExprAttr>("condition");
        if (failed(inspect(condition, bindings, site, true)))
          return failure();
        auto selected = evaluatedStatic(condition, bindings);
        if (failed(selected))
          return failure();
        return inspect(tree.getAs<ac::StaticExprAttr>(
                           cast<BoolAttr>(*selected).getValue() ? "yes" : "no"),
                       bindings, site, true);
      }
      if (op && (op.getValue() == "and_bool" || op.getValue() == "or_bool")) {
        auto lhs = tree.getAs<ac::StaticExprAttr>("lhs");
        if (failed(inspect(lhs, bindings, site, true)))
          return failure();
        auto selected = evaluatedStatic(lhs, bindings);
        if (failed(selected))
          return failure();
        bool value = cast<BoolAttr>(*selected).getValue();
        if ((op.getValue() == "and_bool" && !value) ||
            (op.getValue() == "or_bool" && value))
          return success();
        return inspect(tree.getAs<ac::StaticExprAttr>("rhs"), bindings, site,
                       true);
      }
      return inspect(tree, bindings, site, true);
    }
    if (auto type = dyn_cast_or_null<TypeAttr>(attribute)) {
      if (auto function = dyn_cast<FunctionType>(type.getValue())) {
        for (Type value : function.getInputs())
          if (failed(inspect(TypeAttr::get(value), bindings, site, false)))
            return failure();
        for (Type value : function.getResults())
          if (failed(inspect(TypeAttr::get(value), bindings, site, false)))
            return failure();
      }
      if (auto table = dyn_cast<ac::TableType>(type.getValue())) {
        if (failed(inspect(table.getShape(), bindings, site, false)))
          return failure();
        return inspect(TypeAttr::get(table.getElementType()), bindings, site,
                       false);
      }
      if (auto bits = dyn_cast<ac::BitsType>(type.getValue()))
        return inspect(bits.getWidth(), bindings, site, false);
    }
    if (auto dict = dyn_cast_or_null<DictionaryAttr>(attribute))
      for (auto field : dict)
        if (field.getName() != "origin" && field.getName() != "location" &&
            failed(inspect(field.getValue(), bindings, site, symbolic)))
          return failure();
    if (auto array = dyn_cast_or_null<ArrayAttr>(attribute))
      for (auto child : array)
        if (failed(inspect(child, bindings, site, symbolic)))
          return failure();
    return success();
  };
  auto checkWidth = [&](Type type, const ac::HardwareBindings &bindings,
                        Operation *site,
                        uint64_t multiplicity) -> LogicalResult {
    if (!isa<ac::BitsType, ac::StructType, ac::EnumType, ac::TypeParamType,
             ac::TableType>(type))
      return success();
    auto width = packedWidth(type, bindings);
    if (failed(width))
      return failure();
    if (*width > std::numeric_limits<unsigned>::max() / multiplicity)
      return site->emitOpError()
             << "hardware payload plane including enclosing collections "
                "exceeds the current Runtime bit-width capacity";
    return success();
  };
  std::function<LogicalResult(ac::ModuleOp, const ac::HardwareBindings &,
                              uint64_t)>
      capabilities;
  llvm::DenseSet<Operation *> checkedConcreteDefinitions;
  capabilities = [&](ac::ModuleOp module, const ac::HardwareBindings &bindings,
                     uint64_t multiplicity) -> LogicalResult {
    for (auto &value : bindings.integers) {
      auto actual = value.second;
      if (auto expr = dyn_cast<ac::StaticExprAttr>(actual)) {
        auto evaluated = evaluatedStatic(expr, bindings);
        if (failed(evaluated))
          return failure();
        actual = *evaluated;
      }
      if (failed(checkInteger(actual, module)))
        return failure();
    }
    auto result = module.walk([&](Operation *op) {
      for (auto attr : op->getAttrs())
        if (failed(inspect(attr.getValue(), bindings, op, false)))
          return WalkResult::interrupt();
      for (Type type : op->getResultTypes())
        if (failed(inspect(TypeAttr::get(type), bindings, op, false)))
          return WalkResult::interrupt();
      for (Type type : op->getOperandTypes())
        if (failed(inspect(TypeAttr::get(type), bindings, op, false)))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();
    for (Type type : module.getFunctionType().getInputs())
      if (failed(checkWidth(type, bindings, module, multiplicity)))
        return failure();
    for (Type type : module.getFunctionType().getResults())
      if (failed(checkWidth(type, bindings, module, multiplicity)))
        return failure();
    auto widths = module.walk([&](Operation *op) {
      for (Type type : op->getResultTypes())
        if (failed(checkWidth(type, bindings, op, multiplicity)))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (widths.wasInterrupted())
      return failure();
    for (auto &op : module.getBody().front()) {
      if (auto queue = dyn_cast<ac::InstanceOp>(op);
          queue && ac::isQueueInstance(queue)) {
        const auto *config = queuePlan(&op, bindings);
        if (!config)
          return op.emitOpError()
                 << "queue is absent from the frozen emission plan";
        if (config->depth > uint64_t(std::numeric_limits<int64_t>::max()))
          return op.emitOpError()
                 << "queue depth exceeds emitted signed 64-bit indexing";
        if (!multiplicity ||
            config->tokenCardinality >
                std::numeric_limits<uint64_t>::max() / multiplicity)
          return op.emitOpError() << "queue family token geometry overflows";
        // Tables hold independent scalar wires: round each scalar first, not
        // the packed token. These bytes estimate storage, never hardware
        // layout.
        uint64_t scalarBytes =
            ((config->tokenElementWidth - 1) / 64 + 1) * 3 * sizeof(uint64_t);
        if (config->tokenCardinality >
            std::numeric_limits<uint64_t>::max() / scalarBytes)
          return op.emitOpError() << "queue token representation overflows";
        uint64_t tokenBytes = config->tokenCardinality * scalarBytes;
        uint64_t capacity = std::numeric_limits<std::ptrdiff_t>::max();
        // Seven control planes (168), Current/Pending metadata (32/8), and a
        // completion word (8) per lane; 128 reserves the shared owner. Besides
        // D slots, retain Pending, two data pin planes and two token buffers.
        // Delayed availability adds D u64 deadlines plus tick, maturity cursor
        // and eligible count (24). Pending timing flags use its existing
        // reserve.
        uint64_t laneCapacity = (capacity - 128) / multiplicity;
        bool delayed = config->availabilityLatency > 1;
        uint64_t fixedBytes = delayed ? 240 : 216;
        if (laneCapacity <= fixedBytes)
          return op.emitOpError()
                 << "queue storage including enclosing collections exceeds "
                    "the current Runtime object capacity";
        uint64_t remaining = laneCapacity - fixedBytes;
        if (delayed) {
          if (config->depth > remaining / sizeof(uint64_t))
            return op.emitOpError()
                   << "queue storage including enclosing collections exceeds "
                      "the current Runtime object capacity";
          remaining -= config->depth * sizeof(uint64_t);
        }
        // Subtract reserved token copies before comparing depth: neither D+5
        // nor the combined payload/timing/family allocation can overflow.
        uint64_t tokenCapacity = remaining / tokenBytes;
        if (tokenCapacity < 5 || config->depth > tokenCapacity - 5)
          return op.emitOpError()
                 << "queue storage including enclosing collections exceeds "
                    "the current Runtime object capacity";
        continue;
      }
      if (!isa<ac::InstanceOp, ac::CollectionOp>(op))
        continue;
      auto child = bindOccurrence(&op, bindings);
      if (failed(child))
        return failure();
      for (const auto &binding : child->integers) {
        Attribute value = binding.second;
        if (auto expression = dyn_cast<ac::StaticExprAttr>(value)) {
          auto evaluated = evaluatedStatic(expression, *child);
          if (failed(evaluated))
            return failure();
          value = *evaluated;
        }
        if (failed(checkInteger(value, &op)))
          return failure();
      }
      uint64_t childMultiplicity = multiplicity;
      if (auto collection = dyn_cast<ac::CollectionOp>(op)) {
        auto shape = occurrenceShape(&op, bindings);
        if (failed(shape))
          return failure();
        for (uint64_t extent : *shape) {
          if (childMultiplicity > std::numeric_limits<uint64_t>::max() / extent)
            return op.emitOpError() << "enclosing collection multiplicity "
                                       "overflows target storage indexing";
          childMultiplicity *= extent;
        }
      }
      // Storage arrays are not port planes. Check their concrete host-index
      // capacity too, before a generated std::array bound can overflow.
      auto primitive = primitiveKind(child->owner);
      if (primitive == "sync_mem" || primitive == "sync_mem_dp" ||
          primitive == "byte_mem") {
        Attribute rawDepth = child->integers.lookup("DEPTH");
        if (auto expr = dyn_cast<ac::StaticExprAttr>(rawDepth)) {
          auto value = evaluatedStatic(expr, *child);
          if (failed(value))
            return failure();
          rawDepth = *value;
        }
        auto integer = dyn_cast<ac::MathIntAttr>(rawDepth);
        if (!integer)
          return op.emitOpError() << "memory depth is unresolved";
        StringRef depthText = integer.getCanonicalValue();
        llvm::APInt depthValue;
        if (depthText.starts_with("-") || depthText == "0" ||
            depthText.getAsInteger(10, depthValue))
          return op.emitOpError()
                 << "memory depth exceeds target storage indexing";
        auto typeName = cast<StringAttr>(
            child->owner->getAttrOfType<ArrayAttr>("type_parameters")[0]);
        auto bits =
            packedWidth(child->types.lookup(typeName.getValue()), *child);
        if (failed(bits))
          return failure();
        uint64_t wordBytes = ((*bits - 1) / 64 + 1) * 8;
        uint64_t entryBytes = primitive == "byte_mem" ? 1 : wordBytes;
        uint64_t readPorts = primitive == "sync_mem_dp" ? 2 : 1;
        // Upper bound for the fixed mask/read/pending bookkeeping in the
        // owning kernels; this is a resource estimate, not packed bit layout.
        uint64_t overhead =
            128 + wordBytes * (3 * readPorts + 2) + ((*bits - 1) / 8 + 1);
        uint64_t capacity = std::numeric_limits<std::ptrdiff_t>::max();
        uint64_t perLaneLimit = capacity / childMultiplicity;
        uint64_t depthLimit =
            overhead >= perLaneLimit ? 0 : (perLaneLimit - overhead) / entryBytes;
        bool exceedsDepthLimit = depthValue.ugt(depthLimit);
        // The kernel also carries fixed bookkeeping per entry, so a signed
        // host-index maximum can never denote a constructible memory object.
        bool reachesHostIndexMaximum =
            depthValue.uge(uint64_t(std::numeric_limits<int64_t>::max()));
        if (overhead >= perLaneLimit || exceedsDepthLimit ||
            reachesHostIndexMaximum)
          return op.emitOpError()
                 << "memory storage including enclosing collections exceeds "
                    "the current Runtime object capacity";
      }
      if (auto def = dyn_cast<ac::ModuleOp>(child->owner))
        if (failed(capabilities(def, *child, childMultiplicity)))
          return failure();
    }
    if (module.getParameters().empty() && module.getTypeParameters().empty())
      checkedConcreteDefinitions.insert(module);
    return success();
  };
  if (!plan.targetCapabilitiesValidated) {
    auto root = rootBindings();
    if (failed(root) ||
        failed(capabilities(cast<ac::ModuleOp>(root->owner), *root, 1)))
      return failure();
    // Every non-generic definition emits a concrete scalar wrapper, even when
    // the selected root does not instantiate it. Validate that emitted closure
    // too; unbound generic definitions must not acquire facts from defaults.
    for (const HardwareEmissionDefinitionPlan &planned : plan.definitions) {
      auto definition = cast<ac::ModuleOp>(planned.diagnosticAnchor);
      if (!definition.getParameters().empty() ||
          !definition.getTypeParameters().empty() ||
          checkedConcreteDefinitions.contains(definition))
        continue;
      ac::HardwareBindings bindings;
      bindings.owner = definition;
      if (failed(capabilities(definition, bindings, 1)))
        return failure();
    }
  }
  llvm::StringSet<> cpp, cppNamespaces, rtl, rtlTypes;
  auto reserveCppName = [&](StringRef name, Operation *site) -> LogicalResult {
    SmallVector<StringRef> parts;
    name.drop_front(2).split(parts, "::");
    std::string prefix;
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
      prefix += "::" + parts[i].str();
      if (cpp.contains(prefix))
        return site->emitOpError() << "emitted hardware name collision";
      cppNamespaces.insert(prefix);
    }
    if (cppNamespaces.contains(name) || !cpp.insert(name).second)
      return site->emitOpError() << "emitted hardware name collision";
    return success();
  };
  for (Operation *declaration : plan.nominalDeclarations)
    definitions_.try_emplace(SymbolTable::getSymbolName(declaration).getValue(),
                             declaration);
  // Validate the complete nominal declaration closure shared by both targets.
  for (Operation *operation : plan.nominalDeclarations) {
    Operation &op = *operation;
    Type type;
    if (auto enumeration = dyn_cast<ac::EnumOp>(op))
      type = ac::EnumType::get(op.getContext(), enumeration.getSymNameAttr());
    else if (auto record = dyn_cast<ac::StructOp>(op))
      type = ac::StructType::get(op.getContext(), record.getSymNameAttr());
    else
      continue;
    if (!plan.targetCapabilitiesValidated &&
        failed(checkWidth(type, {}, &op, 1)))
      return failure();
    auto cppName = cppType(type, &op), rtlName = rtlType(type, &op);
    if (failed(cppName) || failed(rtlName) ||
        failed(reserveCppName(*cppName, &op)))
      return failure();
    // RTL typedefs share pycircuit_types, separately from global module names.
    if (!rtlTypes.insert(*rtlName).second)
      return op.emitOpError() << "emitted hardware name collision";
  }
  for (const HardwareEmissionDefinitionPlan &definition : plan.definitions) {
    Operation *site = definition.diagnosticAnchor;
    if (failed(reserveCppName(definition.names.cppQualified, site)) ||
        failed(reserveCppName(definition.names.cppFamilyQualified, site)))
      return failure();
    if (!rtl.insert(definition.names.rtl).second)
      return site->emitOpError() << "emitted hardware name collision";
    names_.try_emplace(site, definition.names);
    definitions_.try_emplace(definition.symbol, site);
  }
  for (const HardwareEmissionPrimitivePlan &primitive : plan.primitives) {
    definitions_.try_emplace(primitive.symbol, primitive.diagnosticAnchor);
  }
  if (physical_) {
    if (!frozenPhysicalProjection)
      return emitPlanError()
             << "physical projection is absent from the frozen plan";
    physicalProjection_ = *frozenPhysicalProjection;
  }
  sourceChecks_ = checks;
  return success();
}
const EmitNames &HardwareEmitContext::names(Operation *definition) const {
  return names_.find(definition)->second;
}
Operation *HardwareEmitContext::lookupDefinition(StringRef symbol) const {
  auto found = definitions_.find(symbol);
  return found == definitions_.end() ? nullptr : found->second;
}
ac::StructOp HardwareEmitContext::lookupStruct(ac::StructType type) const {
  return dyn_cast_or_null<ac::StructOp>(
      lookupDefinition(type.getName().getValue()));
}
FailureOr<ac::EnumDefinitionView>
HardwareEmitContext::resolveEnum(ac::EnumType type, Operation *site) const {
  auto declaration =
      dyn_cast_or_null<ac::EnumOp>(lookupDefinition(type.getName().getValue()));
  if (!declaration)
    return site->emitOpError() << "unresolved enum declaration '"
                               << type.getName().getValue() << "'";
  return ac::validateEnumDeclaration(declaration, site);
}
Operation *HardwareEmitContext::resolveCallee(ac::InstanceOp instance) const {
  return lookupDefinition(instance.getCallee());
}
Operation *
HardwareEmitContext::resolveCallee(ac::CollectionOp collection) const {
  return lookupDefinition(collection.getCallee());
}
StringRef HardwareEmitContext::primitiveKind(Operation *definition) const {
  auto found = primitiveKinds_.find(definition);
  return found == primitiveKinds_.end() ? StringRef() : found->second;
}
namespace {
bool sameBindings(const HardwareEmissionPlan &plan,
                  const HardwareEmissionBindings &frozen,
                  const ac::HardwareBindings &live);
}
ArrayRef<Operation *>
HardwareEmitContext::independentWorkChildren(Operation *definition) const {
  for (const HardwareEmissionDefinitionPlan &planned : plan.definitions)
    if (planned.diagnosticAnchor == definition)
      return planned.independentWorkChildren;
  return {};
}
SmallVector<Operation *> HardwareEmitContext::scheduledOperations(
    Operation *definition, const ac::HardwareBindings &bindings) const {
  auto operations = [](const HardwareEmissionModulePlan &module) {
    SmallVector<Operation *> result;
    llvm::DenseSet<Operation *> included;
    for (const HardwareEmissionScheduleStep &step : module.schedule) {
      // This renderer-facing view describes the definition body. Nested
      // region operations remain represented by their enclosing scheduled op
      // and are lowered recursively by the operation emitter.
      if (step.diagnosticAnchor->getParentOp() != module.diagnosticAnchor)
        continue;
      // Occurrences have their own frozen, source-stable inventory.  Keep
      // that order below: SSA scheduling may legally move stateful children
      // relative to one another, while renderer instance enumeration must not.
      if (isa<ac::InstanceOp, ac::CollectionOp>(step.diagnosticAnchor))
        continue;
      result.push_back(step.diagnosticAnchor);
      included.insert(step.diagnosticAnchor);
    }
    // Effect-only/void children need not appear in the SSA evaluation
    // schedule, but their frozen occurrence inventory still drives target
    // instance emission. Never rediscover them by walking the ACIR body.
    for (const HardwareEmissionChildPlan &child : module.children)
      if (included.insert(child.diagnosticAnchor).second)
        result.push_back(child.diagnosticAnchor);
    return result;
  };
  for (const HardwareEmissionModulePlan &module : plan.modules) {
    if (module.diagnosticAnchor != definition)
      continue;
    if (!sameBindings(plan, module.concreteBindings, bindings))
      continue;
    return operations(module);
  }
  // Generic definitions are emitted once with formal bindings.  A concrete
  // specialization may prune a different subset of children, so the frozen
  // template occurrence inventory—not an arbitrary specialization—is the
  // authoritative renderer view in this scope.
  for (const HardwareEmissionDefinitionPlan &planned : plan.definitions)
    if (planned.diagnosticAnchor == definition) {
      SmallVector<Operation *> result(planned.templateSchedule.begin(),
                                      planned.templateSchedule.end());
      llvm::erase_if(result, [](Operation *operation) {
        return isa<ac::InstanceOp, ac::CollectionOp>(operation);
      });
      llvm::DenseSet<Operation *> included(result.begin(), result.end());
      for (const HardwareEmissionOccurrenceBinding &child :
           planned.templateChildren)
        if (included.insert(child.diagnosticAnchor).second)
          result.push_back(child.diagnosticAnchor);
      return result;
    }
  return {};
}
FailureOr<HardwareEmissionOccurrenceId>
HardwareEmitContext::occurrenceId(Operation *occurrence,
                                  const ac::HardwareBindings &parent) const {
  for (const HardwareEmissionModulePlan &module : plan.modules) {
    if (module.diagnosticAnchor != parent.owner ||
        !sameBindings(plan, module.concreteBindings, parent))
      continue;
    for (const HardwareEmissionChildPlan &child : module.children)
      if (child.diagnosticAnchor == occurrence)
        return child.id;
  }
  if (parent.owner)
    for (const HardwareEmissionDefinitionPlan &definition : plan.definitions)
      if (definition.diagnosticAnchor == parent.owner)
        for (const HardwareEmissionOccurrenceBinding &child :
             definition.templateChildren)
          if (child.diagnosticAnchor == occurrence)
            return child.id;
  return occurrence->emitOpError()
         << "hardware emission occurrence has no frozen identity";
}
namespace {
Operation *bindingOwner(const HardwareEmissionPlan &plan,
                        const HardwareEmissionBindings &bindings) {
  return bindings.owner < plan.definitionAnchors.size()
             ? plan.definitionAnchors[bindings.owner]
             : nullptr;
}
ac::HardwareBindings thawBindings(const HardwareEmissionPlan &plan,
                                  const HardwareEmissionBindings &bindings) {
  ac::HardwareBindings result;
  result.owner = bindingOwner(plan, bindings);
  result.integers = bindings.integers;
  result.types = bindings.types;
  return result;
}
bool sameBindings(const HardwareEmissionPlan &plan,
                  const HardwareEmissionBindings &frozen,
                  const ac::HardwareBindings &live) {
  if (bindingOwner(plan, frozen) != live.owner ||
      frozen.integers.size() != live.integers.size() ||
      frozen.types.size() != live.types.size())
    return false;
  for (const auto &binding : live.integers)
    if (frozen.integers.lookup(binding.getKey()) != binding.getValue())
      return false;
  for (const auto &binding : live.types)
    if (frozen.types.lookup(binding.getKey()) != binding.getValue())
      return false;
  return true;
}
} // namespace
FailureOr<ac::HardwareBindings>
HardwareEmitContext::bindOccurrence(Operation *occurrence,
                                    const ac::HardwareBindings &parent) const {
  for (const HardwareEmissionModulePlan &module : plan.modules) {
    if (module.diagnosticAnchor != parent.owner)
      continue;
    if (!sameBindings(plan, module.concreteBindings, parent))
      continue;
    for (const HardwareEmissionChildPlan &child : module.children)
      if (child.diagnosticAnchor == occurrence)
        return thawBindings(plan, child.concreteBindings);
  }
  if (parent.owner)
    for (const HardwareEmissionDefinitionPlan &definition : plan.definitions)
      if (definition.diagnosticAnchor == parent.owner)
        for (const HardwareEmissionOccurrenceBinding &child :
             definition.templateChildren)
          if (child.diagnosticAnchor == occurrence)
            return thawBindings(plan, child.bindings);
  return occurrence->emitOpError()
         << "hardware emission occurrence is absent from the frozen plan";
}
FailureOr<SmallVector<uint64_t>>
HardwareEmitContext::occurrenceShape(Operation *occurrence,
                                     const ac::HardwareBindings &parent) const {
  for (const HardwareEmissionModulePlan &module : plan.modules) {
    if (module.diagnosticAnchor != parent.owner)
      continue;
    if (!sameBindings(plan, module.concreteBindings, parent))
      continue;
    for (const HardwareEmissionChildPlan &child : module.children)
      if (child.diagnosticAnchor == occurrence) {
        SmallVector<uint64_t> shape(child.shape.begin(), child.shape.end());
        return shape;
      }
  }
  if (parent.owner && parent.integers.empty() && parent.types.empty())
    for (const HardwareEmissionDefinitionPlan &definition : plan.definitions)
      if (definition.diagnosticAnchor == parent.owner)
        for (const HardwareEmissionOccurrenceBinding &child :
             definition.templateChildren)
          if (child.diagnosticAnchor == occurrence && !child.shape.empty())
            return SmallVector<uint64_t>(child.shape.begin(),
                                         child.shape.end());
  return occurrence->emitOpError()
         << "hardware emission occurrence shape is absent from the frozen plan";
}
FailureOr<std::string>
HardwareEmitContext::occurrenceName(Operation *occurrence,
                                    const ac::HardwareBindings &parent) const {
  for (const HardwareEmissionModulePlan &module : plan.modules) {
    if (module.diagnosticAnchor != parent.owner ||
        !sameBindings(plan, module.concreteBindings, parent))
      continue;
    for (const HardwareEmissionChildPlan &child : module.children)
      if (child.diagnosticAnchor == occurrence)
        return child.instanceName;
  }
  if (parent.owner)
    for (const HardwareEmissionDefinitionPlan &definition : plan.definitions)
      if (definition.diagnosticAnchor == parent.owner)
        for (const HardwareEmissionOccurrenceBinding &child :
             definition.templateChildren)
          if (child.diagnosticAnchor == occurrence)
            return child.instanceName;
  return occurrence->emitOpError()
         << "hardware emission occurrence name is absent from the frozen plan";
}
bool HardwareEmitContext::occurrenceIsCollection(Operation *occurrence) const {
  for (const HardwareEmissionModulePlan &module : plan.modules)
    for (const HardwareEmissionChildPlan &child : module.children)
      if (child.diagnosticAnchor == occurrence)
        return !child.shape.empty();
  for (const HardwareEmissionDefinitionPlan &definition : plan.definitions)
    for (const HardwareEmissionOccurrenceBinding &child :
         definition.templateChildren)
      if (child.diagnosticAnchor == occurrence)
        return child.collection;
  return false;
}
const HardwareEmissionQueuePlan *
HardwareEmitContext::queuePlan(Operation *occurrence,
                               const ac::HardwareBindings &parent) const {
  for (const HardwareEmissionModulePlan &module : plan.modules) {
    if (module.diagnosticAnchor != parent.owner ||
        !sameBindings(plan, module.concreteBindings, parent))
      continue;
    for (const HardwareEmissionChildPlan &child : module.children)
      if (child.diagnosticAnchor == occurrence)
        return child.queue ? &*child.queue : nullptr;
  }
  if (parent.owner)
    for (const HardwareEmissionDefinitionPlan &definition : plan.definitions)
      if (definition.diagnosticAnchor == parent.owner)
        for (const HardwareEmissionOccurrenceBinding &child :
             definition.templateChildren)
          if (child.diagnosticAnchor == occurrence)
            return child.queue ? &*child.queue : nullptr;
  return nullptr;
}
FailureOr<SmallVector<Value>>
HardwareEmitContext::outputValues(Operation *definition) const {
  auto found = llvm::find_if(plan.modules, [&](const auto &module) {
    return module.diagnosticAnchor == definition;
  });
  if (found == plan.modules.end())
    for (const HardwareEmissionDefinitionPlan &planned : plan.definitions)
      if (planned.diagnosticAnchor == definition)
        return SmallVector<Value>(planned.outputValues.begin(),
                                  planned.outputValues.end());
  if (found == plan.modules.end())
    return definition->emitOpError()
           << "module outputs are absent from the frozen emission plan";
  SmallVector<Value> result;
  for (const HardwareEmissionPortPlan &output : found->outputs) {
    if (output.value >= found->values.size() ||
        !found->values[output.value].source)
      return definition->emitOpError()
             << "module output has no frozen value source";
    result.push_back(found->values[output.value].source);
  }
  return result;
}
const HardwareEmissionModulePlan *HardwareEmitContext::modulePlan(
    Operation *definition, const ac::HardwareBindings &bindings) const {
  for (const HardwareEmissionModulePlan &module : plan.modules)
    if (module.diagnosticAnchor == definition &&
        (!bindings.owner || sameBindings(plan, module.concreteBindings,
                                         bindings)))
      return &module;
  return nullptr;
}
const HardwareEmissionTypePlan *HardwareEmitContext::concreteValueType(
    Value value, const ac::HardwareBindings &bindings) const {
  for (const HardwareEmissionModulePlan &module : plan.modules) {
    bool matches =
        module.diagnosticAnchor == bindings.owner &&
        module.concreteBindings.integers.size() == bindings.integers.size() &&
        module.concreteBindings.types.size() == bindings.types.size();
    if (matches) {
      for (const auto &binding : bindings.integers)
        matches &= module.concreteBindings.integers.lookup(binding.getKey()) ==
                   binding.getValue();
      for (const auto &binding : bindings.types)
        matches &= module.concreteBindings.types.lookup(binding.getKey()) ==
                   binding.getValue();
    }
    if (!matches)
      continue;
    auto source = llvm::find_if(module.values, [&](const auto &planned) {
      return planned.source == value;
    });
    if (source == module.values.end())
      return nullptr;
    auto found = llvm::find_if(module.values, [&](const auto &planned) {
      return planned.id == source->id;
    });
    return found == module.values.end() ? nullptr : &found->type;
  }
  return nullptr;
}
namespace {
template <class Callback>
bool withEmissionQueries(const HardwareEmissionPlan &plan,
                         const ac::HardwareBindings &bindings,
                         Callback callback) {
  if (!bindings.owner && bindings.integers.empty() && bindings.types.empty() &&
      callback({}, plan.packageTypes))
    return true;
  for (const HardwareEmissionModulePlan &module : plan.modules) {
    if (module.diagnosticAnchor == bindings.owner &&
        module.concreteBindings.integers.size() == bindings.integers.size() &&
        module.concreteBindings.types.size() == bindings.types.size()) {
      bool matches = true;
      for (const auto &binding : bindings.integers)
        matches &= module.concreteBindings.integers.lookup(binding.getKey()) ==
                   binding.getValue();
      for (const auto &binding : bindings.types)
        matches &= module.concreteBindings.types.lookup(binding.getKey()) ==
                   binding.getValue();
      if (matches && callback(module.staticValues, module.types))
        return true;
    }
    for (const HardwareEmissionChildPlan &child : module.children) {
      const auto &actual = child.concreteBindings;
      if (!sameBindings(plan, actual, bindings))
        continue;
      if (callback(child.staticValues, child.types))
        return true;
    }
  }
  if (bindings.integers.empty() && bindings.types.empty())
    for (const HardwareEmissionDefinitionPlan &definition : plan.definitions)
      if ((!bindings.owner || definition.diagnosticAnchor == bindings.owner) &&
          callback(definition.staticValues, definition.types))
        return true;
  return false;
}
} // namespace
FailureOr<Attribute> HardwareEmitContext::evaluatedStatic(
    ac::StaticExprAttr expression, const ac::HardwareBindings &bindings) const {
  // Literals are already frozen in the ACIR attribute itself.  They do not
  // require a query-table row and may legitimately appear in a generic
  // definition rendered with a concrete parent specialization.
  if (auto kind = expression.getTree().getAs<StringAttr>("kind");
      kind && kind.getValue() == "literal") {
    auto literal = expression.getTree().getAs<DictionaryAttr>("value");
    if (literal)
      if (Attribute value = literal.get("value"))
        return value;
  }
  auto canonical = [](Attribute value) {
    std::string text;
    llvm::raw_string_ostream out(text);
    value.print(out);
    return text;
  };
  const std::string requested = canonical(expression);
  Attribute result;
  bool found = withEmissionQueries(
      plan, bindings,
      [&](ArrayRef<HardwareEmissionStaticPlan> values,
          ArrayRef<HardwareEmissionTypeQueryPlan>) {
        auto source = llvm::find_if(values, [&](const auto &value) {
          return value.diagnosticSource == expression ||
                 canonical(value.diagnosticSource) == requested;
        });
        if (source == values.end())
          return false;
        auto item = llvm::find_if(
            values, [&](const auto &value) { return value.id == source->id; });
        if (item == values.end())
          return false;
        result = item->value;
        return true;
      });
  if (!found && !isClosedStatic(expression, bindings))
    return expression;
  if (!found)
    return failure();
  return result;
}
FailureOr<Type>
HardwareEmitContext::resolvedType(Type type,
                                  const ac::HardwareBindings &bindings) const {
  if (auto formal = dyn_cast<ac::TypeParamType>(type)) {
    auto actual = bindings.types.find(formal.getName().getValue());
    if (actual != bindings.types.end())
      return actual->second;
  }
  for (const auto &binding : bindings.types)
    if (binding.getValue() == type)
      return type;
  if (auto table = dyn_cast<ac::TableType>(type)) {
    bool concrete = true;
    for (Attribute extent : table.getShape())
      concrete &= isClosedStatic(cast<ac::StaticExprAttr>(extent), bindings);
    auto element = resolvedType(table.getElementType(), bindings);
    if (concrete && succeeded(element) && *element == table.getElementType())
      return type;
    if (concrete && succeeded(element))
      return ac::TableType::get(type.getContext(), table.getShape(), *element);
    // Generic renderers deliberately retain symbolic extents and formal
    // element types. They are emitted as target parameters, not evaluated in
    // the frozen concrete specialization.
    if (succeeded(element))
      return ac::TableType::get(type.getContext(), table.getShape(), *element);
  }
  if (isa<ac::StructType, ac::EnumType>(type))
    return type;
  if (auto bits = dyn_cast<ac::BitsType>(type)) {
    if (isClosedStatic(bits.getWidth(), bindings))
      return type;
    if (bits.getWidth().getTree().getAs<StringAttr>("kind").getValue() ==
        "reference")
      return type;
  }
  Type result;
  bool found = withEmissionQueries(
      plan, bindings,
      [&](ArrayRef<HardwareEmissionStaticPlan>,
          ArrayRef<HardwareEmissionTypeQueryPlan> types) {
        auto source = llvm::find_if(types, [&](const auto &value) {
          return value.diagnosticSource == type;
        });
        if (source == types.end())
          return false;
        auto item = llvm::find_if(
            types, [&](const auto &value) { return value.id == source->id; });
        if (item == types.end())
          return false;
        result = item->resolved;
        return true;
      });
  if (!found)
    return failure();
  return result;
}
FailureOr<uint64_t>
HardwareEmitContext::packedWidth(Type type,
                                 const ac::HardwareBindings &bindings) const {
  std::optional<uint64_t> result;
  withEmissionQueries(
      plan, bindings,
      [&](ArrayRef<HardwareEmissionStaticPlan>,
          ArrayRef<HardwareEmissionTypeQueryPlan> types) {
        auto source = llvm::find_if(types, [&](const auto &value) {
          return value.diagnosticSource == type &&
                 value.packedWidth.has_value();
        });
        if (source == types.end())
          return false;
        auto item = llvm::find_if(
            types, [&](const auto &value) { return value.id == source->id; });
        if (item == types.end())
          return false;
        result = item->packedWidth;
        return true;
      });
  if (!result)
    return failure();
  return *result;
}
FailureOr<SmallVector<ac::FieldPath>>
HardwareEmitContext::fieldPaths(Type type,
                                const ac::HardwareBindings &bindings) const {
  SmallVector<ac::FieldPath> result;
  bool found = withEmissionQueries(
      plan, bindings,
      [&](ArrayRef<HardwareEmissionStaticPlan>,
          ArrayRef<HardwareEmissionTypeQueryPlan> types) {
        auto source = llvm::find_if(types, [&](const auto &value) {
          return value.diagnosticSource == type;
        });
        if (source == types.end())
          return false;
        auto item = llvm::find_if(
            types, [&](const auto &value) { return value.id == source->id; });
        if (item == types.end())
          return false;
        for (const auto &path : item->fieldPaths) {
          ac::FieldPath materialized;
          for (const std::string &component : path)
            materialized.push_back(
                StringAttr::get(type.getContext(), component));
          result.push_back(std::move(materialized));
        }
        return true;
      });
  if (!found)
    return failure();
  return result;
}
const HardwareEmissionTableViewPlan *
HardwareEmitContext::tableViewPlan(ac::TableViewOp view,
                                   const ac::HardwareBindings &bindings) const {
  for (const HardwareEmissionModulePlan &module : plan.modules) {
    if ((bindings.owner && module.diagnosticAnchor != bindings.owner) ||
        (!bindings.owner &&
         module.diagnosticAnchor != view->getParentOfType<ac::ModuleOp>()) ||
        module.concreteBindings.integers.size() != bindings.integers.size() ||
        module.concreteBindings.types.size() != bindings.types.size())
      continue;
    bool matches = true;
    for (const auto &binding : bindings.integers)
      matches &= module.concreteBindings.integers.lookup(binding.getKey()) ==
                 binding.getValue();
    for (const auto &binding : bindings.types)
      matches &= module.concreteBindings.types.lookup(binding.getKey()) ==
                 binding.getValue();
    if (!matches)
      continue;
    auto found = llvm::find_if(module.tableViews, [&](const auto &planned) {
      return planned.diagnosticAnchor == view.getOperation();
    });
    return found == module.tableViews.end() ? nullptr : &*found;
  }
  return nullptr;
}
ArrayRef<Operation *> HardwareEmitContext::observationLogicalSuffix(
    ac::SourceObserveOp observe, const ac::HardwareBindings &bindings) const {
  for (const HardwareEmissionModulePlan &module : plan.modules) {
    if (module.diagnosticAnchor != bindings.owner ||
        module.concreteBindings.integers.size() != bindings.integers.size() ||
        module.concreteBindings.types.size() != bindings.types.size())
      continue;
    bool matches = true;
    for (const auto &binding : bindings.integers)
      matches &= module.concreteBindings.integers.lookup(binding.getKey()) ==
                 binding.getValue();
    for (const auto &binding : bindings.types)
      matches &= module.concreteBindings.types.lookup(binding.getKey()) ==
                 binding.getValue();
    if (!matches)
      continue;
    auto found = llvm::find_if(plan.observations, [&](const auto &planned) {
      return planned.moduleIndex == &module - plan.modules.data() &&
             planned.diagnosticAnchor == observe.getOperation();
    });
    if (found != plan.observations.end())
      return found->logicalSuffix;
  }
  return {};
}
FailureOr<SmallVector<ac::WireEndpoint>>
HardwareEmitContext::dependencies(Value value, ArrayRef<StringAttr> path,
                                  const ac::HardwareBindings &bindings) const {
  auto matchesPath = [&](ArrayRef<std::string> candidate) {
    if (candidate.size() != path.size())
      return false;
    for (auto [left, right] : llvm::zip_equal(candidate, path))
      if (left != right.getValue())
        return false;
    return true;
  };
  auto materialize = [&](ArrayRef<HardwareEmissionDependencyPlan> rows)
      -> FailureOr<SmallVector<ac::WireEndpoint>> {
    for (const HardwareEmissionDependencyPlan &row : rows) {
      if (row.output.value != value || !matchesPath(row.output.path))
        continue;
      SmallVector<ac::WireEndpoint> result;
      for (const HardwareEmissionDependencyEndpoint &input : row.inputs) {
        ac::FieldPath fieldPath;
        for (const std::string &component : input.path)
          fieldPath.push_back(StringAttr::get(value.getContext(), component));
        result.push_back({input.value, std::move(fieldPath)});
      }
      return result;
    }
    return failure();
  };
  for (const HardwareEmissionModulePlan &module : plan.modules) {
    auto owner = value.getParentRegion()
                     ? value.getParentRegion()->getParentOfType<ac::ModuleOp>()
                     : ac::ModuleOp();
    if ((bindings.owner && module.diagnosticAnchor != bindings.owner) ||
        (!bindings.owner && module.diagnosticAnchor != owner))
      continue;
    bool matches =
        module.concreteBindings.integers.size() == bindings.integers.size() &&
        module.concreteBindings.types.size() == bindings.types.size();
    for (const auto &binding : bindings.integers)
      matches &= module.concreteBindings.integers.lookup(binding.getKey()) ==
                 binding.getValue();
    for (const auto &binding : bindings.types)
      matches &= module.concreteBindings.types.lookup(binding.getKey()) ==
                 binding.getValue();
    if (matches)
      if (auto result = materialize(module.dependencies); succeeded(result))
        return result;
  }
  if (bindings.integers.empty() && bindings.types.empty())
    for (const HardwareEmissionDefinitionPlan &definition : plan.definitions)
      if (!bindings.owner || definition.diagnosticAnchor == bindings.owner)
        if (auto result = materialize(definition.dependencies);
            succeeded(result))
          return result;
  Operation *site = value.getDefiningOp();
  if (!site)
    site = bindings.owner;
  return site->emitOpError()
         << "hardware dependency is absent from the frozen plan";
}
FailureOr<std::string>
HardwareEmitContext::cppFormalName(StringRef name, Operation *site) const {
  auto legalized = identifier(name, site);
  if (failed(legalized))
    return failure();
  return "pyc_formal_" + *legalized;
}
FailureOr<std::string>
HardwareEmitContext::rtlFormalName(StringRef name, Operation *site) const {
  auto legalized = identifier(name, site);
  if (failed(legalized))
    return failure();
  return "pyc_parameter_" + *legalized;
}
FailureOr<std::string> HardwareEmitContext::staticValue(ac::StaticExprAttr expr,
                                                        Operation *site) const {
  return staticTree(*this, expr.getTree(), site, false);
}
bool HardwareEmitContext::isClosedStatic(
    ac::StaticExprAttr expr, const ac::HardwareBindings &bindings) const {
  return closedStatic(expr, bindings, this,
                      bindings.owner ? bindings.owner
                                     : plan.package.operator->());
}
FailureOr<std::string>
HardwareEmitContext::staticRtlValue(ac::StaticExprAttr expr,
                                    Operation *site) const {
  return staticTree(*this, expr.getTree(), site, true);
}
FailureOr<std::string>
HardwareEmitContext::staticValue(ac::StaticExprAttr expr,
                                 const ac::HardwareBindings &bindings,
                                 Operation *site) const {
  return staticTree(*this, expr.getTree(), site, false, &bindings);
}
FailureOr<std::string> HardwareEmitContext::unsignedStaticValue(
    ac::StaticExprAttr expression, Operation *site, bool rtl,
    const ac::HardwareBindings &bindings) const {
  auto kind = expression.getTree().getAs<StringAttr>("kind");
  Attribute frozenValue;
  if (kind && kind.getValue() == "literal") {
    auto literal = expression.getTree().getAs<DictionaryAttr>("value");
    frozenValue = literal ? literal.get("value") : Attribute();
  } else if (isClosedStatic(expression, bindings) &&
      (!kind || kind.getValue() != "reference")) {
    auto value = evaluatedStatic(expression, bindings);
    if (failed(value))
      return failure();
    frozenValue = *value;
  }
  if (frozenValue) {
    auto integer = dyn_cast<ac::MathIntAttr>(frozenValue);
    if (!integer)
      return site->emitOpError()
             << "unsigned static value requires a positive integer fitting u64";
    auto decimal = integer.getCanonicalValue().str();
    uint64_t unsignedValue = 0;
    if (decimal.starts_with("-") || decimal == "0" ||
        StringRef(decimal).getAsInteger(10, unsignedValue))
      return site->emitOpError()
             << "unsigned static value requires a positive integer fitting u64";
    return rtl ? "64'd" + decimal : decimal + "ULL";
  }
  // Queue analysis proves demanded symbolic values positive; prepare retains
  // the signed arithmetic/actual limits for those expressions. Preserve formal
  // names here and convert only the resulting value to the unsigned parameter.
  auto text = staticTree(*this, expression.getTree(), site, rtl, &bindings);
  if (failed(text))
    return failure();
  return rtl ? "$unsigned(64'(" + *text + "))" : "std::uint64_t(" + *text + ")";
}
FailureOr<std::string>
HardwareEmitContext::cppType(Type type, const ac::HardwareBindings &bindings,
                             Operation *site) const {
  auto resolved = resolvedType(type, bindings);
  if (failed(resolved))
    return failure();
  if (auto bits = dyn_cast<ac::BitsType>(*resolved)) {
    auto value = staticValue(bits.getWidth(), bindings, site);
    if (failed(value))
      return failure();
    return "gfsim::Bits<" + *value + ">";
  }
  return cppType(*resolved, site);
}
FailureOr<std::string> HardwareEmitContext::cppType(Type type,
                                                    Operation *site) const {
  if (auto bits = dyn_cast<ac::BitsType>(type)) {
    auto width = staticValue(bits.getWidth(), site);
    if (failed(width))
      return failure();
    return "gfsim::Bits<" + *width + ">";
  }
  if (auto table = dyn_cast<ac::TableType>(type)) {
    auto element = cppType(table.getElementType(), site);
    if (failed(element))
      return failure();
    std::string result = "gfsim::table<" + *element;
    for (auto raw : table.getShape()) {
      auto extent = staticValue(cast<ac::StaticExprAttr>(raw), site);
      if (failed(extent))
        return failure();
      result += ", " + *extent;
    }
    return result + ">";
  }
  if (auto param = dyn_cast<ac::TypeParamType>(type))
    return cppFormalName(param.getName().getValue(), site);
  if (auto enumeration = dyn_cast<ac::EnumType>(type)) {
    if (!lookupDefinition(enumeration.getName().getValue()))
      return failure();
    auto q = qualified(enumeration.getName().getValue(), site, "::");
    if (failed(q))
      return failure();
    return "::" + *q;
  }
  if (auto record = dyn_cast<ac::StructType>(type)) {
    auto q = qualified(record.getName().getValue(), site, "::");
    if (failed(q))
      return failure();
    return "::" + *q;
  }
  return site->emitOpError() << "unsupported C++ hardware type";
}
FailureOr<std::string> HardwareEmitContext::rtlType(Type type,
                                                    Operation *site) const {
  if (auto bits = dyn_cast<ac::BitsType>(type)) {
    auto width = staticTree(*this, bits.getWidth().getTree(), site, true);
    if (failed(width))
      return failure();
    return "logic [(" + *width + ")-1:0]";
  }
  if (auto table = dyn_cast<ac::TableType>(type)) {
    auto width = payloadWidth(table, site, true);
    if (failed(width))
      return failure();
    return "logic [(" + *width + ")-1:0]";
  }
  if (auto param = dyn_cast<ac::TypeParamType>(type))
    return rtlFormalName(param.getName().getValue(), site);
  Operation *declaration;
  if (auto enumeration = dyn_cast<ac::EnumType>(type)) {
    declaration = lookupDefinition(enumeration.getName().getValue());
  } else if (auto record = dyn_cast<ac::StructType>(type)) {
    declaration = lookupDefinition(record.getName().getValue());
  } else {
    return site->emitOpError() << "unsupported RTL hardware type";
  }
  if (!declaration)
    return site->emitOpError() << "unresolved RTL nominal type";
  auto q = qualified(SymbolTable::getSymbolName(declaration).getValue(), site,
                     "_");
  if (failed(q))
    return failure();
  if (rtlPackage)
    rtlPackage->types.insert(declaration);
  return "pycircuit_types::ac_" + *q + "_t";
}
FailureOr<std::string>
HardwareEmitContext::rtlType(Type type, const ac::HardwareBindings &bindings,
                             Operation *site) const {
  auto resolved = resolvedType(type, bindings);
  if (failed(resolved))
    return failure();
  if (isa<ac::BitsType, ac::TableType>(*resolved)) {
    auto width = payloadWidth(*resolved, site, true, bindings);
    if (failed(width))
      return failure();
    return "logic [(" + *width + ")-1:0]";
  }
  return rtlType(*resolved, site);
}
FailureOr<std::string>
HardwareEmitContext::cppInstanceType(ac::InstanceOp instance) const {
  auto *callee = resolveCallee(instance);
  if (!callee)
    return instance.emitOpError() << "unresolved C++ callee";
  std::string result;
  auto kind = primitiveKind(callee);
  if (!kind.empty())
    result = "::gfsim::" + kind.str();
  else
    result = names(callee).cppQualified;
  SmallVector<std::string> actuals;
  auto appendTypes = [&]() -> LogicalResult {
    for (auto raw : instance.getTypeArguments()) {
      auto type = cppType(cast<TypeAttr>(raw).getValue(), instance);
      if (failed(type))
        return failure();
      actuals.push_back(*type);
    }
    return success();
  };
  if (!kind.empty() && failed(appendTypes()))
    return failure();
  ac::HardwareBindings parent;
  parent.owner = instance->getParentOp();
  auto bindings = bindOccurrence(instance, parent);
  if (failed(bindings))
    return failure();
  for (auto raw : callee->getAttrOfType<ArrayAttr>("parameters")) {
    auto name = cast<DictionaryAttr>(raw).getAs<StringAttr>("name");
    auto value = bindings->integers.lookup(name.getValue());
    if (auto integer = dyn_cast<ac::MathIntAttr>(value))
      actuals.push_back(integer.getCanonicalValue().str());
    else {
      auto rendered =
          staticValue(cast<ac::StaticExprAttr>(value), *bindings, instance);
      if (failed(rendered))
        return failure();
      actuals.push_back(*rendered);
    }
  }
  if (kind.empty() && failed(appendTypes()))
    return failure();
  if (!actuals.empty())
    result += "<" + join(actuals, ",") + ">";
  return result;
}
template <class Occurrence>
static FailureOr<std::string>
rtlOccurrenceParameters(const HardwareEmitContext &context,
                        Occurrence instance) {
  SmallVector<std::string> assignments;
  Operation *callee = context.resolveCallee(instance);
  const bool primitive = !context.primitiveKind(callee).empty();
  auto formals = callee->getAttrOfType<ArrayAttr>("parameters");
  if (primitive) {
    for (auto [i, raw] : llvm::enumerate(instance.getParameters())) {
      DictionaryAttr formal = cast<DictionaryAttr>(formals[i]);
      auto name = formal.getAs<StringAttr>("name");
      auto value = staticTree(context, cast<ac::StaticExprAttr>(raw).getTree(),
                              instance, true);
      auto id = identifier(name.getValue(), instance);
      if (failed(value) || failed(id))
        return failure();
      assignments.push_back("." + *id + "(" + *value + ")");
    }
  } else {
    // Generated definitions declare value parameters before type parameters.
    // Positional overrides avoid a named override colliding with a local type
    // name. Reuse binding/substitution to materialize every declared default;
    // empty positional parameter slots are not portable SystemVerilog.
    ac::HardwareBindings parent;
    parent.owner = instance->getParentOp();
    auto bindings = context.bindOccurrence(instance, parent);
    if (failed(bindings))
      return failure();
    for (auto raw : formals) {
      auto name = cast<DictionaryAttr>(raw).getAs<StringAttr>("name");
      auto value = bindings->integers.lookup(name.getValue());
      if (auto integer = dyn_cast<ac::MathIntAttr>(value)) {
        assignments.push_back(integer.getCanonicalValue().str());
      } else {
        auto rendered =
            staticTree(context, cast<ac::StaticExprAttr>(value).getTree(),
                       instance, true, &*bindings);
        if (failed(rendered))
          return failure();
        assignments.push_back(*rendered);
      }
    }
  }
  auto typeFormals = callee->getAttrOfType<ArrayAttr>("type_parameters");
  for (auto [i, raw] : llvm::enumerate(instance.getTypeArguments())) {
    auto type = context.rtlType(cast<TypeAttr>(raw).getValue(), instance);
    if (failed(type))
      return failure();
    auto formalName = cast<StringAttr>(typeFormals[i]).getValue();
    if (primitive) {
      auto id = identifier(formalName, instance);
      if (failed(id))
        return failure();
      assignments.push_back("." + *id + "(" + *type + ")");
    } else {
      assignments.push_back(*type);
    }
  }
  return assignments.empty() ? std::string()
                             : " #( " + join(assignments, ", ") + " )";
}
FailureOr<std::string>
HardwareEmitContext::rtlInstanceParameters(ac::InstanceOp instance) const {
  return rtlOccurrenceParameters(*this, instance);
}
FailureOr<std::string>
HardwareEmitContext::rtlInstanceParameters(ac::CollectionOp collection) const {
  return rtlOccurrenceParameters(*this, collection);
}
FailureOr<std::string> HardwareEmitContext::cppWireExpression(
    Value value, const ac::HardwareBindings &, Operation *site) const {
  auto type = cppType(value.getType(), site);
  if (failed(type))
    return failure();
  return "gfsim::wire<" + *type + ">";
}
} // namespace acir::compiler
namespace acir::compiler {
FailureOr<ac::HardwareBindings> HardwareEmitContext::rootBindings() const {
  if (plan.rootModule >= plan.modules.size())
    return failure();
  return thawBindings(plan, plan.modules[plan.rootModule].concreteBindings);
}
FailureOr<std::string> HardwareEmitContext::rootCppType() const {
  auto bindings = rootBindings();
  if (failed(bindings))
    return failure();
  auto *root = bindings->owner;
  SmallVector<std::string> arguments;
  for (auto raw : root->getAttrOfType<ArrayAttr>("parameters"))
    arguments.push_back(
        cast<ac::MathIntAttr>(
            bindings->integers.lookup(
                cast<DictionaryAttr>(raw).getAs<StringAttr>("name").getValue()))
            .getCanonicalValue()
            .str());
  for (auto raw : root->getAttrOfType<ArrayAttr>("type_parameters")) {
    auto type =
        cppType(bindings->types.lookup(cast<StringAttr>(raw).getValue()),
                *bindings, root);
    if (failed(type))
      return failure();
    arguments.push_back(*type);
  }
  auto result = names(root).cppQualified;
  if (!arguments.empty())
    result += "<" + join(arguments, ", ") + ">";
  return result;
}
FailureOr<std::string> HardwareEmitContext::rootRtlParameters() const {
  auto bindings = rootBindings();
  if (failed(bindings))
    return failure();
  auto *root = bindings->owner;
  SmallVector<std::string> arguments;
  for (auto raw : root->getAttrOfType<ArrayAttr>("parameters")) {
    auto formal = cast<DictionaryAttr>(raw);
    auto name = formal.getAs<StringAttr>("name");
    auto id = rtlFormalName(name.getValue(), root);
    if (failed(id))
      return failure();
    auto value =
        cast<ac::MathIntAttr>(bindings->integers.lookup(name.getValue()))
            .getCanonicalValue();
    arguments.push_back("." + *id + "(" + value.str() + ")");
  }
  for (auto raw : root->getAttrOfType<ArrayAttr>("type_parameters")) {
    auto name = cast<StringAttr>(raw);
    auto id = rtlFormalName(name.getValue(), root);
    auto type = rtlType(bindings->types.lookup(name.getValue()), root);
    if (failed(id) || failed(type))
      return failure();
    arguments.push_back("." + *id + "(" + *type + ")");
  }
  return arguments.empty() ? std::string()
                           : " #( " + join(arguments, ", ") + " )";
}
} // namespace acir::compiler

namespace acir::compiler {
FailureOr<std::string>
HardwareEmitContext::shapeSize(ArrayAttr shape, Operation *site, bool rtl,
                               const ac::HardwareBindings &bindings) const {
  std::string result = "1";
  for (Attribute raw : shape) {
    auto text = staticTree(*this, cast<ac::StaticExprAttr>(raw).getTree(), site,
                           rtl, &bindings);
    if (failed(text))
      return failure();
    result = "(" + result + " * (" + *text + "))";
  }
  return result;
}
FailureOr<std::string>
HardwareEmitContext::payloadWidth(Type type, Operation *site, bool rtl,
                                  const ac::HardwareBindings &bindings) const {
  if (auto formal = dyn_cast<ac::TypeParamType>(type)) {
    auto actual = bindings.types.find(formal.getName().getValue());
    if (actual == bindings.types.end() || actual->second == type) {
      if (rtl) {
        auto name = rtlFormalName(formal.getName().getValue(), site);
        return failed(name) ? FailureOr<std::string>(failure())
                            : FailureOr<std::string>("$bits(" + *name + ")");
      }
      return cppFormalName(formal.getName().getValue(), site);
    }
    return payloadWidth(actual->second, site, rtl, bindings);
  }
  auto resolved = resolvedType(type, bindings);
  if (failed(resolved))
    return failure();
  type = *resolved;
  if (auto table = dyn_cast<ac::TableType>(type)) {
    auto count = shapeSize(table.getShape(), site, rtl, bindings);
    auto element = payloadWidth(table.getElementType(), site, rtl, bindings);
    if (failed(count) || failed(element))
      return failure();
    return "(" + *count + " * (" + *element + "))";
  }
  if (auto bits = dyn_cast<ac::BitsType>(type))
    return staticTree(*this, bits.getWidth().getTree(), site, rtl,
                      bindings.owner ? &bindings : nullptr);
  if (rtl && isa<ac::StructType, ac::EnumType>(type)) {
    // Nominal layouts are verified by common analysis. Render their finite
    // width directly: $bits(package::type) is not portable in packed ranges.
    auto width = packedWidth(type, bindings);
    if (failed(width))
      return failure();
    return std::to_string(*width);
  }
  auto text = rtl ? rtlType(type, site) : cppType(type, bindings, site);
  if (failed(text))
    return failure();
  return rtl ? "$bits(" + *text + ")"
             : "gfsim::hardware_traits<" + *text + ">::width";
}
FailureOr<std::string>
HardwareEmitContext::cppBoundModuleType(Operation *definition,
                                        const ac::HardwareBindings &bindings,
                                        Operation *site, bool kernel) const {
  StringRef kind = primitiveKind(definition);
  if (kernel && kind.empty())
    return site->emitOpError()
           << "only standard storage leaves have a kernel type";
  std::string name =
      kind.empty() ? names(definition).cppQualified : "::gfsim::" + kind.str();
  if (kernel)
    name = "::gfsim::" + (kind == "sync_mem_dp" ? "sync_mem" : kind.str()) +
           "_kernel";
  SmallVector<std::string> actuals;
  auto types = [&]() -> LogicalResult {
    for (auto raw : definition->getAttrOfType<ArrayAttr>("type_parameters")) {
      auto type = bindings.types.lookup(cast<StringAttr>(raw).getValue());
      auto text = cppType(type, bindings, site);
      if (failed(text))
        return failure();
      actuals.push_back(*text);
    }
    return success();
  };
  if (!kind.empty() && failed(types()))
    return failure();
  for (auto raw : definition->getAttrOfType<ArrayAttr>("parameters")) {
    auto attr = bindings.integers.lookup(
        cast<DictionaryAttr>(raw).getAs<StringAttr>("name").getValue());
    if (auto integer = dyn_cast<ac::MathIntAttr>(attr))
      actuals.push_back(integer.getCanonicalValue().str());
    else if (auto expr = dyn_cast<ac::StaticExprAttr>(attr)) {
      auto text = staticValue(expr, bindings, site);
      if (failed(text))
        return failure();
      actuals.push_back(*text);
    } else
      return site->emitOpError() << "missing bound module integer parameter";
  }
  if (kind.empty() && failed(types()))
    return failure();
  if (kernel && kind == "sync_mem_dp")
    actuals.push_back("2");
  return actuals.empty() ? name : name + "<" + join(actuals, ", ") + ">";
}
FailureOr<std::string>
HardwareEmitContext::viewIndex(ac::TableViewOp view, StringRef ordinal,
                               bool rtl,
                               const ac::HardwareBindings &bindings) const {
  auto input = cast<ac::TableType>(view.getInput().getType());
  auto output = cast<ac::TableType>(view.getResult().getType());
  auto kind = view.getKind();
  if (kind == "reshape")
    return ordinal.str();
  auto render = [&](Attribute raw) {
    return staticTree(*this, cast<ac::StaticExprAttr>(raw).getTree(), view, rtl,
                      &bindings);
  };
  auto renderedShape =
      [&](ArrayAttr shape) -> FailureOr<SmallVector<std::string>> {
    SmallVector<std::string> result;
    for (auto raw : shape) {
      auto text = render(raw);
      if (failed(text))
        return failure();
      result.push_back(*text);
    }
    return result;
  };
  auto inShape = renderedShape(input.getShape());
  auto outShape = renderedShape(output.getShape());
  if (failed(inShape) || failed(outShape))
    return failure();
  auto strides = [](ArrayRef<std::string> shape) {
    SmallVector<std::string> result(shape.size(), "1");
    for (size_t i = shape.size(); i > 1; --i)
      result[i - 2] = "(" + result[i - 1] + " * (" + shape[i - 1] + "))";
    return result;
  };
  auto inStride = strides(*inShape), outStride = strides(*outShape);
  SmallVector<std::string> coordinates;
  for (size_t i = 0; i < outShape->size(); ++i)
    coordinates.push_back("((" + ordinal.str() + " / (" + outStride[i] +
                          ")) % (" + (*outShape)[i] + "))");
  auto parameters = view.getParameters();
  std::string result = "0";
  for (size_t axis = 0; axis < inShape->size(); ++axis) {
    std::string coordinate;
    if (kind == "slice") {
      auto offset = render(parameters.getAs<ArrayAttr>("offsets")[axis]);
      auto stride = render(parameters.getAs<ArrayAttr>("strides")[axis]);
      if (failed(offset) || failed(stride))
        return failure();
      // A single output coordinate is always zero; omit a potentially huge
      // (but unobservable) static stride rather than narrowing it in C++.
      auto size = cast<ac::StaticExprAttr>(output.getShape()[axis]);
      bool singleton = false;
      if (isClosedStatic(size, bindings)) {
        auto n = evaluatedStatic(size, bindings);
        singleton = succeeded(n) &&
                    cast<ac::MathIntAttr>(*n).getCanonicalValue() == "1";
      }
      coordinate = singleton ? *offset
                             : "(" + *offset + " + " + coordinates[axis] +
                                   " * (" + *stride + "))";
    } else if (kind == "transpose") {
      coordinate = "0";
      auto axes = parameters.getAs<ArrayAttr>("axes");
      for (size_t j = 0; j < axes.size(); ++j) {
        auto mapped = render(axes[j]);
        if (failed(mapped))
          return failure();
        coordinate += " + ((" + *mapped + ") == " + std::to_string(axis) +
                      " ? " + coordinates[j] + " : 0)";
      }
      coordinate = "(" + coordinate + ")";
    } else if (kind == "rotate") {
      auto selected = render(parameters.get("axis"));
      auto offset = render(parameters.get("offset"));
      if (failed(selected) || failed(offset))
        return failure();
      std::string shift;
      bool closedShape = llvm::all_of(input.getShape(), [&](Attribute raw) {
        return isClosedStatic(cast<ac::StaticExprAttr>(raw), bindings);
      });
      if (closedShape &&
          isClosedStatic(cast<ac::StaticExprAttr>(parameters.get("offset")),
                         bindings) &&
          isClosedStatic(cast<ac::StaticExprAttr>(parameters.get("axis")),
                         bindings)) {
        const auto *planned = tableViewPlan(view, bindings);
        if (!planned)
          return view.emitOpError()
                 << "table view is absent from the frozen emission plan";
        uint64_t stride = 1;
        for (size_t j = axis + 1; j < planned->inputShape.size(); ++j)
          stride *= planned->inputShape[j];
        shift = std::to_string((planned->sourceOrdinalZero / stride) %
                               planned->inputShape[axis]);
      } else {
        auto offsetExpr = cast<ac::StaticExprAttr>(parameters.get("offset"));
        if (isClosedStatic(offsetExpr, bindings)) {
          auto value = evaluatedStatic(offsetExpr, bindings);
          if (failed(value))
            return failure();
          llvm::APSInt integer(
              cast<ac::MathIntAttr>(*value).getCanonicalValue());
          bool negative = integer.isNegative();
          llvm::APInt magnitude = negative
                                      ? -integer.sext(integer.getBitWidth() + 1)
                                      : llvm::APInt(integer);
          if (rtl) {
            llvm::SmallString<256> decimal;
            magnitude.toString(decimal, 10, false);
            auto remainder =
                "(" + std::to_string(std::max(1u, magnitude.getActiveBits())) +
                "'d" + decimal.str().str() + " % (" + (*inShape)[axis] + "))";
            shift = negative ? "(" + remainder + " == 0 ? 0 : (" +
                                   (*inShape)[axis] + ")-" + remainder + ")"
                             : remainder;
          } else {
            SmallVector<std::string> words;
            for (unsigned word = 0;
                 word < std::max(1u, (magnitude.getActiveBits() + 63) / 64);
                 ++word)
              words.push_back(std::to_string(magnitude.getRawData()[word]) +
                              "ULL");
            shift = "pyc_mod_words(std::array<std::uint64_t," +
                    std::to_string(words.size()) + ">{" + join(words, ",") +
                    "}, " + (negative ? "true" : "false") + ", " +
                    (*inShape)[axis] + ")";
          }
        } else
          shift = arithmeticHelper(*this, "mod", rtl).str() + "(" + *offset +
                  ", " + (*inShape)[axis] + ")";
      }
      auto rotated = "((" + coordinates[axis] + " + (" + shift + ")) % (" +
                     (*inShape)[axis] + "))";
      coordinate = "((" + *selected + ") == " + std::to_string(axis) + " ? " +
                   rotated + " : " + coordinates[axis] + ")";
    } else
      return view.emitOpError() << "unverified table view kind";
    result += " + (" + coordinate + ") * (" + inStride[axis] + ")";
  }
  return "(" + result + ")";
}
} // namespace acir::compiler

namespace acir::compiler {
FailureOr<std::string> HardwareEmitContext::cppBoundFamilyType(
    Operation *definition, const ac::HardwareBindings &bindings,
    Operation *site, StringRef count) const {
  if (!primitiveKind(definition).empty())
    return site->emitOpError()
           << "standard leaves use collection_storage of their kernel";
  auto type = cppBoundModuleType(definition, bindings, site);
  if (failed(type))
    return failure();
  auto base = names(definition).cppQualified;
  auto arguments = StringRef(*type).drop_front(base.size());
  std::string result = names(definition).cppFamilyQualified + "<" + count.str();
  if (!arguments.empty())
    result += ", " + arguments.drop_front().drop_back().str();
  return result + ">";
}

FailureOr<uint64_t>
observationPayloadWidth(HardwareEmitContext &context,
                        const ObservationEmissionOccurrence &occurrence) {
  if (occurrence.site.id >= context.plan.observations.size())
    return context.emitPlanError()
           << "observation payload is absent from the frozen plan";
  const auto &frozen = context.plan.observations[occurrence.site.id];
  uint64_t width = 1;
  if (occurrence.site.carriesValue) {
    if (occurrence.site.valueIndex >= frozen.payloadWidths.size())
      return context.emitPlanError()
             << "observation payload width is absent from the frozen plan";
    width = frozen.payloadWidths[occurrence.site.valueIndex];
  }
  if (width == 0 || width > std::numeric_limits<std::uint32_t>::max())
    return occurrence.site.diagnosticAnchor->emitOpError()
           << "observation payload width exceeds Runtime capacity";
  if (occurrence.site.gauge && width > 64)
    return occurrence.site.diagnosticAnchor->emitOpError()
           << "numeric report requires at most 64 bits";
  return width;
}
FailureOr<std::string>
cppObservationDescriptorTable(HardwareEmitContext &context) {
  const ObservationEmissionPlan &plan = context.observationPlan();
  // An unobserved design keeps the empty configuration and emits no table at
  // all, so its generated glue stays byte-identical to the previous emitter.
  if (plan.table.empty())
    return std::string();
  std::string text;
  llvm::raw_string_ostream out(text);
  out << "static constexpr gfsim::ObservationDescriptor "
         "pyc_observation_descriptors["
      << plan.table.size() << "] = {";
  for (auto [index, entry] : llvm::enumerate(plan.table)) {
    const auto &occurrence = plan.occurrences[index];
    auto width = observationPayloadWidth(context, occurrence);
    if (failed(width))
      return failure();
    out << "\n    {" << entry[0] << ", " << entry[1] << ", " << entry[2] << ", "
        << entry[3]
        << ", gfsim::ObservationKind::" << (entry[4] ? "Gauge" : "Event")
        << ", " << occurrence.site.valueIndex << ", "
        << occurrence.site.valueCount << ", "
        << *width << "},";
  }
  out << "\n};\n";
  out.flush();
  return text;
}
FailureOr<std::string>
cppObservationConfigureCall(HardwareEmitContext &context) {
  const ObservationEmissionPlan &plan = context.observationPlan();
  if (plan.table.empty())
    return std::string("observations_.Configure({}, 0)");
  return "observations_.Configure(pyc_observation_descriptors, " +
         std::to_string(plan.events) + ")";
}
FailureOr<std::string> cppObservationChildBase(HardwareEmitContext &context,
                                               ac::ModuleOp definition,
                                               Operation *child,
                                               const ac::HardwareBindings &bindings) {
  const ObservationEmissionPlan &plan = context.observationPlan();
  auto id = context.occurrenceId(child, bindings);
  if (failed(id))
    return failure();
  auto base = plan.childBaseUnits.find(*id);
  if (base == plan.childBaseUnits.end())
    return definition.emitOpError() << "native observation child is unplanned";
  return std::to_string(base->second);
}
FailureOr<uint64_t> cppObservationLaneStride(HardwareEmitContext &context,
                                             ac::ModuleOp definition) {
  const ObservationEmissionPlan &plan = context.observationPlan();
  auto sites = plan.definitions.find(definition);
  return sites == plan.definitions.end()
             ? uint64_t{0}
             : static_cast<uint64_t>(sites->second.size());
}
FailureOr<uint64_t> cppObservationSubtreeWidth(HardwareEmitContext &context,
                                               ac::ModuleOp definition) {
  return context.observationPlan().subtreeWidth.lookup(definition);
}
LogicalResult
cppObservationStaging(HardwareEmitContext &context, ac::ModuleOp definition,
                      const ac::HardwareBindings &bindings,
                      llvm::StringRef object, llvm::StringRef count,
                      const std::function<FailureOr<std::string>(Value)> &value,
                      llvm::raw_ostream &out) {
  const ObservationEmissionPlan &plan = context.observationPlan();
  auto sites = plan.definitions.find(definition);
  if (sites == plan.definitions.end() || sites->second.empty())
    return success();
  // One staged value per descriptor, so every observed operand keeps its own
  // `ObservationDescriptor` and therefore its own globally unique ordinal.
  for (auto [index, site] : llvm::enumerate(sites->second)) {
    auto observe = cast<ac::SourceObserveOp>(site.diagnosticAnchor);
    auto path = value(observe.getPath());
    if (failed(path))
      return failure();
    auto staged =
        site.carriesValue ? value(observe.getValues()[site.valueIndex]) : path;
    if (failed(staged))
      return failure();
    Value observed = site.carriesValue ? observe.getValues()[site.valueIndex]
                                       : observe.getPath();
    const auto *type = context.concreteValueType(observed, bindings);
    if (!type)
      return observe.emitOpError()
             << "observation payload is absent from the frozen type plan";
    uint64_t width = type->packedWidth;
    if (site.gauge && width > 64)
      return observe.emitOpError() << "numeric report requires at most 64 bits";
    auto payload = context.cppType(observed.getType(), bindings, observe);
    if (failed(payload))
      return failure();
    // A Gauge is published as the raw unsigned bit pattern because
    // `ObservationSlots::Precommit` accepts only `Unsigned` gauges; a one-bit
    // Event keeps the source `bool` shape the runner protocol carries.
    out << "    static_assert(gfsim::hardware_traits<" << *payload
        << ">::width == " << width
        << ", \"native observation width must be resolved at emission\");\n"
        << "    for (std::size_t pyc_lane = 0; pyc_lane < " << count
        << "; ++pyc_lane) {\n"
        << "      const auto pyc_observed = " << *staged
        << ".element(pyc_lane).packed();\n"
        << "      const auto pyc_observed_path = " << *path
        << ".element(pyc_lane).packed();\n"
        << "      " << object << (object.empty() ? "" : "template ")
        << "pyc_stage_observation<"
        << (site.gauge
                ? "false"
                : "(gfsim::hardware_traits<" + *payload + ">::width == 1)")
        << ">(pyc_lane, " << index
        << ", pyc_observed, pyc_observed_path.isFullyKnown(), "
        << "pyc_observed_path.isFullyKnown() && "
           "static_cast<bool>(pyc_observed_path.value().value() & 1u));\n"
        << "    }\n";
  }
  return success();
}
} // namespace acir::compiler

namespace acir::compiler {
bool HardwareEmitContext::isSystem() {
  return plan.rootInterface.closedSystem;
}
FailureOr<ObservationEmissionPlan>
observationEmissionPlan(HardwareEmitContext &context) {
  return context.observationPlan();
}
namespace {
FailureOr<llvm::json::Value> metadataValue(Attribute value, Operation *site) {
  if (auto item = dyn_cast<StringAttr>(value))
    return llvm::json::Value(item.getValue());
  if (auto item = dyn_cast<FlatSymbolRefAttr>(value))
    return llvm::json::Value(item.getValue());
  if (auto item = dyn_cast<BoolAttr>(value))
    return llvm::json::Value(item.getValue());
  if (auto item = dyn_cast<IntegerAttr>(value)) {
    if (item.getValue().getActiveBits() > 64)
      return site->emitOpError() << "emission metadata integer exceeds u64";
    return llvm::json::Value(item.getValue().getZExtValue());
  }
  if (auto item = dyn_cast<ac::MathIntAttr>(value))
    return llvm::json::Value(item.getCanonicalValue());
  if (auto items = dyn_cast<ArrayAttr>(value)) {
    llvm::json::Array result;
    for (Attribute item : items) {
      auto converted = metadataValue(item, site);
      if (failed(converted))
        return failure();
      result.push_back(std::move(*converted));
    }
    return llvm::json::Value(std::move(result));
  }
  if (auto items = dyn_cast<DictionaryAttr>(value)) {
    llvm::json::Object result;
    for (auto item : items) {
      auto converted = metadataValue(item.getValue(), site);
      if (failed(converted))
        return failure();
      result[item.getName().strref()] = std::move(*converted);
    }
    return llvm::json::Value(std::move(result));
  }
  return site->emitOpError() << "unsupported emission metadata attribute";
}
std::string cppMetadataString(StringRef value) {
  std::string text;
  llvm::raw_string_ostream out(text);
  out << '"';
  out.write_escaped(value, false);
  out << '"';
  return text;
}
} // namespace
FailureOr<std::string> emissionMetadataJson(Attribute value, Operation *site) {
  auto converted = metadataValue(value, site);
  if (failed(converted))
    return failure();
  std::string text;
  llvm::raw_string_ostream out(text);
  out << *converted;
  return text;
}
FailureOr<std::string>
observationSiteMetadataJson(ac::SourceObserveOp observe) {
  auto identity = observe->getAttrOfType<DictionaryAttr>("ac.observation_id");
  if (failed(ac::detail::verifyObservationID(
          identity, [&] { return observe.emitOpError(); })))
    return failure();
  Builder builder(observe.getContext());
  return emissionMetadataJson(
      builder.getDictionaryAttr(
          {builder.getNamedAttr("site", identity.get("site")),
           builder.getNamedAttr("invocation", identity.get("invocation")),
           builder.getNamedAttr("ordinal", identity.get("ordinal"))}),
      observe);
}
FailureOr<std::string>
cppObservationRunnerMetadata(HardwareEmitContext &context) {
  const ObservationEmissionPlan &plan = context.observationPlan();
  auto root = observationRoot(context);
  if (failed(root))
    return failure();
  std::string text;
  llvm::raw_string_ostream out(text);
  out << "inline std::span<const gfsim::RunnerObservation> "
         "pyc_observation_metadata() {\n";
  if (plan.occurrences.empty()) {
    out << "  return {};\n}\n";
    return text;
  }
  out << "  static const gfsim::RunnerObservation entries[] = {\n";
  for (const auto &entry : plan.occurrences) {
    std::string instance = "root";
    for (const ObservationEmissionPathEntry &path : entry.logicalOwnerPath) {
      instance += "/" + path.instanceName;
      if (path.collection)
        instance += "[" + std::to_string(path.lane) + "]";
    }
    out << "    {" << entry.ordinal << ", "
        << cppMetadataString(entry.site.kind) << ", "
        << cppMetadataString(instance) << ", "
        << cppMetadataString(entry.site.registrationJson) << ", "
        << cppMetadataString(entry.site.siteJson) << ", "
        << cppMetadataString(entry.site.specJson) << ", "
        << (entry.site.carriesValue ? "true" : "false") << ", "
        << cppMetadataString(entry.site.reportName) << ", "
        << entry.site.valueIndex << ", "
        << entry.site.valueCount << "},\n";
  }
  out << "  };\n  return entries;\n}\n";
  return text;
}
} // namespace acir::compiler
