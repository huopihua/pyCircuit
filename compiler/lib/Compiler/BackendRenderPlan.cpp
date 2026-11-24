#include "BackendRenderPlan.h"
#include "BackendPlanning.h"
#include "FinalCppNames.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/JSON.h"

using namespace mlir;

namespace acir::compiler {
namespace {
FailureOr<HardwareRenderMetadata>
metadataFrom(const HardwareEmissionPlan &hardware,
             ac::detail::EmitError emitError) {
  HardwareRenderMetadata result;
  auto owner = freezeFinalSourceOwner(hardware.entry.sourceOwner, emitError);
  if (failed(owner))
    return failure();
  result.implementationDefinition = hardware.entry.implementationDefinition;
  result.logicalDefinition = hardware.entry.logicalDefinition;
  result.entrySource = std::move(*owner);
  result.logicalArguments = hardware.entry.logicalArguments;
  result.resetSemantics = hardware.resetSemantics;
  result.execution = hardware.execution;
  result.systemSamples = hardware.systemSamples;
  result.needsFourState = hardware.needsFourState;
  return result;
}

SmallVector<FinalSourceMembership>
emptyMembership(const HardwareEmissionPlan &hardware,
                ac::detail::EmitError emitError, LogicalResult &status) {
  SmallVector<FinalSourceMembership> result;
  for (const HardwareEmissionSourceUnitPlan &unit : hardware.sourceUnits) {
    auto owner = freezeFinalSourceOwner(unit.sourceOwner, emitError);
    if (failed(owner)) {
      status = failure();
      return {};
    }
    result.push_back({std::move(*owner), {}});
  }
  status = success();
  return result;
}

FailureOr<unsigned> membershipIndex(ArrayRef<FinalSourceMembership> memberships,
                                    const FinalSourceOwner &owner,
                                    ac::detail::EmitError emitError) {
  auto found = llvm::find_if(memberships, [&](const auto &membership) {
    return membership.sourceOwner == owner;
  });
  if (found == memberships.end())
    return emitError() << "native source group has no final SourceOwner";
  return static_cast<unsigned>(found - memberships.begin());
}

FailureOr<std::string>
rootPortMap(const HardwareEmissionPlan &hardware, bool rtl, bool physical,
            ac::detail::EmitError emitError) {
  if (hardware.rootModule >= hardware.modules.size())
    return failure();
  const HardwareEmissionModulePlan &root = hardware.modules[hardware.rootModule];
  const HardwareEmissionRootInterfacePlan &interface = hardware.rootInterface;
  llvm::json::Array inputs, outputs, controls;
  auto describe = [&](const HardwareEmissionPortPlan &port,
                      unsigned ordinal) -> FailureOr<llvm::json::Object> {
    auto wire = legalizeIdentifier(port.name, emitError);
    if (failed(wire))
      return failure();
    llvm::json::Array fields;
    for (const HardwareEmissionPackedField &field : port.type.fields) {
      llvm::json::Array path;
      for (const std::string &part : field.path)
        path.emplace_back(part);
      fields.emplace_back(llvm::json::Object{{"path", std::move(path)},
                                             {"low", field.low},
                                             {"width", field.width}});
    }
    return llvm::json::Object{{"ordinal", ordinal},
                              {"name", port.name},
                              {"wire", *wire},
                              {"width", port.type.packedWidth},
                              {"fields", std::move(fields)}};
  };
  for (unsigned index = 0; index < interface.publicInputCount; ++index) {
    auto row = describe(root.inputs[index], index);
    if (failed(row))
      return failure();
    inputs.emplace_back(std::move(*row));
  }
  llvm::DenseSet<unsigned> companions;
  for (unsigned index = 0; index < interface.publicOutputCount; ++index) {
    const HardwareEmissionPortPlan &port = root.outputs[index];
    auto row = describe(port, index);
    if (failed(row))
      return failure();
    (*row)["availability"] = nullptr;
    if (port.availabilityOutput) {
      const HardwareEmissionPortPlan &companion =
          root.outputs[*port.availabilityOutput];
      auto wire = legalizeIdentifier(companion.name, emitError);
      if (failed(wire))
        return failure();
      (*row)["availability"] = *wire;
      if (companions.insert(*port.availabilityOutput).second)
        controls.emplace_back(llvm::json::Object{{"direction", "output"},
                                                 {"role", "availability"},
                                                 {"wire", *wire},
                                                 {"width", 1}});
    }
    outputs.emplace_back(std::move(*row));
  }
  llvm::json::Object domains{{"clock", nullptr}, {"reset", nullptr}};
  if (interface.clockInput)
    domains["clock"] = *interface.clockInput;
  if (interface.resetInput)
    domains["reset"] = *interface.resetInput;
  if (rtl && !physical &&
      (interface.closedSystem || !hardware.checks.empty())) {
    auto control = [&](StringRef direction, StringRef role, StringRef wire,
                       unsigned width) {
      controls.emplace_back(llvm::json::Object{{"direction", direction},
                                               {"role", role},
                                               {"wire", wire},
                                               {"width", width}});
    };
    control("input", "phase", "pyc_phase", 3);
    control("input", "commit_permission", "pyc_root_commit_ok", 1);
    control("output", "check_complete", "pyc_check_complete", 1);
    control("output", "sampling_error", "pyc_sampling_error", 1);
    control("output", "local_error", "pyc_local_error", 1);
  }
  llvm::json::Value map(llvm::json::Object{{"schema_version", 1},
                                           {"target", rtl ? "verilog" : "cpp"},
                                           {"inputs", std::move(inputs)},
                                           {"outputs", std::move(outputs)},
                                           {"domains", std::move(domains)},
                                           {"controls", std::move(controls)}});
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << map;
  return text;
}
} // namespace

FailureOr<VerifiedCppRenderPlan>
prepareCppRenderPlan(const VerifiedHardwareEmissionPlan &hardware) {
  const HardwareEmissionPlan &common = hardware.get();
  if (!common.package)
    return failure();
  Operation *package = common.package;
  auto emitError = [&]() -> InFlightDiagnostic {
    return package->emitOpError();
  };
  auto parts = lowerCppSourceParts(hardware);
  if (failed(parts))
    return failure();
  auto portMap = rootPortMap(common, false, false, emitError);
  if (failed(portMap))
    return failure();
  parts->portMap = std::move(*portMap);
  parts->sourceCheckCount = common.checks.size();
  LogicalResult membershipStatus = success();
  auto memberships = emptyMembership(common, emitError, membershipStatus);
  if (failed(membershipStatus))
    return failure();
  for (const FinalCppSourceGroup &group : parts->sourceGroups) {
    auto index = membershipIndex(memberships, group.sourceOwner, emitError);
    if (failed(index))
      return failure();
    memberships[*index].generatedFiles.push_back(group.headerPath);
    if (!group.sourcePath.empty())
      memberships[*index].generatedFiles.push_back(group.sourcePath);
  }
  auto maps = emitFinalSourceMaps(common.sourceMaps, memberships, emitError);
  auto metadata = metadataFrom(common, emitError);
  if (failed(maps) || failed(metadata))
    return failure();
  CppRenderPlan plan{std::move(*metadata), std::move(*parts)};
  plan.metadata.sourceMaps = std::move(*maps);
  return verifyAndFreezeCppRenderPlan(std::move(plan), emitError);
}

FailureOr<VerifiedVerilogRenderPlan>
prepareVerilogRenderPlan(const VerifiedHardwareEmissionPlan &hardware,
                         bool physical) {
  const HardwareEmissionPlan &common = hardware.get();
  if (!common.package)
    return failure();
  Operation *package = common.package;
  auto emitError = [&]() -> InFlightDiagnostic {
    return package->emitOpError();
  };
  auto parts = lowerVerilogSourceParts(hardware, physical);
  if (failed(parts))
    return failure();
  auto portMap = rootPortMap(common, true, physical, emitError);
  if (failed(portMap))
    return failure();
  parts->portMap = std::move(*portMap);
  parts->sourceCheckCount = common.checks.size();
  LogicalResult membershipStatus = success();
  auto memberships = emptyMembership(common, emitError, membershipStatus);
  if (failed(membershipStatus))
    return failure();
  for (const FinalVerilogSourceGroup &group : parts->sourceGroups) {
    auto index = membershipIndex(memberships, group.sourceOwner, emitError);
    if (failed(index))
      return failure();
    memberships[*index].generatedFiles.push_back(group.path);
  }
  auto maps = emitFinalSourceMaps(common.sourceMaps, memberships, emitError);
  auto metadata = metadataFrom(common, emitError);
  if (failed(maps) || failed(metadata))
    return failure();
  VerilogRenderPlan plan{std::move(*metadata), std::move(*parts), physical};
  plan.metadata.sourceMaps = std::move(*maps);
  return verifyAndFreezeVerilogRenderPlan(std::move(plan), emitError);
}
} // namespace acir::compiler
