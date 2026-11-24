#ifndef PYCIRCUIT_BACKEND_PLANNING_H
#define PYCIRCUIT_BACKEND_PLANNING_H

#include "FinalCppSourceParts.h"
#include "FinalVerilogSourceParts.h"
#include "HardwareEmissionPlan.h"
#include "pycircuit/Dialect/ACIR/HardwareAnalysis.h"

namespace acir::compiler {

struct ObservationEmissionPlan;

struct BackendPlanningAccess {
  static const ac::HardwareSourceCheckPlan &
  sourceChecks(const VerifiedHardwareEmissionPlan &hardware);
  static std::shared_ptr<const ObservationEmissionPlan> &
  observationPlan(const VerifiedHardwareEmissionPlan &hardware);
  static const ac::HardwarePhysicalProjection *
  physicalProjection(const VerifiedHardwareEmissionPlan &hardware);
  static llvm::StringRef physicalProjectionUnavailableReason(
      const VerifiedHardwareEmissionPlan &hardware);
};

/// Target lowering from a verified backend-neutral plan into complete native
/// source payloads. This migration bridge may read the plan's private,
/// read-only final-ACIR anchors for target-specific lowering, but it must use
/// frozen plan inventories for common scheduling, dependencies, source units,
/// and publication. These functions are private compiler implementation
/// details and are absent from the installed CompilerDev header closure.
mlir::FailureOr<FinalCppSourceParts>
lowerCppSourceParts(const VerifiedHardwareEmissionPlan &hardware);

mlir::FailureOr<FinalVerilogSourceParts>
lowerVerilogSourceParts(const VerifiedHardwareEmissionPlan &hardware,
                        bool physical = false);

} // namespace acir::compiler

#endif // PYCIRCUIT_BACKEND_PLANNING_H
