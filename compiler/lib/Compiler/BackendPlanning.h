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
  static mlir::FailureOr<const ac::HardwarePhysicalProjection *>
  physicalProjection(const VerifiedHardwareEmissionPlan &hardware);
};

/// Operation-level target planning from a verified backend-neutral plan.
/// These functions are private compiler implementation details and are absent
/// from the installed CompilerDev header closure.
mlir::FailureOr<FinalCppSourceParts>
prepareCppSourceParts(const VerifiedHardwareEmissionPlan &hardware);

mlir::FailureOr<FinalVerilogSourceParts>
prepareVerilogSourceParts(const VerifiedHardwareEmissionPlan &hardware,
                          bool physical = false);

} // namespace acir::compiler

#endif // PYCIRCUIT_BACKEND_PLANNING_H
