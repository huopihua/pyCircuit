#ifndef PYCIRCUIT_TEST_HARDWARE_EMIT_CONTEXT_H
#define PYCIRCUIT_TEST_HARDWARE_EMIT_CONTEXT_H

#include "Compiler/BackendPlanning.h"
#include "Compiler/HardwareEmitCommon.h"

#include <memory>

namespace acir::compiler::test {

/// Owns the verified plan for the lifetime of a test-only emission context.
/// Tests must enter emission through the same frozen planning boundary as the
/// public backends instead of reconstructing the retired live-analysis view.
class HardwareEmitTestContext final {
public:
  static mlir::FailureOr<std::unique_ptr<HardwareEmitTestContext>>
  prepare(mlir::ModuleOp package) {
    auto hardware = prepareHardwareEmissionPlan(package);
    if (mlir::failed(hardware))
      return mlir::failure();
    return std::unique_ptr<HardwareEmitTestContext>(
        new HardwareEmitTestContext(std::move(*hardware)));
  }

  HardwareEmitContext &emitter() { return context; }
  const VerifiedHardwareEmissionPlan &hardware() const { return plan; }

private:
  explicit HardwareEmitTestContext(VerifiedHardwareEmissionPlan plan)
      : plan(std::move(plan)),
        context(this->plan.get(), BackendPlanningAccess::sourceChecks(this->plan),
                BackendPlanningAccess::observationPlan(this->plan), nullptr) {}

  VerifiedHardwareEmissionPlan plan;
  HardwareEmitContext context;
};

} // namespace acir::compiler::test

#endif // PYCIRCUIT_TEST_HARDWARE_EMIT_CONTEXT_H
