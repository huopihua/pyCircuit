#ifndef PYCIRCUIT_TEST_HARDWARE_EMIT_CONTEXT_H
#define PYCIRCUIT_TEST_HARDWARE_EMIT_CONTEXT_H

#include "Compiler/BackendPlanning.h"
#include "Compiler/HardwareEmitCommon.h"
#include "mlir/IR/Diagnostics.h"

#include <memory>

namespace acir::compiler::test {

/// Owns the verified plan for the lifetime of a test-only emission context.
/// Tests must enter emission through the same frozen planning boundary as the
/// public backends instead of reconstructing the retired live-analysis view.
class HardwareEmitTestContext final {
public:
  static mlir::FailureOr<std::unique_ptr<HardwareEmitTestContext>>
  prepare(mlir::ModuleOp package) {
    mlir::MLIRContext *sourceContext = package.getContext();
    auto hardware = prepareHardwareEmissionPlan(package);
    if (mlir::failed(hardware))
      return mlir::failure();
    return std::unique_ptr<HardwareEmitTestContext>(
        new HardwareEmitTestContext(std::move(*hardware), sourceContext));
  }

  HardwareEmitContext &emitter() { return context; }
  const VerifiedHardwareEmissionPlan &hardware() const { return plan; }

private:
  explicit HardwareEmitTestContext(VerifiedHardwareEmissionPlan plan,
                                   mlir::MLIRContext *sourceContext)
      : plan(std::move(plan)),
        context(this->plan.get(),
                BackendPlanningAccess::sourceChecks(this->plan),
                BackendPlanningAccess::observationPlan(this->plan), nullptr),
        diagnosticForwarder(std::make_unique<mlir::ScopedDiagnosticHandler>(
            context.mlirContext(),
            [sourceContext](mlir::Diagnostic &diagnostic) {
              std::string text;
              llvm::raw_string_ostream out(text);
              diagnostic.print(out);
              out.flush();
              mlir::emitError(mlir::UnknownLoc::get(sourceContext)) << text;
              return mlir::success();
            })) {}

  VerifiedHardwareEmissionPlan plan;
  HardwareEmitContext context;
  std::unique_ptr<mlir::ScopedDiagnosticHandler> diagnosticForwarder;
};

} // namespace acir::compiler::test

#endif // PYCIRCUIT_TEST_HARDWARE_EMIT_CONTEXT_H
