#include "pycircuit/Dialect/ACIR/SourceTransactionStateAnalysis.h"
#include "pycircuit/Transforms/Passes.h"
#include "pycircuit/Analysis/CompilerPhaseMetrics.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace acir {
namespace {

class AnalyzeTransactionStateSSAPass
    : public PassWrapper<AnalyzeTransactionStateSSAPass,
                         OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeTransactionStateSSAPass)
  AnalyzeTransactionStateSSAPass() = default;
  explicit AnalyzeTransactionStateSSAPass(
      acir::metrics::CompilerPhaseMetrics *metrics)
      : metrics(metrics) {}
  AnalyzeTransactionStateSSAPass(const AnalyzeTransactionStateSSAPass &other)
      : PassWrapper(other), metrics(other.metrics) {}
  StringRef getArgument() const final {
    return "ac-analyze-transaction-state-ssa";
  }
  StringRef getDescription() const final {
    return "Analyze transaction-local candidate state, source paths and "
           "admission controls";
  }
  void runOnOperation() final {
    auto &analysis = getAnalysis<ac::SourceTransactionStateAnalysis>();
    if (metrics)
      analysis.collectMetrics(*metrics);
    if (!analysis.isValid()) {
      signalPassFailure();
      return;
    }
    if (metrics)
      metrics->status = "complete";
    markAllAnalysesPreserved();
  }
private:
  acir::metrics::CompilerPhaseMetrics *metrics = nullptr;
};

} // namespace

std::unique_ptr<Pass> createAnalyzeTransactionStateSSAPass(
    acir::metrics::CompilerPhaseMetrics *metrics) {
  return std::make_unique<AnalyzeTransactionStateSSAPass>(metrics);
}

} // namespace acir
