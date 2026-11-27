#ifndef ACIR_TRANSFORMS_PASSES_H
#define ACIR_TRANSFORMS_PASSES_H
#include <memory>
namespace mlir {
class Pass;
}
namespace acir::metrics {
struct CompilerPhaseMetrics;
}
namespace acir {
std::unique_ptr<mlir::Pass> createNormalizeSourceControlFlowPass();
std::unique_ptr<mlir::Pass> createExtractSourceInterfacePass();
std::unique_ptr<mlir::Pass> createSimplifyRecordWiresPass();
std::unique_ptr<mlir::Pass> createAnalyzeSourceMethodSummariesPass();
std::unique_ptr<mlir::Pass> createAnalyzeTransactionStateSSAPass(
    acir::metrics::CompilerPhaseMetrics *metrics = nullptr);
void registerACIRPasses();
} // namespace acir
#endif
