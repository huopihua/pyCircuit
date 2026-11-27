#ifndef ACIR_LIB_COMPILER_SOURCELINK_H
#define ACIR_LIB_COMPILER_SOURCELINK_H

#include "SourceUnit.h"
#include "pycircuit/Analysis/CompilerPhaseMetrics.h"

namespace acir::compiler {

struct SourceLinkUnit {
  mlir::ModuleOp body;
  mlir::ModuleOp header;
};

// Diagnostic snapshots of the same link pipeline, never source authority.
// Updated only after complete link success; failure preserves prior contents.
struct SourceLinkStages {
  std::string beforeTransactions;
  std::string routedTransactions;
  std::string afterTransactions;
  std::string beforeEntryNormalization;
  std::string normalizedEntry;
};

struct SourceLinkMetrics {
  acir::metrics::CompilerPhaseMetrics transactionPlanning;
  acir::metrics::CompilerPhaseMetrics routing;
  acir::metrics::CompilerPhaseMetrics proof;
  acir::metrics::CompilerPhaseMetrics finalVerification;
};

struct SourceLinkParameterBinding {
  std::string name;
  std::string value; // Canonical signed decimal, never a host-width integer.
};

mlir::FailureOr<SourceHeaderRegistry>
admitSourceLinkUnits(llvm::ArrayRef<SourceLinkUnit> units,
                     ac::detail::EmitError emitError);

enum class SourceLinkTarget { Routed, Hardware };

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
linkSourceUnits(llvm::ArrayRef<SourceLinkUnit> units, llvm::StringRef top,
                  ac::detail::EmitError emitError,
                  SourceLinkTarget target = SourceLinkTarget::Hardware,
                  SourceLinkStages *retainedStages = nullptr,
                  llvm::ArrayRef<SourceLinkParameterBinding> parameters = {},
                  llvm::StringRef resetSemantics = "strict",
                  llvm::StringRef execution = "",
                  SourceLinkMetrics *metrics = nullptr);

} // namespace acir::compiler

#endif // ACIR_LIB_COMPILER_SOURCELINK_H
