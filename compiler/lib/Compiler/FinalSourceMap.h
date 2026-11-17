#ifndef ACIR_LIB_COMPILER_FINALSOURCEMAP_H
#define ACIR_LIB_COMPILER_FINALSOURCEMAP_H

#include "Dialect/ACIR/ACIRSourceContracts.h"
#include "mlir/IR/BuiltinOps.h"
#include "pycircuit/Dialect/ACIR/SourceUnitValidation.h"

#include <string>

namespace acir::compiler {

// Native files emitted for one final SourceOwner. Empty generatedFiles is
// valid for a final declaration unit with no RTL source file.
struct FinalSourceMembership {
  mlir::DictionaryAttr sourceOwner;
  llvm::SmallVector<std::string> generatedFiles;
};

struct FinalSourceOrigin {
  std::string operation;
  std::string origin;
  std::string location;
};

// Frozen provenance for one final SourceOwner. Plan construction owns the only
// IR walk; publication later merges renderer-owned native file membership.
struct FinalSourceMapPlan {
  mlir::DictionaryAttr sourceOwner;
  std::string path;
  llvm::SmallVector<FinalSourceOrigin> origins;
};

struct FinalSourceMap {
  mlir::DictionaryAttr sourceOwner;
  std::string path;
  std::string text;
  llvm::SmallVector<std::string> generatedFiles;
};

mlir::FailureOr<llvm::SmallVector<FinalSourceMapPlan>>
prepareFinalSourceMaps(llvm::ArrayRef<ac::FinalSourceUnitView> units,
                       ac::detail::EmitError emitError);

mlir::LogicalResult verifyFinalSourceMapPlan(const FinalSourceMapPlan &plan,
                                             ac::detail::EmitError emitError);

// Merge renderer-owned native membership into frozen provenance. Membership
// comes from the selected native emitter, never from generated text or paths.
mlir::FailureOr<llvm::SmallVector<FinalSourceMap>>
emitFinalSourceMaps(llvm::ArrayRef<FinalSourceMapPlan> plans,
                    llvm::ArrayRef<FinalSourceMembership> membership,
                    ac::detail::EmitError emitError);

// Validate a saved map's exact native schema and canonical MLIR payloads.
// Historical source authenticity and external file ownership are out of scope.
mlir::LogicalResult
validateFinalSourceMapText(llvm::StringRef text, mlir::MLIRContext &context,
                           ac::detail::EmitError emitError,
                           std::string *canonicalPath = nullptr);

} // namespace acir::compiler

#endif // ACIR_LIB_COMPILER_FINALSOURCEMAP_H
