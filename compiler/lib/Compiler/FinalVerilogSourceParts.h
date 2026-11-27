#ifndef PYCIRCUIT_FINAL_VERILOG_SOURCE_PARTS_H
#define PYCIRCUIT_FINAL_VERILOG_SOURCE_PARTS_H
#include "pycircuit/Dialect/ACIR/HardwareAnalysis.h"
#include <cstdint>
#include <string>
#include <vector>
namespace acir::compiler {
class HardwareEmitContext;
struct FinalVerilogSourceGroup {
  mlir::DictionaryAttr sourceOwner;
  llvm::SmallVector<mlir::FlatSymbolRefAttr> definitions;
  std::string path, text;
};
struct FinalVerilogSourceParts {
  llvm::SmallVector<FinalVerilogSourceGroup, 0> sourceGroups;
  llvm::SmallVector<std::string> standardSources;
  std::string core, runtimeGlue, rootRtlName;
  std::string simulationTop, portMap;
  uint64_t sourceCheckCount = 0;
};
mlir::FailureOr<FinalVerilogSourceParts>
emitVerilogSourceParts(mlir::ModuleOp package, ac::HardwareAnalysis &analysis,
                       bool physical = false);
mlir::FailureOr<FinalVerilogSourceParts>
emitPreparedVerilogSourceParts(HardwareEmitContext &context);
void renderVerilogSourceParts(const FinalVerilogSourceParts &parts,
                              llvm::raw_ostream &out);
mlir::LogicalResult emitVerilog(mlir::ModuleOp package,
                                ac::HardwareAnalysis &analysis,
                                llvm::raw_ostream &out, bool physical = false);
} // namespace acir::compiler
#endif
