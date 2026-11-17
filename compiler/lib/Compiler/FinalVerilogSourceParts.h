#ifndef PYCIRCUIT_FINAL_VERILOG_SOURCE_PARTS_H
#define PYCIRCUIT_FINAL_VERILOG_SOURCE_PARTS_H
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <string>
#include <vector>
namespace acir::compiler {
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
} // namespace acir::compiler
#endif
