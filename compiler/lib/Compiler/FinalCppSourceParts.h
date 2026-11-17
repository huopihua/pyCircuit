#ifndef PYCIRCUIT_FINAL_CPP_SOURCE_PARTS_H
#define PYCIRCUIT_FINAL_CPP_SOURCE_PARTS_H
#include "mlir/IR/BuiltinAttributes.h"
#include <cstdint>
#include <string>
#include <vector>
namespace acir::compiler {
struct FinalCppSourceGroup {
  mlir::DictionaryAttr sourceOwner;
  std::string headerPath, sourcePath, header, source;
};
struct FinalCppSourceParts {
  std::string supportHeader, systemHeader, rootCppName, rootHeaderPath;
  std::vector<FinalCppSourceGroup> sourceGroups;
  std::string simulationMain, simulationConfig, portMap;
  uint64_t sourceCheckCount = 0;
};
} // namespace acir::compiler
#endif
