// Emit source-owned modules from verified final ACIR. Publication is
// transactional in the Python driver; this transport writes nothing before
// complete success.
#include "Compiler/BackendRenderPlan.h"
#include "Compiler/FinalSourceMap.h"
#include "Compiler/HardwareEmissionPlan.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "pycircuit/Dialect/ACIR/ACIRDialect.h"
#include "pycircuit/Dialect/GFIR/GFIRDialect.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry dialects;
  dialects.insert<acir::ac::ACIRDialect, acir::gfir::GFIRDialect>();
  MLIRContext context(dialects);
  context.loadAllAvailableDialects();
  auto emitError = [&] { return mlir::emitError(UnknownLoc::get(&context)); };
  if (argc == 3 && llvm::StringRef(argv[1]) == "--verify-source-map") {
    auto buffer = llvm::MemoryBuffer::getFile(argv[2]);
    if (!buffer) {
      emitError() << "cannot read source map";
      return 1;
    }
    std::string path;
    if (failed(acir::compiler::validateFinalSourceMapText(
            (*buffer)->getBuffer(), context, emitError, &path)))
      return 1;
    llvm::outs() << llvm::formatv(
        "{0}\n", llvm::json::Value(llvm::json::Object{{"path", path}}));
    return 0;
  }
  StringRef target, format = "json", rtlMode = "managed";
  bool optionsValid = argc >= 4 && (argc % 2 == 0);
  for (int i = 2; optionsValid && i + 1 < argc; i += 2) {
    StringRef option(argv[i]);
    if (option == "--target" && target.empty())
      target = argv[i + 1];
    else if (option == "--format")
      format = argv[i + 1];
    else if (option == "--rtl-mode")
      rtlMode = argv[i + 1];
    else
      optionsValid = false;
  }
  if (!optionsValid || (target != "cpp" && target != "verilog") ||
      (format != "json" && format != "source") ||
      (rtlMode != "managed" && rtlMode != "physical") ||
      (target != "verilog" && rtlMode != "managed")) {
    llvm::errs() << "usage: pycircuit-emit FINAL.ac --target cpp|verilog "
                    "[--format json|source] [--rtl-mode managed|physical]\n";
    return 2;
  }
  bool physical = rtlMode == "physical";
  auto package = parseSourceFile<ModuleOp>(argv[1], &context);
  if (!package || failed(mlir::verify(*package)))
    return 1;
  // Materialize and verify the target-neutral plan before either renderer is
  // selected. Existing renderers remain the output oracle during migration;
  // metadata below already consumes only this frozen view.
  auto verifiedPlan = acir::compiler::prepareHardwareEmissionPlan(*package);
  if (failed(verifiedPlan))
    return 1;
  if (target == "cpp") {
    auto renderPlan = acir::compiler::prepareCppRenderPlan(*verifiedPlan);
    if (failed(renderPlan))
      return 1;
    if (format == "source")
      acir::compiler::renderCppSource(*renderPlan, llvm::outs());
    else
      acir::compiler::renderCppJson(*renderPlan, llvm::outs());
  } else {
    auto renderPlan =
        acir::compiler::prepareVerilogRenderPlan(*verifiedPlan, physical);
    if (failed(renderPlan))
      return 1;
    if (format == "source")
      acir::compiler::renderVerilogSource(*renderPlan, llvm::outs());
    else
      acir::compiler::renderVerilogJson(*renderPlan, llvm::outs());
  }
  return 0;
}
