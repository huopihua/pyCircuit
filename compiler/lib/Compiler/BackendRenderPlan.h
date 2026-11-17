#ifndef PYCIRCUIT_BACKEND_RENDER_PLAN_H
#define PYCIRCUIT_BACKEND_RENDER_PLAN_H

#include "FinalCppSourceParts.h"
#include "FinalSourceMap.h"
#include "FinalVerilogSourceParts.h"
#include "HardwareEmissionPlan.h"

#include "llvm/Support/raw_ostream.h"

namespace acir::compiler {

/// Target-neutral publication metadata copied from the verified hardware plan.
/// Target renderers consume this value together with already rendered native
/// source parts; neither layer can reopen the source IR or HardwareAnalysis.
struct HardwareRenderMetadata {
  HardwareEmissionEntryPlan entry;
  ac::HardwareResetSemantics resetSemantics =
      ac::HardwareResetSemantics::Strict;
  std::optional<HardwareEmissionExecutionPlan> execution;
  llvm::SmallVector<std::string, 0> systemSamples;
  bool needsFourState = false;
  llvm::SmallVector<FinalSourceMap, 0> sourceMaps;
};

struct CppRenderPlan {
  HardwareRenderMetadata metadata;
  FinalCppSourceParts parts;
};

struct VerilogRenderPlan {
  HardwareRenderMetadata metadata;
  FinalVerilogSourceParts parts;
  bool physical = false;
};

/// Move-only proof that target source parts, provenance membership, and all
/// publication paths form one closed C++ output inventory.
class VerifiedCppRenderPlan final {
public:
  VerifiedCppRenderPlan(VerifiedCppRenderPlan &&) = default;
  VerifiedCppRenderPlan &operator=(VerifiedCppRenderPlan &&) = default;
  VerifiedCppRenderPlan(const VerifiedCppRenderPlan &) = delete;
  VerifiedCppRenderPlan &operator=(const VerifiedCppRenderPlan &) = delete;

  const CppRenderPlan &get() const { return plan; }
  const CppRenderPlan &operator*() const { return plan; }
  const CppRenderPlan *operator->() const { return &plan; }

private:
  explicit VerifiedCppRenderPlan(CppRenderPlan &&plan)
      : plan(std::move(plan)) {}
  CppRenderPlan plan;

  friend mlir::FailureOr<VerifiedCppRenderPlan>
  verifyAndFreezeCppRenderPlan(CppRenderPlan plan,
                               ac::detail::EmitError emitError);
};

/// Move-only proof that target source parts, provenance membership, and all
/// publication paths form one closed Verilog output inventory.
class VerifiedVerilogRenderPlan final {
public:
  VerifiedVerilogRenderPlan(VerifiedVerilogRenderPlan &&) = default;
  VerifiedVerilogRenderPlan &operator=(VerifiedVerilogRenderPlan &&) = default;
  VerifiedVerilogRenderPlan(const VerifiedVerilogRenderPlan &) = delete;
  VerifiedVerilogRenderPlan &
  operator=(const VerifiedVerilogRenderPlan &) = delete;

  const VerilogRenderPlan &get() const { return plan; }
  const VerilogRenderPlan &operator*() const { return plan; }
  const VerilogRenderPlan *operator->() const { return &plan; }

private:
  explicit VerifiedVerilogRenderPlan(VerilogRenderPlan &&plan)
      : plan(std::move(plan)) {}
  VerilogRenderPlan plan;

  friend mlir::FailureOr<VerifiedVerilogRenderPlan>
  verifyAndFreezeVerilogRenderPlan(VerilogRenderPlan plan,
                                   ac::detail::EmitError emitError);
};

/// Target planning consumes only the verified plan API. A private, uninstalled
/// migration bridge still owns the remaining operation-level source generation;
/// everything after this boundary is pure rendering from frozen values.
mlir::FailureOr<VerifiedCppRenderPlan>
prepareCppRenderPlan(const VerifiedHardwareEmissionPlan &hardware);

mlir::FailureOr<VerifiedVerilogRenderPlan>
prepareVerilogRenderPlan(const VerifiedHardwareEmissionPlan &hardware,
                         bool physical = false);

mlir::LogicalResult verifyCppRenderPlan(const CppRenderPlan &plan,
                                        ac::detail::EmitError emitError);
mlir::FailureOr<VerifiedCppRenderPlan>
verifyAndFreezeCppRenderPlan(CppRenderPlan plan,
                             ac::detail::EmitError emitError);

mlir::LogicalResult verifyVerilogRenderPlan(const VerilogRenderPlan &plan,
                                            ac::detail::EmitError emitError);
mlir::FailureOr<VerifiedVerilogRenderPlan>
verifyAndFreezeVerilogRenderPlan(VerilogRenderPlan plan,
                                 ac::detail::EmitError emitError);

/// Pure renderers. Their signatures intentionally expose neither ModuleOp nor
/// HardwareAnalysis, so rendering cannot silently recover semantic decisions.
void renderCppSource(const VerifiedCppRenderPlan &plan, llvm::raw_ostream &out);
void renderVerilogSource(const VerifiedVerilogRenderPlan &plan,
                         llvm::raw_ostream &out);
void renderCppJson(const VerifiedCppRenderPlan &plan, llvm::raw_ostream &out);
void renderVerilogJson(const VerifiedVerilogRenderPlan &plan,
                       llvm::raw_ostream &out);

} // namespace acir::compiler

#endif // PYCIRCUIT_BACKEND_RENDER_PLAN_H
