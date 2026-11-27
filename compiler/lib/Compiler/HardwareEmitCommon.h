#ifndef PYCIRCUIT_HARDWARE_EMIT_COMMON_H
#define PYCIRCUIT_HARDWARE_EMIT_COMMON_H
#include "pycircuit/Analysis/CompilerPhaseMetrics.h"
#include "pycircuit/Dialect/ACIR/HardwareAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
namespace acir::compiler {
using HardwareDefinitionId = uint32_t;

struct EmitNames {
  std::string cppQualified;
  std::string cppClass;
  std::string cppNamespace;
  std::string cppFamilyClass;
  std::string cppFamilyQualified;
  std::string rtl;
  std::string header;
  std::string source;
  std::string verilog;
};
// Actual definition-local references, recorded while emitting physical RTL.
struct PhysicalVerilogLayout {
  llvm::DenseMap<mlir::Value, std::string> nets;
  llvm::DenseMap<mlir::Operation *, std::string> checkErrors;
};
struct RtlPackageDependencies {
  llvm::SetVector<mlir::Operation *> types;
  bool floorDiv = false;
  bool mod = false;
};

struct HardwareEmissionChild {
  mlir::Operation *occurrence = nullptr;
  mlir::Operation *callee = nullptr;
  std::optional<HardwareDefinitionId> definition;
  bool queue = false;
  std::string primitiveKind;
};

enum class HardwareTableRegionLowering : uint8_t {
  Procedural,
  Generated,
};

/// Frozen definition-local lowering facts for one Table map/match region.
/// Static lane counts are retained for elaboration-cost accounting; generic
/// definitions keep the count symbolic and still share one lowering decision.
struct HardwareTableRegionEmissionPlan {
  HardwareTableRegionLowering lowering = HardwareTableRegionLowering::Generated;
  std::optional<uint64_t> staticLanes;
};

/// Definition-local facts frozen after final hardware verification. Occurrence
/// handles remain available for binding-dependent lowering and diagnostics;
/// static identity, child topology, and work scheduling use DefinitionId.
struct HardwareDefinitionEmissionPlan {
  HardwareDefinitionId id = 0;
  ac::ModuleOp definition;
  llvm::SmallVector<HardwareEmissionChild, 0> children;
  llvm::SmallVector<HardwareDefinitionId, 0> dependencies;
  llvm::SmallVector<mlir::Operation *, 0> independentWorkSubtrees;
  llvm::SmallVector<mlir::Operation *, 0> observations;
  llvm::DenseMap<mlir::Operation *, HardwareTableRegionEmissionPlan>
      tableRegions;
  uint64_t tableSplats = 0;
  bool reachableFromRoot = false;
};

struct ObservationEmissionSite {
  mlir::Operation *op = nullptr;
  unsigned valueIndex = 0;
  bool carriesValue = false;
  bool gauge = false;
  uint64_t registrationKey = 0, siteKey = 0;
};
struct ObservationEmissionChild {
  mlir::Operation *op = nullptr;
  ac::ModuleOp definition;
  uint64_t shape = 1;
};
struct ObservationEmissionOccurrence {
  ObservationEmissionSite site;
  uint64_t ordinal = 0;
  llvm::SmallVector<std::pair<mlir::Operation *, uint64_t>> path;
  llvm::SmallVector<std::pair<mlir::Operation *, uint64_t>> logicalOwnerPath;
};
struct ObservationEmissionPlan {
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<ObservationEmissionSite>>
      definitions;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<ObservationEmissionChild>>
      children;
  llvm::DenseMap<mlir::Operation *, uint64_t> subtreeWidth;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<uint64_t>> childBaseUnits;
  llvm::SmallVector<std::array<uint64_t, 5>> table;
  llvm::SmallVector<ObservationEmissionOccurrence> occurrences;
  uint64_t events = 0;
};

class HardwareEmitContext {
public:
  HardwareEmitContext(mlir::ModuleOp package, ac::HardwareAnalysis &analysis,
                      bool physical = false,
                      acir::metrics::CompilerPhaseMetrics *metrics = nullptr);
  mlir::LogicalResult prepare();
  bool isSystem();
  bool managesChecks();
  bool isPhysical() const { return physical_; }
  acir::metrics::CompilerPhaseMetrics *metrics() const { return metrics_; }
  const ac::HardwarePhysicalProjection &physicalProjection() const {
    return *physicalProjection_;
  }
  // Prepare the verified check lifecycle shared by generated systems and
  // checked module callers; each emitter retains its observation boundary.
  mlir::LogicalResult prepareNativeChecks();
  /// Reject reuse after any package operation, block, attribute, operand,
  /// result type, property, successor, or location changed. This is an exact
  /// identity check, not a replay of source-check planning or proof.
  mlir::LogicalResult verifyPreparedPackage();
  const ac::HardwareSourceCheckPlan &sourceChecks() const;
  bool resetIfOne() const;
  std::string cppReset(llvm::StringRef expression) const;
  std::string rtlReset(llvm::StringRef expression) const;
  const EmitNames &names(mlir::Operation *definition) const;
  std::optional<HardwareDefinitionId>
  definitionId(mlir::Operation *definition) const;
  const HardwareDefinitionEmissionPlan &
  definitionPlan(HardwareDefinitionId id) const;
  const HardwareDefinitionEmissionPlan &
  definitionPlan(ac::ModuleOp definition) const;
  const HardwareTableRegionEmissionPlan &
  tableRegionPlan(mlir::Operation *operation) const;
  llvm::ArrayRef<HardwareDefinitionEmissionPlan> definitionPlans() const {
    return definitionPlans_;
  }
  llvm::ArrayRef<HardwareDefinitionId> rootReachableDefinitions() const {
    return rootReachableDefinitions_;
  }
  llvm::ArrayRef<std::string> standardRtlSources() const {
    return standardRtlSources_;
  }
  const ObservationEmissionPlan &observationPlan() const;
  mlir::FailureOr<std::string> cppFormalName(llvm::StringRef name,
                                             mlir::Operation *site) const;
  mlir::FailureOr<std::string> rtlFormalName(llvm::StringRef name,
                                             mlir::Operation *site) const;
  mlir::FailureOr<std::string> cppType(mlir::Type type,
                                       mlir::Operation *site) const;
  mlir::FailureOr<std::string> cppType(mlir::Type type,
                                       const ac::HardwareBindings &bindings,
                                       mlir::Operation *site) const;
  mlir::FailureOr<std::string> staticValue(ac::StaticExprAttr expression,
                                           const ac::HardwareBindings &bindings,
                                           mlir::Operation *site) const;
  mlir::FailureOr<std::string> rtlType(mlir::Type type,
                                       mlir::Operation *site) const;
  mlir::FailureOr<std::string> staticValue(ac::StaticExprAttr expression,
                                           mlir::Operation *site) const;
  bool isClosedStatic(ac::StaticExprAttr expression,
                      const ac::HardwareBindings &bindings = {}) const;
  mlir::FailureOr<std::string> staticRtlValue(ac::StaticExprAttr expression,
                                              mlir::Operation *site) const;
  mlir::FailureOr<ac::HardwareBindings> rootBindings() const;
  mlir::FailureOr<std::string> rootPortMap(bool rtl = true);
  mlir::FailureOr<std::string> rootCppType() const;
  mlir::FailureOr<std::string> rootRtlParameters() const;
  mlir::FailureOr<std::string> cppInstanceType(ac::InstanceOp instance) const;
  mlir::FailureOr<std::string>
  cppBoundModuleType(mlir::Operation *definition,
                     const ac::HardwareBindings &bindings,
                     mlir::Operation *site, bool kernel = false) const;
  mlir::FailureOr<std::string>
  cppBoundFamilyType(mlir::Operation *definition,
                     const ac::HardwareBindings &bindings,
                     mlir::Operation *site, llvm::StringRef count) const;
  mlir::FailureOr<std::string>
  rtlInstanceParameters(ac::InstanceOp instance) const;
  mlir::FailureOr<std::string>
  rtlInstanceParameters(ac::CollectionOp collection) const;
  mlir::FailureOr<std::string>
  shapeSize(mlir::ArrayAttr shape, mlir::Operation *site, bool rtl = false,
            const ac::HardwareBindings &bindings = {}) const;
  mlir::FailureOr<std::string>
  payloadWidth(mlir::Type type, mlir::Operation *site, bool rtl = false,
               const ac::HardwareBindings &bindings = {}) const;
  mlir::FailureOr<std::string>
  unsignedStaticValue(ac::StaticExprAttr expression, mlir::Operation *site,
                      bool rtl = false,
                      const ac::HardwareBindings &bindings = {}) const;
  mlir::FailureOr<std::string>
  viewIndex(ac::TableViewOp view, llvm::StringRef ordinal, bool rtl = false,
            const ac::HardwareBindings &bindings = {}) const;
  mlir::FailureOr<std::string>
  cppWireExpression(mlir::Value value, const ac::HardwareBindings &bindings,
                    mlir::Operation *site) const;
  mlir::ModuleOp package;
  ac::HardwareAnalysis &analysis;
  PhysicalVerilogLayout physicalLayout;
  RtlPackageDependencies *rtlPackage = nullptr;

private:
  struct PreparedOperationIdentity {
    mlir::Operation *operation = nullptr;
    mlir::Block *parentBlock = nullptr;
    mlir::LocationAttr location;
    mlir::DictionaryAttr attributes;
    mlir::Attribute properties;
    llvm::SmallVector<mlir::Value, 0> operands;
    llvm::SmallVector<mlir::Type, 0> resultTypes;
    llvm::SmallVector<mlir::Block *, 0> blocks;
    llvm::SmallVector<mlir::Type, 0> blockArgumentTypes;
    llvm::SmallVector<mlir::Block *, 0> successors;
  };
  void capturePreparedPackage();
  mlir::LogicalResult prepareDefinitionEmissionPlans(ac::ModuleOp root);
  mlir::LogicalResult prepareImpl(bool allowSourceChecks);
  bool physical_ = false;
  acir::metrics::CompilerPhaseMetrics *metrics_ = nullptr;
  std::optional<ac::HardwarePhysicalProjection> physicalProjection_;
  std::optional<ac::HardwareSourceCheckPlan> sourceChecks_;
  std::optional<ObservationEmissionPlan> observationPlan_;
  llvm::SmallVector<PreparedOperationIdentity, 0> preparedPackage_;
  llvm::DenseMap<mlir::Operation *, EmitNames> names_;
  llvm::DenseMap<mlir::Operation *, HardwareDefinitionId> definitionIds_;
  llvm::SmallVector<HardwareDefinitionEmissionPlan, 0> definitionPlans_;
  llvm::SmallVector<HardwareDefinitionId, 0> rootReachableDefinitions_;
  llvm::SmallVector<std::string, 0> standardRtlSources_;
};
mlir::LogicalResult emitHardwareCppDefinition(HardwareEmitContext &context,
                                              ac::ModuleOp definition,
                                              llvm::raw_ostream &out);
mlir::FailureOr<ObservationEmissionPlan>
observationEmissionPlan(HardwareEmitContext &context);
mlir::FailureOr<uint64_t>
observationPayloadWidth(HardwareEmitContext &context,
                        const ObservationEmissionOccurrence &occurrence);
mlir::FailureOr<std::string> emissionMetadataJson(mlir::Attribute value,
                                                  mlir::Operation *site);
mlir::FailureOr<std::string>
observationSiteMetadataJson(ac::SourceObserveOp observe);
mlir::FailureOr<std::string>
cppObservationRunnerMetadata(HardwareEmitContext &context);
mlir::FailureOr<std::string>
cppObservationDescriptorTable(HardwareEmitContext &context);
mlir::FailureOr<std::string>
cppObservationConfigureCall(HardwareEmitContext &context);
mlir::FailureOr<std::string>
cppObservationChildBase(HardwareEmitContext &context, ac::ModuleOp definition,
                        unsigned childIndex);
mlir::FailureOr<uint64_t> cppObservationLaneStride(HardwareEmitContext &context,
                                                   ac::ModuleOp definition);
mlir::FailureOr<uint64_t>
cppObservationSubtreeWidth(HardwareEmitContext &context,
                           ac::ModuleOp definition);
mlir::LogicalResult cppObservationStaging(
    HardwareEmitContext &context, ac::ModuleOp definition,
    const ac::HardwareBindings &bindings, llvm::StringRef object,
    llvm::StringRef count,
    const std::function<mlir::FailureOr<std::string>(mlir::Value)> &value,
    llvm::raw_ostream &out);
std::string jsonString(llvm::StringRef value);
std::string displayLiteral(llvm::StringRef value);
mlir::LogicalResult emitPhysicalFaultReporter(HardwareEmitContext &context,
                                              llvm::StringRef clock,
                                              llvm::StringRef reset,
                                              llvm::raw_ostream &out);
mlir::LogicalResult emitHardwareVerilogDefinition(HardwareEmitContext &context,
                                                  ac::ModuleOp definition,
                                                  llvm::raw_ostream &out);
} // namespace acir::compiler
#endif
