#ifndef PYCIRCUIT_HARDWARE_EMIT_COMMON_H
#define PYCIRCUIT_HARDWARE_EMIT_COMMON_H
#include "HardwareEmissionPlan.h"
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
struct ObservationEmissionPlan;
using EmitNames = HardwareEmissionDefinitionNames;
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
class HardwareEmitContext {
public:
  HardwareEmitContext(
      const HardwareEmissionPlan &plan,
      const ac::HardwareSourceCheckPlan &sourceChecks,
      std::shared_ptr<const ObservationEmissionPlan> &observationPlanCache,
      const ac::HardwarePhysicalProjection *physicalProjection,
      bool physical = false);
  mlir::LogicalResult prepare();
  bool isSystem();
  bool managesChecks();
  bool isPhysical() const { return physical_; }
  const ac::HardwarePhysicalProjection &physicalProjection() const {
    return *physicalProjection_;
  }
  // Prepare the verified check lifecycle shared by generated systems and
  // checked module callers; each emitter retains its observation boundary.
  mlir::LogicalResult prepareNativeChecks();
  const ac::HardwareSourceCheckPlan &sourceChecks() const;
  const ObservationEmissionPlan &observationPlan() const;
  bool resetIfOne() const;
  std::string cppReset(llvm::StringRef expression) const;
  std::string rtlReset(llvm::StringRef expression) const;
  const EmitNames &names(mlir::Operation *definition) const;
  mlir::Operation *lookupDefinition(llvm::StringRef symbol) const;
  ac::StructOp lookupStruct(ac::StructType type) const;
  mlir::FailureOr<ac::EnumDefinitionView>
  resolveEnum(ac::EnumType type, mlir::Operation *site) const;
  mlir::Operation *resolveCallee(ac::InstanceOp instance) const;
  mlir::Operation *resolveCallee(ac::CollectionOp collection) const;
  llvm::StringRef primitiveKind(mlir::Operation *definition) const;
  llvm::ArrayRef<mlir::Operation *>
  independentWorkChildren(mlir::Operation *definition) const;
  llvm::SmallVector<mlir::Operation *>
  scheduledOperations(mlir::Operation *definition,
                      const ac::HardwareBindings &bindings) const;
  mlir::FailureOr<HardwareEmissionOccurrenceId>
  occurrenceId(mlir::Operation *occurrence,
               const ac::HardwareBindings &parent) const;
  mlir::FailureOr<ac::HardwareBindings>
  bindOccurrence(mlir::Operation *occurrence,
                 const ac::HardwareBindings &parent) const;
  mlir::FailureOr<llvm::SmallVector<uint64_t>>
  occurrenceShape(mlir::Operation *occurrence,
                  const ac::HardwareBindings &parent) const;
  mlir::FailureOr<std::string>
  occurrenceName(mlir::Operation *occurrence,
                 const ac::HardwareBindings &parent) const;
  bool occurrenceIsCollection(mlir::Operation *occurrence) const;
  const HardwareEmissionQueuePlan *
  queuePlan(mlir::Operation *occurrence,
            const ac::HardwareBindings &parent) const;
  mlir::FailureOr<llvm::SmallVector<mlir::Value>>
  outputValues(mlir::Operation *definition) const;
  const HardwareEmissionModulePlan *
  modulePlan(mlir::Operation *definition,
             const ac::HardwareBindings &bindings = {}) const;
  mlir::FailureOr<llvm::SmallVector<ac::WireEndpoint>>
  dependencies(mlir::Value value, llvm::ArrayRef<mlir::StringAttr> path,
               const ac::HardwareBindings &bindings) const;
  const HardwareEmissionTypePlan *
  concreteValueType(mlir::Value value,
                    const ac::HardwareBindings &bindings) const;
  mlir::FailureOr<mlir::Attribute>
  evaluatedStatic(ac::StaticExprAttr expression,
                  const ac::HardwareBindings &bindings) const;
  mlir::FailureOr<mlir::Type>
  resolvedType(mlir::Type type, const ac::HardwareBindings &bindings) const;
  mlir::FailureOr<uint64_t>
  packedWidth(mlir::Type type, const ac::HardwareBindings &bindings) const;
  mlir::FailureOr<llvm::SmallVector<ac::FieldPath>>
  fieldPaths(mlir::Type type, const ac::HardwareBindings &bindings) const;
  const HardwareEmissionTableViewPlan *
  tableViewPlan(ac::TableViewOp view,
                const ac::HardwareBindings &bindings) const;
  llvm::ArrayRef<mlir::Operation *>
  observationLogicalSuffix(ac::SourceObserveOp observe,
                           const ac::HardwareBindings &bindings) const;
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
  mlir::FailureOr<std::string>
  rtlType(mlir::Type type, const ac::HardwareBindings &bindings,
          mlir::Operation *site) const;
  mlir::FailureOr<std::string> staticValue(ac::StaticExprAttr expression,
                                           mlir::Operation *site) const;
  bool isClosedStatic(ac::StaticExprAttr expression,
                      const ac::HardwareBindings &bindings = {}) const;
  mlir::FailureOr<std::string> staticRtlValue(ac::StaticExprAttr expression,
                                              mlir::Operation *site) const;
  mlir::FailureOr<ac::HardwareBindings> rootBindings() const;
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
  const HardwareEmissionPlan &plan;
  mlir::MLIRContext *mlirContext() const { return plan.package->getContext(); }
  mlir::InFlightDiagnostic emitPlanError() const {
    return plan.package->emitOpError();
  }
  const ac::HardwareSourceCheckPlan &frozenSourceChecks;
  std::shared_ptr<const ObservationEmissionPlan> &observationPlanCache;
  const ac::HardwarePhysicalProjection *frozenPhysicalProjection;
  PhysicalVerilogLayout physicalLayout;
  RtlPackageDependencies *rtlPackage = nullptr;

private:
  mlir::LogicalResult prepareImpl(bool allowSourceChecks);
  bool physical_ = false;
  std::optional<ac::HardwarePhysicalProjection> physicalProjection_;
  std::optional<ac::HardwareSourceCheckPlan> sourceChecks_;
  std::shared_ptr<const ObservationEmissionPlan> observationPlan_;
  llvm::DenseMap<mlir::Operation *, EmitNames> names_;
  llvm::StringMap<mlir::Operation *> definitions_;
  llvm::DenseMap<mlir::Operation *, std::string> primitiveKinds_;
};
mlir::LogicalResult emitHardwareCppDefinition(HardwareEmitContext &context,
                                              ac::ModuleOp definition,
                                              llvm::raw_ostream &out);
struct ObservationEmissionSite {
  HardwareEmissionObservationId id = 0;
  mlir::Operation *diagnosticAnchor = nullptr;
  unsigned valueIndex = 0;
  bool carriesValue = false;
  bool gauge = false;
  uint64_t registrationKey = 0, siteKey = 0;
  std::string kind;
  std::string registrationJson;
  std::string siteJson;
  std::string specJson;
  std::string reportName;
  uint64_t valueCount = 0;
};
struct ObservationEmissionChild {
  HardwareEmissionOccurrenceId id = 0;
  mlir::Operation *op = nullptr;
  ac::ModuleOp definition;
  uint64_t shape = 1;
};
struct ObservationEmissionPathEntry {
  HardwareEmissionOccurrenceId id = 0;
  mlir::Operation *diagnosticAnchor = nullptr;
  uint64_t lane = 0;
  std::string instanceName;
  bool collection = false;
  bool operator==(const ObservationEmissionPathEntry &) const = default;
};
struct ObservationEmissionOccurrence {
  ObservationEmissionSite site;
  uint64_t ordinal = 0;
  llvm::SmallVector<ObservationEmissionPathEntry> path;
  llvm::SmallVector<ObservationEmissionPathEntry> logicalOwnerPath;
};
struct ObservationEmissionPlan {
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<ObservationEmissionSite>>
      definitions;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<ObservationEmissionChild>>
      children;
  llvm::DenseMap<mlir::Operation *, uint64_t> subtreeWidth;
  llvm::DenseMap<HardwareEmissionOccurrenceId, uint64_t> childBaseUnits;
  llvm::SmallVector<std::array<uint64_t, 5>> table;
  llvm::SmallVector<ObservationEmissionOccurrence, 0> occurrences;
  uint64_t events = 0;
};
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
                        mlir::Operation *child,
                        const ac::HardwareBindings &bindings);
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
