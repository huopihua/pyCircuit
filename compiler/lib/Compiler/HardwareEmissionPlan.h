#ifndef PYCIRCUIT_HARDWARE_EMISSION_PLAN_H
#define PYCIRCUIT_HARDWARE_EMISSION_PLAN_H

#include "FinalSourceMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "pycircuit/Dialect/ACIR/HardwareAnalysisContext.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace acir::compiler {

/// Backend-neutral, concrete facts derived once from verified final hardware.
///
/// This is an in-process code-generation plan, not another IR. Attribute, type,
/// and operation handles borrow the immutable input context. Renderers must use
/// the materialized values below; operation handles are diagnostic anchors
/// only.
using HardwareEmissionValueId = uint64_t;

enum class HardwareEmissionTypeKind { Bits, Table, Struct, Enum };

struct HardwareEmissionPackedField {
  llvm::SmallVector<std::string, 2> path;
  uint64_t low = 0;
  uint64_t width = 0;

  bool operator==(const HardwareEmissionPackedField &) const = default;
};

struct HardwareEmissionTypePlan {
  HardwareEmissionTypeKind kind = HardwareEmissionTypeKind::Bits;
  mlir::Type resolvedType;
  uint64_t packedWidth = 0;
  llvm::SmallVector<uint64_t, 2> shape;
  llvm::SmallVector<HardwareEmissionPackedField, 2> fields;
  std::string canonicalType;

  bool operator==(const HardwareEmissionTypePlan &other) const {
    return kind == other.kind && packedWidth == other.packedWidth &&
           shape == other.shape && fields == other.fields &&
           canonicalType == other.canonicalType;
  }
};

struct HardwareEmissionBinding {
  std::string name;
  std::string value;

  bool operator==(const HardwareEmissionBinding &) const = default;
};

struct HardwareEmissionModuleKey {
  std::string definition;
  llvm::SmallVector<HardwareEmissionBinding, 2> integers;
  llvm::SmallVector<HardwareEmissionBinding, 2> types;

  bool operator==(const HardwareEmissionModuleKey &) const = default;
};

struct HardwareEmissionDefinitionNames {
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

struct HardwareEmissionOccurrenceBinding {
  mlir::Operation *diagnosticAnchor = nullptr;
  ac::HardwareBindings bindings;
  llvm::SmallVector<uint64_t, 2> shape;
};

struct HardwareEmissionDependencyEndpoint {
  mlir::Value value;
  llvm::SmallVector<std::string, 2> path;
};

struct HardwareEmissionDependencyPlan {
  HardwareEmissionDependencyEndpoint output;
  llvm::SmallVector<HardwareEmissionDependencyEndpoint, 2> inputs;
};

struct HardwareEmissionStaticPlan {
  ac::StaticExprAttr expression;
  mlir::Attribute value;
};

struct HardwareEmissionTypeQueryPlan {
  mlir::Type source;
  mlir::Type resolved;
  std::optional<uint64_t> packedWidth;
  llvm::SmallVector<llvm::SmallVector<std::string, 2>, 2> fieldPaths;
};

struct HardwareEmissionTableViewPlan {
  mlir::Operation *diagnosticAnchor = nullptr;
  uint64_t sourceOrdinalZero = 0;
  llvm::SmallVector<uint64_t, 2> inputShape;
};

struct HardwareEmissionDefinitionPlan {
  mlir::Operation *diagnosticAnchor = nullptr;
  std::string symbol;
  mlir::DictionaryAttr sourceOwner;
  HardwareEmissionDefinitionNames names;
  llvm::SmallVector<HardwareEmissionOccurrenceBinding, 2> templateChildren;
  llvm::SmallVector<mlir::Operation *, 2> independentWorkChildren;
  llvm::SmallVector<HardwareEmissionDependencyPlan, 0> dependencies;
  llvm::SmallVector<HardwareEmissionStaticPlan, 0> staticValues;
  llvm::SmallVector<HardwareEmissionTypeQueryPlan, 0> types;
};

struct HardwareEmissionPrimitivePlan {
  mlir::Operation *diagnosticAnchor = nullptr;
  std::string symbol;
  std::string kind;
  std::string standardSource;
};

enum class HardwareEmissionValueKind {
  ModuleInput,
  OperationResult,
  RegionArgument
};

struct HardwareEmissionValuePlan {
  HardwareEmissionValueId id = 0;
  mlir::Value source;
  HardwareEmissionValueKind kind = HardwareEmissionValueKind::OperationResult;
  HardwareEmissionTypePlan type;
};

struct HardwareEmissionPortPlan {
  std::string name;
  HardwareEmissionTypePlan type;
  HardwareEmissionValueId value = 0;
  std::optional<unsigned> availabilityOutput;
};

struct HardwareEmissionScheduleStep {
  mlir::Operation *diagnosticAnchor = nullptr;
  std::string operation;
  llvm::SmallVector<HardwareEmissionValueId, 4> inputs;
  llvm::SmallVector<HardwareEmissionValueId, 2> outputs;
  bool temporalCut = false;
};

enum class HardwareEmissionChildKind { Module, Primitive };

struct HardwareEmissionQueuePlan {
  uint64_t depth = 0;
  uint64_t tokenCardinality = 0;
  uint64_t tokenElementWidth = 0;
  uint64_t availabilityLatency = 0;
};

struct HardwareEmissionChildPlan {
  mlir::Operation *diagnosticAnchor = nullptr;
  HardwareEmissionChildKind kind = HardwareEmissionChildKind::Module;
  std::string instanceName;
  std::string callee;
  std::string primitiveKind;
  llvm::SmallVector<uint64_t, 2> shape;
  HardwareEmissionModuleKey bindings;
  ac::HardwareBindings concreteBindings;
  std::optional<unsigned> moduleIndex;
  std::optional<HardwareEmissionQueuePlan> queue;
  llvm::SmallVector<HardwareEmissionStaticPlan, 0> staticValues;
  llvm::SmallVector<HardwareEmissionTypeQueryPlan, 0> types;
};

struct HardwareEmissionModulePlan {
  mlir::Operation *diagnosticAnchor = nullptr;
  HardwareEmissionModuleKey key;
  ac::HardwareBindings concreteBindings;
  mlir::DictionaryAttr sourceOwner;
  llvm::SmallVector<HardwareEmissionPortPlan, 0> inputs;
  llvm::SmallVector<HardwareEmissionPortPlan, 0> outputs;
  llvm::SmallVector<HardwareEmissionValuePlan, 0> values;
  llvm::SmallVector<HardwareEmissionScheduleStep, 0> schedule;
  llvm::SmallVector<HardwareEmissionDependencyPlan, 0> dependencies;
  llvm::SmallVector<HardwareEmissionStaticPlan, 0> staticValues;
  llvm::SmallVector<HardwareEmissionTypeQueryPlan, 0> types;
  llvm::SmallVector<HardwareEmissionTableViewPlan, 0> tableViews;
  llvm::SmallVector<HardwareEmissionChildPlan, 2> children;
  unsigned publicInputCount = 0;
  unsigned publicOutputCount = 0;
  llvm::SmallVector<uint64_t, 2> rootInputDefaults;
  std::optional<unsigned> clockInput;
  std::optional<unsigned> resetInput;
  bool closedSystem = false;
};

struct HardwareEmissionEntryArgument {
  std::string kind;
  std::string value;
};

struct HardwareEmissionEntryPlan {
  std::string implementationDefinition;
  std::string logicalDefinition;
  mlir::DictionaryAttr sourceOwner;
  llvm::SmallVector<HardwareEmissionEntryArgument, 2> logicalArguments;
};

struct HardwareEmissionExecutionPlan {
  std::string strategy;
  std::string protocol;
  std::string publicationField;
  std::optional<uint64_t> stateOutput;
};

struct HardwareEmissionCheckPlan {
  uint64_t ordinal = 0;
  uint64_t bindingIndex = 0;
  uint64_t ownerOccurrence = 0;
  std::string kind;
  std::string message;
  std::string location;
  std::string identity;
};

struct HardwareEmissionObservationPlan {
  unsigned moduleIndex = 0;
  mlir::Operation *diagnosticAnchor = nullptr;
  std::string kind;
  std::string identity;
  llvm::SmallVector<uint64_t, 2> payloadWidths;
  llvm::SmallVector<mlir::Operation *, 2> logicalSuffix;
};

struct HardwareEmissionSourceUnitPlan {
  mlir::DictionaryAttr sourceOwner;
  llvm::SmallVector<unsigned, 0> moduleIndices;
  llvm::SmallVector<unsigned, 0> definitionIndices;
  std::string cppHeaderPath;
  std::string cppSourcePath;
  std::string verilogPath;
};

struct HardwareEmissionPlan {
  mlir::ModuleOp package;
  unsigned rootModule = 0;
  HardwareEmissionEntryPlan entry;
  ac::HardwareResetSemantics resetSemantics =
      ac::HardwareResetSemantics::Strict;
  std::optional<HardwareEmissionExecutionPlan> execution;
  llvm::SmallVector<std::string, 0> systemSamples;
  bool needsFourState = false;
  llvm::SmallVector<HardwareEmissionDefinitionPlan, 0> definitions;
  llvm::SmallVector<mlir::Operation *, 0> nominalDeclarations;
  llvm::SmallVector<HardwareEmissionPrimitivePlan, 0> primitives;
  llvm::SmallVector<std::string, 0> standardVerilogSources;
  llvm::SmallVector<HardwareEmissionModulePlan, 0> modules;
  llvm::SmallVector<HardwareEmissionCheckPlan, 0> checks;
  llvm::SmallVector<HardwareEmissionObservationPlan, 0> observations;
  llvm::SmallVector<HardwareEmissionSourceUnitPlan, 0> sourceUnits;
  llvm::SmallVector<FinalSourceMapPlan, 0> sourceMaps;
};

/// Move-only ownership boundary for a completely materialized and verified
/// emission plan. Target policy construction and rendering receive this
/// read-only view. Logical targets cannot reopen HardwareAnalysis through it;
/// physical projection may use one short-lived target-specific analysis and
/// caches only its immutable result.
class VerifiedHardwareEmissionPlan final {
public:
  VerifiedHardwareEmissionPlan(VerifiedHardwareEmissionPlan &&);
  VerifiedHardwareEmissionPlan &operator=(VerifiedHardwareEmissionPlan &&);
  ~VerifiedHardwareEmissionPlan();
  VerifiedHardwareEmissionPlan(const VerifiedHardwareEmissionPlan &) = delete;
  VerifiedHardwareEmissionPlan &
  operator=(const VerifiedHardwareEmissionPlan &) = delete;

  const HardwareEmissionPlan &get() const { return plan; }
  const HardwareEmissionPlan &operator*() const { return plan; }
  const HardwareEmissionPlan *operator->() const { return &plan; }

private:
  struct FrozenBackendState;

  VerifiedHardwareEmissionPlan(HardwareEmissionPlan &&plan,
                               std::unique_ptr<FrozenBackendState> backendState);
  HardwareEmissionPlan plan;
  std::unique_ptr<FrozenBackendState> backendState;

  friend struct BackendPlanningAccess;
  friend mlir::FailureOr<VerifiedHardwareEmissionPlan>
  prepareHardwareEmissionPlan(mlir::ModuleOp package);
  friend mlir::FailureOr<VerifiedHardwareEmissionPlan>
  verifyAndFreezeHardwareEmissionPlan(HardwareEmissionPlan plan);
};

/// Analyze the existing final-stage ac.* hardware package exactly once and
/// freeze every backend-neutral fact needed by later target planning.
mlir::FailureOr<VerifiedHardwareEmissionPlan>
prepareHardwareEmissionPlan(mlir::ModuleOp package);

/// Structural plan verification only. This never reopens HardwareAnalysis or
/// infers a fact from the source IR.
mlir::LogicalResult
verifyHardwareEmissionPlan(const HardwareEmissionPlan &plan);

mlir::FailureOr<VerifiedHardwareEmissionPlan>
verifyAndFreezeHardwareEmissionPlan(HardwareEmissionPlan plan);

} // namespace acir::compiler

#endif // PYCIRCUIT_HARDWARE_EMISSION_PLAN_H
