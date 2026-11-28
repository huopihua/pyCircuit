namespace detail {
FailureOr<bool> memoryAddressesEqual(
    const TransactionStatePlan &plan, BoundValueId left, BoundValueId right,
    TransactionPoint point, const HardwareAnalysis &hardware,
    llvm::function_ref<FailureOr<BoundValueId>(BoundValueId, TransactionPoint)> identity,
    llvm::function_ref<LogicalResult(uint64_t)> pay) {
  if (failed(pay(8)) || left >= plan.values.size() || right >= plan.values.size()) return failure();
  auto a = identity(left, point), b = identity(right, point);
  if (failed(a) || failed(b) || *a >= plan.values.size() || *b >= plan.values.size()) return failure();
  if (*a == *b) return true;
  struct Constant { uint64_t width; MathIntAttr value; };
  auto constant = [&](BoundValueId id) -> FailureOr<std::optional<Constant>> {
    const auto &ref = plan.values[id];
    if (ref.anchor >= plan.instances.size()) return failure();
    auto operation = ref.value.getDefiningOp<BitsConstantOp>();
    if (!operation) return std::optional<Constant>();
    const auto &bindings = plan.instances[ref.anchor].bindings;
    auto type = hardware.resolveType(ref.value.getType(), bindings, operation);
    if (failed(type) || !isa<BitsType>(*type)) return std::optional<Constant>();
    // Pay the complete original closed expression before asking the existing
    // evaluator for its value. Source origin/span attributes are not equality.
    SmallVector<Attribute, 0> pending;
    if (failed(pay(2))) return failure();
    pending.push_back(operation.getValue());
    while (!pending.empty()) {
      auto attribute = pending.pop_back_val();
      if (failed(pay(2))) return failure();
      if (auto text = dyn_cast<StringAttr>(attribute))
        if (failed(pay(text.getValue().size()))) return failure();
      bool admitted = true;
      attribute.walkImmediateSubElements([&](Attribute child) {
        if (failed(pay(2))) { admitted = false; return; }
        pending.push_back(child);
      }, [&](Type) {});
      if (!admitted) return failure();
    }
    if (!hardware.isStaticEvaluable(operation.getValue(), bindings)) return std::optional<Constant>();
    auto width = hardware.getPackedWidth(*type, bindings, operation);
    auto evaluated = hardware.evaluateStatic(operation.getValue(), bindings, operation);
    if (failed(width) || failed(evaluated)) return failure();
    auto integer = dyn_cast<MathIntAttr>(*evaluated);
    if (!integer) return std::optional<Constant>();
    if (failed(pay(integer.getCanonicalValue().size() + *width / 64 + 1))) return failure();
    llvm::APSInt value(integer.getCanonicalValue());
    if (value.isNegative() || value.getActiveBits() > *width) return std::optional<Constant>();
    return std::optional<Constant>(Constant{*width, integer});
  };
  auto lhs = constant(*a), rhs = constant(*b);
  if (failed(lhs) || failed(rhs)) return failure();
  if (!*lhs || !*rhs || (**lhs).width != (**rhs).width) return false;
  return llvm::APSInt::compareValues(llvm::APSInt((**lhs).value.getCanonicalValue()),
                                     llvm::APSInt((**rhs).value.getCanonicalValue())) == 0;
}

FailureOr<bool> memoryReadCoveredBy(
    const TransactionStatePlan &plan, MemoryAccessId earlier, MemoryAccessId later,
    llvm::function_ref<FailureOr<bool>(BoundValueId, BoundValueId, TransactionPoint)> same,
    llvm::function_ref<LogicalResult(uint64_t)> pay) {
  if (failed(pay(8))) return failure();
  if (earlier >= later || later >= plan.memoryAccesses.size()) return false;
  const auto &a = plan.memoryAccesses[earlier], &b = plan.memoryAccesses[later];
  if (a.kind != TransactionMemoryAccessKind::Read || b.kind != TransactionMemoryAccessKind::Read ||
      a.owner != b.owner || a.port != b.port || a.owner >= plan.memoryOwners.size() ||
      plan.memoryOwners[a.owner].layout.kind != "byte_mem" || a.arguments.size() != 1 || b.arguments.size() != 1 ||
      a.point.prefix > b.point.prefix) return false;
  // Resource-demand lowering may narrow the physical contribution of a read
  // to the paths on which its value is actually consumed. Dominance of the
  // source access alone therefore does not authorize a descendant access to
  // reuse that contribution. Reuse only inside the exact source arm; distinct
  // arms retain independent proposals and are merged by the pin materializer.
  if (a.point.path.branch != b.point.path.branch ||
      a.point.path.whenTrue != b.point.path.whenTrue)
    return false;
  bool authored = a.pathFact && a.pathFact.getValue();
  if (!authored) {
    auto equal = same(a.authoredPath, b.authoredPath, b.point);
    if (failed(equal)) return failure();
    authored = *equal;
  }
  if (!authored) return false;
  return same(a.arguments.front(), b.arguments.front(), b.point);
}
} // namespace detail
