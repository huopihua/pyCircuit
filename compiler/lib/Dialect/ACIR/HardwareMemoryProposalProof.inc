  // Each argument has an actual required-use witness. An async read depends on
  // exactly its own address witnesses and source path; unrelated earlier
  // admission cannot feed its combinational address. Requests and writes
  // retain the complete final fire.
  llvm::DenseSet<uint64_t> memoryWitnesses;
  for (const auto &proposal : view.memoryProposals) {
    SourceCheckBudget::PhaseScope proposalPhase(
        budget, "sampled memory proposal proof");
    SourceCheckBudget::LocalScope proposalBudget(budget);
    auto actual = proof.normalize(proposal.path), raw = proof.normalize(proposal.rawPath);
    if (failed(actual) || failed(raw)) return failure();
    auto combine = actual->value.getDefiningOp<BitsBinaryOp>();
    if (!combine || combine.getOpcode() != "and")
      return view.rule->emitOpError() << "Memory proposal requires admission AND raw path";
    auto selected = proof.normalize(combine.getRhs());
    if (failed(selected) || !sameEndpoint(*selected, *raw))
      return view.rule->emitOpError() << "Memory proposal raw path differs from its actual enable";
    auto rawFactors = factors(proposal.rawPath);
    if (failed(rawFactors)) return failure();
    bool witnessed = false;
    for (size_t first = 0; first < view.requirements.size(); ++first) {
      if (failed(budget.debit(1, view.rule))) return failure();
      if (first + proposal.arguments.size() > view.requirements.size()) break;
      const auto &start = view.requirements[first];
      if (start.kind.getValue() != "memory_argument" || start.operand != 0 ||
          start.occurrence != proposal.occurrence || start.location != proposal.location ||
          memoryWitnesses.contains(first)) continue;
      bool matches = true;
      uint64_t lastControl = start.control;
      for (auto [ordinal, argument] : llvm::enumerate(proposal.arguments)) {
        const auto &requirement = view.requirements[first + ordinal];
        if (requirement.kind.getValue() != "memory_argument" || requirement.operand != ordinal ||
            requirement.occurrence != proposal.occurrence || requirement.location != proposal.location ||
            requirement.control != start.control + ordinal || memoryWitnesses.contains(first + ordinal)) {
          matches = false; break;
        }
        auto payload = proof.normalize(requirement.operation->getOperand(0)), expectedPayload = proof.normalize(argument);
        auto requiredPath = factors(requirement.operation->getOperand(2));
        if (failed(payload) || failed(expectedPayload) || failed(requiredPath)) return failure();
        if (requiredPath->alwaysFalse != rawFactors->alwaysFalse || !sameEndpoint(*payload, *expectedPayload)) { matches = false; break; }
        auto contains = [&](const Factors &haystack, const Factor &needle) {
          return llvm::any_of(haystack.atoms, [&](const Factor &atom) {
            return !atom.expandedSelector && !atom.literal && atom.positive == needle.positive &&
                   sameEndpoint(atom.value, needle.value);
          });
        };
        for (const auto &factor : requiredPath->atoms)
          if (!factor.expandedSelector && !factor.literal && !contains(*rawFactors, factor)) matches = false;
        for (const auto &factor : rawFactors->atoms)
          if (!factor.expandedSelector && !factor.literal && !contains(*requiredPath, factor)) matches = false;
        if (!matches) break;
        lastControl = requirement.control;
      }
      if (!matches) continue;
      bool read = proposal.kind.getValue() == "read";
      uint64_t begin = read ? start.control : 0;
      uint64_t count = read ? lastControl + 1 : view.controls.size();
      if (read) {
        Value cursor = combine.getLhs();
        bool exact = true;
        for (uint64_t index = count; index > begin; --index) {
          auto joined = cursor.getDefiningOp<BitsBinaryOp>();
          if (!joined || joined.getOpcode() != "and") {
            exact = false;
            break;
          }
          auto actualAcceptance =
              proof.parse(joined.getRhs(), PrefixMode::Acceptance);
          if (failed(actualAcceptance))
            return failure();
          auto rawControl = normalizedControl(index - 1);
          if (failed(rawControl))
            return failure();
          auto expectedAcceptance = proof.knownValue(
              *rawControl,
              view.controls[index - 1].kind ==
                      SampledControlKind::BranchSelect
                  ? KnownTest::Either
                  : KnownTest::One,
              view.rule);
          if (failed(expectedAcceptance))
            return failure();
          auto same =
              proof.same(*actualAcceptance, *expectedAcceptance, view.rule);
          if (failed(same))
            return failure();
          if (!*same) {
            exact = false;
            break;
          }
          cursor = joined.getLhs();
        }
        auto root = proof.normalize(cursor);
        auto rootValue = failed(root)
                             ? FailureOr<std::optional<bool>>(failure())
                             : literal(*root, analysis, budget);
        if (failed(rootValue))
          return failure();
        if (!exact || !rootValue->value_or(false))
          continue;
      }
      Prefix expected;
      for (uint64_t index = begin; index < count; ++index) {
        auto next = proof.conjunction(expected, acceptances[index], view.rule);
        if (failed(next)) return failure();
        expected = std::move(*next);
      }
      if (!read) {
        auto prefix = proof.parse(combine.getLhs());
        if (failed(prefix) || failed(proof.equal(*prefix, expected, view.rule)))
          continue;
      }
      for (uint64_t ordinal = 0; ordinal < proposal.arguments.size(); ++ordinal) {
        if (failed(charge(budget, view.rule, 1, sizeof(uint64_t) * 3))) return failure();
        memoryWitnesses.insert(first + ordinal);
      }
      witnessed = true; break;
    }
    if (!witnessed)
      return view.rule->emitOpError() << "Memory proposal differs from exact argument witnesses or reached/atomic admission";
  }
  for (auto [ordinal, requirement] : llvm::enumerate(view.requirements)) {
    SourceCheckBudget::PhaseScope requirementPhase(
        budget, "sampled memory witness coverage proof");
    SourceCheckBudget::LocalScope requirementBudget(budget);
    if (failed(budget.debit(1, view.rule))) return failure();
    if (requirement.kind.getValue() == "memory_argument" && !memoryWitnesses.contains(ordinal))
      return view.rule->emitOpError() << "Memory argument witness lacks its exact issue proposal";
  }
