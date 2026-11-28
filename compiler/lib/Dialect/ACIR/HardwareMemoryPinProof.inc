        using Paths = SmallVector<HardwareIdentityEndpoint, 0>;
        struct KnownConstant {
          uint64_t width;
          MathIntAttr value;
        };
        auto knownConstant = [&](const HardwareIdentityEndpoint &endpoint)
            -> FailureOr<std::optional<KnownConstant>> {
          if (failed(step(module)))
            return failure();
          auto bindings = scopeBindings(endpoint.scope);
          if (failed(bindings))
            return failure();
          if (!isa<BitsType>(endpoint.resolvedType))
            return std::optional<KnownConstant>();
          auto constant = endpoint.value.getDefiningOp<BitsConstantOp>();
          if (!constant)
            return std::optional<KnownConstant>();
          auto width = analysis.getPackedWidth(endpoint.resolvedType, *bindings,
                                               constant);
          auto evaluated =
              analysis.evaluateStatic(constant.getValue(), *bindings, constant);
          if (failed(width) || failed(evaluated))
            return failure();
          auto integer = dyn_cast<MathIntAttr>(*evaluated);
          if (!integer)
            return std::optional<KnownConstant>();
          if (failed(step(module, SourceCheckBudget::add(
                                      integer.getCanonicalValue().size(),
                                      *width / 64 + 1))))
            return failure();
          llvm::APSInt number(integer.getCanonicalValue());
          if (number.isNegative() || number.getActiveBits() > *width)
            return std::optional<KnownConstant>();
          return std::optional<KnownConstant>(KnownConstant{*width, integer});
        };
        auto constantsEqual = [&](const HardwareIdentityEndpoint &left,
                                  const HardwareIdentityEndpoint &right)
            -> FailureOr<bool> {
          auto lhs = knownConstant(left), rhs = knownConstant(right);
          if (failed(lhs) || failed(rhs))
            return failure();
          if (!*lhs || !*rhs || (**lhs).width != (**rhs).width)
            return false;
          return llvm::APSInt::compareValues(
                     llvm::APSInt((**lhs).value.getCanonicalValue()),
                     llvm::APSInt((**rhs).value.getCanonicalValue())) == 0;
        };
        auto unionMatches = [&](const HardwareIdentityEndpoint &active,
                                ArrayRef<HardwareIdentityEndpoint> paths) -> LogicalResult {
          SmallVector<HardwareIdentityEndpoint, 0> pending;
          SmallVector<bool, 0> matched(paths.size(), false);
          if (failed(pay(sizeof(HardwareIdentityEndpoint)))) return failure();
          pending.push_back(active);
          size_t matchedCount = 0;
          while (!pending.empty()) {
            auto actual = pending.pop_back_val();
            if (failed(step(module))) return failure();
            // Proposal paths are leaves even if a future path happens to use
            // OR internally. A route union cannot inspect their internals.
            bool found = false;
            for (auto [ordinal, path] : llvm::enumerate(paths)) {
              if (failed(step(module))) return failure();
              if (matched[ordinal] || !same(actual, path)) continue;
              matched[ordinal] = true;
              ++matchedCount;
              found = true;
              break;
            }
            if (found) continue;
            auto zero = literal(actual.value, actual.scope, false);
            if (failed(zero)) return failure();
            if (*zero) continue;
            auto joined = actual.value.getDefiningOp<BitsBinaryOp>();
            if (!joined || joined.getOpcode() != "or")
              return module->emitOpError()
                     << "Memory incoming active union differs from its exact "
                        "data contributors; unmatched "
                     << actual.value << "; expected " << paths.size();
            auto left = normal(joined.getLhs(), actual.scope), right = normal(joined.getRhs(), actual.scope);
            if (failed(left) || failed(right) || failed(pay(2 * sizeof(HardwareIdentityEndpoint)))) return failure();
            pending.push_back(*right); pending.push_back(*left);
          }
          if (matchedCount != paths.size())
            return module->emitOpError() << "Memory incoming active union omits a data contributor";
          return success();
        };
        std::function<FailureOr<Paths>(HardwareIdentityEndpoint, unsigned)> consume =
            [&](HardwareIdentityEndpoint actual, unsigned depth) -> FailureOr<Paths> {
          if (depth > 128 || failed(step(module))) return failure();
          SmallVector<Paths, 0> pieces;
          for (;;) {
            auto selected = memorySelection(actual);
            if (failed(selected)) return failure();
            if (!*selected) {
              // Equal data proposals can fold their nested selects to one
              // unconditional carrier. Authenticate every remaining payload
              // against that carrier, while returning every proposal path so
              // unionMatches still proves the complete active OR separately.
              Paths folded;
              bool sharedPayload = !expected.empty();
              for (const auto &candidate : expected) {
                if (!candidate.argument) {
                  sharedPayload = false;
                  break;
                }
                auto data = normal(
                    candidate.proposal->proposal
                        ->arguments[*candidate.argument],
                    candidate.proposal->scope);
                auto path = normal(candidate.proposal->proposal->path,
                                   candidate.proposal->scope);
                if (failed(data) || failed(path))
                  return failure();
                auto samePayload = constantsEqual(*data, actual);
                if (failed(samePayload))
                  return failure();
                sharedPayload &= *samePayload;
                if (failed(pay(sizeof(HardwareIdentityEndpoint))))
                  return failure();
                folded.push_back(*path);
              }
              if (sharedPayload) {
                expected.clear();
                if (failed(pay(sizeof(Paths))))
                  return failure();
                pieces.push_back(std::move(folded));
                break;
              }
              auto idle = memoryIdle(actual);
              if (failed(idle) || !*idle)
                return module->emitOpError()
                       << "Memory physical pin requires its known safe idle "
                          "carrier";
              break;
            }
            const auto &selection = **selected;
            auto match = expected.end();
            for (auto candidate = expected.begin(); candidate != expected.end(); ++candidate) {
              if (failed(step(module))) return failure();
              auto path = normal(candidate->proposal->proposal->path, candidate->proposal->scope);
              if (failed(path)) return failure();
              if (!same(*path, selection.condition)) continue;
              bool dataMatches;
              if (candidate->argument) {
                auto data = normal(candidate->proposal->proposal->arguments[*candidate->argument], candidate->proposal->scope);
                if (failed(data)) return failure();
                dataMatches = same(*data, selection.payload);
              } else {
                auto one = literal(selection.payload.value, selection.payload.scope, true);
                if (failed(one)) return failure();
                dataMatches = *one;
              }
              if (dataMatches) { match = candidate; break; }
            }
            Paths current;
            if (match != expected.end()) {
              expected.erase(match);
              if (failed(pay(sizeof(HardwareIdentityEndpoint)))) return failure();
              current.push_back(selection.condition);
            } else {
              auto path = dyn_cast<BlockArgument>(selection.condition.value),
                   data = dyn_cast<BlockArgument>(selection.payload.value);
              bool abstract = path && data && selection.condition.scope.path.empty() && selection.payload.scope.path.empty() &&
                  path.getOwner()->getParentOp() == module && data.getOwner() == path.getOwner() &&
                  path.getArgNumber() >= abstractStart && data.getArgNumber() < abstractEnd &&
                  (path.getArgNumber() - abstractStart) % 2 == 0 && data.getArgNumber() == path.getArgNumber() + 1;
              if (abstract) {
                if (failed(pay(sizeof(HardwareIdentityEndpoint)))) return failure();
                current.push_back(selection.condition);
              } else {
                // An actual child incoming pair adds a select(active,data,idle)
                // around its parent's already merged contributions. Authenticate
                // both sides: exact payload origins below, exact active OR above.
                auto nested = consume(selection.payload, depth + 1);
                if (failed(nested) || failed(unionMatches(selection.condition, *nested))) return failure();
                current = std::move(*nested);
              }
            }
            if (failed(pay(sizeof(Paths)))) return failure();
            pieces.push_back(std::move(current));
            actual = selection.fallback;
          }
          Paths result;
          for (const auto &piece : llvm::reverse(pieces)) {
            if (failed(pay(SourceCheckBudget::multiply(piece.size(), sizeof(HardwareIdentityEndpoint))))) return failure();
            llvm::append_range(result, piece);
          }
          return result;
        };
        auto actual = normal(layout.allocation->getOperand(pin), representative.target.scope);
        if (failed(actual)) return failure();
        auto consumed = consume(*actual, 0);
        if (failed(consumed)) return failure();
        if (!expected.empty())
          return module->emitOpError()
                 << "Memory physical pin differs from complete actual "
                    "proposals for operand "
                 << pin << "; matched " << consumed->size() << ", missing "
                 << expected.size() << "; first missing path "
                 << expected.front().proposal->proposal->path
                 << ", occurrence "
                 << expected.front().proposal->proposal->occurrence;
