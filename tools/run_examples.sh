#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
tier=gate
list_only=0
published_only=0
while (($#)); do
  case "$1" in
    --tier) tier="${2:?--tier requires gate or nightly}"; shift 2 ;;
    --list) list_only=1; shift ;;
    --published) published_only=1; tier=nightly; shift ;;
    -h|--help) echo "Usage: $0 [--tier gate|nightly] [--published] [--list]"; exit 0 ;;
    *) pyc_die "unknown option: $1" ;;
  esac
done
[[ "$tier" == gate || "$tier" == nightly ]] || pyc_die "tier must be gate or nightly"
if ((!list_only)); then
  pyc_begin_test_run examples "$tier"
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$(pyc_pythonpath)"
pyc_set_public_helpers
if [[ -n "${PYC_PYTHON_EXECUTABLE:-}" ]]; then
  python_bin="$(cd -- "$(dirname -- "${PYC_PYTHON_EXECUTABLE}")" && pwd)"
  export PATH="${python_bin}:${PATH}"
fi
cd "${PYC_ROOT_DIR}"
build="${PYCIRCUIT_TEST_OUTPUT}/examples-${tier}"
cmake -S "${PYC_ROOT_DIR}/examples" -B "${build}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${PYC_TOOLCHAIN_ROOT}" \
  -Dpycircuit_DIR="${PYC_TOOLCHAIN_ROOT}/share/pycircuit/cmake" \
  -DPython3_EXECUTABLE="${PYC_PYTHON_EXECUTABLE:-$(command -v python3)}" \
  -DPYC_EXAMPLE_TIER="${tier}"
test_args=(-L "^examples$" -L "^${tier}$")
scope="registered-examples-serial-parallel-rtl"
published_names=()
if ((published_only)); then
  published_pattern=""
  while IFS= read -r name; do
    published_names+=("${name}")
    if [[ -n "${published_pattern}" ]]; then
      published_pattern+="|"
    fi
    published_pattern+="${name}"
  done < <(${PYC_PYTHON_EXECUTABLE:-python3} \
    "${PYC_ROOT_DIR}/tools/example_catalog.py" published-tests --format lines)
  published_regex="^(${published_pattern})$"
  test_args+=(-R "${published_regex}")
  scope="published-generated-current-candidate-14"
fi
if ((list_only)); then
  ctest --test-dir "${build}" "${test_args[@]}" \
    --show-only=json-v1 --no-tests=error
  exit 0
fi
ctest --test-dir "${build}" "${test_args[@]}" \
  --show-only=json-v1 --no-tests=error \
  > "${docs_gate_dir}/examples-${tier}-selection.json"
if ((published_only)); then
  "${PYC_PYTHON_EXECUTABLE:-python3}" \
    "${PYC_ROOT_DIR}/tools/example_catalog.py" validate-published-selection \
    --selection "${docs_gate_dir}/examples-${tier}-selection.json"
fi
if ((published_only)); then
  build_targets=()
  for name in "${published_names[@]}"; do
    build_targets+=("pycircuit_${name}" "${name}_verilog_artifacts")
  done
  cmake --build "${build}" --parallel "${PYC_TEST_JOBS:-4}" \
    --target "${build_targets[@]}"
else
  cmake --build "${build}" --parallel "${PYC_TEST_JOBS:-4}"
fi
ctest --test-dir "${build}" "${test_args[@]}" \
  --no-tests=error --output-on-failure \
  --output-junit "${docs_gate_dir}/examples-${tier}.xml"
pyc_finish_test_run examples "$tier" "${scope}"
