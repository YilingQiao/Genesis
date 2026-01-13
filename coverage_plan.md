# Genesis Coverage Report Implementation Plan

## Problem Statement

Genesis is a physics simulation framework built on GSTaichi (a fork of Taichi), which uses JIT compilation for GPU-accelerated kernels. Standard Python coverage tools like `coverage.py` cannot instrument code inside `@ti.kernel` and `@ti.func` decorated functions because this code is compiled at runtime and executed at the C++/GPU level.

**Challenge**: How to measure test coverage for a codebase with 356 JIT-compiled kernels that bypass standard Python coverage instrumentation?

## Requirements

1. **Python Code Coverage**: Measure line-level coverage for all standard Python code
2. **Kernel Execution Coverage**: Track which `@ti.kernel` functions are called during tests
3. **Combined Reporting**: Generate unified reports showing both metrics
4. **Pytest Integration**: Seamless integration with existing pytest test suite
5. **Parallel Test Support**: Work with pytest-xdist parallel test execution

## Technical Constraints

- GSTaichi kernels are JIT-compiled → cannot get line-level coverage inside kernels
- Tests use `pytest-xdist` for parallel execution → coverage must aggregate across processes
- Tests require `gs.init()` before execution → kernel profiler must be enabled after init
- Multi-backend support (CPU/GPU/Metal) → coverage varies by execution environment

## Solution Architecture

### Two-Level Coverage Approach

| Level | Tool | Measures | Granularity |
|-------|------|----------|-------------|
| Python Coverage | `coverage.py` / `pytest-cov` | All Python code except kernel internals | Line-level |
| Kernel Coverage | GSTaichi `KernelProfiler` | Which kernels were executed | Function-level |

### Components

```
tests/coverage/
├── __init__.py              # Package exports
├── kernel_coverage.py       # Kernel tracking using GSTaichi profiler
├── conftest_plugin.py       # Pytest plugin for --kernel-coverage
└── generate_report.py       # Combined HTML report generator

scripts/
└── run_coverage.sh          # Convenience script to run full coverage

pyproject.toml               # Coverage configuration added
```

## Implementation Steps

- [x] **Step 1**: Add coverage dependencies to `pyproject.toml`
  - Added `pytest-cov` and `coverage[toml]>=7.0` to dev dependencies

- [x] **Step 2**: Configure `coverage.py` in `pyproject.toml`
  - Set source directory, branch coverage, parallel mode
  - Exclude `@ti.kernel` and `@ti.func` lines from coverage report
  - Configure HTML and XML output formats

- [x] **Step 3**: Create kernel coverage tracker (`kernel_coverage.py`)
  - AST-based discovery of all `@ti.kernel` decorated functions
  - Integration with GSTaichi's `KernelProfiler` to track execution
  - JSON report generation with per-file breakdown

- [x] **Step 4**: Create pytest plugin (`conftest_plugin.py`)
  - Add `--kernel-coverage` CLI option
  - Patch `gs.init()` to enable kernel profiler after initialization
  - Collect and save kernel coverage at session end

- [x] **Step 5**: Create combined report generator (`generate_report.py`)
  - Merge Python coverage JSON with kernel coverage JSON
  - Generate HTML dashboard with both metrics
  - Calculate weighted combined score (70% Python, 30% Kernel)

- [x] **Step 6**: Create convenience script (`run_coverage.sh`)
  - Single command to run tests with full coverage
  - Handles dependency checking and report generation

- [ ] **Step 7**: Test on subset of tests
  - Verify Python coverage collection works
  - Verify kernel coverage tracking works
  - Validate combined report generation

## Usage

### Quick Start
```bash
# Run all tests with coverage
./scripts/run_coverage.sh

# Run specific tests
./scripts/run_coverage.sh tests/test_rigid_physics.py
```

### Manual Invocation
```bash
# Python coverage only
pytest --cov=genesis --cov-report=html tests/

# Python + Kernel coverage
pytest --cov=genesis \
    -p tests.coverage.conftest_plugin \
    --kernel-coverage \
    tests/
```

### Generate Report from Existing Data
```bash
python tests/coverage/generate_report.py \
    --coverage-data .coverage_data \
    --kernel-data kernel_coverage.json \
    --output coverage_report
```

## Output

Reports are generated in `coverage_report/`:
- `index.html` - Combined dashboard
- `python/index.html` - Detailed Python line coverage
- `kernel_coverage.json` - Kernel execution data

## Limitations

1. **No line coverage inside kernels**: Cannot measure which lines inside `@ti.kernel` functions executed
2. **Kernel name matching heuristic**: GSTaichi adds suffixes to kernel names; matching uses substring comparison
3. **Single-process recommended**: For accurate coverage, run with `--numprocesses=1`
4. **Backend-specific**: Kernel coverage may vary between CPU/GPU/Metal backends

## Future Improvements

1. Add Taichi source-level tracing if GSTaichi adds support
2. Integrate with CI/CD for coverage reporting on PRs
3. Add coverage thresholds and fail-on-low-coverage option
4. Track kernel coverage across multiple backends

## Files Created

| File | Description |
|------|-------------|
| `pyproject.toml` | Added coverage config and dependencies |
| `tests/coverage/__init__.py` | Package initialization |
| `tests/coverage/kernel_coverage.py` | Kernel coverage tracker class |
| `tests/coverage/conftest_plugin.py` | Pytest plugin for --kernel-coverage |
| `tests/coverage/generate_report.py` | Combined report generator |
| `scripts/run_coverage.sh` | Convenience script |
