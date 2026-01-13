"""
Pytest plugin for kernel coverage tracking.

To enable, add to your conftest.py:
    pytest_plugins = ["tests.coverage.conftest_plugin"]

Or run with:
    pytest --kernel-coverage
"""

import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Add kernel coverage options."""
    group = parser.getgroup("kernel-coverage", "Kernel coverage tracking")
    group.addoption(
        "--kernel-coverage",
        action="store_true",
        default=False,
        help="Enable kernel execution tracking",
    )
    group.addoption(
        "--kernel-coverage-report",
        default="kernel_coverage.json",
        help="Path to save kernel coverage report (default: kernel_coverage.json)",
    )


def pytest_configure(config):
    """Configure kernel coverage if enabled."""
    if not config.getoption("--kernel-coverage", False):
        return

    # Only run on main process, not xdist workers
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return

    # IMPORTANT: Patch gstaichi FIRST, before any code imports it
    print("\nKernel coverage: patching gstaichi.init BEFORE anything imports genesis...")
    _patch_gstaichi_init()

    try:
        from tests.coverage.kernel_coverage import get_tracker

        tracker = get_tracker()
        tracker.discover_kernels()
        config._kernel_tracker = tracker
        print(f"Kernel coverage: discovered {sum(len(v) for v in tracker._defined_kernels.values())} kernels")
    except ImportError as e:
        print(f"\nWarning: Could not enable kernel coverage: {e}")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Collect and save kernel coverage at end of test session."""
    config = session.config
    if not config.getoption("--kernel-coverage", False):
        return

    # Only run on main process
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return

    tracker = getattr(config, "_kernel_tracker", None)
    if tracker is None:
        return

    try:
        # Try to enable profiler and collect data
        tracker.enable()
        tracker.collect()

        # Save report
        report_path = config.getoption("--kernel-coverage-report")
        tracker.save_report(report_path)
        tracker.print_summary()
    except Exception as e:
        print(f"\nWarning: Could not generate kernel coverage report: {e}")


# Patch gstaichi.init to automatically enable kernel profiler
_original_ti_init = None


def _patched_ti_init(*args, **kwargs):
    """Patched ti.init that enables kernel profiler."""
    import sys

    global _original_ti_init

    # Inject kernel_profiler=True if not already set
    if "kernel_profiler" not in kwargs:
        kwargs["kernel_profiler"] = True
        # Use stderr because genesis redirects stdout
        print("Kernel coverage: injected kernel_profiler=True into ti.init", file=sys.stderr)

    result = _original_ti_init(*args, **kwargs)

    # Also call enable_after_init to mark profiler as enabled in our tracker
    try:
        from tests.coverage.kernel_coverage import get_tracker

        tracker = get_tracker()
        tracker._profiler_enabled = True
        import gstaichi as ti

        tracker._ti = ti
        print("Kernel coverage: tracker enabled after ti.init", file=sys.stderr)
    except Exception as e:
        print(f"Kernel coverage: failed to enable tracker: {e}", file=sys.stderr)

    return result


def _patch_gstaichi_init():
    """Patch gstaichi.init to enable kernel profiler."""
    global _original_ti_init

    import sys

    if "gstaichi" in sys.modules:
        print("WARNING: gstaichi already imported before patch!")
    if "genesis" in sys.modules:
        print("WARNING: genesis already imported before patch!")

    try:
        import gstaichi as ti

        if _original_ti_init is None:
            _original_ti_init = ti.init
            ti.init = _patched_ti_init
            print(f"Kernel coverage: patched gstaichi.init (id={id(ti.init)}) to enable kernel_profiler")
            print(f"Kernel coverage: original ti.init id={id(_original_ti_init)}")

            # Verify patch is in module
            import gstaichi

            print(f"Kernel coverage: gstaichi.init is now {gstaichi.init}")
    except ImportError:
        pass


# Also patch gs.init for cases where it's called directly
_original_gs_init = None


def _patched_gs_init(*args, **kwargs):
    """Patched gs.init that enables kernel profiler."""
    global _original_gs_init
    result = _original_gs_init(*args, **kwargs)

    try:
        from tests.coverage.kernel_coverage import get_tracker

        tracker = get_tracker()
        tracker.enable_after_init()
    except Exception:
        pass

    return result


def pytest_collection_modifyitems(session, config, items):
    """Patch gs.init to enable kernel profiling after initialization."""
    if not config.getoption("--kernel-coverage", False):
        return

    global _original_gs_init

    try:
        import genesis as gs

        if _original_gs_init is None and hasattr(gs, "init"):
            _original_gs_init = gs.init
            gs.init = _patched_gs_init
    except ImportError:
        pass
