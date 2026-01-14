"""
Pytest plugin for kernel coverage tracking with xdist support.

To enable, add to your conftest.py:
    pytest_plugins = ["tests.coverage.conftest_plugin"]

Or run with:
    pytest -p tests.coverage.conftest_plugin --kernel-coverage
"""

import json
import os
import sys
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


# Patch gstaichi.init to automatically enable kernel profiler
_original_ti_init = None
_kernel_coverage_enabled = False


def _patched_ti_init(*args, **kwargs):
    """Patched ti.init that enables kernel profiler."""
    global _original_ti_init, _collection_done

    # Inject kernel_profiler=True if not already set
    if "kernel_profiler" not in kwargs:
        kwargs["kernel_profiler"] = True

    result = _original_ti_init(*args, **kwargs)

    # Mark profiler as enabled in our tracker
    try:
        from tests.coverage.kernel_coverage import get_tracker

        tracker = get_tracker()
        tracker._profiler_enabled = True
        import gstaichi as ti

        tracker._ti = ti

        # Reset collection flags for this new init/destroy cycle
        # This allows kernel coverage to accumulate across multiple tests
        tracker._collection_complete = False
        _collection_done = False

        # Patch gs.destroy to collect profiler data before destruction
        _patch_gs_destroy()
    except Exception as e:
        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
        print(f"Kernel coverage [{worker_id}]: failed to enable tracker: {e}", file=sys.stderr)

    return result


# Patch gs.destroy to collect profiler data before destruction
_original_gs_destroy = None
_gs_destroy_patched = False
_collection_done = False  # Flag to avoid re-collecting after taichi is destroyed


def _patch_gs_destroy():
    """Patch genesis.destroy to collect profiler data before destroying taichi context."""
    global _original_gs_destroy, _gs_destroy_patched

    if _gs_destroy_patched:
        return

    try:
        import genesis as gs

        if _original_gs_destroy is None:
            _original_gs_destroy = gs.destroy
            gs.destroy = _patched_gs_destroy
            _gs_destroy_patched = True
    except ImportError:
        pass


def _patched_gs_destroy():
    """Patched gs.destroy that collects profiler data before destroying."""
    global _original_gs_destroy, _collection_done

    # Collect profiler data before destroying taichi context
    try:
        from tests.coverage.kernel_coverage import get_tracker

        tracker = get_tracker()
        if tracker._profiler_enabled and tracker._ti is not None:
            tracker.collect()
            _collection_done = True  # Mark collection as done to avoid re-collecting
    except Exception:
        pass  # Silently ignore collection errors

    # Call original destroy
    return _original_gs_destroy()


def _patch_gstaichi_init():
    """Patch gstaichi.init to enable kernel profiler."""
    global _original_ti_init, _kernel_coverage_enabled

    if _kernel_coverage_enabled:
        return  # Already patched

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "controller")

    try:
        import gstaichi as ti

        if _original_ti_init is None:
            _original_ti_init = ti.init
            ti.init = _patched_ti_init
            _kernel_coverage_enabled = True
            print(f"Kernel coverage [{worker_id}]: patched gstaichi.init", file=sys.stderr)
    except ImportError:
        pass


def pytest_configure(config):
    """Configure kernel coverage if enabled - runs on BOTH controller and workers."""
    if not config.getoption("--kernel-coverage", False):
        return

    worker_id = os.environ.get("PYTEST_XDIST_WORKER")

    # IMPORTANT: Patch gstaichi in ALL processes (controller AND workers)
    # This must happen BEFORE any test imports genesis
    _patch_gstaichi_init()

    # Only discover kernels on controller (workers don't need this info)
    if worker_id is None:
        try:
            from tests.coverage.kernel_coverage import get_tracker

            tracker = get_tracker()
            tracker.discover_kernels()
            config._kernel_tracker = tracker
            config._kernel_coverage_report = config.getoption("--kernel-coverage-report")
            total_kernels = sum(len(v) for v in tracker._defined_kernels.values())
            print(f"Kernel coverage: discovered {total_kernels} kernels")
        except ImportError as e:
            print(f"\nWarning: Could not enable kernel coverage: {e}")
    else:
        # Workers need their own tracker instance
        try:
            from tests.coverage.kernel_coverage import get_tracker

            tracker = get_tracker()
            config._kernel_tracker = tracker
            config._worker_id = worker_id
        except ImportError:
            pass


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Collect and save kernel coverage at end of test session."""
    global _collection_done

    config = session.config
    if not config.getoption("--kernel-coverage", False):
        return

    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    tracker = getattr(config, "_kernel_tracker", None)

    if tracker is None:
        return

    try:
        # Only collect if not already done (gs.destroy may have already collected)
        if not _collection_done:
            tracker.collect()
        executed = tracker._executed_kernels

        if worker_id:
            # Worker: save per-worker coverage file
            worker_file = Path(f".kernel_coverage.{worker_id}.json")
            with open(worker_file, "w") as f:
                json.dump({"executed_kernels": list(executed)}, f)
            print(f"Kernel coverage [{worker_id}]: saved {len(executed)} executed kernels", file=sys.stderr)
        else:
            # Controller: merge all worker files and generate final report
            _merge_worker_coverage(config, tracker)

    except Exception as e:
        print(f"\nWarning: Could not generate kernel coverage report: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()


def _merge_worker_coverage(config, tracker):
    """Merge kernel coverage from all workers into final report."""
    # Find all worker coverage files
    worker_files = list(Path(".").glob(".kernel_coverage.gw*.json"))

    all_executed = set()

    for wf in worker_files:
        try:
            with open(wf) as f:
                data = json.load(f)
                all_executed.update(data.get("executed_kernels", []))
            # Clean up worker file
            wf.unlink()
        except Exception as e:
            print(f"Warning: Could not read {wf}: {e}", file=sys.stderr)

    # Also include any kernels from controller (if tests ran there too)
    all_executed.update(tracker._executed_kernels)

    # Update tracker with merged data
    tracker._executed_kernels = all_executed

    # Discover kernels if not already done
    if not tracker._defined_kernels:
        tracker.discover_kernels()

    # Save final report
    report_path = config.getoption("--kernel-coverage-report")
    tracker.save_report(report_path)
    tracker.print_summary()


# Hook for xdist to ensure workers get configured
def pytest_configure_node(node):
    """Called on xdist controller to configure worker nodes."""
    # Pass kernel coverage option to workers via environment
    if node.config.getoption("--kernel-coverage", False):
        node.workerinput["kernel_coverage_enabled"] = True
