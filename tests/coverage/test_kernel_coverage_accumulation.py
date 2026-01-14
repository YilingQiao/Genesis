"""
Regression test for kernel coverage accumulation across multiple gs.init/gs.destroy cycles.

This test verifies that kernel coverage correctly accumulates when tests run multiple
init/destroy cycles, which is the normal pattern when using the initialize_genesis fixture.

IMPORTANT: This test requires the --kernel-coverage flag to run.
Run with: pytest tests/coverage/test_kernel_coverage_accumulation.py --kernel-coverage -n 0
"""

import pytest


# Module-local fixture to override the autouse initialize_genesis fixture.
# Returning None causes initialize_genesis to yield early without initializing Genesis.
@pytest.fixture
def backend():
    """Override the backend fixture to skip initialize_genesis."""
    return None


# Marker kernel name - used to verify cycle 2 collection is working
CYCLE2_MARKER_KERNEL_NAME = "cycle2_marker_kernel"


def test_kernel_coverage_accumulates_across_cycles(pytestconfig, backend):
    """
    Verify kernel coverage accumulates across multiple gs.init/gs.destroy cycles.

    This is a regression test for the fix in Round 2 where _collection_complete
    was blocking collection after the first destroy.

    Test design (per Codex Round 7 review):
    1. After gs.init for cycle 1, call tracker.clear() to fully reset profiler records
    2. Run cycle 1 with Box scene - verify kernels collected
    3. Run cycle 2 with a DETERMINISTIC marker kernel that ONLY runs in cycle 2
    4. Assert marker kernel name appears in new_in_second (deterministic check)

    Determinism: The cycle2_marker_kernel is defined locally and executed ONLY in
    cycle 2, guaranteeing it cannot be pre-covered by any prior test.
    """
    # Skip if --kernel-coverage is not enabled (plugin won't be active)
    if not pytestconfig.getoption("--kernel-coverage", default=False):
        pytest.skip("This test requires --kernel-coverage flag")

    import gstaichi as ti

    import genesis as gs

    from tests.coverage.kernel_coverage import get_tracker

    # Get the global tracker (same one used by conftest_plugin)
    tracker = get_tracker()

    # =========================================================================
    # CYCLE 1: Box-only scene (triggers basic rigid body kernels)
    # =========================================================================
    gs.init(backend=gs.cpu, seed=0)

    # ISOLATION: After gs.init, clear tracker AND profiler records
    # This ensures complete isolation from any prior tests
    tracker.clear()
    tracker._collection_complete = False

    # Record baseline (should be empty now)
    initial_kernels = set(tracker._executed_kernels)
    assert len(initial_kernels) == 0, "Tracker should be cleared after tracker.clear()"

    scene1 = gs.Scene(show_viewer=False)
    scene1.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.5)))
    scene1.build()
    scene1.step()

    gs.destroy()

    # Snapshot kernels after first cycle
    first_cycle_kernels = set(tracker._executed_kernels)

    # ASSERTION 1: First cycle must collect kernels
    assert len(first_cycle_kernels) > 0, f"First cycle should collect kernels. Got: {len(first_cycle_kernels)}"

    # =========================================================================
    # CYCLE 2: Execute deterministic marker kernel that ONLY runs here
    # =========================================================================
    gs.init(backend=gs.cpu, seed=1)

    # Define and execute the marker kernel ONLY in cycle 2
    # This kernel is unique to this test and cycle, guaranteeing determinism
    # The kernel must interact with a Taichi field to be captured by the profiler
    marker_field = ti.field(dtype=ti.i32, shape=(1,))

    @ti.kernel
    def cycle2_marker_kernel(field: ti.template()):
        """Marker kernel that only runs in cycle 2 - used for deterministic testing."""
        # Must write to a field to ensure kernel is compiled and executed
        field[0] = 42

    # Execute the marker kernel to ensure it's recorded by the profiler
    cycle2_marker_kernel(marker_field)
    ti.sync()  # Ensure kernel execution is complete before collection

    # Also run a minimal scene to verify normal kernel collection continues
    scene2 = gs.Scene(show_viewer=False)
    scene2.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.5)))
    scene2.build()
    scene2.step()

    gs.destroy()

    # Snapshot kernels after second cycle
    second_cycle_kernels = set(tracker._executed_kernels)

    # Calculate new kernels added in second cycle
    new_in_second = second_cycle_kernels - first_cycle_kernels

    # =========================================================================
    # ASSERTIONS
    # =========================================================================

    # ASSERTION 2: Accumulation - first cycle's kernels must be subset of second
    assert first_cycle_kernels.issubset(second_cycle_kernels), (
        f"Kernel set should accumulate (first cycle should be subset of second). "
        f"Lost kernels: {first_cycle_kernels - second_cycle_kernels}"
    )

    # ASSERTION 3: Deterministic marker kernel check
    # The cycle2_marker_kernel MUST be in new_in_second (it only runs in cycle 2)
    # This is the KEY regression test - if collection stopped after first destroy,
    # the marker kernel would NOT be recorded.
    marker_found = any(CYCLE2_MARKER_KERNEL_NAME in kernel_name for kernel_name in new_in_second)
    assert marker_found, (
        f"REGRESSION DETECTED: {CYCLE2_MARKER_KERNEL_NAME} not found in new kernels! "
        f"This indicates kernel collection stopped after first gs.destroy(). "
        f"New kernels in cycle 2: {new_in_second}"
    )

    # ASSERTION 4: Strict growth - second cycle MUST add new kernels
    # This is a secondary check - the marker kernel assertion is the primary one
    assert len(new_in_second) > 0, (
        f"REGRESSION DETECTED: Second cycle collected 0 new kernels! "
        f"First cycle: {len(first_cycle_kernels)} kernels, "
        f"Second cycle: {len(second_cycle_kernels)} kernels."
    )

    # Log results for verification
    print("\nKernel coverage accumulation test (isolated & deterministic):")
    print(f"  Baseline cleared:  {len(initial_kernels)} kernels (after tracker.clear())")
    print(f"  After cycle 1:     {len(first_cycle_kernels)} kernels (Box scene)")
    print(f"  After cycle 2:     {len(second_cycle_kernels)} kernels (marker + Box)")
    print(f"  New in cycle 2:    {len(new_in_second)} kernels")
    print(f"  Marker kernel:     {CYCLE2_MARKER_KERNEL_NAME} found = {marker_found}")
    print(f"  Accumulation OK:   {first_cycle_kernels.issubset(second_cycle_kernels)}")
    print("  Regression test:   PASS")
