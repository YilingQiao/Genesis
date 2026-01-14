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


def test_kernel_coverage_accumulates_across_cycles(pytestconfig, backend):
    """
    Verify kernel coverage accumulates across multiple gs.init/gs.destroy cycles.

    This is a regression test for the fix in Round 2 where _collection_complete
    was blocking collection after the first destroy.

    Uses minimal Genesis scenes to exercise actual kernel execution through the
    patched gstaichi.init path that enables kernel profiling.
    """
    # Skip if --kernel-coverage is not enabled (plugin won't be active)
    if not pytestconfig.getoption("--kernel-coverage", default=False):
        pytest.skip("This test requires --kernel-coverage flag")

    import genesis as gs

    from tests.coverage.kernel_coverage import get_tracker

    # Get the global tracker (same one used by conftest_plugin)
    tracker = get_tracker()

    # Record initial state (copy the set to compare later)
    initial_kernels = set(tracker._executed_kernels)

    # First cycle: init with kernel profiler (via patched gstaichi.init), run minimal scene
    gs.init(backend=gs.cpu, seed=0)

    # Create minimal scene that executes some kernels
    scene1 = gs.Scene(show_viewer=False)
    scene1.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.5)))
    scene1.build()

    # Single step is enough to execute kernels
    scene1.step()

    gs.destroy()

    # Snapshot kernels after first cycle
    first_cycle_kernels = set(tracker._executed_kernels)

    # Verify first cycle collected new kernels
    new_in_first = first_cycle_kernels - initial_kernels
    assert len(new_in_first) > 0, (
        f"First cycle should collect new kernels. "
        f"Initial: {len(initial_kernels)}, After first: {len(first_cycle_kernels)}"
    )

    # Second cycle: init again, run different scene configuration
    gs.init(backend=gs.cpu, seed=1)

    # Create a different scene with Plane (potentially different kernels)
    scene2 = gs.Scene(show_viewer=False)
    scene2.add_entity(gs.morphs.Plane())
    scene2.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 1)))
    scene2.build()

    # Run a few steps
    for _ in range(3):
        scene2.step()

    gs.destroy()

    # Snapshot kernels after second cycle
    second_cycle_kernels = set(tracker._executed_kernels)

    # Calculate new kernels added in second cycle
    new_in_second = second_cycle_kernels - first_cycle_kernels

    # Strict accumulation assertion 1: first cycle's kernels must be a subset of second
    # (accumulation means we never lose kernels)
    assert first_cycle_kernels.issubset(second_cycle_kernels), (
        f"Kernel set should accumulate (first cycle should be subset of second). "
        f"First cycle kernels not in second: {first_cycle_kernels - second_cycle_kernels}"
    )

    # Strict accumulation assertion 2: second cycle MUST add new kernels
    # This catches the original regression where collection stopped after first destroy.
    # The Plane morph in cycle 2 triggers different kernels than Box-only in cycle 1.
    assert len(new_in_second) > 0, (
        f"Second cycle MUST collect new kernels (regression test). "
        f"If this fails, kernel collection may have stopped after first gs.destroy(). "
        f"First cycle: {len(first_cycle_kernels)} kernels, "
        f"Second cycle: {len(second_cycle_kernels)} kernels, "
        f"New in second: {len(new_in_second)}"
    )

    # Log results for verification
    print("\nKernel coverage accumulation test:")
    print(f"  Initial:          {len(initial_kernels)} kernels")
    print(f"  After cycle 1:    {len(first_cycle_kernels)} kernels (+{len(new_in_first)})")
    print(f"  After cycle 2:    {len(second_cycle_kernels)} kernels (+{len(new_in_second)})")
    print(f"  First subset of second: {first_cycle_kernels.issubset(second_cycle_kernels)}")
    print(f"  Strict growth verified: {len(new_in_second)} new kernels in cycle 2")
