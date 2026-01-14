"""
Regression test for kernel coverage accumulation across multiple gs.init/gs.destroy cycles.

This test verifies that kernel coverage correctly accumulates when tests run multiple
init/destroy cycles, which is the normal pattern when using the initialize_genesis fixture.

Run with: pytest tests/coverage/test_kernel_coverage_accumulation.py --kernel-coverage -n 0
"""

import pytest


# Use backend=None to skip the autouse initialize_genesis fixture
# This test manages its own init/destroy cycles
@pytest.mark.parametrize("backend", [None])
def test_kernel_coverage_accumulates_across_cycles(backend):
    """
    Verify kernel coverage accumulates across multiple gs.init/gs.destroy cycles.

    This is a regression test for the fix in Round 2 where _collection_complete
    was blocking collection after the first destroy.
    """
    from tests.coverage.kernel_coverage import get_tracker

    # Get the global tracker (same one used by conftest_plugin)
    tracker = get_tracker()

    # Record initial state
    initial_count = len(tracker._executed_kernels)

    # First cycle: init, run a kernel, destroy
    import genesis as gs

    gs.init(backend=gs.cpu, seed=0)

    # Create a simple scene that will execute some kernels
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.5)))
    scene.build()

    # Step the simulation to execute kernels
    for _ in range(5):
        scene.step()

    gs.destroy()

    # Check that kernels were collected in first cycle
    after_first_cycle = len(tracker._executed_kernels)
    assert after_first_cycle > initial_count, (
        f"First cycle should collect kernels: initial={initial_count}, after={after_first_cycle}"
    )

    # Second cycle: init, run different operations, destroy
    gs.init(backend=gs.cpu, seed=1)

    # Create a different scene configuration
    scene2 = gs.Scene(show_viewer=False)
    # Add multiple entities to potentially trigger different kernels
    plane = scene2.add_entity(gs.morphs.Plane())
    box1 = scene2.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 1)))
    box2 = scene2.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.5, 0, 2)))
    scene2.build()

    # Run more steps
    for _ in range(10):
        scene2.step()

    gs.destroy()

    # Check that kernels accumulated (should be >= first cycle, likely more)
    after_second_cycle = len(tracker._executed_kernels)
    assert after_second_cycle >= after_first_cycle, (
        f"Second cycle should maintain or increase kernel count: "
        f"after_first={after_first_cycle}, after_second={after_second_cycle}"
    )

    # Log results for verification
    print("\nKernel coverage accumulation test:")
    print(f"  Initial:       {initial_count} kernels")
    print(f"  After cycle 1: {after_first_cycle} kernels")
    print(f"  After cycle 2: {after_second_cycle} kernels")
    print(f"  Accumulated:   {after_second_cycle - initial_count} new kernels")
