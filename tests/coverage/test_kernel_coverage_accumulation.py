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

    Test design (per Codex reviews):
    1. Clear tracker baseline to isolate from prior tests
    2. Run cycle 1 with Box scene - verify kernels collected
    3. Run cycle 2 with Plane scene - verify NEW kernels collected
    4. The strict growth assertion catches the original regression

    Determinism: By clearing baseline and using different scene configurations,
    we guarantee that each cycle executes different kernels. The Plane morph
    triggers SAP collision detection kernels not used with Box-only scenes.
    """
    # Skip if --kernel-coverage is not enabled (plugin won't be active)
    if not pytestconfig.getoption("--kernel-coverage", default=False):
        pytest.skip("This test requires --kernel-coverage flag")

    import genesis as gs

    from tests.coverage.kernel_coverage import get_tracker

    # Get the global tracker (same one used by conftest_plugin)
    tracker = get_tracker()

    # ISOLATION: Clear tracker baseline to ensure test is independent of prior runs
    tracker._executed_kernels.clear()
    tracker._collection_complete = False

    # Record baseline (should be empty now)
    initial_kernels = set(tracker._executed_kernels)
    assert len(initial_kernels) == 0, "Tracker should be cleared at test start"

    # =========================================================================
    # CYCLE 1: Box-only scene (triggers basic rigid body kernels)
    # =========================================================================
    gs.init(backend=gs.cpu, seed=0)

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
    # CYCLE 2: Plane scene with collision (triggers different kernels)
    # The Plane morph triggers SAP collision detection kernels that Box alone doesn't
    # =========================================================================
    gs.init(backend=gs.cpu, seed=1)

    scene2 = gs.Scene(show_viewer=False)
    # Plane triggers collision detection kernels
    scene2.add_entity(gs.morphs.Plane())
    # Box falling onto plane ensures collision kernels execute
    scene2.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 1)))
    scene2.build()

    # Multiple steps to ensure collision occurs
    for _ in range(5):
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

    # ASSERTION 3: Strict growth - second cycle MUST add new kernels
    # This is the KEY regression test assertion. If _collection_complete wasn't
    # reset on gs.init(), the second cycle would collect 0 new kernels.
    # The Plane+Box collision scene guarantees different kernels from Box-only.
    assert len(new_in_second) > 0, (
        f"REGRESSION DETECTED: Second cycle collected 0 new kernels! "
        f"This indicates kernel collection stopped after first gs.destroy(). "
        f"First cycle: {len(first_cycle_kernels)} kernels, "
        f"Second cycle: {len(second_cycle_kernels)} kernels. "
        f"Expected second > first due to Plane collision kernels."
    )

    # ASSERTION 4: Sanity check - verify meaningful growth
    # The Plane scene should trigger at least 10 new kernels for collision detection
    min_expected_new = 10
    assert len(new_in_second) >= min_expected_new, (
        f"Expected at least {min_expected_new} new kernels from Plane collision scene, "
        f"but got {len(new_in_second)}. This may indicate partial collection failure."
    )

    # Log results for verification
    print("\nKernel coverage accumulation test (isolated & deterministic):")
    print(f"  Baseline cleared:  {len(initial_kernels)} kernels (isolated from prior tests)")
    print(f"  After cycle 1:     {len(first_cycle_kernels)} kernels (Box scene)")
    print(f"  After cycle 2:     {len(second_cycle_kernels)} kernels (Plane+Box collision)")
    print(f"  New in cycle 2:    {len(new_in_second)} kernels")
    print(f"  Accumulation OK:   {first_cycle_kernels.issubset(second_cycle_kernels)}")
    print(f"  Regression test:   PASS (cycle 2 added {len(new_in_second)} new kernels)")
