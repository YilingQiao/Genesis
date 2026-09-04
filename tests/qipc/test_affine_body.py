import pytest

import genesis as gs

from ..utils.assertions import assert_allclose


@pytest.mark.parametrize("backend", [gs.cuda])
def test_freefall_ground_contact_fixed_body_and_reset(show_viewer):
    DT = 1e-2
    GRAVITY = -9.81
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(6.0, -6.0, 3.0),
            camera_lookat=(2.0, 0.0, 0.8),
        ),
        show_viewer=show_viewer,
    )
    falling_box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 2.0),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.QIPC.AffineBody(),
    )
    fixed_box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(2.0, 0.0, 0.5),
            size=(0.1, 0.1, 0.1),
            fixed=True,
        ),
        material=gs.materials.QIPC.AffineBody(),
    )
    dropping_box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(4.0, 0.0, 0.15),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.QIPC.AffineBody(
            rho=500.0,
            kappa=1e7,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.QIPC.AffineBody(),
    )
    scene.build()

    fixed_transform_init = fixed_box.get_transform()

    # Contact-free flight follows the implicit-Euler recurrence z_k = z_0 + g dt^2 k (k + 1) / 2 exactly.
    n_steps_fall = 10
    for _ in range(n_steps_fall):
        scene.step()
    z_expected = 2.0 + GRAVITY * DT**2 * n_steps_fall * (n_steps_fall + 1) / 2
    assert_allclose(falling_box.get_transform()[2, 3], z_expected, tol=1e-6)
    verts_after_fall = falling_box.get_verts()

    # The dropping box reaches the ground and settles on the barrier.
    for _ in range(49):
        scene.step()
    transform_prev = dropping_box.get_transform()
    scene.step()
    transform_curr = dropping_box.get_transform()
    # The barrier keeps the contact penetration-free: every vertex stays strictly above the ground.
    assert dropping_box.get_verts()[:, 2].min() > 0.0
    # At rest on the ground: the residual motion over one step is far below the free-fall scale.
    assert (transform_curr - transform_prev).abs().max() / DT < 0.05
    # A fixed body never moves.
    assert_allclose(fixed_box.get_transform(), fixed_transform_init, tol=gs.EPS)

    # Reset restores the initial state and the trajectory replays.
    scene.reset()
    for _ in range(n_steps_fall):
        scene.step()
    assert_allclose(falling_box.get_verts(), verts_after_fall, tol=1e-9)


@pytest.mark.parametrize("backend", [gs.cuda])
def test_build_rejects_unsupported_configurations(show_viewer):
    scene = gs.Scene(
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 1.0),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.QIPC.AffineBody(),
    )
    with pytest.raises(gs.GenesisException):
        scene.build(n_envs=2)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            requires_grad=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 1.0),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.QIPC.AffineBody(),
    )
    with pytest.raises(gs.GenesisException):
        scene.build()


@pytest.mark.parametrize("backend", [gs.cpu])
def test_build_rejects_cpu_backend(show_viewer):
    scene = gs.Scene(
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 1.0),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.QIPC.AffineBody(),
    )
    with pytest.raises(gs.GenesisException):
        scene.build()
