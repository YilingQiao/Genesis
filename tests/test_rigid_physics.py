import pytest
import xml.etree.ElementTree as ET

import numpy as np
import genesis as gs
from genesis.ext import trimesh

from .utils import simulate_and_check_mujoco_consistency


@pytest.fixture
def xml_path(request, tmp_path, model_name):
    mjcf = request.getfixturevalue(model_name)
    xml_tree = ET.ElementTree(mjcf)
    file_path = tmp_path / f"{model_name}.xml"
    xml_tree.write(file_path, encoding="utf-8", xml_declaration=True)
    return str(file_path)


@pytest.fixture(scope="session")
def box_plan():
    """Generate an XML model for a box on a plane."""
    mjcf = ET.Element("mujoco", model="one_box")
    ET.SubElement(mjcf, "option", timestep="0.01")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3", friction="1. 0.5 0.5")
    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "geom", type="plane", name="floor", pos="0. 0. 0.", size="40. 40. 40.")
    box_body = ET.SubElement(worldbody, "body", name="box", pos="0. 0. 0.3")
    ET.SubElement(box_body, "geom", type="box", size="0.2 0.2 0.2", pos="0. 0. 0.")
    ET.SubElement(box_body, "joint", name="root", type="free")
    return mjcf


@pytest.fixture(scope="session")
def chain_capsule_hinge(asset_tmp_path):  # , enable_mesh):
    enable_mesh = True
    if enable_mesh:
        mesh_path = asset_tmp_path / "capsule.obj"
        tmesh = trimesh.creation.capsule(radius=0.05, height=0.5, count=(3, 3))
        tmesh.export(mesh_path, file_type="obj")

    mjcf = ET.Element("mujoco", model="two_stick_robot")
    ET.SubElement(mjcf, "option", timestep="0.05")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3")
    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(asset, "mesh", name="capsule", refpos="0 0 -0.25", refquat="0.707 0 -0.707 0", file=str(mesh_path))
    worldbody = ET.SubElement(mjcf, "worldbody")
    link0 = ET.SubElement(worldbody, "body", name="body1", pos="0.1 0.2 0.0", quat="0.707 0 0.707 0")
    if enable_mesh:
        ET.SubElement(link0, "geom", type="mesh", mesh="capsule", rgba="0 0 1 0.3")
    else:
        ET.SubElement(link0, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05", rgba="0 0 1 0.3")
    link1 = ET.SubElement(link0, "body", name="body2", pos="0.5 0.2 0.0", quat="0.92388 0 0 0.38268")
    if enable_mesh:
        ET.SubElement(link1, "geom", type="mesh", mesh="capsule")
    else:
        ET.SubElement(link1, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05")
    ET.SubElement(link1, "joint", type="hinge", name="joint1", axis="0 0 1", pos="0.0 0.0 0.0")
    link2 = ET.SubElement(link1, "body", name="body3", pos="0.5 0.2 0.0", quat="0.92388 0 0.38268 0.0")
    if enable_mesh:
        ET.SubElement(link2, "geom", type="mesh", mesh="capsule")
    else:
        ET.SubElement(link2, "geom", type="capsule", fromto="0 0 0 0.5 0 0", size="0.05")
    ET.SubElement(link2, "joint", type="hinge", name="joint2", axis="0 1 0")
    return mjcf


@pytest.mark.parametrize("model_name", ["box_plan"])
@pytest.mark.parametrize(
    "gs_solver",
    [gs.constraint_solver.CG, gs.constraint_solver.Newton],
)
@pytest.mark.parametrize(
    "gs_integrator",
    [gs.integrator.implicitfast, gs.integrator.Euler],
)
@pytest.mark.parametrize("backend", [gs.cpu], indirect=True)
def test_box_plan_dynamics(gs_sim, mj_sim):
    (gs_robot,) = gs_sim.entities
    cube_pos = np.array([0.0, 0.0, 0.6])
    cube_quat = np.random.rand(4)
    cube_quat /= np.linalg.norm(cube_quat)
    qpos = np.concatenate((cube_pos, cube_quat))
    qvel = np.random.rand(gs_robot.n_dofs) * 0.2
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=150)


@pytest.mark.parametrize("model_name", ["chain_capsule_hinge"])
@pytest.mark.parametrize(
    "gs_solver",
    [gs.constraint_solver.CG],
)
@pytest.mark.parametrize(
    "gs_integrator",
    [gs.integrator.implicitfast],
)
@pytest.mark.parametrize("backend", [gs.cpu], indirect=True)
def test_simple_kinematic_chain(gs_sim, mj_sim):
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=500)


@pytest.mark.parametrize("xml_path", ["xml/walker.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast])
@pytest.mark.parametrize("backend", [gs.cpu], indirect=True)
def test_walker(gs_sim, mj_sim):
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=500)
