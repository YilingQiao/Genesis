import numpy as np
from polyscope import imgui

from uipc import Logger, Timer
from uipc.core import Engine, World, Scene
from uipc.geometry import ground, label_surface, trimesh, tetmesh
from uipc.constitution import AffineBodyConstitution, StableNeoHookean, ElasticModuli
from uipc.gui import SceneGUI
from uipc.unit import MPa, GPa
from asset_dir import AssetDir

import trimesh as my_trimesh


class PhysOptim(object):
    def __init__(self, mesh_trimesh_dict, optim_num=0, visualize=True):
        Timer.disable_all()
        Logger.set_level(Logger.Level.Error)

        workspace = AssetDir.output_path(__file__)
        this_folder = AssetDir.folder(__file__)

        engine = Engine("cuda", workspace)
        world = World(engine)

        config = Scene.default_config()
        config["gravity"] = [[0.0], [-0.0], [-9.8]]
        config["dt"] = 0.001
        config["contact"]["d_hat"] = 0.001
        config["newton"]["velocity_tol"] = 0.001
        config["contact"]["enable"] = True
        config["contact"]["friction"]["enable"] = False
        config["line_search"]["max_iter"] = 30
        config["linear_system"]["tol_rate"] = 1e-4
        config["sanity_check"]["enable"] = False
        scene = Scene(config)

        # begin setup the scene
        # t = Transform.Identity()
        # t.rotate(AngleAxis(np.pi/2, Vector3.UnitX()))
        # io = SimplicialComplexIO(t)

        # create constituiton
        abd = AffineBodyConstitution()
        stk = StableNeoHookean()
        # create constraint
        # rm = RotatingMotor()
        scene_contacts = {}
        scene_number = 1
        scene.contact_tabular().default_model(0.1, 1e9)

        # close the collision between different scene
        for i in range(scene_number):
            scene_contacts[i] = scene.contact_tabular().create_subscene(f"contact_model{i}")
        for i in range(scene_number):
            for j in range(scene_number):
                if i != j:
                    scene.contact_tabular().subscene_insert(scene_contacts[i], scene_contacts[j], False)
        # close the collision within the same scene
        cet0 = scene.contact_tabular().create("contact0")
        cet1 = scene.contact_tabular().create("contact1")
        # scene.contact_tabular().insert(cet0, cet1, 0, 0, False)

        cube_mesh0 = tetmesh(mesh_trimesh_dict["cube"][0], mesh_trimesh_dict["cube"][1])
        blob_mesh0 = tetmesh(mesh_trimesh_dict["blob"][0], mesh_trimesh_dict["blob"][1])
        cube_obj = {}
        cube_mesh = {}
        blob_obj = {}
        blob_mesh = {}
        self._mesh_handles = {}
        for i in range(scene_number):
            cube_obj[i] = scene.objects().create(f"cube{i}")

            # testobj = my_trimesh.load(f'{AssetDir.trimesh_path()}/gear0/gear.obj')

            cube_mesh[i] = cube_mesh0
            scene_contacts[i].subscene_append(cube_mesh[i])
            # cet0.apply_to(cube_mesh[i])
            # view(gear_mesh.transforms())[0] = (t.matrix())

            # gear_mesh = io.read(f'{AssetDir.trimesh_path()}/gear0/gear.obj')
            label_surface(cube_mesh[i])
            moduli_box = ElasticModuli.youngs_poisson(1e3 * pow(3, i), 0.45)
            stk.apply_to(cube_mesh[i], moduli_box)  # 100 MPa
            # rm.apply_to(gear_mesh, 100, motor_axis=Vector3.UnitZ(), motor_rot_vel=np.pi)
            cube_obj[i].geometries().create(cube_mesh[i])
            self._mesh_handles[f"cube{i}"] = cube_mesh[i]

            blob_obj[i] = scene.objects().create(f"blob{i}")

            blob_mesh[i] = blob_mesh0
            scene_contacts[i].subscene_append(blob_mesh[i])
            # cet1.apply_to(blob_mesh[i])
            label_surface(blob_mesh[i])
            # moduli_blob = ElasticModuli.youngs_poisson(1e3*pow(10, i), 0.45)
            abd.apply_to(blob_mesh[i], 1e9)  # 100 MPa
            # rm.apply_to(gear_mesh, 100, motor_axis=Vector3.UnitZ(), motor_rot_vel=np.pi)
            blob_obj[i].geometries().create(blob_mesh[i])
            self._mesh_handles[f"blob{i}"] = blob_mesh[i]

        ground_height = 0
        ground_obj = scene.objects().create("ground")
        ground_geo = ground(ground_height, [0, 0, 1])
        ground_obj.geometries().create(ground_geo)

        # end setup the scene

        world.init(scene)
        self._engine = engine
        self._world = world
        self._scene = scene
        self._visualize = visualize
        self._scene_gui = None
        # Lazy import to avoid GUI setup when not visualizing

        # visualize = False
        if visualize:

            sgui = SceneGUI(scene, "split")

            sgui.register()
            sgui.set_edge_width(1)
            self._scene_gui = sgui

            run = True

        else:
            # Non-visual mode does not auto-run. Users should call step().
            pass

    def step(self):
        """Advance one frame and return all vertices' positions and velocities per object.

        Returns a dict: { object_name: { 'positions': (N,3) float64, 'velocities': (N,3) float64 } }
        """
        # Advance simulation
        print("step")
        self._world.advance()
        self._world.retrieve()
        # If GUI is active, let it update too
        if self._scene_gui is not None:
            self._scene_gui.update()

        # Gather full volumetric tet states (all tet objects in scene)
        from uipc import builtin
        from uipc.backend import SceneVisitor
        from uipc.geometry import SimplicialComplexSlot, apply_transform, merge

        state = {}
        visitor = SceneVisitor(self._scene)
        for geo_slot in visitor.geometries():
            if isinstance(geo_slot, SimplicialComplexSlot):
                geo = geo_slot.geometry()
                if geo.dim() == 3:
                    proc_geo = geo
                    if geo.instances().size() >= 1:
                        proc_geo = merge(apply_transform(geo))
                    pos = proc_geo.positions().view().reshape(-1, 3)
                    # tets = proc_geo.tetrahedra().topo().view().reshape(-1, 4)
                    # vel_slot = proc_geo.vertices().find(builtin.velocity)
                    # vel = vel_slot.view().reshape(-1, 3) if vel_slot is not None else np.zeros_like(pos)
                    state[f"tet_{geo_slot.id()}"] = {
                        "positions": pos,
                        # 'velocities': vel,
                        # 'tets': tets,
                    }
                    print(
                        "pos",
                        state[f"tet_{geo_slot.id()}"]["positions"].shape,
                        state[f"tet_{geo_slot.id()}"]["positions"].mean(),
                    )
                    # print("pos", pos.shape, pos.mean())
        return state


if __name__ == "__main__":
    test = PhysOptim()
