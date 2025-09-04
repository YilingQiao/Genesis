import genesis as gs
import numpy as np
import torch
import argparse
from tqdm import tqdm
from batch_scene import PhysOptim
SCENE_POS = (0.5, 0.5, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver", choices=["explicit", "implicit"], default="explicit", help="FEM solver type (default: explicit)"
    )
    parser.add_argument("--dt", type=float, help="Time step (auto-selected based on solver if not specified)")
    parser.add_argument(
        "--substeps", type=int, help="Number of substeps (auto-selected based on solver if not specified)"
    )
    #parser.add_argument("--vis", "-v", action="store_true", help="Show visualization GUI")

    args = parser.parse_args()

    if args.solver == "explicit":
        dt = args.dt if args.dt is not None else 1e-4
        substeps = args.substeps if args.substeps is not None else 5
    else:  # implicit
        dt = args.dt if args.dt is not None else 1e-3
        substeps = args.substeps if args.substeps is not None else 1

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=substeps,
            gravity=(0, 0, -9.81),
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=args.solver == "implicit",
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())

    blob = scene.add_entity(
        morph=gs.morphs.Sphere(pos=tuple(map(sum, zip(SCENE_POS, (-0., -0., 0.3)))), radius=0.1),
        material=gs.materials.FEM.Elastic(E=1.0e4, nu=0.45, rho=1000.0, model="stable_neohookean"),
    )

    cube = scene.add_entity(
        morph=gs.morphs.Box(pos=tuple(map(sum, zip(SCENE_POS, (0., 0., 0)))), size=(0.2, 0.2, 0.2)),
        material=gs.materials.FEM.Elastic(E=1.0e6, nu=0.45, rho=1000.0, model="stable_neohookean"),
    )


    #print(cube.elems)
    #print(cube.init_positions.numpy())
    useGenesis = False
    if useGenesis:
        scene.build()
        for step in range(50000):
            scene.step()
    else:
        mesh_trimesh_dict = {}
        mesh_trimesh_dict['cube'] = [cube.init_positions.numpy(), cube.elems]
        mesh_trimesh_dict['blob'] = [blob.init_positions.numpy(), blob.elems]
        ipc_sim = PhysOptim(mesh_trimesh_dict, optim_num=0, visualize=True)
        for i_step in range(1000000):
            print(f"step {i_step}")
            new_state = ipc_sim.step()
            # new_state is a (N, 3) numpy array of surface vertices


if __name__ == "__main__":
    main()
