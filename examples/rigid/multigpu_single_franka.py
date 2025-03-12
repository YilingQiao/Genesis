import multiprocessing
import os
import genesis as gs
import argparse


def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    backend = gs.cpu if args.cpu else gs.gpu
    print("backend:", backend)
    gs.init(backend=backend, logger_verbose_time=True)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            # constraint_solver=gs.constraint_solver.Newton,
        ),
        show_FPS=False,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        visualize_contact=True,
    )

    ########################## build ##########################
    scene.build()
    for i in range(1000):
        scene.step()


def run(gpu_id, data_part, func):

    # Set environment args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # main script
    func()

    print(f"[GPU {gpu_id}] processing: {data_part}")


if __name__ == "__main__":
    num_gpus = 2
    data_partitions = ["A", "B"]

    processes = []
    for i in range(num_gpus):
        # i = 0
        p = multiprocessing.Process(target=run, args=(i, data_partitions[i], main))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
