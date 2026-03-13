"""Genesis Web GUI demo.

Same scene as examples/viewer_plugin/imgui_joint_control.py,
but rendered headlessly and streamed to a browser via WebSocket.

Usage:
    python examples/web_gui_demo.py
    # Then open http://localhost:8765 in your browser
"""

import argparse
import time

import genesis as gs


def main():
    parser = argparse.ArgumentParser(description="Genesis Web GUI demo")
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-p", "--port", type=int, default=8765)
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(),
        show_viewer=False,
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
    )

    # Same entities as imgui_joint_control.py
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.add_entity(gs.morphs.Box(pos=(0, 0, 1.0), size=(0.2, 0.2, 0.2)))

    # Camera matching the imgui viewer defaults
    scene.add_camera(
        res=(1280, 720),
        pos=(2.0, 2.0, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=30,
    )

    scene.build()

    # Start web server
    from genesis.vis.web.server import GenesisWebServer

    web = GenesisWebServer(scene, port=args.port)
    web.start()

    print(f"Open http://localhost:{args.port} in your browser")

    # Cooperative simulation loop
    while True:
        web.process_commands()
        if web.should_step():
            scene.step()
        web.produce_frame()
        time.sleep(0.01)


if __name__ == "__main__":
    main()
