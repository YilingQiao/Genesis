"""Interactive joint control example using ImGui overlay."""

import time
import genesis as gs
from genesis.ext.pyrender.imgui_overlay import ImGuiOverlayPlugin

gs.init()

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.0, 2.0, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
    ),
    show_viewer=True,
)
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
scene.build()

plugin = ImGuiOverlayPlugin()
scene.viewer._pyrender_viewer.register_plugin(plugin)

while scene.viewer.is_alive():
    if plugin.should_step():
        scene.step()
    time.sleep(0.01)  # Small sleep to prevent busy-waiting and allow GUI responsiveness
