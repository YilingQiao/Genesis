"""
Example demonstrating camera sensors with different rendering backends.

Creating cameras as sensors using add_sensor() with three backends
Rasterizer, Raytracer and BatchRenderer, plus Apollo.
Test the attachment, add light, batch rendering functionalities.
"""

import os
import matplotlib.pyplot as plt
import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.options.sensors import RasterizerCameraOptions, RaytracerCameraOptions, BatchRendererCameraOptions

########################## init ##########################
gs.init(seed=0, precision="32", backend=gs.gpu, logging_level="info")

# Import Apollo camera sensor (after gs.init)
try:
    from apollo_plugin.apollo_camera_sensor import ApolloCameraOptions

    APOLLO_PLUGIN_AVAILABLE = True
except ImportError:
    ApolloCameraOptions = None
    APOLLO_PLUGIN_AVAILABLE = False

########################## check dependencies ##########################
# Try to import LuisaRenderPy to determine if raytracer is available
try:
    import LuisaRenderPy

    ENABLE_RAYTRACER = True
    print("✓ LuisaRenderPy available - Raytracer will be enabled")
except ImportError:
    ENABLE_RAYTRACER = False
    print("⊘ LuisaRenderPy not available - Raytracer will be disabled")

try:
    import gs_madrona

    ENABLE_MADRONA = True
    print("✓ gs_madrona available - BatchRenderer will be enabled")
except ImportError:
    ENABLE_MADRONA = False
    print("⊘ gs_madrona not available - BatchRenderer will be disabled")

# Check if Apollo plugin is available
if APOLLO_PLUGIN_AVAILABLE:
    # Try to import Apollo renderer
    try:
        from gs_apollo import ApolloRenderer as ApolloRendererImpl

        ENABLE_APOLLO = True
        print("✓ gs_apollo available - Apollo renderer will be enabled")
    except ImportError:
        ENABLE_APOLLO = False
        print("⊘ gs_apollo not available - Apollo renderer will be disabled")
else:
    ENABLE_APOLLO = False
    print("⊘ Apollo plugin not available - Apollo renderer will be disabled")

########################## create a scene ##########################
# Choose renderer based on raytracer availability
if ENABLE_RAYTRACER:
    renderer = gs.renderers.RayTracer(
        env_surface=gs.surfaces.Emission(
            emissive_texture=gs.textures.ColorTexture(color=(0.2, 0.3, 0.5)),
        ),
        env_radius=20.0,
    )
else:
    # Use Rasterizer as fallback renderer
    renderer = gs.renderers.Rasterizer()

scene = gs.Scene(
    rigid_options=gs.options.RigidOptions(
        enable_collision=True,
        gravity=(0, 0, -9.8),
    ),
    renderer=renderer,
    show_viewer=False,
)

########################## entities ##########################
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
    surface=gs.surfaces.Rough(color=(0.4, 0.4, 0.4)),
)

sphere = scene.add_entity(
    morph=gs.morphs.Sphere(pos=(0.0, 0.0, 2.0), radius=0.5),
    surface=gs.surfaces.Smooth(color=(1.0, 0.5, 0.5)),
)

box = scene.add_entity(
    morph=gs.morphs.Box(pos=(1.0, 1.0, 1.0), size=(0.3, 0.3, 0.3)),
    surface=gs.surfaces.Rough(color=(0.5, 1.0, 0.5)),
)

########################## Camera Configurations ##########################
# Define common camera parameters
CAMERA_COMMON_KWARGS = dict(
    {
        "up": (0.0, 0.0, 1.0),
        "near": 0.1,
        "far": 100.0,
    }
)

CAMERA_SENSORS_KWARGS = [
    {
        "name": "cam0",
        "pos": (3.0, 0.0, 2.0),
        "lookat": (0.0, 0.0, 1.0),
        "fov": 60.0,
        "attachment": None,  # No attachment
        "lights": [{"pos": (2.0, 2.0, 5.0), "color": (1.0, 1.0, 1.0), "intensity": 0.5}],
    },
    {
        "name": "cam1",
        "pos": (0.0, 1.0, 12.0),
        "lookat": (0.0, 0.0, 1.0),
        "fov": 60.0,
        "attachment": None,
        "lights": [],
    },
    {
        "name": "cam_attached",
        "pos": (0.0, 0.0, 1.0),
        "lookat": (0.0, 0.0, 0.0),
        "fov": 70.0,
        "attachment": {
            "entity_idx": None,
            "link_idx_local": 0,
            "pos_offset": (0.0, 0.0, 0.0),
            "euler_offset": (0.0, 0.0, 0.0),
        },
        "lights": [],
    },
]


# Create camera configurations for all backends
backends = [
    ("raster", RasterizerCameraOptions, True),  # Always enabled
    ("raytrace", RaytracerCameraOptions, ENABLE_RAYTRACER),
    ("batch", BatchRendererCameraOptions, ENABLE_MADRONA),
    ("apollo", ApolloCameraOptions, ENABLE_APOLLO),
]

backend_configs = {}
for backend_name, options_class, enabled in backends:
    if not enabled:
        continue

    configs = []
    for camera_config in CAMERA_SENSORS_KWARGS:
        name = f"{backend_name}_{camera_config['name']}"
        res = (500, 600)

        # Create options with common and backend-specific parameters
        options_kwargs = {
            "res": res,
            "pos": camera_config["pos"],
            "lookat": camera_config["lookat"],
            "up": CAMERA_COMMON_KWARGS["up"],
            "fov": camera_config["fov"],
            "lights": camera_config["lights"],
        }

        # Add backend-specific parameters first (before attachment)
        if backend_name == "apollo":
            options_kwargs.update(
                {
                    "app_mode": "batch_render",
                    "render_mode": "forward",
                    "update_ground_truth_only": False,
                    "scene_description_export_path": "apollo_scene.json",
                    "capture_animation": True,
                }
            )

        # Handle attachment
        attachment = camera_config["attachment"]
        if attachment is not None:
            # For attached cameras, set the entity_idx to the sphere's index
            options_kwargs.update(
                {
                    "entity_idx": sphere.idx,
                    "link_idx_local": attachment["link_idx_local"],
                    "pos_offset": attachment["pos_offset"],
                    "euler_offset": attachment["euler_offset"],
                }
            )

        # Add backend-specific parameters
        if backend_name == "raster":
            options_kwargs.update({"near": CAMERA_COMMON_KWARGS["near"], "far": CAMERA_COMMON_KWARGS["far"]})
        elif backend_name == "raytrace":
            options_kwargs.update(
                {
                    "model": "pinhole",
                    "spp": 64,
                    "denoise": False,
                }
            )
            if attachment is None:  # Only add env surface for non-attached cameras
                options_kwargs.update(
                    {
                        "env_surface": gs.surfaces.Emission(
                            emissive_texture=gs.textures.ColorTexture(color=(0.2, 0.3, 0.5)),
                        ),
                        "env_radius": 20.0,
                    }
                )
        elif backend_name == "batch":
            options_kwargs.update({"use_rasterizer": True})
            if camera_config["lights"]:
                adjusted_lights = [{**light, "directional": False} for light in camera_config["lights"]]
                options_kwargs["lights"] = adjusted_lights

        # Adjust lights for raytracer and apollo (different intensity/color)
        if backend_name == "raytrace" and camera_config["lights"]:
            adjusted_lights = [
                {**light, "color": (10.0, 10.0, 10.0), "intensity": 1.0} for light in camera_config["lights"]
            ]
            options_kwargs["lights"] = adjusted_lights
        elif backend_name == "apollo" and camera_config["lights"]:
            # Apollo may need similar light adjustments as raytracer for consistency
            adjusted_lights = [
                {**light, "color": (10.0, 10.0, 10.0), "intensity": 0.2} for light in camera_config["lights"]
            ]
            options_kwargs["lights"] = adjusted_lights

        options = options_class(**options_kwargs)
        configs.append(
            {
                "name": name,
                "options": options,
                "attachment": camera_config["attachment"],
            }
        )

    backend_configs[backend_name] = configs

########################## Create Cameras ##########################
cameras = {}

for group_name, configs in backend_configs.items():
    print(f"\n=== {group_name.title()} Cameras ===")
    for config in configs:
        camera = scene.add_sensor(config["options"])
        cameras[config["name"]] = camera

    print(f"✓ Created {len(configs)} {group_name.lower()} cameras")


########################## build ##########################
n_envs = 1
scene.build(n_envs=n_envs)  # Build with 1 environment

########################## identify attached cameras ##########################
print("\n=== Identifying Attached Cameras ===")

# Identify cameras that are configured to be attached
attached_cameras = []
for group_name, configs in backend_configs.items():
    for config in configs:
        if config["attachment"] is not None:
            camera = cameras[config["name"]]
            attached_cameras.append(camera)
            print(f"✓ {config['name']} is attached to sphere")

print(f"✓ Identified {len(attached_cameras)} attached cameras")

########################## simulate and render ##########################
print("\n=== Simulation Loop ===")
os.makedirs("camera_sensor_output", exist_ok=True)


# Helper to convert torch tensors to numpy arrays for saving
def to_numpy_for_save(tensor_or_array):
    if hasattr(tensor_or_array, "cpu"):
        return tensor_or_array.cpu().numpy()
    return tensor_or_array


for i in range(100):
    scene.step()

    # Render every 10 steps
    if i % 10 == 0:
        print(f"\n--- Step {i} ---")

        # Read and print camera data
        camera_data = {}
        for cam_name, camera in cameras.items():
            # Read camera data (handles both single and multi-environment cases)
            data = camera.read()
            camera_data[cam_name] = data

            # Determine print suffix based on camera type
            suffix = ""
            if "raytrace" in cam_name:
                suffix = " (auto-rendered on read)"
            elif "apollo" in cam_name:
                suffix = " (Apollo render)"
            elif cam_name.endswith("_cam1") and data.rgb.ndim > 3:
                suffix = " (env 0)"

            print(f"  {cam_name.replace('_', ' ').title()} RGB shape: {data.rgb.shape}{suffix}")

        # Save images (always from environment 0 for visualization)
        for cam_name, data in camera_data.items():
            rgb_data = data.rgb[0] if data.rgb.ndim > 3 else data.rgb
            suffix = "_env0" if n_envs > 1 else ""
            filename = f"camera_sensor_output/{cam_name}{suffix}_step{i:03d}.png"
            plt.imsave(filename, to_numpy_for_save(rgb_data))
