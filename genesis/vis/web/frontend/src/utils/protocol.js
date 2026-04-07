/**
 * WebSocket message protocol constants.
 * Must match genesis/vis/web/protocol.py MsgType values.
 *
 * TypeScript interfaces are auto-generated from Pydantic models:
 *   npm run generate:types
 * See: ../types/protocol.generated.d.ts
 */

// Server -> Client
export const SCENE_INFO = 'scene_info'
export const STATE_UPDATE = 'state_update'

// Client -> Server
export const SIM_CONTROL = 'sim_control'
export const CAMERA_UPDATE = 'camera_update'
export const ENTITY_UPDATE = 'entity_update'
export const VIS_TOGGLE = 'vis_toggle'
export const SET_RESOLUTION = 'set_resolution'
export const SET_TARGET_FPS = 'set_target_fps'

// Sim control actions
export const ACTION_PLAY = 'play'
export const ACTION_PAUSE = 'pause'
export const ACTION_STEP = 'step'
export const ACTION_RESET = 'reset'

// Camera update actions
export const CAM_ORBIT = 'orbit'
export const CAM_PAN = 'pan'
export const CAM_ZOOM = 'zoom'
export const CAM_SET_POSE = 'set_pose'
export const CAM_SET_FOV = 'set_fov'
export const CAM_RESET = 'reset'
