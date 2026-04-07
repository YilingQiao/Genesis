import { describe, it, expect } from 'vitest'
import {
  SCENE_INFO, STATE_UPDATE,
  SIM_CONTROL, CAMERA_UPDATE, ENTITY_UPDATE, VIS_TOGGLE,
  ACTION_PLAY, ACTION_PAUSE, ACTION_STEP, ACTION_RESET,
  CAM_ORBIT, CAM_PAN, CAM_ZOOM, CAM_SET_POSE, CAM_SET_FOV, CAM_RESET,
} from './protocol.js'

describe('protocol constants', () => {
  it('server->client message types match protocol.py', () => {
    expect(SCENE_INFO).toBe('scene_info')
    expect(STATE_UPDATE).toBe('state_update')
  })

  it('client->server message types match protocol.py', () => {
    expect(SIM_CONTROL).toBe('sim_control')
    expect(CAMERA_UPDATE).toBe('camera_update')
    expect(ENTITY_UPDATE).toBe('entity_update')
    expect(VIS_TOGGLE).toBe('vis_toggle')
  })

  it('sim control actions are correct strings', () => {
    expect(ACTION_PLAY).toBe('play')
    expect(ACTION_PAUSE).toBe('pause')
    expect(ACTION_STEP).toBe('step')
    expect(ACTION_RESET).toBe('reset')
  })

  it('camera actions are correct strings', () => {
    expect(CAM_ORBIT).toBe('orbit')
    expect(CAM_PAN).toBe('pan')
    expect(CAM_ZOOM).toBe('zoom')
    expect(CAM_SET_POSE).toBe('set_pose')
    expect(CAM_SET_FOV).toBe('set_fov')
    expect(CAM_RESET).toBe('reset')
  })
})
