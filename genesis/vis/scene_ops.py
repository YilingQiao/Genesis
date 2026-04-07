"""Shared scene data utilities.

Constants and data-extraction helpers used by the ImGui overlay and web protocol.
Rendering operations have moved to SceneController (genesis.vis.controller).
"""

import numpy as np

import genesis as gs

FREE_JOINT_POS_LIMIT = 10.0
QUATERNION_COMPONENT_LIMIT = 1.0


def build_entity_joint_data(entity):
    """Build rich joint metadata for *entity*.

    Handles free joints (7 qpos), spherical joints (4 qpos quaternion), and regular joints correctly
    using ``n_qs`` (not ``n_dofs``) for indexing.

    Returns:
        dict with keys: ``q_names``, ``q_limits_lower``, ``q_limits_upper``, ``q_is_quaternion``, ``quat_groups``,
        ``has_free_joint``, ``free_joint_q_start``.
    """
    q_names = []
    q_limits_lower = []
    q_limits_upper = []
    q_is_quaternion = []
    quat_groups = []
    has_free_joint = False
    free_joint_q_start = -1

    for joint in entity.joints:
        if joint.n_qs == 0 or joint.type == gs.JOINT_TYPE.FIXED:
            continue

        if joint.type == gs.JOINT_TYPE.FREE:
            has_free_joint = True
            free_joint_q_start = len(q_names)
            q_names.extend(
                [
                    f"{joint.name}_x",
                    f"{joint.name}_y",
                    f"{joint.name}_z",
                    f"{joint.name}_qw",
                    f"{joint.name}_qx",
                    f"{joint.name}_qy",
                    f"{joint.name}_qz",
                ]
            )
            q_limits_lower.extend([-FREE_JOINT_POS_LIMIT] * 3 + [-QUATERNION_COMPONENT_LIMIT] * 4)
            q_limits_upper.extend([FREE_JOINT_POS_LIMIT] * 3 + [QUATERNION_COMPONENT_LIMIT] * 4)
            q_is_quaternion.extend([False, False, False, True, True, True, True])
            quat_groups.append([len(q_names) - 4, len(q_names)])
        elif joint.type == gs.JOINT_TYPE.SPHERICAL:
            quat_start = len(q_names)
            q_names.extend(
                [
                    f"{joint.name}_qw",
                    f"{joint.name}_qx",
                    f"{joint.name}_qy",
                    f"{joint.name}_qz",
                ]
            )
            q_limits_lower.extend([-QUATERNION_COMPONENT_LIMIT] * 4)
            q_limits_upper.extend([QUATERNION_COMPONENT_LIMIT] * 4)
            q_is_quaternion.extend([True, True, True, True])
            quat_groups.append([quat_start, quat_start + 4])
        else:
            for i in range(joint.n_qs):
                name = joint.name if joint.n_qs == 1 else f"{joint.name}[{i}]"
                q_names.append(name)
                lo = float(joint.dofs_limit[i, 0])
                hi = float(joint.dofs_limit[i, 1])
                if not np.isfinite(lo):
                    lo = -1e6
                if not np.isfinite(hi):
                    hi = 1e6
                q_limits_lower.append(lo)
                q_limits_upper.append(hi)
                q_is_quaternion.append(False)

    return {
        "q_names": q_names,
        "q_limits_lower": q_limits_lower,
        "q_limits_upper": q_limits_upper,
        "q_is_quaternion": q_is_quaternion,
        "quat_groups": quat_groups,
        "has_free_joint": has_free_joint,
        "free_joint_q_start": free_joint_q_start,
    }
