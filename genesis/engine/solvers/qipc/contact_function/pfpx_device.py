"""PFPx Jacobians (@qd.func device version).

Ports cgq ``gipc_pfpx.h``. Each function writes into an output template array.
PT/EE: (12, 9), PP: (6,), PE: (9, 4). Only the nonzero column is populated.
"""

from __future__ import annotations

import quadrants as qd


@qd.func
def pfpx_pt_device(
    x0x: qd.f64,
    x0y: qd.f64,
    x0z: qd.f64,
    x1x: qd.f64,
    x1y: qd.f64,
    x1z: qd.f64,
    x2x: qd.f64,
    x2y: qd.f64,
    x2z: qd.f64,
    x3x: qd.f64,
    x3y: qd.f64,
    x3z: qd.f64,
    d_hat_sqrt: qd.f64,
    out: qd.template(),
):
    """PT PFPx: writes 12 values into out[0..11] (column 8 of 12x9 matrix)."""
    t2 = 1.0 / d_hat_sqrt
    t12 = -x1x + x0x
    t13 = -x1y + x0y
    t14 = -x1z + x0z
    t15 = -x2x + x1x
    t16 = -x2y + x1y
    t17 = -x2z + x1z
    t18 = -x3x + x1x
    t19 = -x3y + x1y
    t20 = -x3z + x1z
    t21 = -x3x + x2x
    t22 = -x3y + x2y
    t23 = -x3z + x2z
    t33 = t15 * t19 - t16 * t18
    t34 = t15 * t20 - t17 * t18
    t35 = t16 * t20 - t17 * t19
    t44 = 1.0 / (t33 * t33 + t34 * t34 + t35 * t35)
    t46 = t14 * t33 + t12 * t35 - t13 * t34
    t47 = t46 * t46
    t49 = 1.0 / qd.sqrt(t44 * t47)

    out[0] = t2 * t35 * t44 * t46 * t49
    out[1] = -t2 * t34 * t44 * t46 * t49
    out[2] = t2 * t33 * t44 * t46 * t49

    b_d = t44 * t44 * t47
    t44_m = t44 * t46
    t47n = t2 * t49

    out[3] = t47n * (b_d * (t22 * t33 * 2.0 + t23 * t34 * 2.0) + t44_m * ((t35 + t13 * t23) - t14 * t22) * 2.0) * -0.5
    out[4] = t47n * (b_d * (t21 * t33 * 2.0 - t23 * t35 * 2.0) + t44_m * ((t34 + t12 * t23) - t14 * t21) * 2.0) / 2.0
    out[5] = t47n * (b_d * (t21 * t34 * 2.0 + t22 * t35 * 2.0) - t44_m * ((t33 + t12 * t22) - t13 * t21) * 2.0) / 2.0
    out[6] = t47n * (t44_m * (t13 * t20 - t14 * t19) * 2.0 + b_d * (t19 * t33 * 2.0 + t20 * t34 * 2.0)) / 2.0
    out[7] = t47n * (t44_m * (t12 * t20 - t14 * t18) * 2.0 + b_d * (t18 * t33 * 2.0 - t20 * t35 * 2.0)) * -0.5
    out[8] = t47n * (t44_m * (t12 * t19 - t13 * t18) * 2.0 - b_d * (t18 * t34 * 2.0 + t19 * t35 * 2.0)) / 2.0
    out[9] = t47n * (t44_m * (t13 * t17 - t14 * t16) * 2.0 + b_d * (t16 * t33 * 2.0 + t17 * t34 * 2.0)) * -0.5
    out[10] = t47n * (t44_m * (t12 * t17 - t14 * t15) * 2.0 + b_d * (t15 * t33 * 2.0 - t17 * t35 * 2.0)) / 2.0
    out[11] = t47n * (t44_m * (t12 * t16 - t13 * t15) * 2.0 - b_d * (t15 * t34 * 2.0 + t16 * t35 * 2.0)) * -0.5


@qd.func
def pfpx_ee_device(
    x0x: qd.f64,
    x0y: qd.f64,
    x0z: qd.f64,
    x1x: qd.f64,
    x1y: qd.f64,
    x1z: qd.f64,
    x2x: qd.f64,
    x2y: qd.f64,
    x2z: qd.f64,
    x3x: qd.f64,
    x3y: qd.f64,
    x3z: qd.f64,
    d_hat_sqrt: qd.f64,
    out: qd.template(),
):
    """EE PFPx: writes 12 values into out[0..11] (column 8 of 12x9 matrix)."""
    t12 = -x1x + x0x
    t13 = -x1y + x0y
    t14 = -x1z + x0z
    t15 = -x2x + x0x
    t16 = -x2y + x0y
    t17 = -x2z + x0z
    t18 = -x3x + x2x
    t19 = -x3y + x2y
    t20 = -x3z + x2z
    t33 = t15 * t19
    t34 = t16 * t18
    t35 = t15 * t20
    t36 = t17 * t18
    t37 = t16 * t20
    t38 = t17 * t19
    t45 = t12 * t19 - t13 * t18
    t46 = t12 * t20 - t14 * t18
    t47 = t13 * t20 - t14 * t19
    t76 = 1.0 / (t45 * t45 + t46 * t46 + t47 * t47)
    t78 = t17 * t45 + t15 * t47 - t16 * t46
    t77 = t76 * t76
    t79 = t78 * t78
    t82 = (t13 * t45 * 2.0 + t14 * t46 * 2.0) * t77 * t79
    t83 = (t12 * t46 * 2.0 + t13 * t47 * 2.0) * t77 * t79
    t84 = (t19 * t45 * 2.0 + t20 * t46 * 2.0) * t77 * t79
    t85 = (t18 * t46 * 2.0 + t19 * t47 * 2.0) * t77 * t79
    t86 = (t12 * t45 * 2.0 - t14 * t47 * 2.0) * t77 * t79
    t87 = (t18 * t45 * 2.0 - t20 * t47 * 2.0) * t77 * t79
    t78_m = t78 * t76
    t77_inv = 1.0 / d_hat_sqrt * (1.0 / qd.sqrt(t76 * t79))

    out[0] = t77_inv * (t84 - t78_m * ((-t37 + t38) + t47) * 2.0) * -0.5
    out[1] = t77_inv * (t87 - t78_m * ((-t35 + t36) + t46) * 2.0) / 2.0
    out[2] = t77_inv * (t85 + t78_m * ((-t33 + t34) + t45) * 2.0) / 2.0
    out[3] = t77_inv * (t84 + t78_m * (t37 - t38) * 2.0) / 2.0
    out[4] = t77_inv * (t87 + t78_m * (t35 - t36) * 2.0) * -0.5
    out[5] = t77_inv * (t85 - t78_m * (t33 - t34) * 2.0) * -0.5

    t18_v = t13 * t17 - t14 * t16
    out[6] = t77_inv * (t82 - t78_m * (t18_v + t47) * 2.0) / 2.0
    t20_v = t12 * t17 - t14 * t15
    out[7] = t77_inv * (t86 - t78_m * (t20_v + t46) * 2.0) * -0.5
    t19_v = t12 * t16 - t13 * t15
    out[8] = t77_inv * (t83 + t78_m * (t19_v + t45) * 2.0) * -0.5
    out[9] = t77_inv * (t82 - t78_m * t18_v * 2.0) * -0.5
    out[10] = t77_inv * (t86 - t78_m * t20_v * 2.0) / 2.0
    out[11] = t77_inv * (t83 + t78_m * t19_v * 2.0) / 2.0


@qd.func
def pfpx_pp_device(
    x0x: qd.f64,
    x0y: qd.f64,
    x0z: qd.f64,
    x1x: qd.f64,
    x1y: qd.f64,
    x1z: qd.f64,
    d_hat_sqrt: qd.f64,
    out: qd.template(),
):
    """PP PFPx: writes 6 values into out[0..5]."""
    t8 = 1.0 / d_hat_sqrt
    t25 = -x1x + x0x
    t26 = -x1y + x0y
    t27 = -x1z + x0z
    t25_inv = 1.0 / qd.sqrt(t25 * t25 + t26 * t26 + t27 * t27)
    r0 = t8 * (x0x * 2.0 - x1x * 2.0) * t25_inv / 2.0
    r1 = t8 * (x0y * 2.0 - x1y * 2.0) * t25_inv / 2.0
    r2 = t8 * (x0z * 2.0 - x1z * 2.0) * t25_inv / 2.0
    out[0] = r0
    out[1] = r1
    out[2] = r2
    out[3] = -r0
    out[4] = -r1
    out[5] = -r2


@qd.func
def pfpx_pe_device(
    x0x: qd.f64,
    x0y: qd.f64,
    x0z: qd.f64,
    x1x: qd.f64,
    x1y: qd.f64,
    x1z: qd.f64,
    x2x: qd.f64,
    x2y: qd.f64,
    x2z: qd.f64,
    d_hat_sqrt: qd.f64,
    out: qd.template(),
):
    """PE PFPx: writes 9 values into out[0..8] (column 3 of 9x4 matrix)."""
    t8 = 1.0 / d_hat_sqrt
    t18 = -x1x + x0x
    t19 = -x1y + x0y
    t20 = -x1z + x0z
    t21 = -x2x + x0x
    t22 = -x2y + x0y
    t23 = -x2z + x0z
    t24 = -x2x + x1x
    t25 = -x2y + x1y
    t26 = -x2z + x1z
    t43 = 1.0 / (t24 * t24 + t25 * t25 + t26 * t26)
    t45 = t18 * t22 - t19 * t21
    t46 = t18 * t23 - t20 * t21
    t47 = t19 * t23 - t20 * t22
    t44 = t43 * t43
    t51 = t45 * t45 + t46 * t46 + t47 * t47
    t54 = (x1x * 2.0 - x2x * 2.0) * t44 * t51
    t55 = (x1y * 2.0 - x2y * 2.0) * t44 * t51
    t56 = (x1z * 2.0 - x2z * 2.0) * t44 * t51
    t44_sqrt = 1.0 / qd.sqrt(t43 * t51)

    t51_sc = t8 * t43 * t44_sqrt
    out[0] = t51_sc * (t25 * t45 * 2.0 + t26 * t46 * 2.0) / 2.0
    out[1] = t51_sc * (t24 * t45 * 2.0 - t26 * t47 * 2.0) * -0.5
    out[2] = t51_sc * (t24 * t46 * 2.0 + t25 * t47 * 2.0) * -0.5

    t51_sc2 = t8 * t44_sqrt
    out[3] = t51_sc2 * (t54 + t43 * (t22 * t45 * 2.0 + t23 * t46 * 2.0)) * -0.5
    out[4] = t51_sc2 * (t55 - t43 * (t21 * t45 * 2.0 - t23 * t47 * 2.0)) * -0.5
    out[5] = t51_sc2 * (t56 - t43 * (t21 * t46 * 2.0 + t22 * t47 * 2.0)) * -0.5
    out[6] = t51_sc2 * (t54 + t43 * (t19 * t45 * 2.0 + t20 * t46 * 2.0)) / 2.0
    out[7] = t51_sc2 * (t55 - t43 * (t18 * t45 * 2.0 - t20 * t47 * 2.0)) / 2.0
    out[8] = t51_sc2 * (t56 - t43 * (t18 * t46 * 2.0 + t19 * t47 * 2.0)) / 2.0
