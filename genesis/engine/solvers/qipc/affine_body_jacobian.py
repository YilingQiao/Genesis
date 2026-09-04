"""ABD affine Jacobian chain rule -- J*q, J^T*g, J^T*H*J.

Port of cgq ``generated/gipc_abd_jacobian.h``. cgq ships these as unrolled
codegen; the same expressions have a closed form that is written out here as
loops, which is both shorter and easier to check against cgq by inspection.

A 12-DOF affine body maps a rest position ``x_bar`` to world space as
``x = t + A @ x_bar``, with the DOF vector packed row-major::

    q = [ t(3) | A_row0(3) | A_row1(3) | A_row2(3) ]

so ``x[a] = q[a] + sum_c A[a, c] * x_bar[c]``. Writing the *extended* rest
basis ``e(x_bar) = [1, x_bar_0, x_bar_1, x_bar_2]`` and

    idx(a, k) = a               if k == 0
                3 + 3*a + (k-1) otherwise

the 3x12 Jacobian is exactly ``J[a, idx(a, k)] = e[k]`` (all other entries
zero), which collapses the three chain-rule products to

    (J @ q)[a]          = sum_k q[idx(a, k)] * e[k]
    (J^T @ g)[idx(a,k)] = g[a] * e[k]
    (Jx^T H Jy)[idx(a,k), idx(b,l)] = H[a, b] * ex[k] * ey[l]

Note the two-sided form takes *two* rest positions: contact Hessian blocks
couple a vertex on the row body with a (generally different) vertex on the
column body, so the left and right bases differ.
"""

from __future__ import annotations

import quadrants as qd


@qd.func
def abd_extended_basis(x_bar: qd.template(), out: qd.template()):
    """Fill ``out`` (4-vec) with the extended rest basis ``[1, x, y, z]``."""
    out[0] = qd.f64(1.0)
    for c in qd.static(range(3)):
        out[c + 1] = x_bar[c]


@qd.func
def abd_dof_index(a: qd.i32, k: qd.i32) -> qd.i32:
    """Row-major DOF slot for spatial component ``a`` and basis index ``k``."""
    ret = a
    if k > 0:
        ret = 3 + 3 * a + (k - 1)
    return ret


@qd.func
def abd_J_mul_q(x_bar: qd.template(), q: qd.template(), out: qd.template()):
    """World position of a rest point: ``out(3) = J(x_bar) @ q(12)``."""
    e = qd.Vector.zero(qd.f64, 4)
    abd_extended_basis(x_bar, e)
    for a in qd.static(range(3)):
        acc = qd.f64(0.0)
        for k in qd.static(range(4)):
            acc = acc + q[abd_dof_index(a, k)] * e[k]
        out[a] = acc


@qd.func
def abd_JT_mul_g(x_bar: qd.template(), g: qd.template(), out: qd.template()):
    """Lift a vertex-space gradient to body DOFs: ``out(12) = J^T @ g(3)``."""
    e = qd.Vector.zero(qd.f64, 4)
    abd_extended_basis(x_bar, e)
    for a in qd.static(range(3)):
        for k in qd.static(range(4)):
            out[abd_dof_index(a, k)] = g[a] * e[k]


@qd.func
def abd_JT_H_J(
    x_bar: qd.template(),
    y_bar: qd.template(),
    H: qd.template(),
    out: qd.template(),
):
    """Lift a 3x3 vertex-space Hessian block to a 12x12 body-DOF block.

    ``out = J(x_bar)^T @ H @ J(y_bar)``. Accumulates nothing -- ``out`` is
    fully overwritten.
    """
    ex = qd.Vector.zero(qd.f64, 4)
    ey = qd.Vector.zero(qd.f64, 4)
    abd_extended_basis(x_bar, ex)
    abd_extended_basis(y_bar, ey)
    for a in range(3):
        for b in range(3):
            hab = H[a, b]
            for k in range(4):
                for lo in range(4):
                    out[abd_dof_index(a, k), abd_dof_index(b, lo)] = hab * ex[k] * ey[lo]
