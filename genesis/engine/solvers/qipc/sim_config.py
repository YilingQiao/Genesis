"""SimConfig -- solver parameters as a @qd.data_oriented SimSystem."""

from __future__ import annotations

import numpy as np
import quadrants as qd

from genesis.engine.solvers.qipc.sim_system import SimSystem


@qd.data_oriented
class SimConfig(SimSystem):
    """Solver parameter container.

    ``dt`` is a host-side float (becomes a compile-time constant via template
    mapping).  ``gravity_np`` is retained on host for the ABD mass-computation
    pipeline at init time.
    """

    def __init__(
        self,
        dt: float = 0.01,
        gravity: tuple[float, float, float] = (0.0, -9.8, 0.0),
        *,
        max_newton: int = 100,
        newton_min: int = 2,
        vel_tol: float = 1e-2,
        max_ls_iter: int = 100,
        pcg_tol: float = 1e-4,
        pcg_max_iter: int = 200,
        tol: float = 1e-3,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.gravity_np = np.array(gravity, dtype=np.float64)
        self.max_newton = max_newton
        self.newton_min = newton_min
        self.vel_tol = vel_tol
        self.max_ls_iter = max_ls_iter
        self.pcg_tol = pcg_tol
        self.pcg_max_iter = pcg_max_iter
        self.tol = tol

    def do_build(self) -> None:
        pass
