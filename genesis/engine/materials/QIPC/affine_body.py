from genesis.typing import PositiveFloat

from .base import Base


class AffineBody(Base):
    """
    A near-rigid body simulated by the affine-body Incremental Potential Contact (IPC) solver.

    The body carries 12 degrees of freedom (a translation and a linear map), with rigidity enforced by a stiffness
    penalty rather than exactly. Contact between QIPC entities is penetration-free.

    Note
    ----
    Contact is frictionless: bodies rest and stack, but nothing holds tangentially.

    Parameters
    ----------
    rho : float, optional
        Material density (kg/m^3). Defaults to 1000.
    kappa : float, optional
        Rigidity stiffness (Pa). A higher value keeps the body closer to exactly rigid under load but makes each step
        harder to solve, costing solver iterations; a lower value solves faster and lets the body visibly stretch and
        shear under strong contact forces. Defaults to 1e8.
    """

    rho: PositiveFloat = 1000.0
    kappa: PositiveFloat = 1e8
