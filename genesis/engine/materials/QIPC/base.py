from typing import TYPE_CHECKING

from ..base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.qipc_entity import QIPCEntity


class Base(Material["QIPCEntity"]):
    """
    The base class of QIPC materials, simulated by the affine-body Incremental Potential Contact (IPC) solver.

    Note
    ----
    This class should *not* be instantiated directly.
    """
