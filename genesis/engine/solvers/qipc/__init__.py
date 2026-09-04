"""Affine-body Incremental Potential Contact (IPC) solver core, vendored from Genesis-Embodied-AI/qipc.

The package holds the solver systems and device kernels of qipc (commit 7b74152), the quadrants port of the
cuda-graph-qipc engine, with the standalone frontend removed: 'QIPCSolver' assembles the systems directly from Genesis
entities. Every buffer is fp64 ('qd.f64') by design, so the solve is precision-independent from the Genesis backend.

Two upstream patterns survive inside these modules because quadrants' fastcache forces them (see 'sim_system.py'):
host bookkeeping lives outside 'self.__dict__' of '@qd.data_oriented' instances, and every value mutated inside the
step graph is a device 'qd.ndarray' scalar. Plain Python attributes on these classes are compile-time constants.
"""

from .affine_body_dynamics import AffineBodyDynamics
from .affine_body_preconditioner import ABDPreconditioner
from .contact_system import ContactSystem
from .global_linear_system import GlobalLinearSystem
from .global_surface_manager import GlobalSurfaceManager
from .global_vertex_manager import GlobalVertexManager
from .linear_pcg import LinearPCG
from .sim_config import SimConfig
from .sim_engine import SimEngine
