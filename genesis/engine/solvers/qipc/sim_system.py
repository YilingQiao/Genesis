from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from genesis.engine.solvers.qipc.sim_engine import SimEngine

T = TypeVar("T", bound="SimSystem")


class _SystemState:
    """Host-only bookkeeping for one SimSystem, held outside its ``__dict__``."""

    __slots__ = ("engine", "valid")

    def __init__(self) -> None:
        self.engine: SimEngine | None = None
        self.valid: bool = True


# Keyed by ``id(system)``; entries are dropped by ``weakref.finalize`` when the
# system dies, so an id can never be observed stale after reuse.
#
# This must NOT live on the instance ``__dict__``. quadrants walks the ``__dict__``
# of every ``@qd.data_oriented`` kernel argument twice: once to build the fastcache
# key, where a ``None``-valued member is rejected outright and disqualifies the
# entire kernel call ([FASTCACHE][PARAM_INVALID]), and once to bake primitive
# members in as compile-time constants, which would silently freeze ``valid``.
# See road-map Standing rules 4 and 6. ``LBVH`` uses the same pattern.
_SYSTEM_STATE: dict[int, _SystemState] = {}


class SimSystem(ABC):
    """Base class for all solver subsystems. Mirrors cgq SimSystem."""

    def __init__(self) -> None:
        key = id(self)
        _SYSTEM_STATE[key] = _SystemState()
        weakref.finalize(self, _SYSTEM_STATE.pop, key, None)

    @property
    def _state(self) -> _SystemState:
        state = _SYSTEM_STATE.get(id(self))
        if state is None:
            raise RuntimeError(
                f"{type(self).__name__} has no SimSystem state; its __init__ must call super().__init__()"
            )
        return state

    @property
    def engine(self) -> SimEngine:
        engine = self._state.engine
        assert engine is not None, "SimSystem not bound to an engine"
        return engine

    def is_valid(self) -> bool:
        return self._state.valid

    def find(self, system_type: type[T]) -> T | None:
        """Find a sibling system by type. Returns None if not registered."""
        return self.engine.find(system_type)

    def require(self, system_type: type[T]) -> T:
        """Find a sibling system by type. Asserts it exists."""
        result = self.find(system_type)
        assert result is not None, f"Required system {system_type.__name__} not found"
        return result

    @abstractmethod
    def do_build(self) -> None:
        """Called by SimEngine.build_systems() after all systems are registered.
        Subsystems can use find()/require() here to wire cross-system references."""
        ...

    def _set_engine(self, engine: SimEngine) -> None:
        self._state.engine = engine

    def _invalidate(self) -> None:
        self._state.valid = False
