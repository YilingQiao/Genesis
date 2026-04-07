"""Genesis Web GUI -- browser-based simulation viewer and control panel."""


def __getattr__(name):
    if name == "GenesisWebServer":
        try:
            from .server import GenesisWebServer

            return GenesisWebServer
        except ImportError as e:
            raise ImportError(
                "Genesis Web GUI requires additional dependencies. Install them with: pip install genesis[web]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
