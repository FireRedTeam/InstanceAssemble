from contextlib import contextmanager
from typing import Iterable
from peft.tuners.tuners_utils import BaseTunerLayer

@contextmanager
def enable_lora(modules: Iterable[BaseTunerLayer], on: float = 1.0, off: float = 0.0):
    """
    Temporarily enable LoRA adapters by setting their scales to `on`,
    and restore to `off` on exit.

    Args:
        modules: Iterable of PEFT BaseTunerLayer modules.
        on:  scale used inside the context (default 1.0).
        off: scale restored after exit (default 0.0).
    """
    mods = [m for m in modules if isinstance(m, BaseTunerLayer)]
    try:
        for m in mods:
            for name in getattr(m, "active_adapters", ()):
                m.set_scale(name, on)
        yield
    finally:
        for m in mods:
            for name in getattr(m, "active_adapters", ()):
                m.set_scale(name, off)
