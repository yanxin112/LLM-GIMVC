from .freecsl import FreeCSLAdapter
from .jga_imvc import JGAIMVCAdapter
from .mica import MICAAdapter


BASELINE_REGISTRY = {
    "mica": MICAAdapter,
    "jga_imvc": JGAIMVCAdapter,
    "freecsl": FreeCSLAdapter,
}


def get_baseline_adapter(method):
    method = method.lower()
    if method not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown external baseline: {method}")
    return BASELINE_REGISTRY[method]()


def is_external_baseline(method):
    return method.lower() in BASELINE_REGISTRY
