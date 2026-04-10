"""
NumPy 2.0 compatibility shims.

Import this module BEFORE any other library in every page and utility file:

    import utils.compat  # noqa: F401

Why this is needed
------------------
NumPy 2.0 (released 2024-06) made two breaking changes that affect older
compiled libraries (scipy < 1.13, shap < 0.45, some sklearn plugins, etc.):

1. Type aliases removed: np.float_, np.bool, np.int_, np.unicode_, etc.
   These are patched back onto the np namespace below.

2. Internal C-extension layout changed: numpy.core was reorganised into
   numpy._core.  NumPy 2.0 ships a thin Python stub for numpy.core, but it
   does NOT register numpy.core.multiarray (and siblings) in sys.modules as
   importable sub-packages.  Libraries that do
       import numpy.core.multiarray
   or the equivalent C-level PyImport_ImportModule call therefore raise
       ImportError: numpy.core.multiarray failed to import
   Registering the submodules explicitly in sys.modules fixes the
   Python-level case.

Note: binary (C-extension ABI) incompatibilities cannot be patched at the
Python level.  If a library was compiled against NumPy 1.x and its .so
references symbols that no longer exist in the NumPy 2.x shared library,
you must downgrade NumPy:
    pip install "numpy>=1.24,<2.0" --force-reinstall
or upgrade the offending library to a NumPy-2.0-aware build.
"""

import sys
import numpy as np

# ── 1. Restore removed type-alias attributes ──────────────────────────────────
# Keys are the removed alias names; values are the canonical numpy attribute
# names (strings) that replaced them.  Using strings avoids evaluating
# np.<attr> at dict-creation time, which would raise AttributeError if numpy
# is only partially loaded or if the target itself doesn't exist.
_ALIASES = {
    "unicode_": "str_",
    "str0":     "bytes_",
    "string_":  "bytes_",
    "bool":     "bool_",
    "bool8":    "bool_",
    "int":      "int_",
    "int0":     "intp",
    "uint":     "uint64",
    "float":    "float64",
    "float_":   "float64",
    "float96":  "longdouble",
    "complex":  "complex128",
    "complex_": "complex128",
    "object":   "object_",
}
for _alias, _target in _ALIASES.items():
    if not hasattr(np, _alias) and hasattr(np, _target):
        try:
            setattr(np, _alias, getattr(np, _target))
        except (AttributeError, TypeError):
            pass

# ── 2. Ensure numpy.core.* are registered in sys.modules ─────────────────────
# NumPy 2.0 keeps numpy.core as a compat stub module but does NOT register its
# children (numpy.core.multiarray, etc.) as importable sub-packages.
# We do that here so that `import numpy.core.multiarray` succeeds.
try:
    import numpy._core as _npc  # always present in NumPy 1.20+ and 2.x

    # Register numpy.core itself if absent (it usually IS present in NumPy 2.x)
    sys.modules.setdefault("numpy.core", _npc)

    # Register each submodule only if not already present
    _SUBMODULES = [
        "multiarray",
        "numeric",
        "umath",
        "fromnumeric",
        "arrayprint",
        "defchararray",
        "records",
        "function_base",
        "getlimits",
        "shape_base",
        "einsumfunc",
        "numerictypes",
        "overrides",
        "_multiarray_umath",
        "_ufunc_config",
        "multiarray_tests",
    ]
    for _sub in _SUBMODULES:
        _key = f"numpy.core.{_sub}"
        if _key not in sys.modules and hasattr(_npc, _sub):
            sys.modules[_key] = getattr(_npc, _sub)

except Exception:
    pass  # If numpy._core doesn't exist we're on an old numpy — no patch needed
