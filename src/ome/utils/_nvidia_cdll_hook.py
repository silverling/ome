"""
This file hijack the CDLL loading mechanism in ctypes to redirect loading of
`libnvrtc.so.12` to the one bundled in the `nvidia-cuda-nvrtc-cu12` package.
"""

import ctypes
import ctypes.util
import functools
from pathlib import Path

import nvidia

_real_CDLL_new = ctypes.CDLL.__new__
_real_find_library = ctypes.util.find_library

libnvrtc_path = Path(nvidia.__file__).parent / "cuda_nvrtc" / "lib" / "libnvrtc.so.12"
LIB_MAP = {"libnvrtc.so.12": str(libnvrtc_path)}


def _remap(name):
    return LIB_MAP.get(name, name)


@functools.wraps(_real_CDLL_new)
def _CDLL_new(cls, name, *args, **kwargs):
    obj = _real_CDLL_new(cls)
    obj.__init__(_remap(name), *args, **kwargs)
    return obj


@functools.wraps(_real_find_library)
def _find_library(name):
    return _real_find_library(_remap(name))


ctypes.CDLL.__new__ = staticmethod(_CDLL_new)  # ty: ignore[invalid-assignment]
ctypes.util.find_library = _find_library
