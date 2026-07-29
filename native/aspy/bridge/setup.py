"""Build the optional AsPy torch-npu bridge into an external directory."""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension


def required_directory(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"{variable} must point to an existing directory")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"{variable} is not a directory: {path}")
    return path


op_api = required_directory("SPIKINGJELLY_NPU_ASPY_OP_API")
cann = required_directory("ASCEND_TOOLKIT_HOME") / "aarch64-linux"
if not cann.is_dir():
    raise RuntimeError(f"qualified CANN architecture directory is missing: {cann}")

setup(
    name="spikingjelly-npu-aspy-native",
    version="0.1.0",
    ext_modules=[
        NpuExtension(
            "_spikingjelly_npu_aspy",
            [str(Path(__file__).with_name("aspy_bridge.cpp"))],
            include_dirs=[str(op_api / "include"), str(cann / "include")],
            library_dirs=[str(op_api / "lib"), str(cann / "lib64")],
            libraries=["cust_opapi", "nnopbase"],
            extra_link_args=[f"-Wl,-rpath,{op_api / 'lib'}"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
