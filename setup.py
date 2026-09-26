"""Builds the optional C++ extension nomo._core.

The extension is an accelerator only: if no C++ toolchain is available (common on Windows),
installation continues and the bit-identical pure-Python kernel is used instead.
"""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


class OptionalBuildExt(build_ext):
    def run(self):
        try:
            super().run()
        except Exception as exc:  # noqa: BLE001
            print(f"warning: nomo._core not built ({exc}); using the pure-Python kernel")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: nomo._core not built ({exc}); using the pure-Python kernel")


setup(
    ext_modules=[Pybind11Extension("nomo._core", ["csrc/lif_kernel.cpp"], cxx_std=17,
                                   extra_compile_args=["-O3"], optional=True)],
    cmdclass={"build_ext": OptionalBuildExt},
)
