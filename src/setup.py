import sys

from cx_Freeze import setup, Executable

sys.setrecursionlimit(2000)

build_options = {
    "packages": ["PyQt5", 'torch', "easyocr"],
    "include_files": ["gui/icons", "gui/app_window.ui", "gui/instance_window.ui"],
    "excludes": []
}

exe = Executable(
    "run_gui_app.py",
    base="Win32GUI" if sys.platform == "win32" else None,
    target_name="TarMass"
)

setup(
    name='TarMass',
    version='1.0.0',
    description='Tardigrade biomass estimation tool.',
    options={"build_exe": build_options},
    executables=[exe]
)
