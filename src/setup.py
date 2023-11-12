import sys

from cx_Freeze import setup, Executable

sys.setrecursionlimit(2000)

build_options = {
    "packages": ["PyQt5", 'torch', "easyocr", "pandas"],
    "include_files": ["gui/icons", "gui/app_window.ui", "gui/instance_window.ui", "../docs/user_manual.pdf"],
    "excludes": [
        "keypoints_detector/datasets",
        "keypoints_detector/datasets/kpt_rcnn/runs",
        "keypoints_detector/datasets/kpt_rcnn/EDA.ipynb",
        "keypoints_detector/datasets/kpt_rcnn/dataset.py",
        "keypoints_detector/datasets/kpt_rcnn/hyperparam_search.py",
        "keypoints_detector/datasets/kpt_rcnn/test.py",
        "keypoints_detector/datasets/kpt_rcnn/train.py",
        "keypoints_detector/datasets/yolo/params.yaml",
        "keypoints_detector/datasets/yolo/tardi-pose.yaml",
        "keypoints_detector/datasets/yolo/train.py",
        "keypoints_detector/datasets/yolo/yolov8m-pose.pt"

    ]
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
