"""Honest capability boundary for a future independently validated NVOF adapter."""


def availability():
    # Finding a CUDA adapter or driver DLL does not validate SDK initialization,
    # vector units or an output-grid configuration. Never advertise that as use.
    return {
        "available": False,
        "implemented": False,
        "reason": "NVIDIA Optical Flow SDK后端尚未实现并验证，使用本地DIS双向光流",
    }
