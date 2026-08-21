"""Top-view RGB rendering for the insertion env (vision modality).

Uses MuJoCo's offscreen EGL/GLFW backend with a single cached Renderer
(no context manager: ``with renderer(...)`` is not reliably implemented
across mujoco builds / EGL platforms and can abort the process).  If
rendering fails at any point, the module permanently falls back to a
deterministic placeholder so kernels stay alive without a display while
preserving the observation dimensionality.
"""

from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

_renderer_cache: dict[tuple[int, int], object] = {}
_dead: set[tuple[int, int]] = set()


def _placeholder(size: int) -> np.ndarray:
    # deterministic placeholder: green field + dark slot stripe
    img = np.full((size, size, 3), 0.55, dtype=np.float32)
    img[..., 1] = 0.75
    img[:, int(size * 0.45):int(size * 0.55), :] = 0.25
    return img


def render_top(env, size: int = 64) -> np.ndarray:
    """Return an (size, size, 3) float image of the workspace from above."""
    mj = getattr(env, "mujoco", None)
    model = getattr(env, "model", None)
    data = getattr(env, "data", None)
    if mj is None or model is None or data is None:
        return _placeholder(size)

    try:
        import mujoco
        cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "top")
        if cam_id < 0:
            return _placeholder(size)
        renderer_cls = getattr(mujoco, "Renderer", None)
        if renderer_cls is None:
            return _placeholder(size)
        key = (id(model), size)
        if key in _dead:
            return _placeholder(size)
        r = _renderer_cache.get(key)
        if r is None:
            try:
                r = renderer_cls(model, height=size, width=size)
            except Exception:
                _dead.add(key)
                return _placeholder(size)
            _renderer_cache[key] = r
        r.update_scene(data, camera=cam_id)
        img = r.render()
        return img.astype(np.float32) / 255.0
    except Exception:
        key = (id(model), size)
        r = _renderer_cache.pop(key, None)
        if r is not None:
            try:
                r.close()
            except Exception:
                pass
        _dead.add(key)
        return _placeholder(size)
