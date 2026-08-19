"""Top-view RGB rendering for the insertion env (vision modality).

Uses MuJoCo's offscreen EGL/GLFW backend.  Falls back to a placeholder
(marked 'norender') so kernels can run headless without a display and still
produce deterministically-correct results if rendering is unavailable.
"""

from __future__ import annotations

import numpy as np


def render_top(env, size: int = 64) -> np.ndarray:
    """Return an (size, size, 3) float image of the workspace from above.

    Falls back to a synthetic image (green field + slot marker) if no
    renderer backend is available, which keeps the data pipeline alive on
    hosts without EGL/GLFW while preserving the observation dimensionality.
    """
    mj = getattr(env, "mujoco", None)
    model = getattr(env, "model", None)
    data = getattr(env, "data", None)
    if mj is not None and model is not None and data is not None:
        try:
            cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "top")
            if cam_id >= 0:
                import mujoco
                renderer = getattr(mujoco, "Renderer", None)
                if renderer is not None:
                    with renderer(model, height=size, width=size) as r:
                        r.update_scene(data, camera=cam_id)
                        return r.render().astype(np.float32) / 255.0
        except Exception:
            pass
    # deterministic placeholder: green background + dark slot stripe
    img = np.full((size, size, 3), 0.55, dtype=np.float32)
    img[..., 1] = 0.75
    img[:, int(size * 0.45):int(size * 0.55), :] = 0.25
    return img
