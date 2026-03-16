"""
权重导出：支持 .pt (torch) 与 .npz / .npy (numpy)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _to_numpy(arr: Any) -> Any:
    if hasattr(arr, "numpy"):
        return arr.detach().cpu().numpy()
    if hasattr(arr, "__array__"):
        return np.asarray(arr)
    return arr


def _npz_safe_key(key: str) -> str:
    """np.savez 的 key 必须是合法标识符，将 '.' 替换为 '_'。"""
    return key.replace(".", "_")


def export_weights(
    weights: Dict[str, Any],
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
) -> None:
    """
    将权重字典写入文件。

    - format 为 None：按 path 扩展名推断（.npz / .npy -> numpy，否则 .pt）。
    - format "pt"：torch.save(weights, path)，value 可为 Tensor 或 ndarray。
    - format "npz"：np.savez(path, **weights)，key 中 '.' 会替换为 '_'；value 转为 ndarray。
    - format "npy"：仅当 len(weights)==1 时保存单个数组为 .npy；否则报错（多数组请用 .npz）。

    加载：.pt -> torch.load(path)；.npz -> d = np.load(path); d.files 为 key 列表；.npy -> np.load(path)。
    """
    path_str = str(path)
    if format is None:
        if path_str.endswith(".npz"):
            format = "npz"
        elif path_str.endswith(".npy"):
            format = "npy"
        else:
            format = "pt"

    if format == "pt":
        if not _TORCH_AVAILABLE:
            raise RuntimeError("export_weights(..., format='pt') requires torch")
        out: Dict[str, Any] = {}
        for k, v in weights.items():
            if hasattr(v, "numpy"):
                out[k] = v if v.is_cuda else v.clone()
            elif hasattr(v, "__array__"):
                out[k] = torch.from_numpy(np.asarray(v)).clone()
            else:
                out[k] = v
        torch.save(out, path_str)
        return

    if not _NP_AVAILABLE:
        raise RuntimeError("export_weights(..., format='npz'/'npy') requires numpy")

    np_weights = {k: _to_numpy(v) for k, v in weights.items()}

    if format == "npy":
        if len(np_weights) != 1:
            raise ValueError(
                "export_weights(..., format='npy') 仅支持单数组，当前共 %d 个 key，请改用 .npz"
                % len(np_weights)
            )
        np.save(path_str, next(iter(np_weights.values())))
        return

    if format == "npz":
        # np.savez 的 keyword 必须是合法标识符
        safe = {_npz_safe_key(k): v for k, v in np_weights.items()}
        np.savez(path_str, **safe)
        return

    raise ValueError("format 应为 None / 'pt' / 'npz' / 'npy'，当前 %r" % format)
