import hashlib
import json
from typing import Any


def _to_serializable(obj: Any):
    """
    将对象转换为可 JSON 序列化、且跨平台稳定的表示：
    - 使用排序的字典键
    - 将 set/tuple 转换为列表
    - 将 numpy/torch 标量转换为内置类型
    - 浮点统一为 repr 字符串，避免 0 与 -0、平台格式差异
    """
    try:
        import numpy as _np  # type: ignore
    except Exception:  # pragma: no cover
        _np = None
    try:
        import torch as _torch  # type: ignore
    except Exception:  # pragma: no cover
        _torch = None

    if _np is not None and isinstance(obj, _np.generic):
        obj = obj.item()
    if _torch is not None and hasattr(_torch, 'Tensor') and isinstance(obj, _torch.Tensor):
        obj = obj.detach().cpu().tolist()

    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, float):
        # 使用 repr 确保跨平台一致性，并避免 -0.0 与 0.0 差异
        if obj == 0.0:
            obj = 0.0
        return repr(obj)
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    # 兜底：转字符串
    return repr(obj)


def stable_hash(obj: Any, prefix: str = '', algo: str = 'sha256', length: int = 16) -> str:
    """
    基于规范化 JSON 的跨平台稳定哈希：
    - algo: 'sha256'|'md5'|'sha1' 等 hashlib 支持的算法
    - length: 返回哈希前缀长度
    - prefix: 业务前缀，例如 'TrainPL_'
    """
    normalized = _to_serializable(obj)
    data = json.dumps(normalized, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
    h = hashlib.new(algo)
    h.update(data.encode('utf-8'))
    digest = h.hexdigest()
    if length and length > 0:
        digest = digest[:int(length)]
    return f"{prefix}{digest}"


