#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_state.py — sürümlü, tür-korumalı çalışma-zamanı durum kodlaması.

Checkpoint'ler güvenilir yerel üretim dosyalarıdır; rastgele pickle yüklemek yerine
veri türleri açıkça kodlanır. Desteklenmeyen nesne sessizce atlanmaz. Bu modül
organların mimarisini yeniden kurmaz; onların *gelecek davranışı etkileyen* dinamik
alanlarını taşır. Ağırlıklar mevcut NPZ yolunda kalır, burada ikinci bir kopya
zorunluluğu yoktur.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import fields, is_dataclass
import base64
import importlib
import math
import random
from typing import Any

import numpy as np


class StateError(TypeError):
    pass


def encode(value: Any) -> Any:
    """Desteklenen bir çalışma-zamanı değerini JSON'a uygun, tür-korumalı yap."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, np.ndarray):
        return {"__ndarray__": encode(value.tolist()),
                "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, np.random.Generator):
        return {"__np_generator__": encode(value.bit_generator.state)}
    if isinstance(value, random.Random):
        return {"__py_random__": encode(value.getstate())}
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, deque):
        return {"__deque__": encode(list(value)), "maxlen": value.maxlen}
    if isinstance(value, Counter):
        return {"__counter__": encode(dict(value))}
    if isinstance(value, OrderedDict):
        return {"__ordered_dict__": encode(list(value.items()))}
    if isinstance(value, dict):
        # Normal string-keyed dict'leri kompakt tut; diğer anahtarları da kaybetme.
        if all(isinstance(k, str) for k in value):
            return {str(k): encode(v) for k, v in value.items()}
        return {"__dict_items__": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"__tuple__": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, set):
        return {"__set__": [encode(v) for v in value]}
    if isinstance(value, frozenset):
        return {"__frozenset__": [encode(v) for v in value]}
    if is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        return {"__dataclass__": f"{cls.__module__}:{cls.__qualname__}",
                "fields": {f.name: encode(getattr(value, f.name)) for f in fields(value)}}
    raise StateError(f"unsupported state value: {type(value).__module__}.{type(value).__qualname__}")


def _import_dataclass(path: str):
    try:
        module_name, qualname = str(path).split(":", 1)
        # Checkpoint iç sınıfları dışında rastgele modül yükleme; bu dosyalar Nora'nın
        # kendi üretim snapshot'ları içindir.
        if not (module_name.startswith("nora_") or module_name == "__main__"):
            raise StateError(f"dataclass module rejected: {module_name}")
        obj = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj
    except StateError:
        raise
    except Exception as exc:
        raise StateError(f"cannot restore dataclass {path}: {exc}") from exc


def decode(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        raise StateError(f"invalid encoded state: {type(value).__name__}")
    if "__float__" in value:
        tag = value["__float__"]
        return float("nan") if tag == "nan" else float("-inf" if tag == "-inf" else "inf")
    if "__ndarray__" in value:
        arr = np.asarray(decode(value["__ndarray__"]), dtype=value.get("dtype") or None)
        shape = tuple(int(x) for x in value.get("shape") or ())
        return arr.reshape(shape) if shape else arr
    if "__np_generator__" in value:
        rng = np.random.default_rng()
        rng.bit_generator.state = decode(value["__np_generator__"])
        return rng
    if "__py_random__" in value:
        rng = random.Random()
        rng.setstate(decode(value["__py_random__"]))
        return rng
    if "__bytes__" in value:
        return base64.b64decode(value["__bytes__"])
    if "__deque__" in value:
        return deque(decode(value["__deque__"]), maxlen=value.get("maxlen"))
    if "__counter__" in value:
        return Counter(decode(value["__counter__"]))
    if "__ordered_dict__" in value:
        return OrderedDict((decode(k), decode(v)) for k, v in decode(value["__ordered_dict__"]))
    if "__dict_items__" in value:
        return {decode(k): decode(v) for k, v in value["__dict_items__"]}
    if "__tuple__" in value:
        return tuple(decode(v) for v in value["__tuple__"])
    if "__set__" in value:
        return {decode(v) for v in value["__set__"]}
    if "__frozenset__" in value:
        return frozenset(decode(v) for v in value["__frozenset__"])
    if "__dataclass__" in value:
        cls = _import_dataclass(value["__dataclass__"])
        kwargs = {k: decode(v) for k, v in (value.get("fields") or {}).items()}
        try:
            return cls(**kwargs)
        except Exception as exc:
            raise StateError(f"cannot construct {cls.__name__}: {exc}") from exc
    return {str(k): decode(v) for k, v in value.items()}


def capture_attrs(obj: Any, names, *, strict: bool = True) -> dict:
    """Alanları kodla; strict modda tek bir destek dışı alan bile işlemeyi durdurur."""
    out, skipped = {}, []
    for name in names:
        if not hasattr(obj, name):
            continue
        try:
            out[str(name)] = encode(getattr(obj, name))
        except (StateError, ValueError, TypeError) as exc:
            if strict:
                raise StateError(f"{type(obj).__name__}.{name}: {exc}") from exc
            skipped.append(str(name))
    result = {"values": out}
    if skipped:
        result["skipped"] = skipped
    return result


def restore_attrs(obj: Any, data: dict | None, *, strict: bool = True) -> list[str]:
    if not data:
        return []
    values = data.get("values", data)
    skipped = list(data.get("skipped") or [])
    if strict and skipped:
        raise StateError(f"checkpoint skipped fields: {skipped}")
    for name, value in (values or {}).items():
        try:
            setattr(obj, str(name), decode(value))
        except Exception as exc:
            if strict:
                raise StateError(f"cannot restore {type(obj).__name__}.{name}: {exc}") from exc
            skipped.append(str(name))
    return skipped


def capture_rng(rng: Any) -> dict:
    return {"value": encode(rng)}


def restore_rng(current: Any, data: dict | None) -> Any:
    if not data:
        return current
    restored = decode(data.get("value"))
    # Var olan üreteç nesnesini korumak, başka organların referanslarını bozmaz.
    if isinstance(current, np.random.Generator) and isinstance(restored, np.random.Generator):
        current.bit_generator.state = restored.bit_generator.state
        return current
    if isinstance(current, random.Random) and isinstance(restored, random.Random):
        current.setstate(restored.getstate())
        return current
    return restored


__all__ = ["StateError", "encode", "decode", "capture_attrs", "restore_attrs",
           "capture_rng", "restore_rng"]
