import hashlib
import json
import os
import pickle
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any


# Prefer this over hydra.utils.instantiate, which does not propagate exception tracebacks, which makes debugging harder.
def instantiate(cfg, **kwargs):
    if cfg is None:
        return None

    cfg = cfg.copy()
    cls = cfg.pop("_target_")

    # Resolve the class if it is a string
    if isinstance(cls, str):
        module_name, class_name = cls.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)

    cfg.update(kwargs)
    obj = cls(**cfg)
    return obj


def _to_json_safe(value):
    """Recursively convert values into a JSON-serializable form for stable cache keys."""
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        mod = getattr(value, "__module__", "")
        qual = getattr(value, "__qualname__", getattr(value, "__name__", repr(value)))
        return f"{mod}.{qual}"
    return value


def normalize_hparams(hparams):
    """
    Recursively convert any dict entries with a `_target_` key whose value is a class/type
    into a string "module.ClassName".
    """
    if isinstance(hparams, dict):
        hparams = dict(hparams)  # shallow copy

        if "_target_" in hparams and isinstance(hparams["_target_"], type):
            cls = hparams["_target_"]
            hparams["_target_"] = f"{cls.__module__}.{cls.__name__}"

        for k, v in hparams.items():
            hparams[k] = normalize_hparams(v)

    elif isinstance(hparams, list):
        hparams = [normalize_hparams(item) for item in hparams]

    return hparams


def load_or_instantiate(cfg: dict[str, Any], name_prefix: str, cache_dir: Path, rebuild=False, save=True, ignore=[], **kwargs) -> Any:
    """Load, or create and cache, an object specified by a given configuration.
    Args:
        cache_dir (Path): Directory to store cached objects.
        name_prefix (str): Prefix for the cached object file names.
        config (dict[str, Any]): Configuration dictionary. _target_ = class to instantiate. Other kwargs are passed to the class.
        rebuild (bool): If True, forces the cache to be rebuilt even if the cached file exists.
        ignore (list[str]): List of config keys to ignore when looking for cache hits.
        kwargs: Additional keyword arguments to pass to the class constructor. Overrides any values in the config.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = deepcopy(cfg)
    cfg.update(kwargs)  # Update cfg with kwargs, allowing to override config values

    # Create dictionary for cache lookup

    cache_key = deepcopy(cfg)

    for k in ignore:
        cache_key.pop(k)

    if not isinstance(cache_key["_target_"], str):
        # Convert class to string. Assume string is on the format <class 'module.ClassName'>.
        cache_key["_target_"] = str(cache_key["_target_"]).split("'")[1]

    cache_key = _to_json_safe(cache_key)
    cache_key = {k: v for k, v in sorted(cache_key.items())}

    s = json.dumps(cache_key, sort_keys=True)  # Sort to ensure consistent hashing
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    data_file = cache_dir / f"{name_prefix}_{h}.pkl"
    meta_file = data_file.with_suffix(".json")
    tmp_suffix = f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
    data_tmp_file = Path(f"{str(data_file)}{tmp_suffix}")
    meta_tmp_file = Path(f"{str(meta_file)}{tmp_suffix}")

    if data_file.exists() and not rebuild:
        with data_file.open("rb") as f:
            obj = pickle.load(f)
        return obj, data_file

    obj = instantiate(cfg)

    if save:
        with meta_tmp_file.open("w") as f:
            json.dump(cache_key, f)
        with data_tmp_file.open("wb") as f:
            pickle.dump(obj, f)

        os.replace(data_tmp_file, data_file)
        os.replace(meta_tmp_file, meta_file)

    return obj, data_file
