import json
import os
import threading
import time
from typing import Any

from app.schemas import InferenceRequest
from config import get_settings

settings = get_settings()

_inference_store: list[InferenceRequest] = []
_store_lock = threading.Lock()
_dirty = False

FLUSH_INTERVAL_SECONDS = settings.vault_flush_interval_seconds

# Serialization
def _serialize(record: InferenceRequest) -> dict[str, Any]:
    return json.loads(record.model_dump_json())

# Atomic disk write 
def _write_to_disk() -> None:
    vault_path = settings.vault_path
    os.makedirs(os.path.dirname(vault_path), exist_ok=True)
    tmp_path = vault_path + ".tmp"
    with _store_lock:
        payload = [_serialize(r) for r in _inference_store]
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp_path, vault_path)



# Background batch flush thread
class _BatchVaultWriter(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="VaultFlushThread")
        self._stop_event = threading.Event()

    def run(self) -> None:
        global _dirty
        while not self._stop_event.is_set():
            time.sleep(FLUSH_INTERVAL_SECONDS)
            if _dirty:
                try:
                    _write_to_disk()
                    _dirty = False
                except Exception:
                    pass

    def stop(self) -> None:
        self._stop_event.set()


_batch_writer = _BatchVaultWriter()


# Public API
def initialize_vault() -> None:
    global _dirty
    vault_path = settings.vault_path
    os.makedirs(os.path.dirname(vault_path), exist_ok=True)

    if not os.path.exists(vault_path):
        with open(vault_path, "w") as f:
            json.dump([], f)
    else:
        with open(vault_path, "r") as f:
            raw_records: list[dict] = json.load(f)
        with _store_lock:
            for raw in raw_records:
                try:
                    _inference_store.append(InferenceRequest(**raw))
                except Exception:
                    continue

    if not _batch_writer.is_alive():
        _batch_writer.start()


def flush_vault() -> None:
    """
    Explicit flush — called on app shutdown to guarantee no data loss.
    Bypasses the dirty flag so it always writes.
    """
    global _dirty
    _write_to_disk()
    _dirty = False


def save_inference(data: InferenceRequest) -> bool:
    """
    O(1) — appends to memory and marks dirty.
    Disk write happens asynchronously via the background thread.
    """
    global _dirty
    try:
        with _store_lock:
            if len(_inference_store) >= settings.max_vault_records:
                _inference_store.pop(0)
            _inference_store.append(data)
        _dirty = True
        return True
    except Exception:
        return False


def get_all_inferences() -> list[InferenceRequest]:
    with _store_lock:
        return list(_inference_store)


def get_inference_by_id(request_id: str) -> InferenceRequest | None:
    with _store_lock:
        return next(
            (r for r in _inference_store if r.request_id == request_id),
            None,
        )


def delete_inference(request_id: str) -> bool:
    global _inference_store, _dirty
    with _store_lock:
        original_length = len(_inference_store)
        _inference_store = [r for r in _inference_store if r.request_id != request_id]
        changed = len(_inference_store) < original_length
    if changed:
        _dirty = True
    return changed
