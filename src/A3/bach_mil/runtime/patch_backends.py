from __future__ import annotations

import contextlib
import ctypes
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import torch
import torch.nn as nn

from ..data.label_map import PHOTO_LABELS
from ..models.encoder import build_patch_model
from ..utils.device import get_torch_device
from ..utils.io import load_json

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 2
ACL_DEVICE = 0
ACL_HOST = 1
ACL_DTYPE_TO_NUMPY = {
    0: np.float32,
    1: np.float16,
    2: np.int8,
    3: np.int32,
    4: np.uint8,
    6: np.int16,
    7: np.uint16,
    8: np.uint32,
    9: np.int64,
    10: np.uint64,
    11: np.float64,
    12: np.bool_,
}


def _acl_dtype_to_numpy(dtype_code: int) -> np.dtype:
    return ACL_DTYPE_TO_NUMPY.get(int(dtype_code), np.float32)


def _resolve_meta_json(model_path: str | Path, meta_json: str | Path | None) -> Path | None:
    if meta_json is not None:
        path = Path(meta_json)
        return path if path.exists() else None
    model_path = Path(model_path)
    candidates = [
        model_path.with_suffix('.meta.json'),
        model_path.with_name(f'{model_path.stem}.meta.json'),
        model_path.with_suffix('.json'),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _ensure_numpy_batch(batch: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(batch, torch.Tensor):
        arr = batch.detach().cpu().numpy()
    else:
        arr = np.asarray(batch)
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    if arr.ndim != 4:
        raise ValueError(f'expected NCHW batch, got shape={arr.shape}')
    return arr


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32)
    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)


def cosine_similarity_mean(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(len(a), -1)
    b = np.asarray(b, dtype=np.float32).reshape(len(b), -1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.clip(denom, a_min=1e-12, a_max=None)
    return float(np.mean((a * b).sum(axis=1) / denom))


def _parse_torch_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    name = str(name).lower()
    if name in {'none', '', 'null'}:
        return None
    if name == 'fp16':
        return torch.float16
    if name == 'bf16':
        return torch.bfloat16
    raise ValueError(f'unsupported amp dtype: {name}')


def _dtype_name(dtype: torch.dtype | None) -> str:
    if dtype is None:
        return 'none'
    if dtype == torch.float16:
        return 'fp16'
    if dtype == torch.bfloat16:
        return 'bf16'
    return str(dtype)


def _numpy_view_from_ptr(ptr: int, *, size_bytes: int, dtype: np.dtype, shape: Sequence[int]) -> tuple[ctypes.Array, np.ndarray]:
    dtype = np.dtype(dtype)
    item_count = int(size_bytes // max(1, dtype.itemsize))
    buf_type = ctypes.c_ubyte * int(size_bytes)
    backing = buf_type.from_address(int(ptr))
    view = np.frombuffer(backing, dtype=dtype, count=item_count).reshape(tuple(int(x) for x in shape))
    return backing, view


@dataclass
class BatchOutputs:
    features: np.ndarray
    logits: np.ndarray

    @property
    def probs(self) -> np.ndarray:
        return softmax_numpy(self.logits)


class PatchExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        features = self.model.forward_features(x)
        logits = self.model.classifier(features)
        return features, logits


def load_patch_model_from_ckpt(
    *,
    ckpt_path: str | Path,
    model_name: str,
    backbone_pool: str,
    backbone_init_values: float | None,
    default_classes: Sequence[str] | None = None,
) -> tuple[nn.Module, list[str], dict]:
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    classes = list(ckpt.get('classes', list(default_classes or PHOTO_LABELS)))
    model = build_patch_model(
        model_name=model_name,
        num_classes=len(classes),
        pretrained=False,
        backbone_pool=backbone_pool,
        backbone_init_values=backbone_init_values,
    )
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    meta = {
        'classes': classes,
        'feature_dim': int(getattr(model, 'feature_dim', 0)),
        'ckpt_path': str(ckpt_path),
        'metrics': ckpt.get('metrics'),
    }
    return model, classes, meta


class PatchBackendBase(ABC):
    backend_name = 'base'

    def __init__(self, *, classes: Sequence[str], feature_dim: int, input_size: int = 224):
        self.classes = list(classes)
        self.feature_dim = int(feature_dim)
        self.input_size = int(input_size)

    @abstractmethod
    def predict_batch(self, batch: torch.Tensor | np.ndarray) -> BatchOutputs:
        raise NotImplementedError

    def close(self) -> None:
        return None


class PytorchPatchBackend(PatchBackendBase):
    backend_name = 'pytorch'

    def __init__(
        self,
        *,
        ckpt_path: str | Path,
        model_name: str,
        backbone_pool: str,
        backbone_init_values: float | None,
        default_classes: Sequence[str] | None = None,
        input_size: int = 224,
        device: str | torch.device | None = None,
        amp_dtype: str | None = None,
        channels_last: bool = False,
        jit_compile: bool = False,
    ):
        model, classes, meta = load_patch_model_from_ckpt(
            ckpt_path=ckpt_path,
            model_name=model_name,
            backbone_pool=backbone_pool,
            backbone_init_values=backbone_init_values,
            default_classes=default_classes,
        )
        self.device = torch.device(device) if device is not None else get_torch_device()
        self.amp_dtype = _parse_torch_dtype(amp_dtype)
        self.amp_dtype_name = _dtype_name(self.amp_dtype)
        self.channels_last_requested = bool(channels_last)
        self.channels_last = bool(channels_last) and self.device.type != 'npu'
        self.jit_compile = bool(jit_compile)
        if self.jit_compile and self.device.type == 'npu' and hasattr(torch, 'npu') and hasattr(torch.npu, 'set_compile_mode'):
            torch.npu.set_compile_mode(jit_compile=True)
        self.model = model.to(self.device)
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.ckpt_path = str(ckpt_path)
        super().__init__(classes=classes, feature_dim=meta['feature_dim'], input_size=input_size)

    def _autocast_context(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        if self.device.type == 'npu' and hasattr(torch, 'npu') and hasattr(torch.npu, 'amp'):
            return torch.npu.amp.autocast(dtype=self.amp_dtype)
        if self.device.type == 'cuda':
            return torch.cuda.amp.autocast(dtype=self.amp_dtype)
        return contextlib.nullcontext()

    def predict_batch(self, batch: torch.Tensor | np.ndarray) -> BatchOutputs:
        arr = _ensure_numpy_batch(batch)
        x = torch.from_numpy(arr).to(self.device)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            with self._autocast_context():
                feature_tensor = self.model.forward_features(x)
                logit_tensor = self.model.classifier(feature_tensor)
            features = feature_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            logits = logit_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        return BatchOutputs(features=features, logits=logits)


class OnnxPatchBackend(PatchBackendBase):
    backend_name = 'onnx'

    def __init__(
        self,
        *,
        onnx_path: str | Path,
        meta_json: str | Path | None = None,
        default_classes: Sequence[str] | None = None,
        input_size: int = 224,
        providers: Sequence[str] | None = None,
    ):
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        meta_path = _resolve_meta_json(onnx_path, meta_json)
        meta = load_json(meta_path) if meta_path is not None else {}
        available = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ['CPUExecutionProvider'] if p in available] or available
        self.session = ort.InferenceSession(str(onnx_path), providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name
        output_names = [x.name for x in self.session.get_outputs()]
        self.features_name = meta.get('features_output_name') or ('features' if 'features' in output_names else output_names[0])
        self.logits_name = meta.get('logits_output_name') or ('logits' if 'logits' in output_names else output_names[-1])
        self.static_batch = None
        if meta.get('input_shape'):
            batch_dim = meta['input_shape'][0]
            if isinstance(batch_dim, int) and batch_dim > 0:
                self.static_batch = int(batch_dim)
        else:
            shape = self.session.get_inputs()[0].shape
            if shape and isinstance(shape[0], int):
                self.static_batch = int(shape[0])
        classes = list(meta.get('classes') or list(default_classes or PHOTO_LABELS))
        feature_dim = int(meta.get('feature_dim') or meta.get('output_shapes', {}).get(self.features_name, [0, 0])[-1] or 0)
        self.onnx_path = str(onnx_path)
        self.meta_path = str(meta_path) if meta_path is not None else None
        super().__init__(classes=classes, feature_dim=feature_dim, input_size=input_size)

    def _run_once(self, arr: np.ndarray) -> BatchOutputs:
        n = arr.shape[0]
        if self.static_batch is not None and n != self.static_batch:
            if n > self.static_batch:
                raise ValueError(f'chunk batch {n} exceeds static batch {self.static_batch}')
            pad = np.zeros((self.static_batch - n, *arr.shape[1:]), dtype=np.float32)
            arr_run = np.concatenate([arr, pad], axis=0)
        else:
            arr_run = arr
        outputs = self.session.run([self.features_name, self.logits_name], {self.input_name: arr_run})
        features = np.asarray(outputs[0], dtype=np.float32)[:n]
        logits = np.asarray(outputs[1], dtype=np.float32)[:n]
        return BatchOutputs(features=features, logits=logits)

    def predict_batch(self, batch: torch.Tensor | np.ndarray) -> BatchOutputs:
        arr = _ensure_numpy_batch(batch)
        if self.static_batch is None or arr.shape[0] <= self.static_batch:
            return self._run_once(arr)
        outputs = []
        for start in range(0, arr.shape[0], self.static_batch):
            outputs.append(self._run_once(arr[start:start + self.static_batch]))
        return BatchOutputs(
            features=np.concatenate([x.features for x in outputs], axis=0),
            logits=np.concatenate([x.logits for x in outputs], axis=0),
        )


class OmPatchBackend(PatchBackendBase):
    backend_name = 'om'

    def __init__(
        self,
        *,
        om_path: str | Path,
        meta_json: str | Path | None = None,
        default_classes: Sequence[str] | None = None,
        input_size: int = 224,
        device_id: int = 0,
        execution_mode: str = 'sync',
        host_io_mode: str = 'legacy',
        output_mode: str = 'both',
    ):
        import acl

        self.acl = acl
        self.om_path = str(om_path)
        self.execution_mode = str(execution_mode).lower()
        if self.execution_mode not in {'sync', 'async'}:
            raise ValueError(f'unsupported OM execution_mode={execution_mode}')
        self.host_io_mode = str(host_io_mode).lower()
        if self.host_io_mode not in {'legacy', 'buffer_reuse'}:
            raise ValueError(f'unsupported OM host_io_mode={host_io_mode}')
        self.output_mode = str(output_mode).lower()
        if self.output_mode not in {'both', 'features_only', 'logits_only'}:
            raise ValueError(f'unsupported OM output_mode={output_mode}')
        meta_path = _resolve_meta_json(om_path, meta_json)
        if meta_path is None:
            raise FileNotFoundError(f'OM backend requires meta json, missing for {om_path}')
        self.meta = load_json(meta_path)
        self.meta_path = str(meta_path)
        self.device_id = int(device_id)
        self.input_shape = list(self.meta['input_shape'])
        self.output_shapes = {k: list(v) for k, v in self.meta['output_shapes'].items()}
        self.output_names = list(self.meta['output_order'])
        self.static_batch = int(self.input_shape[0])
        self._init_acl()
        model_id, ret = self._unpack_with_ret(self.acl.mdl.load_from_file(self.om_path))
        self._check_ret(ret, f'load_from_file({self.om_path})')
        self.model_id = model_id
        self.model_desc = self.acl.mdl.create_desc()
        self._check_ret(self.acl.mdl.get_desc(self.model_desc, self.model_id), 'mdl.get_desc')
        self.output_numpy_dtypes = self._resolve_output_numpy_dtypes()
        self.input_dataset = self.acl.mdl.create_dataset()
        self.output_dataset = self.acl.mdl.create_dataset()
        self.input_ptrs: list[int] = []
        self.input_bufs: list[int] = []
        self.output_device_ptrs: list[int] = []
        self.output_host_ptrs: list[int] = []
        self.output_host_backings: list[ctypes.Array] = []
        self.output_host_views: list[np.ndarray] = []
        self.output_sizes: list[int] = []
        self.output_bufs: list[int] = []
        self.input_host_staging: np.ndarray | None = None
        self._last_input_bytes: bytes | None = None
        self.stream = None
        if self.execution_mode == 'async':
            self._check_ret(self.acl.rt.set_context(self.context), 'acl.rt.set_context')
            self.stream, ret = self._unpack_with_ret(self.acl.rt.create_stream())
            self._check_ret(ret, 'acl.rt.create_stream')
        self._build_io_buffers()
        classes = list(self.meta.get('classes') or list(default_classes or PHOTO_LABELS))
        feature_dim = int(self.meta.get('feature_dim') or self.output_shapes[self.output_names[0]][-1])
        super().__init__(classes=classes, feature_dim=feature_dim, input_size=input_size)

    @staticmethod
    def _unpack_with_ret(result):
        if isinstance(result, tuple):
            if len(result) == 2 and isinstance(result[1], int):
                return result[0], int(result[1])
            if result and isinstance(result[-1], int):
                return result[:-1], int(result[-1])
        if isinstance(result, int):
            return None, int(result)
        return result, 0

    def _check_ret(self, ret: int, op: str) -> None:
        if int(ret) != 0:
            raise RuntimeError(f'ACL call failed: {op}, ret={ret}')

    def _init_acl(self) -> None:
        self._check_ret(self.acl.init(), 'acl.init')
        self._check_ret(self.acl.rt.set_device(self.device_id), f'acl.rt.set_device({self.device_id})')
        self.run_mode, ret = self.acl.rt.get_run_mode()
        self._check_ret(ret, 'acl.rt.get_run_mode')
        self.context, ret = self.acl.rt.create_context(self.device_id)
        self._check_ret(ret, f'acl.rt.create_context({self.device_id})')

    def _resolve_output_numpy_dtypes(self) -> list[np.dtype]:
        meta_dtypes = self.meta.get('output_dtypes', {})
        resolved: list[np.dtype] = []
        get_output_data_type = getattr(self.acl.mdl, 'get_output_data_type', None)
        for idx, name in enumerate(self.output_names):
            dtype = None
            if get_output_data_type is not None:
                dtype_code = get_output_data_type(self.model_desc, idx)
                dtype = _acl_dtype_to_numpy(dtype_code)
            if dtype is None:
                dtype_name = str(meta_dtypes.get(name, 'float32')).lower()
                dtype = {
                    'float16': np.float16,
                    'float32': np.float32,
                    'int32': np.int32,
                    'int64': np.int64,
                }.get(dtype_name, np.float32)
            resolved.append(dtype)
        return resolved

    def _build_io_buffers(self) -> None:
        self.input_buffer_size = int(self.acl.mdl.get_input_size_by_index(self.model_desc, 0))
        self.input_device_ptr, ret = self.acl.rt.malloc(self.input_buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
        self._check_ret(ret, 'acl.rt.malloc(input:0)')
        input_buf = self.acl.create_data_buffer(self.input_device_ptr, self.input_buffer_size)
        _, ret = self._unpack_with_ret(self.acl.mdl.add_dataset_buffer(self.input_dataset, input_buf))
        self._check_ret(ret, 'acl.mdl.add_dataset_buffer(input:0)')
        self.input_ptrs.append(self.input_device_ptr)
        self.input_bufs.append(input_buf)
        if self.host_io_mode == 'buffer_reuse':
            self.input_host_staging = np.empty(tuple(self.input_shape), dtype=np.float32)

        for idx, name in enumerate(self.output_names):
            size = int(self.acl.mdl.get_output_size_by_index(self.model_desc, idx))
            device_ptr, ret = self.acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            self._check_ret(ret, f'acl.rt.malloc(output:{idx})')
            host_ptr, ret = self.acl.rt.malloc_host(size)
            self._check_ret(ret, f'acl.rt.malloc_host(output:{idx})')
            buf = self.acl.create_data_buffer(device_ptr, size)
            _, ret = self._unpack_with_ret(self.acl.mdl.add_dataset_buffer(self.output_dataset, buf))
            self._check_ret(ret, f'acl.mdl.add_dataset_buffer(output:{idx})')
            self.output_device_ptrs.append(device_ptr)
            self.output_host_ptrs.append(host_ptr)
            self.output_sizes.append(size)
            self.output_bufs.append(buf)
            if self.host_io_mode == 'buffer_reuse':
                backing, host_view = _numpy_view_from_ptr(
                    host_ptr,
                    size_bytes=size,
                    dtype=self.output_numpy_dtypes[idx],
                    shape=self.output_shapes[name],
                )
                self.output_host_backings.append(backing)
                self.output_host_views.append(host_view)

    def _read_output_from_host(self, idx: int, name: str) -> np.ndarray:
        dtype = self.output_numpy_dtypes[idx]
        if self.host_io_mode == 'buffer_reuse':
            arr = np.array(self.output_host_views[idx], copy=True)
        else:
            raw = self.acl.util.ptr_to_bytes(self.output_host_ptrs[idx], self.output_sizes[idx])
            arr = np.frombuffer(raw, dtype=dtype).reshape(self.output_shapes[name]).copy()
        return arr.astype(np.float32, copy=False)

    def _prepare_input_ptr(self, arr_run: np.ndarray) -> int:
        arr_run = np.ascontiguousarray(arr_run)
        if self.host_io_mode == 'buffer_reuse':
            if self.input_host_staging is None:
                raise RuntimeError('input_host_staging is not initialized')
            np.copyto(self.input_host_staging, arr_run)
            return int(self.input_host_staging.ctypes.data)
        bytes_data = arr_run.tobytes()
        self._last_input_bytes = bytes_data
        return int(self.acl.util.bytes_to_ptr(bytes_data))

    def _should_read_output(self, name: str) -> bool:
        if self.output_mode == 'both':
            return True
        if self.output_mode == 'features_only':
            return name == 'features'
        if self.output_mode == 'logits_only':
            return name == 'logits'
        return True

    def _empty_output(self, name: str, n: int) -> np.ndarray:
        shape = list(self.output_shapes[name])
        shape[0] = int(n)
        return np.zeros(shape, dtype=np.float32)

    def _read_output(self, idx: int, name: str) -> np.ndarray:
        size = self.output_sizes[idx]
        self._check_ret(
            self.acl.rt.memcpy(
                self.output_host_ptrs[idx],
                size,
                self.output_device_ptrs[idx],
                size,
                ACL_MEMCPY_DEVICE_TO_HOST,
            ),
            f'acl.rt.memcpy(d2h:{name})',
        )
        return self._read_output_from_host(idx, name)

    def _run_once(self, arr: np.ndarray) -> BatchOutputs:
        n = arr.shape[0]
        if n > self.static_batch:
            raise ValueError(f'chunk batch {n} exceeds static batch {self.static_batch}')
        if n != self.static_batch:
            pad = np.zeros((self.static_batch - n, *arr.shape[1:]), dtype=np.float32)
            arr_run = np.concatenate([arr, pad], axis=0)
        else:
            arr_run = arr
        self._check_ret(self.acl.rt.set_context(self.context), 'acl.rt.set_context')
        if arr_run.nbytes != self.input_buffer_size:
            raise RuntimeError(f'input bytes mismatch: got={arr_run.nbytes}, expect={self.input_buffer_size}')
        src_ptr = self._prepare_input_ptr(arr_run)
        copy_kind = ACL_MEMCPY_HOST_TO_DEVICE
        if self.run_mode == ACL_DEVICE:
            copy_kind = ACL_MEMCPY_HOST_TO_DEVICE
        if self.execution_mode == 'async' and self.stream is not None:
            self._check_ret(
                self.acl.rt.memcpy_async(
                    self.input_device_ptr,
                    self.input_buffer_size,
                    src_ptr,
                    arr_run.nbytes,
                    copy_kind,
                    self.stream,
                ),
                'acl.rt.memcpy_async(h2d:input)',
            )
            self._check_ret(
                self.acl.mdl.execute_async(self.model_id, self.input_dataset, self.output_dataset, self.stream),
                'acl.mdl.execute_async',
            )
            for idx, name in enumerate(self.output_names):
                if not self._should_read_output(name):
                    continue
                self._check_ret(
                    self.acl.rt.memcpy_async(
                        self.output_host_ptrs[idx],
                        self.output_sizes[idx],
                        self.output_device_ptrs[idx],
                        self.output_sizes[idx],
                        ACL_MEMCPY_DEVICE_TO_HOST,
                        self.stream,
                    ),
                    f'acl.rt.memcpy_async(d2h:{name})',
                )
            self._check_ret(self.acl.rt.synchronize_stream(self.stream), 'acl.rt.synchronize_stream')
            outputs = {}
            for idx, name in enumerate(self.output_names):
                if self._should_read_output(name):
                    outputs[name] = self._read_output_from_host(idx, name)[:n]
                else:
                    outputs[name] = self._empty_output(name, n)
        else:
            self._check_ret(
                self.acl.rt.memcpy(
                    self.input_device_ptr,
                    self.input_buffer_size,
                    src_ptr,
                    arr_run.nbytes,
                    copy_kind,
                ),
                'acl.rt.memcpy(h2d:input)',
            )
            self._check_ret(self.acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset), 'acl.mdl.execute')
            outputs = {}
            for idx, name in enumerate(self.output_names):
                if self._should_read_output(name):
                    outputs[name] = self._read_output(idx, name)[:n]
                else:
                    outputs[name] = self._empty_output(name, n)
        features = outputs['features'] if 'features' in outputs else outputs[self.output_names[0]]
        logits = outputs['logits'] if 'logits' in outputs else outputs[self.output_names[-1]]
        return BatchOutputs(features=features, logits=logits)

    def predict_batch(self, batch: torch.Tensor | np.ndarray) -> BatchOutputs:
        arr = _ensure_numpy_batch(batch)
        if arr.shape[0] <= self.static_batch:
            return self._run_once(arr)
        outputs = []
        for start in range(0, arr.shape[0], self.static_batch):
            outputs.append(self._run_once(arr[start:start + self.static_batch]))
        return BatchOutputs(
            features=np.concatenate([x.features for x in outputs], axis=0),
            logits=np.concatenate([x.logits for x in outputs], axis=0),
        )

    def close(self) -> None:
        unload = getattr(self.acl.mdl, 'unload', None)
        if unload is not None and getattr(self, 'model_id', None) is not None:
            unload(self.model_id)
        for buf in getattr(self, 'input_bufs', []):
            self.acl.destroy_data_buffer(buf)
        for ptr in getattr(self, 'input_ptrs', []):
            self.acl.rt.free(ptr)
        for buf in getattr(self, 'output_bufs', []):
            self.acl.destroy_data_buffer(buf)
        for ptr in getattr(self, 'output_device_ptrs', []):
            self.acl.rt.free(ptr)
        for ptr in getattr(self, 'output_host_ptrs', []):
            self.acl.rt.free_host(ptr)
        if getattr(self, 'input_dataset', None) is not None:
            self.acl.mdl.destroy_dataset(self.input_dataset)
        if getattr(self, 'output_dataset', None) is not None:
            self.acl.mdl.destroy_dataset(self.output_dataset)
        if getattr(self, 'model_desc', None) is not None:
            self.acl.mdl.destroy_desc(self.model_desc)
        if getattr(self, 'stream', None) is not None:
            self.acl.rt.destroy_stream(self.stream)
        if getattr(self, 'context', None) is not None:
            self.acl.rt.destroy_context(self.context)
        self.acl.rt.reset_device(self.device_id)
        self.acl.finalize()


def create_patch_backend(
    *,
    backend: str,
    ckpt_path: str | Path | None,
    model_name: str,
    backbone_pool: str,
    backbone_init_values: float | None,
    default_classes: Sequence[str] | None = None,
    input_size: int = 224,
    onnx_path: str | Path | None = None,
    om_path: str | Path | None = None,
    meta_json: str | Path | None = None,
    device_id: int = 0,
    providers: Sequence[str] | None = None,
    torch_device: str | torch.device | None = None,
    amp_dtype: str | None = None,
    channels_last: bool = False,
    jit_compile: bool = False,
    om_execution_mode: str = 'sync',
    om_host_io_mode: str = 'legacy',
    om_output_mode: str = 'both',
) -> PatchBackendBase:
    backend = str(backend).lower()
    if backend == 'auto':
        if om_path is not None and Path(om_path).exists():
            backend = 'om'
        elif onnx_path is not None and Path(onnx_path).exists():
            backend = 'onnx'
        else:
            backend = 'pytorch'

    if backend == 'pytorch':
        if ckpt_path is None:
            raise ValueError('pytorch backend requires ckpt_path')
        return PytorchPatchBackend(
            ckpt_path=ckpt_path,
            model_name=model_name,
            backbone_pool=backbone_pool,
            backbone_init_values=backbone_init_values,
            default_classes=default_classes,
            input_size=input_size,
            device=torch_device,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            jit_compile=jit_compile,
        )
    if backend == 'onnx':
        if onnx_path is None:
            raise ValueError('onnx backend requires onnx_path')
        return OnnxPatchBackend(
            onnx_path=onnx_path,
            meta_json=meta_json,
            default_classes=default_classes,
            input_size=input_size,
            providers=providers,
        )
    if backend == 'om':
        if om_path is None:
            raise ValueError('om backend requires om_path')
        return OmPatchBackend(
            om_path=om_path,
            meta_json=meta_json,
            default_classes=default_classes,
            input_size=input_size,
            device_id=device_id,
            execution_mode=om_execution_mode,
            host_io_mode=om_host_io_mode,
            output_mode=om_output_mode,
        )
    raise ValueError(f'unknown backend={backend}')
