__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Python standard library imports
from typing import AnyStr, Dict, Any
from dataclasses import dataclass, asdict
import logging
# 3rd Party imports
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.nn import GraphConv
from enum import Enum
# Library imports
from play import Play
from dataset import DatasetException
from dataset.graph.pyg_datasets import PyGDatasets
from deeplearning.training.gnn_training import GNNTraining
from deeplearning.model.graph.graph_base_model import GraphBaseModel
from deeplearning.model.graph.graph_conv_model import GraphConvModel
from dataset.graph.graph_data_loader import GraphDataLoader
from deeplearning.block.graph import GraphException
import python


@dataclass
class GNNExecutionConfiguration:
    processor: AnyStr
    mixed_precision: bool
    hidden_dimension: int
    checkpoint: bool
    pin_memory: bool
    num_workers: int
    neighbors_sampling: AnyStr
    batching: bool

    def __str__(self) -> AnyStr:
        return (f'{self.processor=}, {self.mixed_precision=}, {self.hidden_dimension=}, {self.checkpoint=}, '
                f'{self.pin_memory=}, {self.num_workers=}, {self.neighbors_sampling=}, {self.batching=}')

    def asdict(self) -> Dict[AnyStr, Any]:
        return asdict(self)


"""
import time, tracemalloc, os
import psutil
from functools import wraps

_PROC = psutil.Process(os.getpid())
def monitor_memory(*, 
                   print_report=True, 
                   return_report=False, 
                   top=0,                  # top N allocation lines (tracemalloc)
                   tracemalloc_frames=25,  # stack depth for tracemalloc
                   gpu=False,              # track PyTorch CUDA memory
                   gpu_device=None):       # int or torch.device
 
 def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # ---------- pre ----------
            start_t = time.perf_counter()
            rss_start = None
            if _PROC:
                try: rss_start = _PROC.memory_info().rss
                except Exception: pass

            tracemalloc.start(tracemalloc_frames)
            snap_before = tracemalloc.take_snapshot() if top else None

            gpu_ok = False
            gpu_start = gpu_end = gpu_peak = None
            if gpu:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA not available")
                    dev = gpu_device if gpu_device is not None else torch.cuda.current_device()
                    torch.cuda.reset_peak_memory_stats(dev)
                    torch.cuda.synchronize()
                    gpu_start = torch.cuda.memory_allocated(dev)
                    gpu_ok = True
                except Exception:
                    gpu_ok = False  # silently skip if unavailable

            # ---------- run ----------
            ok = False
            err = None
            result = None
            try:
                result = fn(*args, **kwargs)
                ok = True
            except Exception as e:
                err = e
            # ---------- post ----------
            elapsed = time.perf_counter() - start_t

            cur_py, peak_py = tracemalloc.get_traced_memory()
            snap_after = tracemalloc.take_snapshot() if top else None
            tracemalloc.stop()

            rss_end = rss_delta = None
            if _PROC:
                try:
                    rss_end = _PROC.memory_info().rss
                    if rss_start is not None:
                        rss_delta = rss_end - rss_start
                except Exception:
                    pass

            if gpu_ok:
                try:
                    import torch
                    dev = gpu_device if gpu_device is not None else torch.cuda.current_device()
                    torch.cuda.synchronize()
                    gpu_end  = torch.cuda.memory_allocated(dev)
                    gpu_peak = torch.cuda.max_memory_allocated(dev)
                except Exception:
                    gpu_ok = False

            top_lines = None
            if top and snap_before and snap_after:
                stats = snap_after.compare_to(snap_before, 'lineno')
                top_lines = [str(s) for s in stats[:top]]

            report = {
                "elapsed_sec": elapsed,
                "rss_start_bytes": rss_start,
                "rss_end_bytes": rss_end,
                "rss_delta_bytes": rss_delta,
                "py_current_bytes": cur_py,
                "py_peak_bytes": peak_py,
                "top_allocations": top_lines,
                "gpu": None if not gpu_ok else {
                    "start_bytes": gpu_start,
                    "end_bytes": gpu_end,
                    "peak_bytes": gpu_peak,
                    "delta_bytes": None if (gpu_start is None or gpu_end is None) else (gpu_end - gpu_start),
                },
            }

            if print_report:
                def fmt(b): return "n/a" if b is None else f"{_b2mb(b):.2f} MB"
                print(
                    f"[mem] {fn.__name__}: {elapsed:.3f}s | "
                    f"RSS Δ {fmt(rss_delta)} (start {fmt(rss_start)}, end {fmt(rss_end)}) | "
                    f"Py peak {fmt(peak_py)}"
                )
                if report["gpu"] is not None:
                    g = report["gpu"]
                    print(f"[mem][GPU] start {fmt(g['start_bytes'])} | end {fmt(g['end_bytes'])} | peak {fmt(g['peak_bytes'])}")
                if top_lines:
                    print("[mem] Top allocations:")
                    for line in top_lines:
                        print("   ", line)

            if ok:
                return (result, report) if return_report else result
            else:
                # still print/recorded report; re-raise original exception
                raise err
        return wrapper
    return decorator
    
-------------------------
def monitor_memory_mps():
    @wraps
    def wrapper(*args, **kwargs) -> Dict[AnyStr, Any]:
        # ---------- Tracking ----------
        gpu_start = gpu_end = gpu_peak = None

        # MPS availability check varies by version
        if torch.backends.mps.is_available() and torch.mps.is_available():
            torch.mps.synchronize()
                        # MPS exposes current/driver allocated; no official "peak"   
            gpu_start = torch.mps.current_allocated_memory()
            drv_start = torch.mps.driver_allocated_memory()
            
            result = fn(*args, **kwargs)
     
            # ---------- Post-run GPU collection ----------
            torch.mps.synchronize()
            gpu_end = torch.mps.current_allocated_memory()
                
                # No official peak API on MPS; you can approximate with max(start,end)
            gpu_peak = max(gpu_start or 0, gpu_end or 0)
            drv_end = torch.mps.driver_allocated_memory()

            return {
                "start_bytes": gpu_start,
                "end_bytes": gpu_end,
                "peak_bytes": gpu_peak,
                "delta_bytes": None if (gpu_start is None or gpu_end is None) else (gpu_end - gpu_start),
            }
        else:
            raise MemoryException('MPS driver is not available')
        return wrapper
    return monitor_memory_mps


===========================

def monitor_memory(..., gpu=False, gpu_backend="auto", gpu_device=None):
    ...
    def wrapper(*args, **kwargs):
        ...
        # ---------- GPU / Accelerator tracking ----------
        acc_kind = None
        gpu_start = gpu_end = gpu_peak = None

        if gpu:
            try:
                import torch

                # pick backend
                if gpu_backend in ("auto", "cuda") and torch.cuda.is_available():
                    acc_kind = "cuda"
                    dev = gpu_device if gpu_device is not None else torch.cuda.current_device()
                    torch.cuda.reset_peak_memory_stats(dev)
                    torch.cuda.synchronize()
                    gpu_start = torch.cuda.memory_allocated(dev)

                elif gpu_backend in ("auto", "mps"):
                    # MPS availability check varies by version
                    mps_ok = False
                    try:
                        mps_ok = torch.backends.mps.is_available()
                    except Exception:
                        try:
                            mps_ok = torch.mps.is_available()
                        except Exception:
                            mps_ok = False

                    if mps_ok:
                        acc_kind = "mps"
                        torch.mps.synchronize()
                        # MPS exposes current/driver allocated; no official "peak"
                        gpu_start = torch.mps.current_allocated_memory()
                        # (optional) You can also record driver memory:
                        # drv_start = torch.mps.driver_allocated_memory()
                # else: leave acc_kind=None (no accelerator)

            except Exception:
                acc_kind = None

        # ---------- run the function ----------
        ...
        # ---------- Post-run GPU collection ----------
        if acc_kind == "cuda":
            try:
                import torch
                dev = gpu_device if gpu_device is not None else torch.cuda.current_device()
                torch.cuda.synchronize()
                gpu_end  = torch.cuda.memory_allocated(dev)
                gpu_peak = torch.cuda.max_memory_allocated(dev)
            except Exception:
                pass

        elif acc_kind == "mps":
            try:
                import torch
                torch.mps.synchronize()
                gpu_end = torch.mps.current_allocated_memory()
                # No official peak API on MPS; you can approximate with max(start,end)
                gpu_peak = max(gpu_start or 0, gpu_end or 0)
                # (optional) drv_end = torch.mps.driver_allocated_memory()
            except Exception:
                pass

        report["gpu"] = None if acc_kind is None else {
            "backend": acc_kind,
            "start_bytes": gpu_start,
            "end_bytes": gpu_end,
            "peak_bytes": gpu_peak,
            "delta_bytes": None if (gpu_start is None or gpu_end is None) else (gpu_end - gpu_start),
        }


@monitor_memory(print_report=True, return_report=False, top=5, gpu=False)
def heavy_fn(n=5_000_000):
    # Example: allocate a big list briefly
    data = [i for i in range(n)]
    return sum(data[:100])

# Just run (prints a summary)
heavy_fn()

# Or capture the report:
@monitor_memory(return_report=True, top=3)
def foo():
    return [0]*10_0000

res, report = foo()
print("Peak Python MB:", report["py_peak_bytes"] / (1024*1024))
"""


class GNNMemoryOptimization(object):
    def __init__(self, gnn: GraphBaseModel) -> None:
        self.gnn = gnn


if __name__ == '__main__':
    gnn_configuration = GNNExecutionConfiguration(processor='mps',
                                                  mixed_precision=True,
                                                  hidden_dimension=64,
                                                  checkpoint=True,
                                                  pin_memory=True,
                                                  num_workers=6,
                                                  neighbors_sampling="NodeNeighbors",
                                                  batching=True)
    logging.info(gnn_configuration.asdict())

