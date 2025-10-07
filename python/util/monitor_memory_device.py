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
from functools import wraps
# 3rd Party imports
import torch
__all__ = ['monitor_memory_device']

"""
Memory profiling for CPU and GPU (mps and cuda drivers). The profiling collect 
- initial application memory allocation
- optional driver memory allocation
- peak memory usage
- final application memory allocation
- optional final driver memory allocation
- memory consumption.

The collection of stats is implemented as a Python decorator.
"""
def _monitor_memory_cpu(fn) :
    @wraps(fn)
    def wrapper(*args, **kwargs) -> (Any, Dict[AnyStr, Any]):
        import os
        import psutil
        # ---------- Pre-execution collection ----------
        proc = psutil.Process(os.getpid())
        rss_start = proc.memory_info().rss

        # -------- Execution ----------
        result = fn(*args, **kwargs)

        # ---------- Post-execution collection ----------
        rss_end = proc.memory_info().rss

        # ---------- Reporting ----------
        memory_report = {
            'device': 'CPU',
            'start_bytes': rss_start,
            'end_bytes': rss_end,
            'peak_bytes': rss_end,
            'delta_bytes': rss_end - rss_start
        }
        return result, memory_report
    return wrapper


def _monitor_memory_cuda(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs) -> (Any, Dict[AnyStr, Any]):
        if torch.cuda.is_available():
            # ---------- CUDA Pre-execution collection ----------
            dev = torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(dev)

            # -------- Execution ----------
            result = fn(*args, **kwargs)

            # ---------- Post-execution collection ----------
            torch.cuda.synchronize()
            gpu_start = torch.cuda.memory_allocated(dev)

            torch.cuda.synchronize()
            gpu_end = torch.cuda.memory_allocated(dev)
            gpu_peak = torch.cuda.max_memory_allocated(dev)

            # ---------- Reporting ----------
            memory_report = {
                'device': 'CUDA',
                'start_bytes': gpu_start,
                'end_bytes': gpu_end,
                'peak_bytes': gpu_peak,
                'delta_bytes': gpu_end - gpu_start
            }
            return result, memory_report
        else:
            raise MemoryError('CUDA driver is not available')
    return wrapper


def _monitor_memory_mps(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs) -> (Any, Dict[AnyStr, Any]):
        if torch.backends.mps.is_available():
            # ---------- Pre-execution collection ----------
            torch.mps.synchronize()
            # MPS exposes current/driver allocated; no official "peak"
            gpu_start = torch.mps.current_allocated_memory()
            drv_start = torch.mps.driver_allocated_memory()

            # -------- Execution ----------
            result = fn(*args, **kwargs)

            # ---------- Post-execution collection ----------
            torch.mps.synchronize()
            gpu_end = torch.mps.current_allocated_memory()

            # No official peak API on MPS; you can approximate with max(start,end)
            gpu_peak = max(gpu_start or 0, gpu_end or 0)
            drv_end = torch.mps.driver_allocated_memory()
            # Generate report as a dictionary
            memory_report = {
                'device': 'MPS',
                'start_bytes': gpu_start,
                'end_bytes': gpu_end,
                'peak_bytes': gpu_peak,
                'start_drv_bytes': drv_start,
                'end_drv_bytes': drv_end,
                'delta_bytes': gpu_end - gpu_start,
                'delta_drv_bytes': drv_end - drv_start
            }
            return result, memory_report
        else:
            raise MemoryError('MPS driver is not available')
    return wrapper


"""
    Generic lambda for collecting the memory consumption data for the right device (cpu, mps or cuda). 
    It reverts to cpu if cuda or mps is not detected.
"""
monitor_memory_device = _monitor_memory_cuda if torch.cuda.is_available() else _monitor_memory_mps \
    if torch.backends.mps.is_available() else _monitor_memory_cpu
