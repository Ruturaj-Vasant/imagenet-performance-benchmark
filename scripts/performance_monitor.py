#!/usr/bin/env python3
import csv
import os
import socket
import time
import platform
from datetime import datetime

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

try:
    from thop import profile  # type: ignore
except Exception:  # pragma: no cover
    profile = None


class PerformanceMonitor:
    """
    Lightweight metrics helper to summarize runtime, model complexity, and system info.
    Safe on CPU, CUDA, and MPS. External deps (psutil, thop) are optional.
    """

    def __init__(self, torch, model, batch_size, arch, dataset_root, world_size=1, distributed=False):
        self.torch = torch
        self.model = model
        self.batch_size = int(batch_size)
        self.arch = arch
        self.dataset_root = dataset_root
        self.world_size = world_size
        self.distributed = distributed
        self._t0 = None
        self._t1 = None

    # ----- timing -----
    def start(self):
        self._t0 = time.perf_counter()
        if self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()

    def stop(self):
        self._t1 = time.perf_counter()

    # ----- collectors -----
    def collect_system(self):
        torch = self.torch
        device_type = (
            'cuda' if torch.cuda.is_available() else
            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
            'cpu'
        )
        gpu_name = None
        if device_type == 'cuda':
            try:
                gpu_name = self.torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = None

        cpu_name = platform.processor() or platform.machine()
        # Additional system info
        os_name = platform.system()
        kernel = platform.release()
        python_version = platform.python_version()

        # RAM and disk info
        ram_total_gb = ram_available_gb = disk_usage_pct = cpu_util = 'N/A'
        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                ram_total_gb = round(vm.total / (1024 ** 3), 3)
                ram_available_gb = round(vm.available / (1024 ** 3), 3)
                disk = psutil.disk_usage('/')
                disk_usage_pct = round(disk.percent, 2)
                cpu_util = round(psutil.cpu_percent(interval=0.1), 2)
            except Exception:
                pass

        return {
            'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            'host': socket.gethostname(),
            'device_type': device_type,
            'gpu_name': gpu_name or 'N/A',
            'cpu_name': cpu_name or 'N/A',
            'torch_version': getattr(self.torch, '__version__', 'unknown'),
            'os': os_name,
            'kernel': kernel,
            'python_version': python_version,
            'ram_total_gb': ram_total_gb,
            'ram_available_gb': ram_available_gb,
            'disk_usage_pct': disk_usage_pct,
            'cpu_util': cpu_util,
        }

    def collect_model_complexity(self, input_size=(3, 224, 224)):
        params_m = sum(p.numel() for p in self.model.parameters()) / 1e6
        flops_g = None
        if profile is not None:
            try:
                dummy = self.torch.randn(1, *input_size)
                if self.torch.cuda.is_available():
                    dummy = dummy.cuda(non_blocking=True)
                elif hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
                    dummy = dummy.to('mps')
                macs, _ = profile(self.model, inputs=(dummy,), verbose=False)
                flops_g = macs / 1e9
            except Exception:
                flops_g = None
        # Model file size (if available)
        model_size_mb = 'N/A'
        try:
            # If model has a file attribute, or can be saved to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=True) as tmpf:
                self.torch.save(self.model.state_dict(), tmpf.name)
                tmpf.flush()
                model_size_mb = round(os.path.getsize(tmpf.name) / (1024 ** 2), 4)
        except Exception:
            pass
        # Arithmetic intensity: FLOPs per MB model size
        arithmetic_intensity = 'N/A'
        try:
            if flops_g is not None and isinstance(model_size_mb, (int, float)) and model_size_mb > 0:
                arithmetic_intensity = round(flops_g / model_size_mb, 4)
        except Exception:
            pass
        return {
            'params_m': round(params_m, 4),
            'flops_g_per_image': round(flops_g, 4) if flops_g is not None else 'N/A',
            'model_size_mb': model_size_mb,
            'arithmetic_intensity': arithmetic_intensity,
        }

    def collect_runtime(self, train_stats=None, val_stats=None):
        import statistics
        total_time = None
        if self._t0 is not None and self._t1 is not None:
            total_time = self._t1 - self._t0

        # Memory and RAM
        peak_gpu_mem_mb = None
        if self.torch.cuda.is_available():
            try:
                peak_gpu_mem_mb = self.torch.cuda.max_memory_allocated() / 1e6
            except Exception:
                peak_gpu_mem_mb = None
        process_rss_gb = None
        if psutil is not None:
            try:
                process_rss_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
            except Exception:
                process_rss_gb = None

        # GPU utilization via nvidia-smi
        gpu_util = 'N/A'
        if self.torch.cuda.is_available():
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2
                )
                util_lines = result.stdout.strip().splitlines()
                if util_lines:
                    # Average utilization across all visible GPUs
                    utils = [float(line.strip()) for line in util_lines if line.strip()]
                    if utils:
                        gpu_util = round(sum(utils) / len(utils), 2)
            except Exception:
                gpu_util = 'N/A'

        out = {
            'total_time_s': round(total_time, 4) if total_time is not None else 'N/A',
            'peak_gpu_mem_mb': round(peak_gpu_mem_mb, 2) if peak_gpu_mem_mb is not None else 'N/A',
            'process_rss_gb': round(process_rss_gb, 3) if process_rss_gb is not None else 'N/A',
            'gpu_util': gpu_util,
        }

        def _add(prefix, stats):
            if not stats:
                return
            out[f'{prefix}_batches'] = int(stats.get('batches', 0))
            out[f'{prefix}_batch_time_avg_s'] = round(stats.get('batch_time_avg_s', 0.0), 6)
            data_avg = stats.get('data_time_avg_s')
            out[f'{prefix}_data_time_avg_s'] = round(data_avg, 6) if data_avg is not None else 'N/A'
            # Batch time variance
            batch_times = stats.get('batch_times', None)
            batch_time_std = 'N/A'
            if batch_times and isinstance(batch_times, (list, tuple)) and len(batch_times) > 1:
                try:
                    batch_time_std = round(statistics.stdev(batch_times), 6)
                except Exception:
                    batch_time_std = 'N/A'
            out[f'{prefix}_batch_time_std_s'] = batch_time_std
            # Epoch time
            if stats.get('epoch_time_s') is not None:
                out[f'{prefix}_epoch_time_s'] = round(stats['epoch_time_s'], 6)
            if stats.get('batches') and stats.get('epoch_time_s'):
                images = self.batch_size * stats['batches']
                out[f'{prefix}_throughput_img_s'] = round(images / stats['epoch_time_s'], 3)
            if 'acc1' in stats:
                out[f'{prefix}_acc1'] = round(float(stats['acc1']), 3)
            if 'acc5' in stats:
                out[f'{prefix}_acc5'] = round(float(stats['acc5']), 3)
            # Achieved GFLOPs/s if available
            flops_g = None
            if hasattr(self, '_last_flops_g_per_image'):
                flops_g = self._last_flops_g_per_image
            achieved_gflops = 'N/A'
            try:
                if flops_g is not None and isinstance(flops_g, (int, float)) and stats.get('batch_time_avg_s', 0.0) > 0:
                    achieved_gflops = round((float(flops_g) * self.batch_size) / float(stats['batch_time_avg_s']), 6)
            except Exception:
                achieved_gflops = 'N/A'
            out[f'{prefix}_achieved_gflops_s'] = achieved_gflops

        # Store last flops_g_per_image for achieved_gflops_s computation
        # Try to get from model complexity
        try:
            mcomplex = self.collect_model_complexity()
            flops_g = mcomplex.get('flops_g_per_image')
            if isinstance(flops_g, (int, float, float)):
                self._last_flops_g_per_image = flops_g
            else:
                self._last_flops_g_per_image = None
        except Exception:
            self._last_flops_g_per_image = None

        _add('train', train_stats)
        _add('val', val_stats)
        return out

    def achieved_tflops(self, flops_g_per_image, batch_time_avg_s):
        try:
            return round((float(flops_g_per_image) * self.batch_size) / float(batch_time_avg_s) / 1000.0, 6)
        except Exception:
            return 'N/A'

    def write_csv(self, path, row_dict):
        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        # Ensure all keys needed for enhanced logging are present, even if N/A
        extra_keys = [
            'model_size_mb', 'arithmetic_intensity', 'cpu_util', 'gpu_util', 'ram_total_gb', 'ram_available_gb', 'disk_usage_pct'
        ]
        for k in extra_keys:
            if k not in row_dict:
                row_dict[k] = 'N/A'
        header_needed = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
            if header_needed:
                writer.writeheader()
            writer.writerow(row_dict)
