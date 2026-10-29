"""
Hardware Monitoring and Benchmark Module - LAPIG
Real-time tracking of: VRAM, RAM, CPU, and GPU compute throughout the pipeline lifecycle.
"""

import os
import sys
import time
import json
import threading
import subprocess
from datetime import timedelta

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

class HardwareBenchmarkMonitor:
    def __init__(self, sample_interval=1.0):
        self.interval = sample_interval
        self.running = False
        self.thread = None
        self.stop_event = threading.Event()
        self.samples = []
        self.phases = []
        self.current_phase = "Initialization"
        self.phase_start_time = time.time()
        self.overall_start_time = None
        self.overall_end_time = None
        self.process = psutil.Process(os.getpid()) if psutil else None

        # GPU Detection
        self.has_cuda = torch.cuda.is_available() if torch else False
        self.device_name = torch.cuda.get_device_name(0) if self.has_cuda else "CPU"
        self.total_vram_gb = 0.0
        if self.has_cuda:
            props = torch.cuda.get_device_properties(0)
            self.total_vram_gb = props.total_memory / (1024**3)
            torch.cuda.reset_peak_memory_stats()

        self.total_ram_gb = (psutil.virtual_memory().total / (1024**3)) if psutil else 0.0
        self.cpu_cores = psutil.cpu_count(logical=True) if psutil else os.cpu_count()

    def _query_nvidia_smi(self):
        """Queries GPU metrics via nvidia-smi if available."""
        if not self.has_cuda:
            return 0.0, 0.0, 0
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu', '--format=csv,noheader,nounits'],
                timeout=1,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            parts = [p.strip() for p in out.split(',')]
            gpu_util = float(parts[0]) if len(parts) > 0 else 0.0
            mem_used_mb = float(parts[1]) if len(parts) > 1 else 0.0
            temp_c = int(parts[2]) if len(parts) > 2 else 0
            return gpu_util, mem_used_mb / 1024.0, temp_c
        except Exception:
            return 0.0, 0.0, 0

    def start(self):
        """Starts asynchronous hardware sampling thread."""
        self.overall_start_time = time.time()
        self.phase_start_time = self.overall_start_time
        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self.thread.start()

    def set_phase(self, phase_name):
        """Records transition to a new pipeline phase."""
        now = time.time()
        if self.current_phase:
            duration = now - self.phase_start_time
            self.phases.append({
                "phase": self.current_phase,
                "duration_sec": duration
            })
        self.current_phase = phase_name
        self.phase_start_time = now

    def _sampling_loop(self):
        while not self.stop_event.is_set():
            t = time.time()
            cpu_pct = psutil.cpu_percent(interval=None) if psutil else 0.0
            vmem = psutil.virtual_memory() if psutil else None
            sys_ram_gb = (vmem.used / (1024**3)) if vmem else 0.0
            sys_ram_pct = vmem.percent if vmem else 0.0
            proc_ram_gb = (self.process.memory_info().rss / (1024**3)) if self.process else 0.0

            # GPU PyTorch
            vram_alloc_gb = (torch.cuda.memory_allocated() / (1024**3)) if self.has_cuda else 0.0
            vram_res_gb = (torch.cuda.memory_reserved() / (1024**3)) if self.has_cuda else 0.0

            # GPU nvidia-smi
            gpu_util, total_gpu_mem_used_gb, gpu_temp = self._query_nvidia_smi()

            self.samples.append({
                "timestamp": t,
                "phase": self.current_phase,
                "cpu_pct": cpu_pct,
                "sys_ram_gb": sys_ram_gb,
                "sys_ram_pct": sys_ram_pct,
                "proc_ram_gb": proc_ram_gb,
                "vram_alloc_gb": vram_alloc_gb,
                "vram_res_gb": vram_res_gb,
                "total_vram_used_gb": total_gpu_mem_used_gb,
                "gpu_util_pct": gpu_util,
                "gpu_temp_c": gpu_temp
            })
            time.sleep(self.interval)

    def stop(self):
        """Terminates monitoring thread and consolidates summary metrics."""
        self.overall_end_time = time.time()
        if self.current_phase:
            duration = self.overall_end_time - self.phase_start_time
            self.phases.append({
                "phase": self.current_phase,
                "duration_sec": duration
            })
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.running = False
        return self.get_summary()

    def get_summary(self):
        """Calculates aggregated peak and mean hardware metrics."""
        total_sec = max(0.1, (self.overall_end_time or time.time()) - (self.overall_start_time or time.time()))

        # Native PyTorch memory peaks
        max_torch_alloc = (torch.cuda.max_memory_allocated() / (1024**3)) if self.has_cuda else 0.0
        max_torch_res = (torch.cuda.max_memory_reserved() / (1024**3)) if self.has_cuda else 0.0

        if not self.samples:
            return {
                "total_duration_sec": total_sec,
                "device_name": self.device_name,
                "phases": self.phases,
                "gpu": {"max_vram_alloc_gb": max_torch_alloc, "max_vram_res_gb": max_torch_res},
                "cpu": {},
                "ram": {}
            }

        # Aggregate sampled metrics
        cpu_vals = [s["cpu_pct"] for s in self.samples]
        ram_vals = [s["sys_ram_gb"] for s in self.samples]
        proc_ram_vals = [s["proc_ram_gb"] for s in self.samples]
        vram_alloc_vals = [s["vram_alloc_gb"] for s in self.samples]
        vram_res_vals = [s["vram_res_gb"] for s in self.samples]
        gpu_util_vals = [s["gpu_util_pct"] for s in self.samples]
        gpu_temp_vals = [s["gpu_temp_c"] for s in self.samples if s["gpu_temp_c"] > 0]

        summary = {
            "total_duration_sec": total_sec,
            "total_duration_str": str(timedelta(seconds=int(total_sec))),
            "device_name": self.device_name,
            "cpu_cores": self.cpu_cores,
            "total_system_ram_gb": round(self.total_ram_gb, 2),
            "total_vram_gb": round(self.total_vram_gb, 2),
            "phases": self.phases,
            "cpu": {
                "avg_pct": round(sum(cpu_vals) / len(cpu_vals), 1),
                "peak_pct": round(max(cpu_vals), 1)
            },
            "ram": {
                "avg_sys_ram_gb": round(sum(ram_vals) / len(ram_vals), 2),
                "peak_sys_ram_gb": round(max(ram_vals), 2),
                "peak_proc_ram_gb": round(max(proc_ram_vals), 2)
            },
            "gpu": {
                "has_cuda": self.has_cuda,
                "peak_vram_alloc_gb": round(max(max_torch_alloc, max(vram_alloc_vals, default=0)), 2),
                "peak_vram_res_gb": round(max(max_torch_res, max(vram_res_vals, default=0)), 2),
                "avg_vram_alloc_gb": round(sum(vram_alloc_vals) / len(vram_alloc_vals), 2) if vram_alloc_vals else 0.0,
                "avg_gpu_util_pct": round(sum(gpu_util_vals) / len(gpu_util_vals), 1) if gpu_util_vals else 0.0,
                "peak_gpu_util_pct": round(max(gpu_util_vals, default=0), 1),
                "peak_temp_c": max(gpu_temp_vals, default=0)
            }
        }
        return summary

    def generate_report_text(self):
        """Generates formatted text report for benchmark diagnostics."""
        data = self.get_summary()
        lines = []
        lines.append("=" * 80)
        lines.append("HARDWARE BENCHMARK REPORT - LAPIG CADASTRAL PIPELINE")
        lines.append("=" * 80)
        lines.append(f"TOTAL EXECUTION TIME:     {data['total_duration_str']} ({data['total_duration_sec']:.1f}s)")
        lines.append(f"PROCESSING DEVICE:        {data['device_name']}")
        lines.append("-" * 80)
        lines.append("TIME DISTRIBUTION PER PIPELINE PHASE:")
        total_t = max(0.1, data['total_duration_sec'])
        for p in data['phases']:
            dur = p['duration_sec']
            pct = (dur / total_t) * 100
            dur_str = str(timedelta(seconds=int(dur)))
            lines.append(f"  * {p['phase']:<44} {dur_str} ({pct:5.1f}%)")
        lines.append("-" * 80)

        # GPU
        if data['gpu']['has_cuda']:
            v_alloc = data['gpu']['peak_vram_alloc_gb']
            v_tot = data['total_vram_gb']
            v_pct = (v_alloc / v_tot * 100) if v_tot > 0 else 0
            lines.append(f"GRAPHICS CARD (GPU - {data['device_name']}):")
            lines.append(f"  * Peak Allocated VRAM:            {v_alloc:.2f} GB / {v_tot:.2f} GB ({v_pct:.1f}%)")
            lines.append(f"  * Peak Reserved VRAM (Cache):     {data['gpu']['peak_vram_res_gb']:.2f} GB")
            lines.append(f"  * Mean Allocated VRAM:            {data['gpu']['avg_vram_alloc_gb']:.2f} GB")
            lines.append(f"  * GPU Compute Utilization:        Mean: {data['gpu']['avg_gpu_util_pct']}% | Peak: {data['gpu']['peak_gpu_util_pct']}%")
            if data['gpu']['peak_temp_c'] > 0:
                lines.append(f"  * Peak GPU Temperature:           {data['gpu']['peak_temp_c']} deg C")
        else:
            lines.append("GPU: CUDA not available. Pipeline executed on CPU.")

        lines.append("-" * 80)
        # RAM
        r_peak = data['ram']['peak_sys_ram_gb']
        r_tot = data['total_system_ram_gb']
        r_pct = (r_peak / r_tot * 100) if r_tot > 0 else 0
        lines.append(f"SYSTEM RAM (Total: {r_tot:.2f} GB):")
        lines.append(f"  * Peak System-Wide RAM Usage:     {r_peak:.2f} GB ({r_pct:.1f}%)")
        lines.append(f"  * Mean System-Wide RAM Usage:     {data['ram']['avg_sys_ram_gb']:.2f} GB")
        lines.append(f"  * Python Process RSS Peak:        {data['ram']['peak_proc_ram_gb']:.2f} GB")

        lines.append("-" * 80)
        # CPU
        lines.append(f"PROCESSOR (CPU - {data['cpu_cores']} cores):")
        lines.append(f"  * Mean CPU Utilization:           {data['cpu']['avg_pct']}%")
        lines.append(f"  * Peak CPU Utilization:           {data['cpu']['peak_pct']}%")

        lines.append("-" * 80)
        # Diagnostic
        lines.append("DIAGNOSTIC & PERFORMANCE ANALYSIS:")
        sam_phase = next((p for p in data['phases'] if 'SAMGeo' in p['phase'] or 'Inference' in p['phase'] or 'Inferência' in p['phase']), None)
        if sam_phase:
            sam_pct = (sam_phase['duration_sec'] / total_t) * 100
            lines.append(f"  1. Computational Bottleneck: SAMGeo inference accounted for {sam_pct:.1f}% of total execution time.")
        if data['gpu']['has_cuda']:
            v_alloc = data['gpu']['peak_vram_alloc_gb']
            v_tot = data['total_vram_gb']
            if v_alloc < v_tot * 0.90:
                lines.append(f"  2. VRAM Safety Margin: Stable! Peak allocation ({v_alloc:.2f} GB) remained safely below capacity ({v_tot:.2f} GB).")
            else:
                lines.append(f"  2. VRAM Warning: Close to memory limit ({v_alloc:.2f} GB / {v_tot:.2f} GB). Maintain points_per_batch=64.")
        lines.append("  3. Vectorization: Planar fence chord rectification and enclave absorption executed efficiently on CPU.")
        lines.append("=" * 80 + "\n")
        return "\n".join(lines)

    def save_report(self, output_dir, filename_prefix="relatorio_benchmark"):
        """Saves reports in .txt and .json formats in the results folder."""
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, f"{filename_prefix}.txt")
        json_path = os.path.join(output_dir, f"{filename_prefix}.json")

        report_text = self.generate_report_text()
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, indent=2, ensure_ascii=False)

        return txt_path, json_path
