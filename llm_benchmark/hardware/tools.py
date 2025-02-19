import os
import csv
import torch
import datetime

from .benchmark import flops_benchmark, memory_bandwidth_benchmark
from .cpu import create_cpu_config, get_cpu_info
from .cuda import create_cuda_config
from .hpu import get_hpu_info
from .hpu import create_hpu_config
from .rocm import create_rocm_config

from llm_benchmark.utils.device_utils import get_available_devices


def get_device_benchmarks(device):
    benchmarks = {}

    device = torch.device(device)
    actual_flops = flops_benchmark(device)
    benchmarks["actual_flops"] = actual_flops
    benchmarks["memory_bandwidth"] = memory_bandwidth_benchmark(device)

    return benchmarks


def get_hardware_info(output_dir=None):
    if output_dir:
        output_dir = os.path.join(output_dir, "hardware")
        os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    devices = get_available_devices()

    hw_info = {}
    for device in devices:
        raw_info = ""
        if device == "cpu":
            device_info = get_cpu_info()
            device_info.update(get_device_benchmarks(device))
        elif device in ["cuda", "gpu"]:
            from .cuda import get_gpu_info
            device_info, raw_info = get_gpu_info()
            for info in device_info:
                info.update(get_device_benchmarks(f"{device}:{info['device_id']}"))
        elif device == "hpu":
            device_info = create_hpu_config()
            # device_info.update(get_device_benchmarks(device))

        hw_info[device] = device_info

        if output_dir and len(device_info):
            csv_file_path = os.path.join(output_dir, f"{device}_info_{timestamp}.csv")
            file_exists = os.path.isfile(csv_file_path)
            # Open the CSV file in append mode
            with open(csv_file_path, "a", newline="") as csvfile:
                fieldnames = (
                    list(device_info[0].keys())
                    if isinstance(device_info, list)
                    else list(device_info.keys())
                )
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write headers if the file is newly created
                if not file_exists:
                    writer.writeheader()

                # Write the summary data
                if isinstance(device_info, list):
                    for info in device_info:
                        writer.writerow(info)
                else:
                    writer.writerow(device_info)

            if isinstance(raw_info, str) and len(raw_info):
                with open(csv_file_path.replace(".csv", ".dump"), "w") as fp:
                    fp.write(raw_info)

            print(f"Hardware info saved to {output_dir}")

    return hw_info

def create_device_config(device="cpu"):

    if device == "cuda":
        return create_cuda_config()
    elif device == "hpu":
        return create_hpu_config()
    elif device == "rocm":
        return create_rocm_config()
    else:
        return create_cpu_config()
        # raise ValueError(f"Invalid device: {device}")

