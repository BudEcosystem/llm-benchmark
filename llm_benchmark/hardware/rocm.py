import re
import subprocess

try:
    from pyrsmi import rocml
except ImportError:
    print(
        "pyrsmi import failed, the system either don't have rocm support or is not configured properly."
    )

from llm_benchmark.hardware.constants import DeviceInfo

# def filter_nvidia_smi(filters: list = None):
#     result = subprocess.run(
#         ["nvidia-smi", "-q"],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         universal_newlines=True,
#     )
#     if result.returncode != 0:
#         raise ValueError(result.stderr)

#     if filters is None or not len(filters):
#         return result.stdout

#     lines = result.stdout.splitlines()
#     gpu_info_list = []
#     gpu_info = {}
#     for line in lines:
#         line = line.strip()
#         if any(key.lower() in line.lower() for key in filters):
#             # Extract key-value pairs from the relevant lines
#             match = re.match(r"(.+?)\s*:\s*(.+)", line)
#             if match:
#                 key = match.group(1).strip().lower().replace(" ", "_").replace(".", "_")
#                 value = match.group(2).strip()
#                 if not key.startswith("gpu"):
#                     key = f"gpu_{key}"

#                 if key in gpu_info:
#                     gpu_info_list.append(gpu_info)
#                     gpu_info = {}

#                 gpu_info[key] = value

#     if len(gpu_info):
#         gpu_info_list.append(gpu_info)

#     return gpu_info_list


# def get_extra_gpu_attributes():
#     """Uses pycuda to extract additional GPU specs like CUDA cores and SM count."""
#     import pycuda.driver as cuda
#     import pycuda.autoinit

#     num_gpus = cuda.Device.count()
#     gpu_specs = []

#     for gpu_id in range(num_gpus):
#         device = cuda.Device(gpu_id)
#         attributes = device.get_attributes()

#         # CUDA cores and other properties depend on the number of SMs (Streaming Multiprocessors)
#         sm_count = attributes[cuda.device_attribute.MULTIPROCESSOR_COUNT]

#         gpu_info = {
#             "sm_count": sm_count,
#             "compute_capability": device.compute_capability(),
#         }

#         for key in (
#             "MAX_THREADS_PER_BLOCK",
#             "MAX_THREADS_PER_MULTIPROCESSOR",
#             "MAX_BLOCKS_PER_MULTIPROCESSOR",
#             "TOTAL_CONSTANT_MEMORY",
#             "GPU_OVERLAP",
#             "SHARED_MEMORY_PER_BLOCK",
#             "MAX_SHARED_MEMORY_PER_BLOCK",
#             "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
#             "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
#             "REGISTERS_PER_BLOCK",
#             "MAX_REGISTERS_PER_BLOCK",
#             "MAX_REGISTERS_PER_MULTIPROCESSOR",
#             "INTEGRATED",
#             "CAN_MAP_HOST_MEMORY",
#             "COMPUTE_MODE",
#             "CONCURRENT_KERNELS",
#             "ECC_ENABLED",
#             "PCI_BUS_ID",
#             "PCI_DEVICE_ID",
#             "TCC_DRIVER",
#             "GLOBAL_MEMORY_BUS_WIDTH",
#             "L2_CACHE_SIZE",
#             "ASYNC_ENGINE_COUNT",
#             "PCI_DOMAIN_ID",
#             "STREAM_PRIORITIES_SUPPORTED",
#             "GLOBAL_L1_CACHE_SUPPORTED",
#             "LOCAL_L1_CACHE_SUPPORTED",
#             "MANAGED_MEMORY",
#             "MULTI_GPU_BOARD",
#             "MULTI_GPU_BOARD_GROUP_ID",
#             "SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO",
#             "PAGEABLE_MEMORY_ACCESS",
#             "CONCURRENT_MANAGED_ACCESS",
#             "COMPUTE_PREEMPTION_SUPPORTED",
#             "MAX_PERSISTING_L2_CACHE_SIZE",
#         ):
#             try:
#                 gpu_info[key.lower()] = attributes[getattr(cuda.device_attribute, key)]
#             except AttributeError:
#                 pass

#         gpu_specs.append(gpu_info)

#     return gpu_specs


def get_cores_info(device_id):
    numa_info = {
        "topo_numa_affinity": rocml.smi_get_device_topo_numa_affinity(device_id),
        "topo_numa_node_number": rocml.smi_get_device_topo_numa_node_number(device_id),
        "device_utilization": rocml.smi_get_device_utilization(device_id)
    }
    return numa_info


# def get_throttle_reasons(device_id):
#     handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    
#     throttle_reasons = {
#         pynvml.nvmlClocksThrottleReasonGpuIdle: "GPUIdle",
#         pynvml.nvmlClocksThrottleReasonHwSlowdown: "HwSlowdown",
#         pynvml.nvmlClocksThrottleReasonSwPowerCap: "SwPowerCap",
#         pynvml.nvmlClocksThrottleReasonUserDefinedClocks: "UserDefinedClocks",
#         pynvml.nvmlClocksThrottleReasonApplicationsClocksSetting: "ApplicationClocksSetting",
#         pynvml.nvmlClocksThrottleReasonAll: "All",
#         # pynvml.nvmlClocksThrottleReasonUnknown: "Unknown",
#         pynvml.nvmlClocksThrottleReasonNone: None,
#     }
#     code = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
#     return throttle_reasons.get(code, str(code))


def get_temp_and_power_info(device_id):

    tp_info = { }

    tp_keys = {
        'rocm_fan_rpms': rocml.smi_get_device_fan_rpms,
        'rocm_fan_speed': rocml.smi_get_device_fan_speed,
        'rocm_fan_speed_max': rocml.smi_get_device_fan_speed_max,
        'rocm_avg_power': rocml.smi_get_device_average_power,
        'rocm_compute_partition': rocml.smi_get_device_compute_partition,
    }
    
    for key, func in tp_keys.items():
        try:
            tp_info[key] = func(device_id)
        except Exception as e:
            tp_info[key] = str(e)
   
    return tp_info


def get_pci_info(device_id):

    pcie_current_info = {
        "pcie_id": rocml.smi_get_device_pci_id(device_id),
        "pcie_replay_counter": rocml.smi_get_device_pci_replay_counter(device_id),
    }
    try:
        pcie_current_info["pcie_replay_bandwidth"] = rocml.smi_get_device_pci_replay_counter(device_id)
    except Exception as e:
        pcie_current_info["pcie_replay_bandwidth"] = str(e)
    
    try:
        pcie_current_info["pcie_replay_throughput"] = rocml.smi_get_device_pci_replay_counter(device_id)
    except Exception as e:
        pcie_current_info["pcie_replay_throughput"] = str(e)

    return pcie_current_info


def get_memory_info(device_id):
    mem_info = {}
    try:
        mem_info.update(
            {
                "rocm_memory_total": rocml.smi_get_device_memory_total(device_id) / (1024**2),
                "rocm_memory_used": rocml.smi_get_device_memory_used(device_id) / (1024**2),
                "rocm_memory_busy": rocml.smi_get_device_memory_busy(device_id) / (1024**2),
                "rocm_memory_partition": rocml.smi_get_device_memory_partition(device_id),
            }
        )
        return mem_info
    except Exception as e:
        print(
            f"ROCm memory extraction attempt failed, falling back to legacy parsing. Error: {e}"
        )

def get_rocm_info():
    rocml.smi_initialize()

    device_count = rocml.smi_get_device_count()
    devices = [rocml.smi_get_device_name(device_index) for device_index in range(device_count)]

    
    # extra_attrs = get_extra_gpu_attributes()

    # if len(devices) - 1 != device_count and len(extra_attrs) != device_count:
    #     raise ValueError(
    #         f"Mismatch in gpu counts b/w backend, got {len(devices)}, {device_count} & {len(extra_attrs)}."
    #     )
    # devices = devices[1:]

    # def extract_value(pattern, text, default="N/A"):
    #     match = re.search(pattern, text)
    #     return match.group(1).strip() if match else default

    rocm_info = []
    try:
        for device_idx in range(device_count):
            rocm_info.append(
                {
                    "device_idx": device_idx,
                    "device_id": rocml.smi_get_device_id(device_idx),
                    "device_uuid": rocml.smi_get_device_uuid(device_idx),
                    "device_unique_id": rocml.smi_get_device_unique_id(device_idx),
                    "product_name": rocml.smi_get_device_name(device_idx).replace(" ", "_").upper(),
                    "kernel_version": rocml.smi_get_kernel_version(),
                    # "product_brand": pynvml.nvmlDeviceGetBrand(handle),
                    # "architecture": extract_value(
                    #     r"Product Architecture\s+:\s+(.+)", devices[device_idx]
                    # ),
                    # "multi_gpu_board": pynvml.nvmlDeviceGetMultiGpuBoard(handle),
                    # "virtualization_mode": extract_value(
                    #     r"Virtualization Mode\s+:\s+(.+)", devices[device_idx]
                    # ),
                    # "host_vgpu_mode": extract_value(
                    #     r"Host VGPU Mode\s+:\s+(.+)", devices[device_idx]
                    # ),
                    # **extra_attrs[device_idx],
                    **get_cores_info(device_idx),
                    **get_temp_and_power_info(device_idx),
                    **get_memory_info(device_idx),
                    **get_pci_info(device_idx),
                }
            )
        return rocm_info, None
    except Exception as e:
        print(f"GPU spec extractismi_outputon failed with {e}")
    finally:
        rocml.smi_shutdown()



def create_rocm_config():

    dev_info, _ = get_rocm_info()
    print(dev_info[0])
    print(dev_info[0]['product_name'])
    
    device_info = DeviceInfo[dev_info[0]['product_name']].value
    device_config = {
        "name": device_info.name,
        "mem_per_GPU_in_GB": device_info.mem_per_GPU_in_GB,
        "hbm_bandwidth_in_GB_per_sec": device_info.hbm_bandwidth_in_GB_per_sec,
        "intra_node_bandwidth_in_GB_per_sec": device_info.intra_node_bandwidth_in_GB_per_sec,
        "intra_node_min_message_latency": device_info.intra_node_min_message_latency,
        "peak_fp16_TFLOPS": device_info.peak_fp16_TFLOPS,
        "peak_i8_TFLOPS": device_info.peak_i8_TFLOPS,
        "peak_i4_TFLOPS": device_info.peak_i4_TFLOPS,
        "inter_node_bandwidth_in_GB_per_sec": device_info.inter_node_bandwidth_in_GB_per_sec,
        "available_count": len(dev_info)
    }
    return device_config, dev_info