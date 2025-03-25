import re
import subprocess

try:
    import pynvml
except ImportError:
    print(
        "pynvml import failed, the system either don't have cuda support or is not configured properly."
    )

from llm_benchmark.hardware.constants import DeviceInfo

def filter_nvidia_smi(filters: list = None):
    result = subprocess.run(
        ["nvidia-smi", "-q"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr)

    if filters is None or not len(filters):
        return result.stdout

    lines = result.stdout.splitlines()
    gpu_info_list = []
    gpu_info = {}
    for line in lines:
        line = line.strip()
        if any(key.lower() in line.lower() for key in filters):
            # Extract key-value pairs from the relevant lines
            match = re.match(r"(.+?)\s*:\s*(.+)", line)
            if match:
                key = match.group(1).strip().lower().replace(" ", "_").replace(".", "_")
                value = match.group(2).strip()
                if not key.startswith("gpu"):
                    key = f"gpu_{key}"

                if key in gpu_info:
                    gpu_info_list.append(gpu_info)
                    gpu_info = {}

                gpu_info[key] = value

    if len(gpu_info):
        gpu_info_list.append(gpu_info)

    return gpu_info_list


def get_extra_gpu_attributes():
    """Uses pycuda to extract additional GPU specs like CUDA cores and SM count."""
    import pycuda.driver as cuda
    import pycuda.autoinit

    num_gpus = cuda.Device.count()
    gpu_specs = []

    for gpu_id in range(num_gpus):
        device = cuda.Device(gpu_id)
        attributes = device.get_attributes()

        # CUDA cores and other properties depend on the number of SMs (Streaming Multiprocessors)
        sm_count = attributes[cuda.device_attribute.MULTIPROCESSOR_COUNT]

        gpu_info = {
            "sm_count": sm_count,
            "compute_capability": device.compute_capability(),
        }

        for key in (
            "MAX_THREADS_PER_BLOCK",
            "MAX_THREADS_PER_MULTIPROCESSOR",
            "MAX_BLOCKS_PER_MULTIPROCESSOR",
            "TOTAL_CONSTANT_MEMORY",
            "GPU_OVERLAP",
            "SHARED_MEMORY_PER_BLOCK",
            "MAX_SHARED_MEMORY_PER_BLOCK",
            "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
            "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
            "REGISTERS_PER_BLOCK",
            "MAX_REGISTERS_PER_BLOCK",
            "MAX_REGISTERS_PER_MULTIPROCESSOR",
            "INTEGRATED",
            "CAN_MAP_HOST_MEMORY",
            "COMPUTE_MODE",
            "CONCURRENT_KERNELS",
            "ECC_ENABLED",
            "PCI_BUS_ID",
            "PCI_DEVICE_ID",
            "TCC_DRIVER",
            "GLOBAL_MEMORY_BUS_WIDTH",
            "L2_CACHE_SIZE",
            "ASYNC_ENGINE_COUNT",
            "PCI_DOMAIN_ID",
            "STREAM_PRIORITIES_SUPPORTED",
            "GLOBAL_L1_CACHE_SUPPORTED",
            "LOCAL_L1_CACHE_SUPPORTED",
            "MANAGED_MEMORY",
            "MULTI_GPU_BOARD",
            "MULTI_GPU_BOARD_GROUP_ID",
            "SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO",
            "PAGEABLE_MEMORY_ACCESS",
            "CONCURRENT_MANAGED_ACCESS",
            "COMPUTE_PREEMPTION_SUPPORTED",
            "MAX_PERSISTING_L2_CACHE_SIZE",
        ):
            try:
                gpu_info[key.lower()] = attributes[getattr(cuda.device_attribute, key)]
            except AttributeError:
                pass

        gpu_specs.append(gpu_info)

    return gpu_specs


def get_cores_info(device_id, current_only: bool = False):
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    current_clocks = {
        "gpu_mem_clock_current": pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_MEM
        ),
        "gpu_sm_clock_current": pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_SM
        ),
        "gpu_graphics_clock_current": pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_GRAPHICS
        ),
        "gpu_utilization": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
    }

    if current_only:
        return current_clocks

    # Memory clocks
    supported_mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
    min_mem_clock = min(supported_mem_clocks) if supported_mem_clocks else None
    app_mem_clock = pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_MEM)
    max_mem_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)

    # SM clocks
    app_sm_clock = pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM)
    max_sm_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)

    # Graphics clocks
    supported_graphics_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(
        handle, max_mem_clock
    )
    min_graphics_clock = (
        min(supported_graphics_clocks) if supported_graphics_clocks else None
    )
    app_graphics_clock = pynvml.nvmlDeviceGetApplicationsClock(
        handle, pynvml.NVML_CLOCK_GRAPHICS
    )
    max_graphics_clock = pynvml.nvmlDeviceGetMaxClockInfo(
        handle, pynvml.NVML_CLOCK_GRAPHICS
    )

    clock_info = {
        **current_clocks,
        "gpu_sm_clock_app": app_sm_clock,
        "gpu_sm_clock_max": max_sm_clock,
        "gpu_memory_freq_min": min_mem_clock,
        "gpu_memory_freq_app": app_mem_clock,
        "gpu_memory_freq_max": max_mem_clock,
        "gpu_graphics_freq_min": min_graphics_clock,
        "gpu_graphics_freq_app": app_graphics_clock,
        "gpu_graphics_freq_max": max_graphics_clock,
    }

    try:
        clock_info["auto_boost_clocks_enabled"] = ",".join(
            map(str, pynvml.nvmlDeviceGetAutoBoostedClocksEnabled(handle))
        )
    except pynvml.NVMLError as e:
        clock_info["auto_boost_clocks_enabled"] = str(e)

    return clock_info


def get_throttle_reasons(device_id):
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    
    throttle_reasons = {
        pynvml.nvmlClocksThrottleReasonGpuIdle: "GPUIdle",
        pynvml.nvmlClocksThrottleReasonHwSlowdown: "HwSlowdown",
        pynvml.nvmlClocksThrottleReasonSwPowerCap: "SwPowerCap",
        pynvml.nvmlClocksThrottleReasonUserDefinedClocks: "UserDefinedClocks",
        pynvml.nvmlClocksThrottleReasonApplicationsClocksSetting: "ApplicationClocksSetting",
        pynvml.nvmlClocksThrottleReasonAll: "All",
        # pynvml.nvmlClocksThrottleReasonUnknown: "Unknown",
        pynvml.nvmlClocksThrottleReasonNone: None,
    }
    code = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
    return throttle_reasons.get(code, str(code))


def get_temp_and_power_info(device_id, current_only: bool = False):
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    tp_info = {
        "gpu_current_temperature": pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        ),
        "gpu_memory_current_temperature": None,
        "gpu_power_draw": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
        "gpu_power_mode": pynvml.nvmlDeviceGetPowerManagementMode(handle),
        "gpu_power_state": pynvml.nvmlDeviceGetPowerState(handle),
        "gpu_performance_state": pynvml.nvmlDeviceGetPerformanceState(handle),
    }

    try:
        tp_info["gpu_fan_speed"] = pynvml.nvmlDeviceGetFanSpeed(handle)
    except pynvml.NVMLError as e:
        tp_info["gpu_fan_speed"] = str(e)

    if current_only:
        # Only add memory current temperature if available
        temp_info = filter_nvidia_smi(["Memory Current Temp"])
        if len(temp_info) > device_id:
            tp_info["gpu_memory_current_temperature"] = temp_info[device_id].get(
                "gpu_memory_current_temp"
            )
            if tp_info["gpu_memory_current_temperature"] is not None:
                tp_info["gpu_memory_current_temperature"] = (
                    tp_info["gpu_memory_current_temperature"]
                    .strip()
                    .replace("C", "")
                    .strip()
                )
    else:
        tp_info.update(
            {
                "gpu_shutdown_temperature": pynvml.nvmlDeviceGetTemperatureThreshold(
                    handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN
                ),
                "gpu_slowdown_temperature": pynvml.nvmlDeviceGetTemperatureThreshold(
                    handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN
                ),
                "gpu_current_power_limit": pynvml.nvmlDeviceGetPowerManagementLimit(
                    handle
                )
                / 1000,
                "gpu_default_power_limit": pynvml.nvmlDeviceGetPowerManagementDefaultLimit(
                    handle
                )
                / 1000,
            }
        )

        power_limit_constraints = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(
            handle
        )
        if len(power_limit_constraints) >= 2:
            tp_info.update(
                {
                    "gpu_min_power_limit": power_limit_constraints[0] / 1000,
                    "gpu_max_power_limit": power_limit_constraints[1] / 1000,
                }
            )

        temp_info = filter_nvidia_smi(
            [
                "GPU T.Limit Temp",
                "GPU Max Operating Temp",
                "GPU Target Temperature",
                "Memory Current Temp",
                "Memory Max Operating Temp",
            ]
        )
        if len(temp_info) > device_id:
            tp_info.update(
                {
                    key: value.strip().replace("C", "").strip()
                    for key, value in temp_info[device_id].items()
                }
            )

    return tp_info


def get_pci_info(device_id, current_only: bool = False):
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    pcie_current_info = {
        "pcie_generation_current": pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle),
        "link_width_current": pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle),
        "tx_throughput": pynvml.nvmlDeviceGetPcieThroughput(
            handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
        ),
        "rx_throughput": pynvml.nvmlDeviceGetPcieThroughput(
            handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
        ),
    }

    if current_only:
        return pcie_current_info

    # PCIe device information that remains static
    pcie_static_info = {
        "pcie_generation_max": pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle),
        "link_width_max": pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle),
    }

    pci_info = {
        **pcie_current_info,
        **pcie_static_info,
        "pcie_host_max": None,
        "pcie_device_current": None,
        "pcie_device_max": None,
    }

    devices = re.split(r"\nGPU [0-9A-Fa-f:.]+", filter_nvidia_smi())
    if len(devices) <= device_id + 1:
        return pci_info

    lines = devices[device_id + 1].splitlines()
    capturing = False
    for i, line in enumerate(lines):
        line = line.strip()

        if line.startswith("PCIe Generation"):
            capturing = True
            continue

        if capturing:
            if line == "":
                break

        match = re.match(r"\s*Device Current\s*:\s*(\d+)", line)
        if match and pci_info["pcie_device_current"] is None:
            pci_info["pcie_device_current"] = match.group(1)

        match = re.match(r"\s*Device Max\s*:\s*(\d+)", line)
        if match and pci_info["pcie_device_max"] is None:
            pci_info["pcie_device_max"] = match.group(1)

        match = re.match(r"\s*Host Max\s*:\s*(N/A|\d+)", line)
        if match and pci_info["pcie_host_max"] is None:
            pci_info["pcie_host_max"] = match.group(1)

    return pci_info


def get_memory_info(device_id):
    mem_info = {}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

        # Get FB memory info
        fb_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_info.update(
            {
                "gpu_fb_memory_total": fb_memory_info.total / (1024**2),
                "gpu_fb_memory_used": fb_memory_info.used / (1024**2),
                "gpu_fb_memory_free": fb_memory_info.free / (1024**2),
            }
        )
        # Get BAR1 memory info
        bar1_memory_info = pynvml.nvmlDeviceGetBAR1MemoryInfo(handle)
        mem_info.update(
            {
                "gpu_bar1_memory_total": bar1_memory_info.bar1Total / (1024**2),
                "gpu_bar1_memory_used": bar1_memory_info.bar1Used / (1024**2),
                "gpu_bar1_memory_free": bar1_memory_info.bar1Free / (1024**2),
            }
        )
        return mem_info
    except Exception as e:
        print(
            f"GPU memory extraction attempt #1 failed, falling back to legacy parsing. Error: {e}"
        )

    devices = re.split(r"\nGPU [0-9A-Fa-f:.]+", filter_nvidia_smi())
    if len(devices) <= device_id + 1:
        return mem_info

    lines = devices[device_id + 1].splitlines()
    capturing_fb = False
    capturing_bar1 = False
    for line in lines:
        line = line.strip()

        # Start capturing FB Memory Usage
        if line.startswith("FB Memory Usage"):
            capturing_fb = True
            capturing_bar1 = False
            continue

        # Start capturing BAR1 Memory Usage
        if line.startswith("BAR1 Memory Usage"):
            capturing_bar1 = True
            capturing_fb = False
            continue

        if capturing_fb:
            if "Total" in line:
                match = re.match(r"Total\s*:\s*(\d+)\s*MiB", line)
                if match:
                    mem_info["gpu_fb_memory_total"] = match.group(1).strip()
            # elif "Reserved" in line:
            #     match = re.match(r"Reserved\s*:\s*(\d+)\s*MiB", line)
            #     if match:
            #         mem_info["gpu_fb_memory_reserved"] = match.group(1).strip()
            elif "Used" in line:
                match = re.match(r"Used\s*:\s*(\d+)\s*MiB", line)
                if match:
                    mem_info["gpu_fb_memory_used"] = match.group(1).strip()
            elif "Free" in line:
                match = re.match(r"Free\s*:\s*(\d+)\s*MiB", line)
                if match:
                    mem_info["gpu_fb_memory_free"] = match.group(1).strip()

        elif capturing_bar1:
            if "Total" in line:
                match = re.match(r"Total\s*:\s*(\d+)\s*MiB", line)
                if match:
                    mem_info["gpu_bar1_memory_total"] = match.group(1).strip()
            elif "Used" in line:
                match = re.match(r"Used\s*:\s*(\d+)\s*MiB", line)
                if match:
                    mem_info["gpu_bar1_memory_used"] = match.group(1).strip()
            elif "Free" in line:
                match = re.match(r"Free\s*:\s*(\d+)\s*MiB", line)
                if match:
                    mem_info["gpu_bar1_memory_free"] = match.group(1).strip()

    return mem_info


def get_gpu_info():
    pynvml.nvmlInit()

    smi_output = filter_nvidia_smi()
    devices = re.split(r"\nGPU [0-9A-Fa-f:.]+", smi_output)

    device_count = pynvml.nvmlDeviceGetCount()
    extra_attrs = get_extra_gpu_attributes()

    if len(devices) - 1 != device_count and len(extra_attrs) != device_count:
        raise ValueError(
            f"Mismatch in gpu counts b/w backend, got {len(devices)}, {device_count} & {len(extra_attrs)}."
        )
    devices = devices[1:]

    def extract_value(pattern, text, default="N/A"):
        match = re.search(pattern, text)
        return match.group(1).strip() if match else default

    gpu_info = []
    try:
        for device_id in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            gpu_info.append(
                {
                    "device_id": device_id,
                    "product_name": pynvml.nvmlDeviceGetName(handle).decode("utf-8").replace(" ", "_").replace("-", "_").upper(),
                    "product_brand": pynvml.nvmlDeviceGetBrand(handle),
                    "architecture": extract_value(
                        r"Product Architecture\s+:\s+(.+)", devices[device_id]
                    ),
                    "multi_gpu_board": pynvml.nvmlDeviceGetMultiGpuBoard(handle),
                    "virtualization_mode": extract_value(
                        r"Virtualization Mode\s+:\s+(.+)", devices[device_id]
                    ),
                    "host_vgpu_mode": extract_value(
                        r"Host VGPU Mode\s+:\s+(.+)", devices[device_id]
                    ),
                    **extra_attrs[device_id],
                    **get_cores_info(device_id),
                    **get_temp_and_power_info(device_id),
                    **get_memory_info(device_id),
                    **get_pci_info(device_id),
                }
            )

    except Exception as e:
        print(f"GPU spec extraction failed with {e}")
    finally:
        pynvml.nvmlShutdown()

    return gpu_info, smi_output


def create_cuda_config():

    dev_info, _ = get_gpu_info()
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
    }
    return device_config, dev_info