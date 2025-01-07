import os
import re
import psutil
import subprocess
from typing import Optional

from .constants import FlopsPerCycle

def get_numa_info():
    numa_info = []

    # Get the number of NUMA nodes
    node_dir = "/sys/devices/system/node/"
    if not os.path.exists(node_dir):
        raise EnvironmentError("NUMA information not available on this system.")
    
    node_dirs = [d for d in os.listdir(node_dir) if d.startswith("node")]

    for node in node_dirs:
        node_path = os.path.join(node_dir, node)
        cpu_list_file = os.path.join(node_path, "cpulist")

        # Read the list of CPUs assigned to this NUMA node
        with open(cpu_list_file, 'r') as f:
            cpu_list = f.read().strip()
        
        numa_info.append({
            "name": node,
            "cpus": cpu_list,
        })

    return numa_info

def get_memcpy_bandwidth(numa_count):
    try:
        # Run the command
        membind_value = 1 if numa_count > 1 else 0
        command = ["numactl", "--cpunodebind=0", "--membind={}".format(membind_value), "mbw", "1000"]
        
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check if the command ran successfully
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with error: {result.stderr}")

        # Extract the line with MEMCPY method
        output = result.stdout
        memcpy_line = None

        for line in output.splitlines():
            if "MEMCPY" in line:
                memcpy_line = line
        
        if not memcpy_line:
            raise ValueError("MEMCPY result not found in output")

        # Use regular expression to extract the average bandwidth (MB/s)
        match = re.search(r'(\d+\.\d+) MiB/s', memcpy_line)
        if match:
            avg_bandwidth = float(match.group(1)) / 1024
            print(f"Average Bandwidth (MEMCPY): {avg_bandwidth} GB/s")
            return avg_bandwidth
        else:
            raise ValueError("Failed to extract bandwidth from MEMCPY result")

    except Exception as e:
        print(f"Error: {e}")

def get_memory_bandwidth():
    """Fetch memory information for bus width, clock speed, and data rate multiplier."""

    try:
        try:
            cmd = "sudo dmidecode --type memory"
            with open(os.devnull, 'w') as devnull:
                output = subprocess.run(cmd.split(), stderr=devnull, stdout=subprocess.PIPE, text=True)
        except Exception as e:
            cmd = "dmidecode --type memory"
            with open(os.devnull, 'w') as devnull:
                output = subprocess.run(cmd.split(), stderr=devnull, stdout=subprocess.PIPE, text=True)
        
        # Extracting bus width and clock speed
        bus_width_match = re.search(r'Width:\s*(\d+)', output.stdout)
        clock_speed_match = re.search(r'Speed:\s*(\d+)', output.stdout)
        memory_type_match = re.search(r'Type: (DDR[0-9]*)', output.stdout)
        
        if memory_type_match:
            memory_type = memory_type_match.group(1)
            
            if "DDR" in memory_type:
                data_rate_multiplier = 2
            elif "DDR2" in memory_type:
                data_rate_multiplier = 4
            elif "DDR3" in memory_type:
                data_rate_multiplier = 8
            elif "DDR4" in memory_type:
                data_rate_multiplier = 16
            elif "DDR5" in memory_type:
                data_rate_multiplier = 32
            else:
                data_rate_multiplier = 2
                # raise Exception("Unsupported memory type")
        else:
            data_rate_multiplier = 2
            # raise Exception("Could not find memory type")
        
        if bus_width_match and clock_speed_match:
            bus_width_bits = int(bus_width_match.group(1))
            clock_speed_mhz = int(clock_speed_match.group(1))
        else:
            # Default values
            bus_width_bits = 64  # Default bus width in bits
            clock_speed_mhz = 1600  # Default clock speed in MHz
            print("Using default values for bus width and clock speed.")

        bandwidth_mb_per_sec = bus_width_bits * clock_speed_mhz * data_rate_multiplier / 8
        bandwidth_gb_per_sec = bandwidth_mb_per_sec / 1024
        return bandwidth_gb_per_sec
    except Exception as e:
        print("Error executing command:", e)
        return None

def get_flops_per_cycle(isa_info):
    flops_per_cycle = FlopsPerCycle.DEFAULT
    if isa_info.get("AMX"):
        # Estimate for AMX (assume 1024 FLOPs per cycle per core for AMX)
        flops_per_cycle = FlopsPerCycle.AMX
    elif isa_info.get("AVX512"):
        # Estimate for AVX-512 (assume 32 FLOPs per cycle per core for AVX-512)
        flops_per_cycle = FlopsPerCycle.AVX512
    elif isa_info.get("AVX2"):
        # Estimate for AVX2 (assume 16 FLOPs per cycle per core for AVX2)
        flops_per_cycle = FlopsPerCycle.AVX2
    elif isa_info.get("AVX"):
        # Estimate for AVX (assume 8 FLOPs per cycle per core for AVX)
        flops_per_cycle = FlopsPerCycle.AVX
    else:
        # Assume basic floating point performance (4 FLOPs per cycle per core)
        flops_per_cycle = FlopsPerCycle.DEFAULT

    return flops_per_cycle.value


def get_flops_per_second(
    clock_speed_mhz: float, num_cores: int, flops_per_cycle: float
) -> float:
    # Calculate the FLOPs
    flops = flops_per_cycle * clock_speed_mhz * num_cores * 1e6

    # Convert FLOPs to TFLOPs
    tflops = flops / 1e12

    return tflops


def get_cache_and_isa_info():
    lscpu_info = {}

    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True)
        output = result.stdout

        lscpu_info["AVX"] = False
        lscpu_info["AMX"] = False

        for line in output.splitlines():
            if "L1d cache" in line:
                lscpu_info["l1_data_cache_size"] = line.split(":")[1].strip()
            if "L1i cache" in line:
                lscpu_info["l1_instruction_cache_size"] = line.split(":")[1].strip()
            if "L2 cache" in line:
                lscpu_info["l2_cache_size"] = line.split(":")[1].strip()
            if "L3 cache" in line:
                lscpu_info["l3_cache_size"] = line.split(":")[1].strip()
            if "Model name" in line:
                lscpu_info["model_name"] = line.split(":")[1].strip()

            if "Flags" in line or "flags" in line:
                flags = line.split(":")[1].strip().split()
                if "avx" in flags or "avx2" in flags or "avx512f" in flags:
                    lscpu_info["AVX"] = True
                if "amx_bf16" in flags or "amx_tile" in flags or "amx_int8" in flags:
                    lscpu_info["AMX"] = True
    except Exception as e:
        print(f"CPU extraction failed with {str(e)}")
    finally:
        return lscpu_info


def get_cores_and_mem_info(pid: Optional[int] = None, current_only: bool = False):
    cm_info = {}

    if pid is not None:
        try:
            proc = psutil.Process(pid)

            cm_info["cpu_utilization"] = proc.cpu_percent(interval=1.0)
            per_core_util = psutil.cpu_percent(interval=None, percpu=True)
            cpu_core_affinity = proc.cpu_affinity()
            cm_info["cpu_core_affinity"] = ",".join(map(str, cpu_core_affinity))
            cm_info["cpu_utilization_per_core"] = ",".join(
                map(str, [per_core_util[core] for core in cpu_core_affinity])
            )

            cpu_freq = psutil.cpu_freq(percpu=True)
            cm_info["cpu_freq_current_per_core"] = ",".join(
                map(str, [cpu_freq[core].current for core in cpu_core_affinity])
            )

            cm_info["cpu_memory_used"] = proc.memory_info().rss
            cm_info["cpu_memory_utilization"] = proc.memory_percent()
            mem_info = psutil.virtual_memory()
        except psutil.NoSuchProcess as e:
            print(f"CPU extraction failed with {str(e)}")
            return cm_info
    else:
        cm_info["cpu_utilization"] = psutil.cpu_percent(interval=1.0)

        mem_info = psutil.virtual_memory()
        cm_info["cpu_memory_used"] = mem_info.used
        cm_info["cpu_memory_utilization"] = mem_info.percent

    cm_info["cpu_memory_used"] = cm_info["cpu_memory_used"] / (1024 ** 3)
    cm_info["cpu_memory_total"] = mem_info.total / (1024 ** 3)
    cm_info["cpu_memory_available"] = mem_info.available / (1024 ** 3)

    cpu_freq = psutil.cpu_freq()
    avg_load = psutil.getloadavg()

    cm_info.update(
        {
            "cpu_freq_current": cpu_freq.current,
            "cpu_load_1min": avg_load[0],
            "cpu_load_5min": avg_load[1],
            "cpu_load_15min": avg_load[2],
        }
    )

    if current_only:
        return cm_info

    num_physical_cores = psutil.cpu_count(logical=False)
    num_virtual_cores = psutil.cpu_count(logical=True)
    #get numa node info
    numa_info = get_numa_info()
    
    cm_info.update(
        {
            "cpu_freq_min": cpu_freq.min,
            "cpu_freq_max": cpu_freq.max,
            "num_physical_cores": num_physical_cores,
            "num_virtual_cores": num_virtual_cores,
            "threads_per_core": num_virtual_cores / num_physical_cores,
            "numa_count": len(numa_info),
            "numa_cores": "|".join([numa_info[i]["cpus"] for i in range(len(numa_info))]),
            "memcpy_bandwidth": get_memcpy_bandwidth(len(numa_info)),
            "mem_bandwidth_GBs": get_memory_bandwidth(),
        }
    )

    return cm_info


def get_temp_and_power_info(current_only: bool = False):
    temps = psutil.sensors_temperatures()
    core_temp = temps.get("coretemp", [])

    tp_info = {
        "cpu_temp_current": sum(int(temp.current) for temp in core_temp)
        / max(1, len(core_temp)),
        "cpu_temp_high": sum(int(temp.high) for temp in core_temp)
        / max(1, len(core_temp)),
    }

    return tp_info


def get_cpu_info():
    # cpu_info = subprocess.check_output(["lscpu"], text=True)
    # if "Intel" in cpu_info:
    #     print("CPU is Intel")
    # else:
    #     print("CPU is not Intel")
    #     return {}
    
    info = get_cores_and_mem_info()
    info.update(get_temp_and_power_info())

    cache_isa_info = get_cache_and_isa_info()

    info["flops_per_cycle"] = get_flops_per_cycle(cache_isa_info)
    info["tflops_current"] = get_flops_per_second(
        info["cpu_freq_current"], info["num_physical_cores"], info["flops_per_cycle"]
    )
    info["tflops_min"] = get_flops_per_second(
        info["cpu_freq_min"], info["num_physical_cores"], info["flops_per_cycle"]
    )
    info["tflops_max"] = get_flops_per_second(
        info["cpu_freq_max"], info["num_physical_cores"], info["flops_per_cycle"]
    )

    return {**info, **cache_isa_info}


def create_cpu_config():

    dev_info = get_cpu_info()
    
    device_config = {}
    device_config["name"] = dev_info['model_name']
    device_config["mem_per_GPU_in_GB"] = dev_info["cpu_memory_total"] / dev_info["numa_count"]
    device_config["hbm_bandwidth_in_GB_per_sec"] = dev_info["mem_bandwidth_GBs"]
    device_config["intra_node_bandwidth_in_GB_per_sec"] = dev_info["memcpy_bandwidth"]
    device_config["intra_node_min_message_latency"] = 8e-06
    device_config["peak_fp16_TFLOPS"] = dev_info['tflops_max'] if dev_info['tflops_max'] > 0.0 else dev_info['tflops_current']
    device_config["peak_i8_TFLOPS"] = 250
    device_config["peak_i4_TFLOPS"] = 500
    device_config["inter_node_bandwidth_in_GB_per_sec"] = 200
    device_config["available_count"] = dev_info["numa_count"]
    
    return device_config, dev_info
