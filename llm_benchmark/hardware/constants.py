from enum import Enum


class FlopsPerCycle(float, Enum):
    AMX = 1024.0
    AVX512 = 32.0
    AVX2 = 16.0
    AVX = 8.0
    DEFAULT = 4.0
class GPUInfo:
    def __init__(self, name, mem_per_GPU_in_GB, hbm_bandwidth_in_GB_per_sec,
                 intra_node_bandwidth_in_GB_per_sec, intra_node_min_message_latency,
                 peak_fp16_TFLOPS, peak_i8_TFLOPS, peak_i4_TFLOPS,
                 inter_node_bandwidth_in_GB_per_sec):
        self.name = name
        self.mem_per_GPU_in_GB = mem_per_GPU_in_GB
        self.hbm_bandwidth_in_GB_per_sec = hbm_bandwidth_in_GB_per_sec
        self.intra_node_bandwidth_in_GB_per_sec = intra_node_bandwidth_in_GB_per_sec
        self.intra_node_min_message_latency = intra_node_min_message_latency
        self.peak_fp16_TFLOPS = peak_fp16_TFLOPS
        self.peak_i8_TFLOPS = peak_i8_TFLOPS
        self.peak_i4_TFLOPS = peak_i4_TFLOPS
        self.inter_node_bandwidth_in_GB_per_sec = inter_node_bandwidth_in_GB_per_sec


class DeviceInfo(Enum):
    NVIDIA_A100_80GB_PCIE = GPUInfo(
        name="NVIDIA_A100_80GB_PCIE",
        mem_per_GPU_in_GB=80,
        hbm_bandwidth_in_GB_per_sec=1935,
        intra_node_bandwidth_in_GB_per_sec=300,
        intra_node_min_message_latency=8e-06,
        peak_fp16_TFLOPS=312,
        peak_i8_TFLOPS=624,
        peak_i4_TFLOPS=1248,
        inter_node_bandwidth_in_GB_per_sec=200
    )
    NVIDIA_A100_SXM4_80GB = GPUInfo(
        name="NVIDIA_A100_SXM4_80GB",
        mem_per_GPU_in_GB=80,
        hbm_bandwidth_in_GB_per_sec=1935,
        intra_node_bandwidth_in_GB_per_sec=300,
        intra_node_min_message_latency=8e-06,
        peak_fp16_TFLOPS=312,
        peak_i8_TFLOPS=624,
        peak_i4_TFLOPS=1248,
        inter_node_bandwidth_in_GB_per_sec=200
    )
    TESLA_V100_PCIE_16GB = GPUInfo(
        name="TESLA_V100_PCIE_16GB",
        mem_per_GPU_in_GB=16,
        hbm_bandwidth_in_GB_per_sec=900,
        intra_node_bandwidth_in_GB_per_sec=32,
        intra_node_min_message_latency=8e-06,
        peak_fp16_TFLOPS=30,
        peak_i8_TFLOPS=30,
        peak_i4_TFLOPS=1248,
        inter_node_bandwidth_in_GB_per_sec=200
    )
    HL_225 = GPUInfo(
        name="HL_225",
        mem_per_GPU_in_GB=96,
        hbm_bandwidth_in_GB_per_sec=2450,
        intra_node_bandwidth_in_GB_per_sec=300,
        intra_node_min_message_latency=8e-06,
        peak_fp16_TFLOPS=432,
        peak_i8_TFLOPS=864,
        peak_i4_TFLOPS=1728,
        inter_node_bandwidth_in_GB_per_sec=200
    )
    GAUDI2 = GPUInfo(
        name="GAUDI2",
        mem_per_GPU_in_GB=96,
        hbm_bandwidth_in_GB_per_sec=2450,
        intra_node_bandwidth_in_GB_per_sec=300,
        intra_node_min_message_latency=8e-06,
        peak_fp16_TFLOPS=432,
        peak_i8_TFLOPS=864,
        peak_i4_TFLOPS=1728,
        inter_node_bandwidth_in_GB_per_sec=200
    )