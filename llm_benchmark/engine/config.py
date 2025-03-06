

class EngineConfig:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1, # set 0 if not supported
        pipeline_parallel_size: int = 1, # set 0 if not supported
        kv_cache_dtype: str = "float16",
        distributed_backend: str = "mp", # mp, ray
        gpu_memory_utilization: float = 0.8,
        swap_space: int = 4, # in GB
        cpu_kv_cache_size: int = 40, # in GB
        block_size: int = 16,
        max_num_batched_tokens: int = 1024,
        max_num_seqs: int = 1024,
        enable_prefix_cache: bool = True,
        enable_chunked_prefill: bool = False,
        chunked_prefill_size: int = 8192,
        max_prefill_tokens: int = 16384,
        disable_sliding_window: bool = False,
        quantization: str = None, # fp16, int8, int4, etc.
        rope_scaling: dict = None, # {type: linear, factor: 1.0},
        enforce_eager: bool = False,
        max_seq_len_to_capture: int = 4096,
        num_scheduler_steps: int = 1,
        scheduler_delay_factor: float = 0.0,
        schedule_policy: str = "lpm", # "lpm", "random", "fcfs", "dfs-weight"
        schedule_conservativeness: float = 0.0,
        stream_interval: float = 0.0,
        attention_backend: str = "flash_attn", # flash_attn, flash_infer, torch_sdpa
        sampling_backend: str = "flashinfer", # "flashinfer", "pytorch"
        enable_torch_compile: bool = False,
        hpu_prompt_bs: tuple = None, # (min, max, step)
        hpu_decode_bs: tuple = None, # (min, max, step)
        hpu_prompt_seq: tuple = None, # (min, max, step)
        hpu_decode_block: tuple = None, # (min, max, step)
    ):

        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.kv_cache_dtype = kv_cache_dtype
        self.distributed_backend = distributed_backend
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space = swap_space
        self.cpu_kv_cache_size = cpu_kv_cache_size
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.enable_prefix_cache = enable_prefix_cache
        self.enable_chunked_prefill = enable_chunked_prefill
        self.chunked_prefill_size = chunked_prefill_size
        self.max_prefill_tokens = max_prefill_tokens
        self.disable_sliding_window = disable_sliding_window
        self.quantization = quantization
        self.rope_scaling = rope_scaling
        self.enforce_eager = enforce_eager
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.num_scheduler_steps = num_scheduler_steps
        self.scheduler_delay_factor = scheduler_delay_factor
        self.schedule_policy = schedule_policy
        self.schedule_conservativeness = schedule_conservativeness
        self.stream_interval = stream_interval
        self.attention_backend = attention_backend
        self.sampling_backend = sampling_backend
        self.enable_torch_compile = enable_torch_compile
        self.hpu_prompt_bs = hpu_prompt_bs
        self.hpu_decode_bs = hpu_decode_bs
        self.hpu_prompt_seq = hpu_prompt_seq
        self.hpu_decode_block = hpu_decode_block
        
    def get_config(self):
        return self.__dict__
        
        
def get_config_from_vllm(config: dict, envs: dict):
    hpu_prompt_bs = (envs.get("VLLM_PROMPT_BS_BUCKET_MIN", None),
                     envs.get("VLLM_PROMPT_BS_BUCKET_MAX", None),
                     envs.get("VLLM_PROMPT_BS_BUCKET_STEP", None))
    hpu_decode_bs = (envs.get("VLLM_DECODE_BS_BUCKET_MIN", None),
                     envs.get("VLLM_DECODE_BS_BUCKET_MAX", None),
                     envs.get("VLLM_DECODE_BS_BUCKET_STEP", None))
    hpu_prompt_seq = (envs.get("VLLM_PROMPT_SEQ_BUCKET_MIN", None),
                      envs.get("VLLM_PROMPT_SEQ_BUCKET_MAX", None),
                      envs.get("VLLM_PROMPT_SEQ_BUCKET_STEP", None))
    hpu_decode_block = (envs.get("VLLM_DECODE_BLOCK_BUCKET_MIN", None),
                         envs.get("VLLM_DECODE_BLOCK_BUCKET_MAX", None),
                         envs.get("VLLM_DECODE_BLOCK_BUCKET_STEP", None))
    return EngineConfig(
        model_name=config.get("model"),
        tensor_parallel_size=config.get("tensor_parallel_size"),
        pipeline_parallel_size=config.get("pipeline_parallel_size"),
        kv_cache_dtype=config.get("kv_cache_dtype"),
        distributed_backend=config.get("distributed_executor_backend", "mp"),
        gpu_memory_utilization=config.get("gpu_memory_utilization"),
        swap_space=config.get("swap_space"),
        cpu_kv_cache_size=envs.get("VLLM_CPU_KVCACHE_SPACE", 4),
        block_size=config.get("block_size"),
        max_num_batched_tokens=config.get("max_num_batched_tokens"),
        max_num_seqs=config.get("max_num_seqs"),
        enable_prefix_cache=config.get("enable_prefix_caching", False),
        enable_chunked_prefill=config.get("enable_chunked_prefill", False),
        chunked_prefill_size=None,
        max_prefill_tokens=None,
        disable_sliding_window=config.get("disable_sliding_window", False),
        quantization=config.get("quantization"),
        rope_scaling=config.get("rope_scaling", None),
        enforce_eager=config.get("enforce_eager", False),
        max_seq_len_to_capture=config.get("max_seq_len_to_capture"),
        num_scheduler_steps=config.get("num_scheduler_steps"),
        scheduler_delay_factor=config.get("scheduler_delay_factor"),
        schedule_policy=None,
        schedule_conservativeness=None,
        stream_interval=None,
        attention_backend=envs.get("VLLM_ATTENTION_BACKEND"),
        sampling_backend=None,
        enable_torch_compile=None,
        hpu_prompt_bs=hpu_prompt_bs,
        hpu_decode_bs=hpu_decode_bs,
        hpu_prompt_seq=hpu_prompt_seq,
        hpu_decode_block=hpu_decode_block,
    )

def get_config_from_sglang(config: dict, envs: dict):
    return EngineConfig(
        model_name=config.get("model_path"),
        tensor_parallel_size=config.get("tp_size"),
        pipeline_parallel_size=None,
        kv_cache_dtype=config.get("kv_cache_dtype"),
        distributed_backend=None,
        gpu_memory_utilization=None,
        swap_space=None,
        cpu_kv_cache_size=None,
        block_size=None,
        max_num_batched_tokens=config.get("max_total_tokens"),
        max_num_seqs=config.get("max_running_requests"),
        enable_prefix_cache=True,
        enable_chunked_prefill=True,
        chunked_prefill_size=config.get("chunked_prefill_size"),
        max_prefill_tokens=config.get("max_prefill_tokens"),
        disable_sliding_window=None,
        quantization=config.get("quantization"),
        rope_scaling=None,
        enforce_eager=None,
        max_seq_len_to_capture=None,
        num_scheduler_steps=config.get("num_continuous_decode_steps"),
        scheduler_delay_factor=None,
        schedule_policy=config.get("schedule_policy"),
        schedule_conservativeness=config.get("schedule_conservativeness"),
        stream_interval=config.get("stream_interval"),
        attention_backend=config.get("attention_backend"),
        sampling_backend=config.get("sampling_backend"),
        enable_torch_compile=config.get("enable_torch_compile"),
        hpu_prompt_bs=None,
        hpu_decode_bs=None,
        hpu_prompt_seq=None,
        hpu_decode_block=None,
    )

def get_config_from_litellm_proxy(config: dict, envs: dict):
    return EngineConfig(
        model_name=config.get("model"),
        tensor_parallel_size=config.get("tp_size"),
        pipeline_parallel_size=None,
        kv_cache_dtype=config.get("kv_cache_dtype"),
        distributed_backend=None,
        gpu_memory_utilization=None,
        swap_space=None,
        cpu_kv_cache_size=None,
        block_size=None,
        max_num_batched_tokens=config.get("max_total_tokens"),
        max_num_seqs=config.get("max_running_requests"),
        enable_prefix_cache=True,
        enable_chunked_prefill=True,
        chunked_prefill_size=config.get("chunked_prefill_size"),
        max_prefill_tokens=config.get("max_prefill_tokens"),
        disable_sliding_window=None,
        quantization=config.get("quantization"),
        rope_scaling=None,
        enforce_eager=None,
        max_seq_len_to_capture=None,
        num_scheduler_steps=config.get("num_continuous_decode_steps"),
        scheduler_delay_factor=None,
        schedule_policy=config.get("schedule_policy"),
        schedule_conservativeness=config.get("schedule_conservativeness"),
        stream_interval=config.get("stream_interval"),
        attention_backend=config.get("attention_backend"),
        sampling_backend=config.get("sampling_backend"),
        enable_torch_compile=config.get("enable_torch_compile"),
        hpu_prompt_bs=None,
        hpu_decode_bs=None,
        hpu_prompt_seq=None,
        hpu_decode_block=None,
    )

def get_engine_config(engine: str, config: dict, envs: dict):
    if engine == "vllm":
        return get_config_from_vllm(config, envs)
    elif engine == "sglang":
        return get_config_from_sglang(config, envs)
    elif engine == "litellm_proxy":
        return get_config_from_litellm_proxy(config, envs)
    else:
        raise ValueError(f"Engine {engine} not supported")
