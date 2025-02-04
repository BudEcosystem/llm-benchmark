# LLM Benchmark

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Run

```bash
python auto_benchmark.py --model meta-llama/Meta-Llama-3-8B-Instruct --docker-image IMAGE_ID --input-tokens 100 --output-tokens 100 --concurrency 1
```


## Add profiler in model class

```python
from llm_benchmark.profiler.common.device_timer import DeviceTimer
from llm_benchmark.profiler.common.timer_stats_store import TimerStatsStore
from llm_benchmark.profiler.utils.record_function_tracer import RecordFunctionTracer

class LlamaForCausalLM(nn.Module, SupportsLoRA):
    def __init__():
        self.timer_stats_store = TimerStatsStore(
            profile_method="record_function"
        )
        self.record_function_tracer = RecordFunctionTracer("/root/results")
    
    def forward():
        with self.record_function_tracer:
            model_output = self.model(input_ids, positions, kv_caches, attn_metadata, intermediate_tensors)
    

class LlamaDecoderLayer(nn.Module):
    def __init__():
        
        self.input_layernorm_timer = DeviceTimer("input_layernorm")
        self.attn_timer = DeviceTimer("attn")
        self.mlp_timer = DeviceTimer("mlp")
        self.post_layernorm_timer = DeviceTimer("post_layernorm")
    
    def forward():
        # Self Attention
        with self.input_layernorm_timer:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual)
        with self.attn_timer:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
            )
        # Fully Connected
        with self.post_layernorm_timer:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
        with self.mlp_timer:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

```
