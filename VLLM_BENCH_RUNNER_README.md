# vLLM Bench Runner

A script to automate running `vllm bench serve` with multiple configuration combinations from a YAML file.

## Overview

`vllm_bench_runner.py` reads a YAML configuration file and executes the native `vllm bench serve` command for all combinations of benchmark parameters. This is useful for:
- Testing multiple input/output token lengths
- Benchmarking different concurrency levels
- Running systematic performance tests across parameter spaces

## Prerequisites

- vLLM must be installed and available in your PATH
- A running vLLM server (the script assumes the server is already running)

## Installation

No installation required. The script is standalone and uses only standard Python libraries plus PyYAML.

```bash
# If PyYAML is not installed
pip install pyyaml
```

## Usage

### Basic Usage

```bash
python3 vllm_bench_runner.py --config example/vllm-bench-config.yaml
```

### Resume from Checkpoint

If a benchmark run is interrupted, you can resume from where it left off:

```bash
python3 vllm_bench_runner.py --config example/vllm-bench-config.yaml --resume
```

### Specify Custom Checkpoint Path

```bash
python3 vllm_bench_runner.py --config my-config.yaml --checkpoint /path/to/checkpoint.json
```

### Set Custom Timeout

Each benchmark has a default timeout of 600 seconds (10 minutes). You can adjust this:

```bash
# Set timeout to 30 minutes (1800 seconds)
python3 vllm_bench_runner.py --config my-config.yaml --timeout 1800

# Set timeout to 5 minutes (300 seconds) for quick tests
python3 vllm_bench_runner.py --config my-config.yaml --timeout 300
```

## Configuration File Format

Create a YAML file with the following structure:

```yaml
# Model to benchmark (required)
model: meta-llama/Llama-3.1-8B-Instruct

# Base URL of the running vLLM server (required)
base_url: http://localhost:8000

# Backend type (optional, default: openai)
backend: openai

# Dataset name (optional, default: random)
dataset_name: random

# Random seed for reproducibility (optional, default: 1234)
seed: 1234

# ===== TOKEN LENGTH CONFIGURATION =====
# You can specify token lengths in TWO ways:

# Option 1: Use token_pairs (similar to auto_benchmark)
# Each pair specifies "input_tokens,output_tokens"
# This creates EXACTLY the specified pairs (no cartesian product)
token_pairs:
  - 50,200
  - 100,150
  - 200,300
  - 400,600
  - 800,1200

# Option 2: Use separate input/output lists (creates all combinations)
# This creates the cartesian product: input_len × output_len
# Comment out these if using token_pairs above
random_input_len:
  - 400
  - 800
  - 1200

random_output_len:
  - 600
  - 1200

# Concurrency levels to test (can be single value or list)
# Note: num_prompts is automatically set equal to max_concurrency
max_concurrency:
  - 1
  - 8
  - 16
  - 32
  - 64
  - 128
  - 256

# Directory to save results (optional, default: ./vllm_bench_results)
result_dir: ./vllm_bench_results
```

### Token Length Configuration Options

**Option 1: token_pairs (Recommended)**
- Similar to `auto_benchmark.py` format
- Specifies exact input/output combinations as `"input,output"`
- No cartesian product - creates EXACTLY the pairs you specify
- Example: 5 pairs × 7 concurrency = 35 total configurations

**Option 2: random_input_len × random_output_len**
- Separate lists for input and output lengths
- Creates ALL combinations (cartesian product)
- Example: 3 inputs × 2 outputs × 7 concurrency = 42 total configurations

**Priority**: If `token_pairs` is present, it takes priority and `random_input_len`/`random_output_len` are ignored.

### Multi-Model Benchmarking

You can benchmark multiple models by providing a list:

```yaml
model:
  - meta-llama/Llama-3.1-8B-Instruct
  - mistralai/Mistral-7B-Instruct-v0.3
```

### Multi-Server Benchmarking

You can benchmark multiple servers by providing a list:

```yaml
base_url:
  - http://localhost:8000
  - http://localhost:8001
```

## Generated Command

For each configuration combination, the script generates and executes a command like:

```bash
vllm bench serve \
  --backend openai \
  --base-url http://localhost:8000 \
  --dataset-name random \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --seed 1234 \
  --num-prompts 256 \
  --random-input-len 400 \
  --random-output-len 600 \
  --max-concurrency 256 \
  --save-result \
  --result-dir ./vllm_bench_results
```

## Output

### Results Directory

Results are saved in the `result_dir` specified in the config (default: `./vllm_bench_results/`).

**Individual JSON Files**

Each benchmark run creates a JSON file with the format:
```
{backend}-{request_rate}qps-concurrency{max_concurrency}-{model}-{timestamp}.json
```

Example result file content:
```json
{
  "date": "20251005-122334",
  "endpoint_type": "openai",
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "num_prompts": 256,
  "max_concurrency": 256,
  "duration": 52.89,
  "completed": 256,
  "total_input_tokens": 102400,
  "total_output_tokens": 153600,
  "request_throughput": 4.84,
  "output_throughput": 2904.76,
  "mean_ttft_ms": 95.91,
  "median_ttft_ms": 95.91,
  "p99_ttft_ms": 95.91,
  "mean_tpot_ms": 88.14,
  "median_tpot_ms": 88.14,
  "p99_tpot_ms": 88.14,
  "mean_itl_ms": 88.14,
  "median_itl_ms": 87.91,
  "p99_itl_ms": 92.94
}
```

**Aggregated CSV Summary**

After all benchmarks complete, the script automatically creates an aggregated CSV file:
```
{result_dir}/benchmark_summary.csv
```

This CSV contains one row per benchmark configuration with columns:
- **Configuration**: model, input_tokens, output_tokens, concurrency, seed
- **Performance**: duration, completed, request_throughput, output_throughput, total_token_throughput
- **Latency Metrics**: mean_ttft_ms, median_ttft_ms, p99_ttft_ms (TTFT, TPOT, ITL, E2EL)
- **Status**: status (completed/failed/timeout), start_time, end_time, error
- **Reference**: result_file (name of the corresponding JSON file)

**Note**: The `num_prompts` parameter is automatically set equal to `max_concurrency` for each benchmark run.

This CSV makes it easy to:
- Compare results across different configurations
- Import into Excel, Pandas, or other analysis tools
- Generate plots and reports
- Track performance trends

### Checkpoint Files

Checkpoint files are saved to `{result_dir}/checkpoints/checkpoint_{timestamp}.json` and contain:
- Configuration for each benchmark run
- Status (pending/running/completed/failed)
- Start and end timestamps
- Error messages (if failed)

This allows you to resume interrupted benchmark runs.

## Example Workflow

1. **Start your vLLM server:**
   ```bash
   vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
   ```

2. **Create your config file** (or use the example):
   ```bash
   cp example/vllm-bench-config.yaml my-benchmark.yaml
   # Edit my-benchmark.yaml as needed
   ```

3. **Run the benchmarks:**
   ```bash
   python3 vllm_bench_runner.py --config my-benchmark.yaml
   ```

4. **If interrupted, resume:**
   ```bash
   python3 vllm_bench_runner.py --config my-benchmark.yaml --resume
   ```

## Comparison with auto_benchmark.py

| Feature | vllm_bench_runner.py | auto_benchmark.py |
|---------|---------------------|-------------------|
| Server deployment | Assumes server running | Deploys with Docker |
| Benchmark tool | Native vLLM CLI | Python benchmark functions |
| Hardware profiling | No | Yes (optional) |
| Model profiling | No | Yes (optional) |
| Warmup runs | No | Yes |
| Checkpointing | Yes | Yes |
| Config format | YAML | YAML |
| Use case | Simple vLLM benchmarks | Complex multi-engine benchmarks |

## Troubleshooting

### "vllm: command not found"

Make sure vLLM is installed and available in your PATH:
```bash
pip install vllm
# Or activate your vLLM environment
source /path/to/vllm-env/bin/activate
```

### Server connection errors

Make sure your vLLM server is running and accessible at the URL specified in `base_url`.

Test connectivity:
```bash
curl http://localhost:8000/v1/models
```

### Benchmark hangs or times out

If benchmarks are timing out, you have several options:

1. **Increase the timeout** (default is 600 seconds):
   ```bash
   python3 vllm_bench_runner.py --config my-config.yaml --timeout 1800
   ```

2. **Check if server is responding**: The script will wait for the server to respond. If the server is not running or not reachable, the benchmark will timeout.

3. **Resume from checkpoint**: If a benchmark times out, you can resume from where it left off:
   ```bash
   python3 vllm_bench_runner.py --config my-config.yaml --resume
   ```

### Permission errors

Make sure the result directory is writable:
```bash
chmod +x vllm_bench_runner.py
mkdir -p ./vllm_bench_results
```
