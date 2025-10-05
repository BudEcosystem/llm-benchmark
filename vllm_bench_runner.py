#!/usr/bin/env python3
"""
Script to run vLLM bench serve with configurations from a YAML file.
This script reads a YAML config file and executes vLLM's native bench serve command
for all combinations of parameters.

Example usage:
    python vllm_bench_runner.py --config vllm-bench-config.yaml
"""

import os
import sys
import csv
import glob
import yaml
import json
import hashlib
import argparse
import subprocess
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


def load_checkpoint(ckpt_path: str) -> Dict:
    """Load checkpoint file to resume from previous runs."""
    if os.path.isdir(ckpt_path):
        filepaths = sorted(Path(ckpt_path).iterdir(), key=lambda t: t.stat().st_mtime)
        ckpt_path = filepaths[-1] if len(filepaths) else None

    if not os.path.isfile(ckpt_path):
        print(f"No checkpoints found in {ckpt_path} for resuming.")
        return {}

    print(f"Resuming benchmarking from checkpoint {ckpt_path}.")
    with open(ckpt_path, "r") as fp:
        return json.load(fp)


def save_checkpoint(checkpoint: Dict, savepath: str):
    """Save checkpoint to file."""
    Path(savepath).parent.mkdir(exist_ok=True, parents=True)
    with open(savepath, "w") as fp:
        json.dump(checkpoint, fp, indent=4)


def generate_combinations(config_section: Dict) -> tuple:
    """Generate all combinations from array parameters in config."""
    fixed_params = {}
    array_params = {}

    for key, value in config_section.items():
        if isinstance(value, list):
            array_params[key] = value
        else:
            fixed_params[key] = value

    # Generate all possible combinations for array parameters
    if array_params:
        keys, values = zip(*array_params.items())
        combinations = list(itertools.product(*values))
    else:
        combinations = [()]  # No combinations to generate

    return fixed_params, array_params, combinations, keys if array_params else []


def load_config(config_file: str) -> Dict:
    """Load and parse YAML configuration file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ["model", "base_url"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    return config


def create_benchmark_configs(config: Dict) -> List[Dict]:
    """Create all benchmark configuration combinations from the YAML config."""
    configs = []

    # Get models as list
    models = config["model"] if isinstance(config["model"], list) else [config["model"]]

    # Get base_url as list
    base_urls = config["base_url"] if isinstance(config["base_url"], list) else [config["base_url"]]

    # Get concurrency levels
    concurrencies = config.get("max_concurrency", [256])
    if not isinstance(concurrencies, list):
        concurrencies = [concurrencies]

    # Check if token_pairs is specified (takes priority over separate input/output lists)
    if "token_pairs" in config:
        # Use token_pairs format: each pair is "input,output"
        token_pairs = config["token_pairs"]
        if not isinstance(token_pairs, list):
            token_pairs = [token_pairs]

        # Parse token pairs
        io_combinations = []
        for pair in token_pairs:
            input_len, output_len = map(int, pair.split(","))
            io_combinations.append((input_len, output_len))
    else:
        # Use separate random_input_len and random_output_len lists (full cartesian product)
        input_lens = config.get("random_input_len", [400])
        if not isinstance(input_lens, list):
            input_lens = [input_lens]

        output_lens = config.get("random_output_len", [600])
        if not isinstance(output_lens, list):
            output_lens = [output_lens]

        # Create all combinations of input and output lengths
        io_combinations = [(input_len, output_len) for input_len in input_lens for output_len in output_lens]

    # Generate all combinations
    for model in models:
        for base_url in base_urls:
            for input_len, output_len in io_combinations:
                for concurrency in concurrencies:
                    bench_config = {
                        "model": model,
                        "base_url": base_url,
                        "backend": config.get("backend", "openai"),
                        "dataset_name": config.get("dataset_name", "random"),
                        "seed": config.get("seed", 1234),
                        "random_input_len": input_len,
                        "random_output_len": output_len,
                        "max_concurrency": concurrency,
                        "result_dir": config.get("result_dir", "./vllm_bench_results"),
                    }
                    configs.append(bench_config)

    return configs


def build_vllm_command(bench_config: Dict) -> List[str]:
    """Build the vLLM bench serve command from configuration.

    Note: num_prompts is set equal to max_concurrency for consistency.
    """
    # Use concurrency as num_prompts
    num_prompts = bench_config["max_concurrency"]

    cmd = [
        "vllm", "bench", "serve",
        "--backend", bench_config["backend"],
        "--base-url", bench_config["base_url"],
        "--dataset-name", bench_config["dataset_name"],
        "--model", bench_config["model"],
        "--seed", str(bench_config["seed"]),
        "--num-prompts", str(num_prompts),
        "--random-input-len", str(bench_config["random_input_len"]),
        "--random-output-len", str(bench_config["random_output_len"]),
        "--max-concurrency", str(bench_config["max_concurrency"]),
        "--save-result",
        "--result-dir", bench_config["result_dir"],
    ]

    return cmd


def get_config_hash(bench_config: Dict) -> str:
    """Generate a unique hash for a benchmark configuration."""
    config_str = json.dumps(bench_config, sort_keys=True)
    return hashlib.sha1(config_str.encode()).hexdigest()


def run_benchmark(bench_config: Dict, checkpoint: Dict, checkpoint_path: str, timeout: int = 600) -> Dict:
    """Run a single vLLM bench serve command.

    Args:
        bench_config: Configuration for this benchmark run
        checkpoint: Checkpoint dictionary to track progress
        checkpoint_path: Path to save checkpoint file
        timeout: Timeout in seconds for the benchmark command (default: 600)
    """
    config_hash = get_config_hash(bench_config)

    # Check if already completed
    if config_hash in checkpoint and checkpoint[config_hash].get("status") == "completed":
        print(f"Skipping already completed benchmark: {bench_config}")
        return checkpoint

    # Initialize checkpoint entry
    if config_hash not in checkpoint:
        checkpoint[config_hash] = {
            "config": bench_config,
            "status": "pending",
            "start_time": None,
            "end_time": None,
        }

    print(f"\n{'='*80}")
    print(f"Running benchmark with configuration:")
    print(f"  Model: {bench_config['model']}")
    print(f"  Base URL: {bench_config['base_url']}")
    print(f"  Input Length: {bench_config['random_input_len']}")
    print(f"  Output Length: {bench_config['random_output_len']}")
    print(f"  Concurrency: {bench_config['max_concurrency']}")
    print(f"{'='*80}\n")

    # Build command
    cmd = build_vllm_command(bench_config)
    print(f"Executing: {' '.join(cmd)}")
    print(f"Timeout: {timeout} seconds\n")

    # Update checkpoint
    checkpoint[config_hash]["status"] = "running"
    checkpoint[config_hash]["start_time"] = datetime.now().isoformat()
    save_checkpoint(checkpoint, checkpoint_path)

    try:
        # Run the command with timeout and real-time output
        # Using stdout=None and stderr=None to show output in real-time
        result = subprocess.run(
            cmd,
            timeout=timeout,
            check=True,
            text=True
        )

        # Update checkpoint
        checkpoint[config_hash]["status"] = "completed"
        checkpoint[config_hash]["end_time"] = datetime.now().isoformat()

        print(f"\n✓ Benchmark completed successfully")

    except subprocess.TimeoutExpired as e:
        print(f"\n✗ Benchmark timed out after {timeout} seconds")
        print(f"  The benchmark exceeded the maximum allowed time.")
        print(f"  You can increase the timeout with --timeout <seconds>")

        checkpoint[config_hash]["status"] = "timeout"
        checkpoint[config_hash]["end_time"] = datetime.now().isoformat()
        checkpoint[config_hash]["error"] = f"Timeout after {timeout} seconds"

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Benchmark failed with error:")
        print(f"  Return code: {e.returncode}")

        checkpoint[config_hash]["status"] = "failed"
        checkpoint[config_hash]["end_time"] = datetime.now().isoformat()
        checkpoint[config_hash]["error"] = f"Process failed with return code {e.returncode}"

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        checkpoint[config_hash]["status"] = "failed"
        checkpoint[config_hash]["end_time"] = datetime.now().isoformat()
        checkpoint[config_hash]["error"] = str(e)

    # Save checkpoint after each run
    save_checkpoint(checkpoint, checkpoint_path)

    return checkpoint


def find_result_json_files(result_dir: str, config_hash: str, bench_config: Dict) -> Optional[str]:
    """Find the JSON result file created by vLLM bench serve for a specific configuration.

    vLLM bench serve creates files like: openai-infqps-concurrency{N}-{model}-{timestamp}.json
    """
    model_name = bench_config["model"].split("/")[-1]  # Get last part of model name
    concurrency = bench_config["max_concurrency"]

    # Pattern to match vLLM bench serve output files
    # Format: {backend}-{rate}qps-concurrency{N}-{model}-{timestamp}.json
    pattern = os.path.join(result_dir, f"*concurrency{concurrency}-{model_name}-*.json")

    matching_files = glob.glob(pattern)

    if matching_files:
        # Return the most recent file if multiple matches
        return max(matching_files, key=os.path.getctime)

    return None


def create_aggregated_summary(checkpoint: Dict, result_dir: str) -> None:
    """Create an aggregated CSV summary from all benchmark results.

    Args:
        checkpoint: Checkpoint dictionary with all benchmark configurations and statuses
        result_dir: Directory where results are saved
    """
    summary_rows = []

    for config_hash, run_data in checkpoint.items():
        bench_config = run_data.get("config", {})
        status = run_data.get("status", "unknown")

        # Base row with configuration
        row = {
            "model": bench_config.get("model", ""),
            "base_url": bench_config.get("base_url", ""),
            "input_tokens": bench_config.get("random_input_len", ""),
            "output_tokens": bench_config.get("random_output_len", ""),
            "concurrency": bench_config.get("max_concurrency", ""),
            "seed": bench_config.get("seed", ""),
            "status": status,
            "start_time": run_data.get("start_time", ""),
            "end_time": run_data.get("end_time", ""),
            "error": run_data.get("error", ""),
        }

        # Try to find and load the JSON result file
        if status == "completed":
            json_file = find_result_json_files(result_dir, config_hash, bench_config)

            if json_file and os.path.exists(json_file):
                try:
                    with open(json_file, "r") as f:
                        result_data = json.load(f)

                    # Add metrics from vLLM bench serve result
                    # These match the format from your example output
                    metrics_to_extract = [
                        "date", "endpoint_type", "label", "request_rate", "burstiness",
                        "duration", "completed", "total_input_tokens", "total_output_tokens",
                        "request_throughput", "request_goodput", "output_throughput",
                        "total_token_throughput",
                        "mean_ttft_ms", "median_ttft_ms", "std_ttft_ms", "p99_ttft_ms",
                        "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p99_tpot_ms",
                        "mean_itl_ms", "median_itl_ms", "std_itl_ms", "p99_itl_ms",
                        "mean_e2el_ms", "median_e2el_ms", "std_e2el_ms", "p99_e2el_ms",
                    ]

                    for metric in metrics_to_extract:
                        row[metric] = result_data.get(metric, "")

                    row["result_file"] = os.path.basename(json_file)

                except Exception as e:
                    print(f"Warning: Could not parse result file {json_file}: {e}")
                    row["result_file"] = os.path.basename(json_file) if json_file else ""
            else:
                row["result_file"] = "not_found"

        summary_rows.append(row)

    if not summary_rows:
        print("No benchmark results to aggregate")
        return

    # Save to CSV
    csv_path = os.path.join(result_dir, "benchmark_summary.csv")

    # Check if file exists to determine if we need headers
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        # Get all unique field names from all rows
        all_fields = set()
        for row in summary_rows:
            all_fields.update(row.keys())

        fieldnames = sorted(list(all_fields))

        # Reorder to put important columns first
        priority_fields = [
            "model", "input_tokens", "output_tokens", "concurrency",
            "status", "duration", "completed", "request_throughput", "output_throughput",
            "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
            "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
        ]

        # Put priority fields first, then the rest
        ordered_fields = [f for f in priority_fields if f in fieldnames]
        remaining_fields = [f for f in fieldnames if f not in priority_fields]
        fieldnames = ordered_fields + remaining_fields

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")

        # Write header if new file
        if not file_exists:
            writer.writeheader()

        # Write all rows
        for row in summary_rows:
            writer.writerow(row)

    print(f"\n✓ Aggregated summary saved to: {csv_path}")
    print(f"  Total benchmarks: {len(summary_rows)}")
    print(f"  Completed: {sum(1 for r in summary_rows if r['status'] == 'completed')}")
    print(f"  Failed: {sum(1 for r in summary_rows if r['status'] == 'failed')}")
    print(f"  Timeout: {sum(1 for r in summary_rows if r['status'] == 'timeout')}")


def main():
    parser = argparse.ArgumentParser(
        description="Run vLLM bench serve with configurations from YAML file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (default: auto-generated in result_dir)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each benchmark run (default: 600)"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Create benchmark configurations
    bench_configs = create_benchmark_configs(config)
    print(f"\nGenerated {len(bench_configs)} benchmark configuration(s)")

    # Setup checkpoint
    result_dir = config.get("result_dir", "./vllm_bench_results")
    os.makedirs(result_dir, exist_ok=True)

    checkpoint_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}.json")

    # Load checkpoint if resuming
    checkpoint = {}
    if args.resume:
        checkpoint = load_checkpoint(args.checkpoint or checkpoint_dir)

    # Run benchmarks
    print(f"\nStarting benchmark runs...")
    print(f"Results will be saved to: {result_dir}")
    print(f"Checkpoint file: {checkpoint_path}")
    print(f"Timeout per benchmark: {args.timeout} seconds\n")

    for i, bench_config in enumerate(bench_configs, 1):
        print(f"\n{'#'*80}")
        print(f"Benchmark {i}/{len(bench_configs)}")
        print(f"{'#'*80}")

        checkpoint = run_benchmark(bench_config, checkpoint, checkpoint_path, timeout=args.timeout)

    # Create aggregated CSV summary
    create_aggregated_summary(checkpoint, result_dir)

    # Print summary
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")

    completed = sum(1 for v in checkpoint.values() if v.get("status") == "completed")
    failed = sum(1 for v in checkpoint.values() if v.get("status") == "failed")
    timeout = sum(1 for v in checkpoint.values() if v.get("status") == "timeout")

    print(f"Total configurations: {len(bench_configs)}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Timeout: {timeout}")
    print(f"\nResults saved to: {result_dir}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Aggregated CSV: {os.path.join(result_dir, 'benchmark_summary.csv')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
