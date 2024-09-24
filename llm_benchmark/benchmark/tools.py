import os
import csv
import shutil
import datetime

from llm_benchmark.benchmark.vllm_benchmark.benchmark_serving import (
    run_benchmark as vllm_run_benchmark,
)
from llm_benchmark.benchmark.llmperf.token_benchmark_ray import (
    run_token_benchmark as llmperf_run_benchmark,
)
from llm_benchmark.profiler.constants import VllmProfileLayer
from llm_benchmark.profiler.record_function_tracer import RecordFunctionTracer


def get_profiler_result(result_dir: str):
    record_function_tracer = RecordFunctionTracer(result_dir, get_all=True)
    profile_stats = record_function_tracer.get_operation_time_stats()
    
    return profile_stats


def create_summary(results, results_dir):
    summary_list = []
    layers = VllmProfileLayer.get_available_profile_names()

    for result in results:
        summary = {}
        summary["Engine"] = result["engine"]
        summary["Engine Config ID"] = result["engine_config_id"]
        summary["Run ID"] = result["run_id"]
        summary["Model"] = result["model"]
        summary["Mean Input Tokens"] = result["input_tokens"]
        summary["Mean Output Tokens"] = result["output_tokens"]
        summary["Concurrent Requests"] = result["concurrency"]
        summary["Completed Requests"] = result["completed"]
        summary["Duration (s)"] = round(result["duration"], 2)
        summary["Request Throughput (req/min)"] = round(
            result["request_throughput_per_min"], 2
        )
        summary["Output Token Throughput (tok/s)"] = round(
            result["output_throughput"], 2
        )
        summary["Output Token Throughput per User (tok/s)"] = round(
            result["output_throughput_per_user"], 2
        )
        summary["Mean End to End Latency (s)"] = round(
            result["mean_end_to_end_latency"], 2
        )
        summary["Mean TTFT (ms)"] = round(result["mean_ttft_ms"], 2)
        summary["P95 TTFT (ms)"] = round(result["p95_ttft_ms"], 2)
        summary["Mean Inter Token Latency (ms)"] = round(result["mean_itl_ms"], 2)
        summary["P95 Inter Token Latency (ms)"] = round(result["p95_itl_ms"], 2)

        for layer in layers:
            summary[f"{layer}_min"] = result[layer]["min"] if layer in result else ""
            summary[f"{layer}_max"] = result[layer]["max"] if layer in result else ""
            summary[f"{layer}_mean"] = result[layer]["mean"] if layer in result else ""
            summary[f"{layer}_median"] = (
                result[layer]["median"] if layer in result else ""
            )
            summary[f"{layer}_std"] = result[layer]["std"] if layer in result else ""

        summary_list.append(summary)

    if len(summary_list) == 0:
        print("No results to save")
        return

    # Define the CSV file path
    # filename = f"{results[0]['model']}"
    # filename = re.sub(r"[^\w\d-]+", "-", filename)
    # filename = re.sub(r"-{2,}", "-", filename)

    csv_file_path = os.path.join(
        results_dir, results[0]["model"].replace("/", "--"), f"summary_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    )
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # Check if the file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_file_path)

    # Open the CSV file in append mode
    with open(csv_file_path, "a", newline="") as csvfile:
        fieldnames = list(summary_list[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write headers if the file is newly created
        if not file_exists:
            writer.writeheader()

        # Write the summary data
        for summary in summary_list:
            writer.writerow(summary)

    print(f"Benchmark summary saved to {csv_file_path}")
    return summary


def format_vllm_result(result):
    formatted_result = {}
    formatted_result["model"] = result["model_id"]
    # formatted_result["concurrency"] = result["concurrency"]
    # formatted_result["input_tokens"] = result["input_tokens"]
    # formatted_result["output_tokens"] = result["output_tokens"]
    formatted_result["total_input_tokens"] = result["total_input_tokens"]
    formatted_result["total_output_tokens"] = result["total_output_tokens"]
    formatted_result["completed"] = result["completed"]
    formatted_result["request_throughput"] = result["request_throughput"]
    formatted_result["output_throughput"] = result["output_throughput"]
    formatted_result["total_token_throughput"] = result["total_token_throughput"]
    # formatted_result["mean_request_throughput"] = result["mean_request_throughput"]
    formatted_result["mean_ttft_ms"] = result["mean_ttft_ms"]
    formatted_result["p95_ttft_ms"] = result["p95_ttft_ms"]
    formatted_result["mean_tpot_ms"] = result["mean_tpot_ms"]
    formatted_result["p95_tpot_ms"] = result["p95_tpot_ms"]
    formatted_result["mean_itl_ms"] = result["mean_itl_ms"]
    formatted_result["p95_itl_ms"] = result["p95_itl_ms"]

    return formatted_result


def format_llmperf_result(result):
    formatted_result = {}
    formatted_result["model"] = result["model"]
    formatted_result["concurrency"] = result["num_concurrent_requests"]
    formatted_result["input_tokens"] = result["mean_input_tokens"]
    formatted_result["output_tokens"] = result["mean_output_tokens"]
    formatted_result["completed"] = result["results"]["num_completed_requests"]
    formatted_result["duration"] = result["results"]["end_to_end_latency_s"]["max"]
    formatted_result["request_throughput_per_min"] = result["results"][
        "num_completed_requests_per_min"
    ]
    formatted_result["output_throughput"] = result["results"][
        "mean_output_throughput_token_per_s"
    ]
    formatted_result["output_throughput_per_user"] = result["results"][
        "request_output_throughput_token_per_s"
    ]["mean"]
    formatted_result["mean_end_to_end_latency"] = result["results"][
        "end_to_end_latency_s"
    ]["mean"]
    formatted_result["mean_ttft_ms"] = result["results"]["ttft_s"]["mean"] * 1000
    formatted_result["p95_ttft_ms"] = (
        result["results"]["ttft_s"]["quantiles"]["p95"] * 1000
    )
    formatted_result["mean_itl_ms"] = (
        result["results"]["inter_token_latency_s"]["mean"] * 1000
    )
    formatted_result["p95_itl_ms"] = (
        result["results"]["inter_token_latency_s"]["quantiles"]["p95"] * 1000
    )
    return formatted_result


def run_benchmark(
    model: str,
    base_url: str,
    input_token: int,
    output_token: int,
    concurrency: int,
    benchmark_script: str,
    result_dir: str,
    run_id: str,
):
    # Set environment variables directly
    os.environ["OPENAI_API_KEY"] = "secret_abcdefg"
    os.environ["OPENAI_API_BASE"] = base_url

    result_dir = os.path.join(result_dir, model.replace("/", "--"))

    traces_dir = f"{result_dir}/profiler_traces/"
    if os.path.exists(traces_dir):
        shutil.rmtree(traces_dir)
    os.makedirs(traces_dir, exist_ok=True)

    print(
        "Running benchmark for model: ",
        model,
        "with input token: ",
        input_token,
        "and output token: ",
        output_token,
        "and concurrency: ",
        concurrency,
        "run id: ",
        run_id,
    )

    if benchmark_script == "vllm":
        result_output = vllm_run_benchmark(
            model, input_token, output_token, concurrency, base_url
        )
        result_output = format_vllm_result(result_output)
    else:
        result_output = llmperf_run_benchmark(
            model, concurrency, concurrency, input_token, 0, output_token, 0
        )
        result_output = format_llmperf_result(result_output)

    profiler_stats = get_profiler_result(result_dir)

    run_id_dir = os.path.join(result_dir, 'traces', run_id)
    os.makedirs(run_id_dir, exist_ok=True)
    for file in os.listdir(traces_dir):
        if file.startswith("profiler_trace_") and file.endswith(".json"):
            shutil.move(os.path.join(traces_dir, file), run_id_dir)

    return {**result_output, **profiler_stats}
