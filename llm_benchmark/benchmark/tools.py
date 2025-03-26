import os
import csv
import shutil
from typing import Optional

from llm_benchmark.benchmark.vllm_benchmark.benchmark_serving import (
    run_benchmark as vllm_run_benchmark,
)
from llm_benchmark.benchmark.llmperf.token_benchmark_ray import (
    run_token_benchmark as llmperf_run_benchmark,
)
from llm_benchmark.benchmark.litellm_proxy.token_benchmark_ray import (
    run_token_benchmark as litellm_run_benchmark,
)
from llm_benchmark.profiler.constants import VllmProfileLayer
from llm_benchmark.profiler.record_function_tracer import RecordFunctionTracer
from llm_benchmark.utils.common import combine_multiple_datasets

from .schemas import BenchmarkResultSchema


def get_profiler_result(result_dir: str):
    record_function_tracer = RecordFunctionTracer(result_dir, get_all=True)
    profile_stats = record_function_tracer.get_operation_time_stats()
    return profile_stats


def create_summary(results, results_dir, profiler_result: bool = False):
    summary_list = []
    layers = VllmProfileLayer.get_available_profile_names()

    for result in results:
        summary = {}
        summary["Engine"] = result["engine"]
        summary["engine_config_id"] = result["engine_config_id"]
        summary["run_id"] = result["run_id"]
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

        if profiler_result:
            for layer in layers:
                for prefix in ("cpu.", "cuda."):
                    name = prefix + layer
                summary[f"{name}_min"] = result[name]["min"] if name in result else ""
                summary[f"{name}_max"] = result[name]["max"] if name in result else ""
                summary[f"{name}_mean"] = result[name]["mean"] if name in result else ""
                summary[f"{name}_median"] = (
                    result[name]["median"] if name in result else ""
                )
                summary[f"{name}_std"] = result[name]["std"] if name in result else ""

        summary_list.append(summary)

    if len(summary_list) == 0:
        print("No results to save")
        return

    # Define the CSV file path
    # filename = f"{results[0]['model']}"
    # filename = re.sub(r"[^\w\d-]+", "-", filename)
    # filename = re.sub(r"-{2,}", "-", filename)

    csv_file_path = os.path.join(
        results_dir, results[0]["model"].replace("/", "--"), "summary.csv"
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
    # formatted_result = {}
    # formatted_result["model"] = result["model_id"]
    # formatted_result["concurrency"] = result["concurrency"]
    # formatted_result["input_tokens"] = result["input_tokens"]
    # formatted_result["output_tokens"] = result["output_tokens"]
    # formatted_result["total_input_tokens"] = result["total_input_tokens"]
    # formatted_result["total_output_tokens"] = result["total_output_tokens"]
    # formatted_result["completed"] = result["completed"]
    # formatted_result["request_throughput"] = result["request_throughput"]
    # formatted_result["output_throughput"] = result["output_throughput"]
    # formatted_result["total_token_throughput"] = result["total_token_throughput"]
    # formatted_result["output_throughput_per_user"] = result["mean_output_throughput_per_user"]
    # formatted_result["mean_end_to_end_latency"] = result["mean_e2el_ms"]
    # formatted_result["mean_ttft_ms"] = result["mean_ttft_ms"]
    # formatted_result["p95_ttft_ms"] = result["p95_ttft_ms"]
    # formatted_result["mean_tpot_ms"] = result["mean_tpot_ms"]
    # formatted_result["p95_tpot_ms"] = result["p95_tpot_ms"]
    # formatted_result["mean_itl_ms"] = result["mean_itl_ms"]
    # formatted_result["p95_itl_ms"] = result["p95_itl_ms"]
    
    benchmark_result = BenchmarkResultSchema(
        model=result["model_id"],
        concurrency=result["concurrency"],
        duration=result["duration"],
        successful_requests=result["successful_requests"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        total_input_tokens=result["total_input_tokens"],
        total_output_tokens=result["total_output_tokens"],
        request_throughput=result["request_throughput"],
        input_throughput=result["input_throughput"],
        output_throughput=result["output_throughput"],
        output_throughput_per_user=result["mean_output_throughput_per_user"],
        # p25_throughput=result["percentiles_output_throughput_per_user"][0],
        # p75_throughput=result["percentiles_output_throughput_per_user"][1],
        # p95_throughput=result["percentiles_output_throughput_per_user"][2],
        # p99_throughput=result["percentiles_output_throughput_per_user"][3],
        # min_throughput=result["min_output_throughput"],
        # max_throughput=result["max_output_throughput"],
        mean_ttft_ms=result["mean_ttft_ms"],
        median_ttft_ms=result["median_ttft_ms"],
        # p25_ttft_ms=result["percentiles_ttft_ms"][0],
        # p75_ttft_ms=result["percentiles_ttft_ms"][1],
        # p95_ttft_ms=result["percentiles_ttft_ms"][2],
        # p99_ttft_ms=result["percentiles_ttft_ms"][3],
        min_ttft_ms=result["min_ttft_ms"],
        max_ttft_ms=result["max_ttft_ms"],
        mean_tpot_ms=result["mean_tpot_ms"],
        median_tpot_ms=result["median_tpot_ms"],
        # p25_tpot_ms=result["percentiles_tpot_ms"][0],
        # p75_tpot_ms=result["percentiles_tpot_ms"][1],
        # p95_tpot_ms=result["percentiles_tpot_ms"][2],
        # p99_tpot_ms=result["percentiles_tpot_ms"][3],
        min_tpot_ms=result["min_tpot_ms"],
        max_tpot_ms=result["max_tpot_ms"],
        mean_itl_ms=result["mean_itl_ms"],
        median_itl_ms=result["median_itl_ms"],
        # p25_itl_ms=result["percentiles_itl_ms"][0],
        # p75_itl_ms=result["percentiles_itl_ms"][1],
        # p95_itl_ms=result["percentiles_itl_ms"][2],
        # p99_itl_ms=result["percentiles_itl_ms"][3],
        min_itl_ms=result["min_itl_ms"],
        max_itl_ms=result["max_itl_ms"],
        mean_e2el_ms=result["mean_e2el_ms"],
        median_e2el_ms=result["median_e2el_ms"],
        # p25_e2el_ms=result["percentiles_e2el_ms"][0],
        # p75_e2el_ms=result["percentiles_e2el_ms"][1],
        # p95_e2el_ms=result["percentiles_e2el_ms"][2],
        # p99_e2el_ms=result["percentiles_e2el_ms"][3],
        min_e2el_ms=result["min_e2el_ms"],
        max_e2el_ms=result["max_e2el_ms"],
        error_messages=result["errors"],
    )
    return benchmark_result.model_dump()


def format_llmperf_result(result):
    num_completed_requests = result["results"]["num_completed_requests"]
    num_requests_completed_per_min = result["results"]["num_completed_requests_per_min"]
    benchmark_result = BenchmarkResultSchema(
        model=result["model"],
        concurrency=result["num_concurrent_requests"],
        duration=result["results"]["end_to_end_latency_s"]["max"],
        successful_requests=num_completed_requests,
        input_tokens=result["mean_input_tokens"],
        output_tokens=result["mean_output_tokens"],
        total_input_tokens=result["results"]["number_input_tokens"]["mean"]*num_completed_requests,
        total_output_tokens=result["results"]["number_output_tokens"]["mean"]*num_completed_requests,
        request_throughput=num_requests_completed_per_min,
        input_throughput=result["results"]["number_input_tokens"]["mean"]*num_requests_completed_per_min,
        output_throughput=result["results"]["mean_output_throughput_token_per_s"],
        output_throughput_per_user=result["results"]["request_output_throughput_token_per_s"]["mean"],
        p25_throughput=result["results"]["request_output_throughput_token_per_s"]["quantiles"]["p25"],
        p75_throughput=result["results"]["request_output_throughput_token_per_s"]["quantiles"]["p75"],
        p95_throughput=result["results"]["request_output_throughput_token_per_s"]["quantiles"]["p95"],
        p99_throughput=result["results"]["request_output_throughput_token_per_s"]["quantiles"]["p99"],
        min_throughput=result["results"]["request_output_throughput_token_per_s"]["min"],
        max_throughput=result["results"]["request_output_throughput_token_per_s"]["max"],
        mean_ttft_ms=result["results"]["ttft_s"]["mean"]*1000,
        median_ttft_ms=result["results"]["ttft_s"]["median"]*1000,
        p25_ttft_ms=result["results"]["ttft_s"]["quantiles"]["p25"]*1000,
        p75_ttft_ms=result["results"]["ttft_s"]["quantiles"]["p75"]*1000,
        p95_ttft_ms=result["results"]["ttft_s"]["quantiles"]["p95"]*1000,
        p99_ttft_ms=result["results"]["ttft_s"]["quantiles"]["p99"]*1000,
        min_ttft_ms=result["results"]["ttft_s"]["min"]*1000,
        max_ttft_ms=result["results"]["ttft_s"]["max"]*1000,
        mean_tpot_ms=result["results"]["tpot_s"]["mean"]*1000,
        median_tpot_ms=result["results"]["tpot_s"]["median"]*1000,
        p25_tpot_ms=result["results"]["tpot_s"]["quantiles"]["p25"]*1000,
        p75_tpot_ms=result["results"]["tpot_s"]["quantiles"]["p75"]*1000,
        p95_tpot_ms=result["results"]["tpot_s"]["quantiles"]["p95"]*1000,
        p99_tpot_ms=result["results"]["tpot_s"]["quantiles"]["p99"]*1000,
        min_tpot_ms=result["results"]["tpot_s"]["min"]*1000,
        max_tpot_ms=result["results"]["tpot_s"]["max"]*1000,
        mean_itl_ms=result["results"]["inter_token_latency_s"]["mean"]*1000,
        median_itl_ms=result["results"]["inter_token_latency_s"]["median"]*1000,
        p25_itl_ms=result["results"]["inter_token_latency_s"]["quantiles"]["p25"]*1000,
        p75_itl_ms=result["results"]["inter_token_latency_s"]["quantiles"]["p75"]*1000,
        p95_itl_ms=result["results"]["inter_token_latency_s"]["quantiles"]["p95"]*1000,
        p99_itl_ms=result["results"]["inter_token_latency_s"]["quantiles"]["p99"]*1000,
        min_itl_ms=result["results"]["inter_token_latency_s"]["min"]*1000,
        max_itl_ms=result["results"]["inter_token_latency_s"]["max"]*1000,
        mean_e2el_ms=result["results"]["end_to_end_latency_s"]["mean"]*1000,
        median_e2el_ms=result["results"]["end_to_end_latency_s"]["median"]*1000,
        p25_e2el_ms=result["results"]["end_to_end_latency_s"]["quantiles"]["p25"]*1000,
        p75_e2el_ms=result["results"]["end_to_end_latency_s"]["quantiles"]["p75"]*1000,
        p95_e2el_ms=result["results"]["end_to_end_latency_s"]["quantiles"]["p95"]*1000,
        p99_e2el_ms=result["results"]["end_to_end_latency_s"]["quantiles"]["p99"]*1000,
        min_e2el_ms=result["results"]["end_to_end_latency_s"]["min"]*1000,
        max_e2el_ms=result["results"]["end_to_end_latency_s"]["max"]*1000,
        error_messages=result["results"].get("error_msg", []),
    )
    # formatted_result = {}
    # formatted_result["model"] = result["model"]
    # formatted_result["concurrency"] = result["num_concurrent_requests"]
    # formatted_result["completed"] = result["results"]["num_completed_requests"]
    # formatted_result["duration"] = result["results"]["end_to_end_latency_s"]["max"]
    # formatted_result["request_throughput_per_min"] = result["results"][
    #     "num_completed_requests_per_min"
    # ]
    # formatted_result["output_throughput"] = result["results"][
    #     "mean_output_throughput_token_per_s"
    # ]
    # formatted_result["output_throughput_per_user"] = result["results"][
    #     "request_output_throughput_token_per_s"
    # ]["mean"]
    # formatted_result["mean_end_to_end_latency"] = result["results"][
    #     "end_to_end_latency_s"
    # ]["mean"]
    # formatted_result["mean_ttft_ms"] = result["results"]["ttft_s"]["mean"] * 1000
    # formatted_result["p95_ttft_ms"] = (
    #     result["results"]["ttft_s"]["quantiles"]["quantiles"]["p95"] * 1000
    # )
    # formatted_result["mean_itl_ms"] = (
    #     result["results"]["inter_token_latency_s"]["mean"] * 1000
    # )
    # formatted_result["p95_itl_ms"] = (
    #     result["results"]["inter_token_latency_s"]["quantiles"]["quantiles"]["p95"] * 1000
    # )
    return benchmark_result.model_dump()


def run_benchmark(
    model: str,
    base_url: str,
    input_token: int,
    output_token: int,
    concurrency: int,
    benchmark_script: str,
    result_dir: str = None,
    run_id: str = None,
    profiler_result: bool = False,
    env_values: Optional[dict] = None,
    latency_factors: Optional[dict] = None,
    datasets: Optional[list] = None,
):
    # Set environment variables directly
    # TODO: Removed it because litellm_proxy requires actual api key
    # and this was overriding the one in the envs
    # If required to set for other engines, can be set as env
    # os.environ["OPENAI_API_KEY"] = "secret_abcdefg"
    # os.environ["OPENAI_API_BASE"] = base_url
    
    sampled_prompts = combine_multiple_datasets(datasets, concurrency)

    if result_dir is not None:
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
            model, input_token, output_token, concurrency, base_url, sampled_prompts
        )
        result_output = format_vllm_result(result_output)
    elif benchmark_script == "llmperf":
        result_output = llmperf_run_benchmark(
            model, concurrency, concurrency, input_token, 0, output_token, 0, sampled_prompts=sampled_prompts
        )
        result_output = format_llmperf_result(result_output)
    elif benchmark_script == "litellm_proxy":
        litellm_master_key = env_values.get("LITELLM_MASTER_KEY")
        if litellm_master_key is None:
            raise ValueError("LITELLM_MASTER_KEY is not set in engine config")
        request_metadata = {
            "api_key": "fake-api-key",
            "litellm_proxy_url": base_url,
            "litellm_master_key": litellm_master_key
        }
        result_output = litellm_run_benchmark(
            model, concurrency, concurrency, input_token, 0, output_token, 0, llm_api="mock_litellm_proxy", request_metadata=request_metadata, latency_factors=latency_factors, sampled_prompts=sampled_prompts
        )
        result_output = format_llmperf_result(result_output)

    profiler_stats = {}
    if profiler_result:
        if result_dir is not None:
            profiler_stats = get_profiler_result(result_dir)

    # run_id_dir = os.path.join(result_dir, 'traces', run_id)
    # os.makedirs(run_id_dir, exist_ok=True)
    # for file in os.listdir(traces_dir):
    #     if file.startswith("profiler_trace_") and file.endswith(".json"):
    #         shutil.move(os.path.join(traces_dir, file), run_id_dir)

    return {**result_output, **profiler_stats}
