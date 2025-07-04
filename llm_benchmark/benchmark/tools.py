import os
import csv
import shutil
from typing import Optional
from uuid import UUID

from llm_benchmark.benchmark.vllm_benchmark.benchmark_serving import (
    run_benchmark as vllm_run_benchmark,
)
from llm_benchmark.benchmark.llmperf.token_benchmark_ray import (
    run_token_benchmark as llmperf_run_benchmark,
)
from llm_benchmark.benchmark.litellm_proxy.token_benchmark_ray import (
    run_token_benchmark as litellm_run_benchmark,
)
from llm_benchmark.benchmark.budlatent.benchmark_embedding_servings import (
    run_benchmark as budlatent_run_benchmark,
)
from llm_benchmark.profiler.constants import VllmProfileLayer
from llm_benchmark.profiler.record_function_tracer import RecordFunctionTracer
from llm_benchmark.utils.common import combine_multiple_datasets

from .schemas import BenchmarkResultSchema, BenchmarkRequestMetrics


def get_profiler_result(result_dir: str):
    record_function_tracer = RecordFunctionTracer(result_dir, get_all=True)
    profile_stats = record_function_tracer.get_operation_time_stats()
    return profile_stats


def create_summary(results, results_dir, profiler_result: bool = False):
    summary_list = []
    layers = VllmProfileLayer.get_available_profile_names()

    for result in results:
        summary = {}
        summary["engine"] = result["engine"]
        summary["engine_config_id"] = result["engine_config_id"]
        summary["run_id"] = result["run_id"]
        summary["status"] = result["status"]

        if "error_messages" in result:
            result.pop("error_messages")
        if "individual_responses" in result:
            result.pop("individual_responses")
        summary = { **summary, **result }

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


def format_vllm_result(result, individual_responses):
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
        mean_output_throughput_per_user=result["mean_output_throughput_per_user"],
        p25_output_throughput_per_user=result["p25_output_throughput_per_user"],
        p75_output_throughput_per_user=result["p75_output_throughput_per_user"],
        p95_output_throughput_per_user=result["p95_output_throughput_per_user"],
        p99_output_throughput_per_user=result["p99_output_throughput_per_user"],
        min_output_throughput_per_user=result["min_output_throughput_per_user"],
        max_output_throughput_per_user=result["max_output_throughput_per_user"],
        mean_ttft_ms=result["mean_ttft_ms"],
        median_ttft_ms=result["median_ttft_ms"],
        p25_ttft_ms=result["p25_ttft_ms"],
        p75_ttft_ms=result["p75_ttft_ms"],
        p95_ttft_ms=result["p95_ttft_ms"],
        p99_ttft_ms=result["p99_ttft_ms"],
        min_ttft_ms=result["min_ttft_ms"],
        max_ttft_ms=result["max_ttft_ms"],
        mean_tpot_ms=result["mean_tpot_ms"],
        median_tpot_ms=result["median_tpot_ms"],
        p25_tpot_ms=result["p25_tpot_ms"],
        p75_tpot_ms=result["p75_tpot_ms"],
        p95_tpot_ms=result["p95_tpot_ms"],
        p99_tpot_ms=result["p99_tpot_ms"],
        min_tpot_ms=result["min_tpot_ms"],
        max_tpot_ms=result["max_tpot_ms"],
        mean_itl_ms=result["mean_itl_ms"],
        median_itl_ms=result["median_itl_ms"],
        p25_itl_ms=result["p25_itl_ms"],
        p75_itl_ms=result["p75_itl_ms"],
        p95_itl_ms=result["p95_itl_ms"],
        p99_itl_ms=result["p99_itl_ms"],
        min_itl_ms=result["min_itl_ms"],
        max_itl_ms=result["max_itl_ms"],
        mean_e2el_ms=result["mean_e2el_ms"],
        median_e2el_ms=result["median_e2el_ms"],
        p25_e2el_ms=result["p25_e2el_ms"],
        p75_e2el_ms=result["p75_e2el_ms"],
        p95_e2el_ms=result["p95_e2el_ms"],
        p99_e2el_ms=result["p99_e2el_ms"],
        min_e2el_ms=result["min_e2el_ms"],
        max_e2el_ms=result["max_e2el_ms"],
        error_messages=result["errors"],
    )
    
    request_metrics = []
    for metrics in individual_responses:
        request_metrics.append(
            BenchmarkRequestMetrics.model_validate(metrics, from_attributes=True).model_dump(mode="json")
        )
    
    return benchmark_result.model_dump(), request_metrics


def format_llmperf_result(result, individual_responses):
    num_completed_requests = result["results"]["num_completed_requests"]
    num_requests_completed_per_min = result["results"]["num_completed_requests_per_min"]
    total_input_tokens = sum([metric["number_input_tokens"] for metric in individual_responses])
    total_output_tokens = sum([metric["number_output_tokens"] for metric in individual_responses])
    benchmark_result = BenchmarkResultSchema(
        model=result["model"],
        concurrency=result["num_concurrent_requests"],
        duration=result["results"]["end_to_end_latency_s"]["max"],
        successful_requests=num_completed_requests,
        input_tokens=result["mean_input_tokens"],
        output_tokens=result["mean_output_tokens"],
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        request_throughput=num_requests_completed_per_min,
        input_throughput=result["results"]["number_input_tokens"]["mean"]*num_requests_completed_per_min,
        output_throughput=result["results"]["mean_output_throughput_token_per_s"],
        mean_output_throughput_per_user=result["results"]["request_output_throughput_token_per_s"]["mean"],
        p25_output_throughput_per_user=result["results"]["request_output_throughput_token_per_s"]["quantiles"]["p25"],
        p75_output_throughput_per_user=result["results"]["request_output_throughput_token_per_s"]["quantiles"]["p75"],
        p95_output_throughput_per_user=result["results"]["request_output_throughput_token_per_s"]["quantiles"]["p95"],
        p99_output_throughput_per_user=result["results"]["request_output_throughput_token_per_s"]["quantiles"]["p99"],
        min_output_throughput_per_user=result["results"]["request_output_throughput_token_per_s"]["min"],
        max_output_throughput_per_user=result["results"]["request_output_throughput_token_per_s"]["max"],
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
    request_metrics = []
    for metrics in individual_responses:
        request_metrics.append(BenchmarkRequestMetrics(**metrics).model_dump(mode="json"))
    return benchmark_result.model_dump(), request_metrics


def format_budlatent_result(result):
    formatted_result = {}
    formatted_result["model"] = result["model"]
    formatted_result["completed"] = result["completed"]
    formatted_result["request_throughput"] = result["request_throughput"]
    formatted_result["mean_end_to_end_latency"] = result["mean_e2el_ms"]
    formatted_result["duration"] = result["duration"]

    benchmark_result = BenchmarkResultSchema(
        model=result["model"],
        concurrency=result["concurrency"],
        duration=result["duration"],
        successful_requests=result["completed"],
        input_tokens=result["num_tokens"],
        # output_tokens=result["output_tokens"],
        request_throughput=result["request_throughput"],
        mean_e2el_ms=result["mean_e2el_ms"],
    )

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
    benchmark_id: Optional[UUID] = None,
    seed: Optional[int] = None,
    min_input_tokens: Optional[int] = None,
    max_input_tokens: Optional[int] = None,
    min_output_tokens: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
):
    if benchmark_script != "litellm_proxy":
        os.environ["OPENAI_API_KEY"] = "secret_abcdefg"
        os.environ["OPENAI_API_BASE"] = base_url
    
    sampled_prompts = combine_multiple_datasets(
        concurrency,
        seed,
        datasets
    )

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
        result_output, individual_responses = vllm_run_benchmark(
            model, input_token, output_token, concurrency, base_url, sampled_prompts=sampled_prompts, benchmark_id=benchmark_id
        )
        # print(f"local model result_output: {result_output}" )
        result_output, individual_responses = format_vllm_result(result_output, individual_responses)
    elif benchmark_script == "llmperf":
        input_deviation = (max_input_tokens - min_input_tokens)/2 if max_input_tokens is not None and min_input_tokens is not None else 0
        output_deviation = (max_output_tokens - min_output_tokens)/2 if max_output_tokens is not None and min_output_tokens is not None else 0
        result_output, individual_responses = llmperf_run_benchmark(
            model, concurrency, concurrency, input_token, input_deviation, output_token, output_deviation, sampled_prompts=sampled_prompts, benchmark_id=benchmark_id
        )
        result_output, individual_responses = format_llmperf_result(result_output, individual_responses)
    elif benchmark_script == "budlatent":
        result_output = budlatent_run_benchmark(
            model, input_token, output_token, concurrency, base_url, sampled_prompts=sampled_prompts, benchmark_id=benchmark_id
        )
        result_output = format_budlatent_result(result_output)
        individual_responses = []  # Budlatent doesn't track individual responses yet
    elif benchmark_script == "litellm_proxy":
        litellm_master_key = env_values.get("LITELLM_MASTER_KEY")
        if litellm_master_key is None:
            raise ValueError("LITELLM_MASTER_KEY is not set in engine config")
        request_metadata = {
            "api_key": "fake-api-key",
            "litellm_proxy_url": base_url,
            "litellm_master_key": litellm_master_key
        }
        result_output, individual_responses = litellm_run_benchmark(
            model, concurrency, concurrency, input_token, 0, output_token, 0, llm_api="mock_litellm_proxy", request_metadata=request_metadata, latency_factors=latency_factors, sampled_prompts=sampled_prompts, benchmark_id=benchmark_id
        )
        result_output, individual_responses = format_llmperf_result(result_output, individual_responses)

    profiler_stats = {}
    if profiler_result:
        if result_dir is not None:
            profiler_stats = get_profiler_result(result_dir)

    # run_id_dir = os.path.join(result_dir, 'traces', run_id)
    # os.makedirs(run_id_dir, exist_ok=True)
    # for file in os.listdir(traces_dir):
    #     if file.startswith("profiler_trace_") and file.endswith(".json"):
    #         shutil.move(os.path.join(traces_dir, file), run_id_dir)

    return {**result_output, **profiler_stats, "individual_responses" : individual_responses}
