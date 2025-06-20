"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""

import argparse
import asyncio
import json
import os
import random
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from uuid import UUID

from datasets import load_dataset

import numpy as np
from .backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from .backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_request_throughput: float
    mean_output_throughput_per_user: float
    median_output_throughput_per_user: float
    std_output_throughput_per_user: float
    percentiles_output_throughput_per_user: List[Tuple[float, float]]
    min_output_throughput_per_user: float
    max_output_throughput_per_user: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    min_ttft_ms: float
    max_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    min_tpot_ms: float
    max_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    min_itl_ms: float
    max_itl_ms: float
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]
    min_e2el_ms: float
    max_e2el_ms: float


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


def sample_random_positive_int(mean: int, stddev: int) -> int:
    ret = -1
    while ret <= 0:
        ret = int(random.gauss(mean, stddev))
    return ret


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    mean_input_len: int,
    seed:int,
    fixed_output_len: int,
    input_column: str = "input",
    output_column: Optional[str] = None,
    stddev_input_len: int = None,
) -> List[Tuple[str, int, int]]:
    if mean_input_len is not None and mean_input_len < 4:
        raise ValueError("mean_input_len must be at least 4")
    
    if stddev_input_len is None:
        stddev_input_len = 0.1*(mean_input_len)

    try:
        if os.path.isfile(dataset_path):
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
                if isinstance(dataset, list) and len(dataset) > 0:
                    if isinstance(dataset[0], dict):
                        column_names = dataset[0].keys()
                    else:
                        raise TypeError("Invalid data format: expected list of dicts.")
                else:
                    raise ValueError("Invalid dataset type: expected non-empty list.")
        else:
            dataset = load_dataset(dataset_path)["train"]
            column_names = dataset.column_names
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")

    if input_column not in column_names:
        raise ValueError(f"'{input_column}' is not a valid input column name.")
    if output_column is not None and output_column not in column_names:
        raise ValueError(f"'{output_column}' is not a valid output column name.")

    if input_column == "conversations":
        try:
            dataset = [data for data in dataset if len(data["conversations"]) >= 2]
            dataset = [
                (data["conversations"][0]["value"], data["conversations"][1]["value"])
                for data in dataset
            ]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Error processing conversations column: {e}")
    else:
        if output_column is None:
            raise ValueError("Provide the Output Column")
        else:
            dataset = [
                (data.get(input_column, ""), data.get(output_column, ""))
                for data in dataset
            ]
            
    random.seed(seed)
    random.shuffle(dataset)
    filtered_dataset: List[Tuple[str, int, int]] = []

    for prompt, completion in dataset:
        try:
            prompt_token_ids = tokenizer(prompt).input_ids
            prompt_len = len(prompt_token_ids)
            completion_token_ids = tokenizer(completion).input_ids
            output_len = len(completion_token_ids) if fixed_output_len is None else fixed_output_len

            random_length = sample_random_positive_int(mean_input_len, stddev_input_len)

            if prompt_len == random_length:
                filtered_dataset.append((prompt, prompt_len, output_len))
            elif prompt_len > random_length:
                sliced_prompt = tokenizer.decode(
                    prompt_token_ids[:random_length], skip_special_tokens=True
                )
                filtered_dataset.append((sliced_prompt, random_length, output_len))

            if len(filtered_dataset) >= num_requests:
                break
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

    while len(filtered_dataset) < num_requests:
        filtered_dataset.extend(filtered_dataset[:num_requests - len(filtered_dataset)])

    if not filtered_dataset:
        raise ValueError("No samples found in the dataset with the given input length.")

    return filtered_dataset




def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
    stddev_input_len: Optional[int]=None
) -> List[Tuple[str, str, int, int]]:
    assert input_len > prefix_len, "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."

    with open(dataset_path) as f:
        poem_lines = f.readlines()

    poem_token_ids = tokenizer(poem_lines).input_ids
    average_poem_len = sum(len(token_ids) for token_ids in poem_token_ids) / len(poem_token_ids)

    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [{"role": "user", "content": base_prompt}]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False
    )
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert input_len > base_prompt_offset, f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    assert prefix_len > base_prompt_offset, f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."

    num_prefix_lines = round((prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    sampled_requests: List[Tuple[str, str, int, int]] = []
    for _ in range(num_requests):
        random_len = sample_random_positive_int(input_len, stddev_input_len)
        num_input_lines = max(0, min(len(poem_lines), round((random_len - base_prompt_offset) / average_poem_len)))

        if num_input_lines < num_prefix_lines:
            num_input_lines = num_prefix_lines

        try:
            sampled_lines = "".join(
                prefix_lines + random.sample(poem_lines, num_input_lines - num_prefix_lines)
            )
        except ValueError:
            sampled_lines = "".join(prefix_lines + poem_lines[:num_input_lines - num_prefix_lines])

        prompt = f"{base_prompt}{sampled_lines}"
        message = [{"role": "user", "content": prompt}]
        prompt_formatted = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False
        )
        prompt_len = len(tokenizer(prompt_formatted).input_ids)
        sampled_requests.append((prompt, prompt_formatted, prompt_len, output_len))

    return sampled_requests



def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    input_lens = np.random.randint(
        int(input_len * range_ratio),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    input_requests = []
    for i in range(num_prompts):
        prompt = tokenizer.decode(
            [(offsets[i] + i + j) % tokenizer.vocab_size for j in range(input_lens[i])]
        )
        input_requests.append((prompt, int(input_lens[i]), int(output_lens[i]), None))

    return input_requests


def transform_sampled_prompts(sampled_prompts: dict, tokenizer: PreTrainedTokenizerBase, fixed_output_len: Optional[int]=100) -> List[Tuple[str, int, int]]:
    filtered_prompts = []
    for dataset_id, value in sampled_prompts.items():
        random.shuffle(value)
        for each in value:
            prompt = each["prompt"]
            completion = each["response"]
            prompt_token_ids = tokenizer(prompt).input_ids
            prompt_len = len(prompt_token_ids)
            completion_token_ids = tokenizer(completion).input_ids
            output_len = len(completion_token_ids) if fixed_output_len is None else fixed_output_len
            filtered_prompts.append((prompt, prompt_len, output_len, dataset_id))
    return filtered_prompts

async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request
    
        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentiles: List[float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    total_request_throughput = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    request_duration:List[float] = []
    reqs_output_throughputs: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            output_len = len(
                tokenizer(outputs[i].generated_text, add_special_tokens=False).input_ids
            )
            actual_output_lens.append(output_len)
            outputs[i].output_len = output_len
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            request_duration.append(outputs[i].latency)
            total_request_throughput += output_len / outputs[i].latency
            reqs_output_throughputs.append(outputs[i].req_output_throughput)
            completed += 1

        else:
            actual_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
        
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_request_throughput=total_request_throughput / completed if completed > 0 else 0,
        mean_output_throughput_per_user = np.mean(reqs_output_throughputs or 0),
        std_output_throughput_per_user = np.std(reqs_output_throughputs or 0),
        median_output_throughput_per_user = np.median(reqs_output_throughputs,0),
        percentiles_output_throughput_per_user=[
            (p, np.percentile(reqs_output_throughputs or 0, p)) for p in selected_percentiles
        ],
        min_output_throughput_per_user=np.min(reqs_output_throughputs or [0]),
        max_output_throughput_per_user=np.max(reqs_output_throughputs or [0]),
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles
        ],
        min_ttft_ms=np.min(ttfts or [0]) * 1000,
        max_ttft_ms=np.max(ttfts or [0]) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[
            (p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles
        ],
        min_tpot_ms=np.min(tpots or [0]) * 1000,
        max_tpot_ms=np.max(tpots or [0]) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles
        ],
        min_itl_ms=np.min(itls or [0]) * 1000,
        max_itl_ms=np.max(itls or [0]) * 1000,
        mean_e2el_ms=np.median(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
        min_e2el_ms=np.min(e2els or [0]) * 1000,
        max_e2el_ms=np.max(e2els or [0]) * 1000,
    )
    return metrics, outputs


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int, Optional[str]]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentiles: List[str],
    benchmark_id: Optional[UUID] = None,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len, dataset_id = input_requests[0]
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        best_of=best_of,
        use_beam_search=use_beam_search,
    )
    
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        print("Initial test run completed. Starting main benchmark run...")

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")

    print(f"Traffic request rate: {request_rate}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len, dataset_id = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
            dataset_id=dataset_id,
            benchmark_id=benchmark_id,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, outputs = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentiles=selected_percentiles,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))    
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total Token throughput (tok/s):", metrics.total_token_throughput
        )
    )

    result = {
        "duration": benchmark_duration,
        "successful_requests": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        # "request_throughput": metrics.request_throughput,
        # "output_throughput": metrics.output_throughput,
        "output_throughput_per_user":[output.req_output_throughput for output in outputs],
        # "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": [output.output_len for output in outputs],
        "e2els": [output.latency for output in outputs],
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        **asdict(metrics)
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function print and add statistics of the specified
        # metric.
        if metric_attribute_name == "output_throughput_per_user":
            print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
            print(
                "{:<40} {:<10.2f}".format(
                    f"Mean {metric_name} (tok/s):",
                    getattr(metrics, f"mean_{metric_attribute_name}"),
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    f"Median {metric_name} (tok/s):",
                    getattr(metrics, f"median_{metric_attribute_name}"),
                )
            )
            result[f"mean_{metric_attribute_name}"] = getattr(
                metrics, f"mean_{metric_attribute_name}"
            )
            result[f"median_{metric_attribute_name}"] = getattr(
                metrics, f"median_{metric_attribute_name}"
            )
            result[f"std_{metric_attribute_name}"] = getattr(
                metrics, f"std_{metric_attribute_name}"
            )
            for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}"):
                p_word = str(int(p)) if int(p) == p else str(p)
                print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (tok/s):", value))
                result[f"p{p_word}_{metric_attribute_name}"] = value
                
        else:
            print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
            print(
                "{:<40} {:<10.2f}".format(
                    f"Mean {metric_name} (ms):",
                    getattr(metrics, f"mean_{metric_attribute_name}_ms"),
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    f"Median {metric_name} (ms):",
                    getattr(metrics, f"median_{metric_attribute_name}_ms"),
                )
            )
            result[f"mean_{metric_attribute_name}_ms"] = getattr(
                metrics, f"mean_{metric_attribute_name}_ms"
            )
            result[f"median_{metric_attribute_name}_ms"] = getattr(
                metrics, f"median_{metric_attribute_name}_ms"
            )
            result[f"std_{metric_attribute_name}_ms"] = getattr(
                metrics, f"std_{metric_attribute_name}_ms"
            )
            for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
                p_word = str(int(p)) if int(p) == p else str(p)
                print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
                result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")
    process_one_metric("output_throughput_per_user","OTPU","Output Throughput per user")

    print("=" * 50)

    return result, outputs


def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    sampled_prompts = args.sampled_prompts
    benchmark_id = args.benchmark_id

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    tokenizer = get_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code)

    if sampled_prompts:
        input_requests = transform_sampled_prompts(
            sampled_prompts,
            tokenizer,
            fixed_output_len=args.mean_output_len
        )

    elif args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next "
            "release. Please use '--dataset-name' and "
            "'--dataset-path' in the future runs.",
            stacklevel=2,
        )
        # input_requests = sample_sharegpt_requests(
        #     dataset_path=args.dataset,
        #     num_requests=args.num_prompts,
        #     tokenizer=tokenizer,
        #     fixed_output_len=args.sharegpt_output_len,
        # )
        input_requests = sample_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            mean_input_len = args.mean_input_len,
            stddev_input_len=args.std_input_len,
            fixed_output_len=args.mean_output_len,
            input_column=args.input_column,
            output_column=args.output_column,
            seed=args.seed
        )

    elif args.dataset_name == "hf":
        # input_requests = sample_sharegpt_requests(
        #     dataset_path=args.dataset_path,
        #     num_requests=args.num_prompts,
        #     tokenizer=tokenizer,
        #     fixed_output_len=args.sharegpt_output_len,
        # )
        input_requests = sample_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            mean_input_len = args.mean_input_len,
            stddev_input_len=args.std_input_len,
            fixed_output_len=args.mean_output_len,
            input_column=args.input_column,
            output_column=args.output_column,
            seed=args.seed
            
        )

    elif args.dataset_name == "sonnet":
        # Do not format the prompt, pass to message directly
        if args.backend == "openai-chat":
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.mean_input_len,
                stddev_input_len = args.std_input_len,
                output_len=args.mean_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [
                (prompt, prompt_len, output_len)
                for prompt, prompt_formatted, prompt_len, output_len in input_requests
            ]
        else:
            assert (
                tokenizer.chat_template or tokenizer.default_chat_template
            ), "Tokenizer/model must have chat template for sonnet dataset."
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.mean_input_len,
                stddev_input_len = args.std_input_len,
                output_len=args.mean_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [
                (prompt_formatted, prompt_len, output_len)
                for prompt, prompt_formatted, prompt_len, output_len in input_requests
            ]

    elif args.dataset_name == "random":
        input_requests = sample_random_requests(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
        )

    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    benchmark_result, individual_responses = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            benchmark_id=benchmark_id,
        )
    )

    result_json: Dict[str, Any] = {}

    # Setup
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json["date"] = current_dt
    result_json["backend"] = backend
    result_json["model_id"] = model_id
    result_json["tokenizer_id"] = tokenizer_id
    result_json["best_of"] = args.best_of
    result_json["use_beam_search"] = args.use_beam_search
    result_json["num_prompts"] = args.num_prompts
    result_json["input_tokens"] = args.random_input_len
    result_json["output_tokens"] = args.random_output_len
    result_json["concurrency"] = args.num_prompts

    # Metadata
    if args.metadata:
        for item in args.metadata:
            if "=" in item:
                kvstring = item.split("=")
                result_json[kvstring[0].strip()] = kvstring[1].strip()
            else:
                raise ValueError(
                    "Invalid metadata format. Please use KEY=VALUE format."
                )

    # Traffic
    result_json["request_rate"] = (
        args.request_rate if args.request_rate < float("inf") else "inf"
    )

    # Merge with benchmark result
    result_json = {**result_json, **benchmark_result}

    if args.save_result:
        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = (
            f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"  # noqa
        )
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)

    return result_json, individual_responses


def get_args():
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the dataset, will be deprecated in the " "next release.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="hf",
        choices=["hf", "sonnet", "random"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None, help="Path to the dataset."
    )
    parser.add_argument(
        "--input-column",type=str,default="input",help="provide the valid input column of the dataset"
    )
    parser.add_argument(
        "--output-column",type=str,default=None,help="provide the valid Output column of the dataset"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and " "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
    "--mean-input-len",type=int,default=200,help="Specify the mean input token length."
    )
    
    parser.add_argument(
        "--std-input-len",type=int,default=None,help="The standard deviation of number of tokens to send in the prompt for the request."
    )

    parser.add_argument(
        "--mean-output-len",
        type=int,
        default=100,
        help="Output length for each request. Overrides the output length "
    )
    # parser.add_argument(
    #     "--sonnet-input-len",
    #     type=int,
    #     default=550,
    #     help="Number of input tokens per request, used only for sonnet dataset.",
    # )
    # parser.add_argument(
    #     "--sonnet-output-len",
    #     type=int,
    #     default=150,
    #     help="Number of output tokens per request, used only for sonnet dataset.",
    # )
    parser.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help="Number of prefix tokens per request, used only for sonnet dataset.",
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random sampling.",
    )
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random sampling.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random sampling.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    # parser.add_argument(
    #     "--percentile-metrics",
    #     type=str,
    #     default="ttft,tpot,itl,e2el",
    #     help="Comma-seperated list of selected metrics to report percentils. "
    #     "This argument specifies the metrics to report percentiles. "
    #     'Allowed metric names are "ttft", "tpot", "itl", "e2el(End to End Latency)".,"otpr(Output Throughput per request) '
    #     'Default value is "ttft,tpot,itl".',
    # )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-seperated list of percentiles for selected metrics. "
        'To report 25-th, 50-th, and 75-th percentiles, use "25,50,75". '
        'Default value is "99". '
        'Use "--percentile-metrics" to select metrics.',
    )

    parser.add_argument(
        "--sampled-prompts",
        type=json.loads,  # Convert JSON string to Python list of dicts
        default=[],
        help="Provide a list of sampled prompts as JSON string (Example: '[{\"id\": 1, \"prompt\": \"Q1\", \"response\": \"A1\"}]')"
    )
    parser.add_argument(
        "--benchmark_id",
        type=str,
        default=None,
        help="UUID of the benchmark run. If not provided, will be set to None."
    )
    args = parser.parse_args()

    return args


def run_benchmark(model, input_len, output_len, num_prompts, base_url, sampled_prompts: Optional[dict] = None, benchmark_id: Optional[UUID] = None):
    # args = get_args()
    class BenchmarkArgs:
        def __init__(self, model, input_len, output_len, num_prompts, base_url, sampled_prompts: Optional[dict] = None, benchmark_id: Optional[UUID] = None):
            self.model = model
            self.tokenizer = "Qwen/Qwen2.5-0.5B-Instruct"
            self.num_prompts = num_prompts
            self.seed = 42
            self.disable_tqdm = False
            self.backend = "vllm"
            self.percentile_metrics = "ttft,tpot,itl,e2el"
            self.metric_percentiles = "25,75,95,99"
            self.base_url = base_url
            self.endpoint = "/chat/completions"
            self.best_of = 1
            self.use_beam_search = False
            self.dataset = None
            self.dataset_name = "random"
            self.dataset_path = None
            self.input_column = "input"
            self.output_column = None
            self.mean_input_len = 200
            self.std_input_len = None
            self.mean_output_len = 100
            # self.sonnet_input_len = 550
            # self.sonnet_output_len = 150
            self.sonnet_prefix_len = 200
            self.random_input_len = input_len
            self.random_output_len = output_len
            self.random_range_ratio = 1.0
            self.request_rate = float("inf")
            self.trust_remote_code = True
            self.profile = False
            self.save_result = False
            self.metadata = None
            self.result_dir = "./results"
            self.result_filename = None
            self.sampled_prompts = sampled_prompts
            self.benchmark_id = benchmark_id

    args = BenchmarkArgs(model, input_len, output_len, num_prompts, base_url, sampled_prompts, benchmark_id)

    return main(args)


if __name__ == "__main__":
    args = get_args()
    main(args)