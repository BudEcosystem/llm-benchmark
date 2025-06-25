import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union
import aiohttp
import asyncio
import argparse
import time
import json
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset


def sample_random_positive_int(mean: int, stddev: int) -> int:
    ret = -1
    while ret <= 0:
        ret = int(random.gauss(mean, stddev))
    return ret


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    mean_input_len: int,
    seed: int,
    fixed_output_len: Optional[int] = None,
    stddev_input_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    if stddev_input_len is None:
        stddev_input_len = 0.1 * (mean_input_len)

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
    random.seed(seed)
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

        random_length = sample_random_positive_int(mean_input_len, stddev_input_len)
        if prompt_len == random_length:
            filtered_dataset.append((prompt, prompt_len, output_len))
        elif prompt_len > random_length:
            sliced_prompt = tokenizer.decode(
                prompt_token_ids[:random_length], skip_special_tokens=True
            )
            filtered_dataset.append((sliced_prompt, random_length, output_len))

    if not filtered_dataset:
        raise ValueError("No samples found in the dataset with the given input length.")

    while len(filtered_dataset) < num_requests:
        filtered_dataset.extend(
            filtered_dataset[: num_requests - len(filtered_dataset)]
        )

    return filtered_dataset


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    mean_input_len: int,
    seed: int,
    fixed_output_len: int,
    input_column: str = "input",
    output_column: Optional[str] = None,
    stddev_input_len: int = None,
) -> List[Tuple[str, int, int]]:
    if mean_input_len is not None and mean_input_len < 4:
        raise ValueError("mean_input_len must be at least 4")

    if stddev_input_len is None:
        stddev_input_len = 0.1 * (mean_input_len)

    try:
        if os.path.isfile(dataset_path):
            with open(dataset_path, "r") as f:
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
            output_len = (
                len(completion_token_ids)
                if fixed_output_len is None
                else fixed_output_len
            )

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
        filtered_dataset.extend(
            filtered_dataset[: num_requests - len(filtered_dataset)]
        )

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
    stddev_input_len: Optional[int] = None,
) -> List[Tuple[str, str, int, int]]:
    assert input_len > prefix_len, (
        "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."
    )

    with open(dataset_path) as f:
        poem_lines = f.readlines()

    poem_token_ids = tokenizer(poem_lines).input_ids
    average_poem_len = sum(len(token_ids) for token_ids in poem_token_ids) / len(
        poem_token_ids
    )

    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [{"role": "user", "content": base_prompt}]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False
    )
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert input_len > base_prompt_offset, (
        f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    )
    assert prefix_len > base_prompt_offset, (
        f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."
    )

    num_prefix_lines = round((prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    sampled_requests: List[Tuple[str, str, int, int]] = []
    for _ in range(num_requests):
        random_len = sample_random_positive_int(input_len, stddev_input_len)
        num_input_lines = max(
            0,
            min(
                len(poem_lines),
                round((random_len - base_prompt_offset) / average_poem_len),
            ),
        )

        if num_input_lines < num_prefix_lines:
            num_input_lines = num_prefix_lines

        try:
            sampled_lines = "".join(
                prefix_lines
                + random.sample(poem_lines, num_input_lines - num_prefix_lines)
            )
        except ValueError:
            sampled_lines = "".join(
                prefix_lines + poem_lines[: num_input_lines - num_prefix_lines]
            )

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
        input_requests.append((prompt, int(input_lens[i]), int(output_lens[i])))

    return input_requests


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


def create_request_payload(text: Union[str, List[str]], model: str) -> Dict[str, Any]:
    """Create the request payload for the embedding API."""
    return {
        "input": text if isinstance(text, list) else [text],
        "model": model,
    }


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    _pbar: Optional[tqdm] = None,
) -> tuple[float, int]:
    """Send a POST request asynchronously and measure response time."""
    start_time = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=10 * 60) as response:
            elapsed_time = time.perf_counter() - start_time
            response.raise_for_status()
            response = await response.json()  # Ensure the response is fully read
            try:
                data = response["data"]
                assert len(data) == len(payload["input"]), (
                    "The length of the data is not equal to the length of the input"
                )
                assert isinstance(data[0].get("embedding"), list), (
                    "The embedding is not a list"
                )

                return (
                    elapsed_time,
                    len(data[0]["embedding"]),
                    response.get("usage", {}).get("prompt_tokens", 0),
                )
            except Exception as e:
                print(f"Error processing response: {e}")
            return None, None, None
    except asyncio.TimeoutError:
        print("Request timed out after 10 minutes")
        return None, None, None
    except Exception as e:
        print(f"Request failed: {e}")
        return None, None, None
    finally:
        if _pbar is not None:
            _pbar.update(1)


async def warmup_server(
    url: str,
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    num_tokens: int,
    num_requests: int,
):
    """Warm up the server with a specified number of requests."""
    print(f"Sending {num_requests} warm-up requests...")
    async with aiohttp.ClientSession() as session:
        payload = [
            create_request_payload([prompt], model)
            for prompt, _, _ in sample_random_requests(
                num_tokens, num_tokens, num_requests, 1, tokenizer
            )
        ]

        for i in tqdm(range(num_requests), desc="Warm-up requests"):
            elapsed_time, embedding_dim, prompt_tokens = await send_request(
                session, url, payload[min(i, len(payload) - 1)]
            )


async def benchmark_fn(
    url: str,
    model: str,
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    input_column: str,
    output_column: str,
    num_tokens: int,
    num_requests: int,
    max_concurrent: int,
    request_rate: float,
    batch_size: int,
    seed: int,
) -> List[float]:
    """Benchmark the embedding server with customization options."""
    pbar = tqdm(total=num_requests, desc="Benchmarking requests")

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(max_concurrent)

        # Generate dataset
        input_requests = sample_sharegpt_requests(
            dataset_path=dataset_path,
            num_requests=num_requests * batch_size,
            tokenizer=tokenizer,
            mean_input_len=num_tokens,
            seed=seed,
        )

        async def task(payload):
            async with semaphore:
                return await send_request(session, url, payload, pbar)

        tasks = []
        inputs = []
        async for request in get_request(input_requests, request_rate):
            if len(inputs) < batch_size:
                prompt, _, _ = request
                inputs.append(prompt)
            if len(inputs) >= batch_size:
                payload = create_request_payload(inputs[:batch_size], model)
                tasks.append(task(payload))
                inputs = inputs[batch_size:]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

    pbar.close()

    return [timing for timing, _, _ in responses if timing is not None]


async def benchmark(
    args, tokenizer: PreTrainedTokenizerBase, selected_percentiles: List[float]
) -> pd.DataFrame:
    """Run all benchmark combinations and return results as DataFrame."""

    # Convert single values to lists for consistent processing
    params = {
        "max_concurrent": args.num_requests,
        "request_rate": args.request_rate,
        "num_tokens": args.num_tokens,
        "warmup_requests": args.warmup_requests,
        "requests": args.num_requests,
        "batch_size": args.batch_size,
    }

    if args.warmup_requests > 0:
        # Run warmup
        await warmup_server(
            args.url,
            args.model,
            tokenizer,
            params["num_tokens"],
            params["warmup_requests"],
        )

    benchmark_start_time = time.perf_counter()

    # Run benchmark
    timings = await benchmark_fn(
        args.url,
        args.model,
        args.dataset,
        tokenizer,
        args.input_column,
        args.output_column,
        params["num_tokens"],
        params["requests"],
        params["max_concurrent"],
        params["request_rate"],
        params["batch_size"],
        args.seed,
    )

    benchmark_duration = time.perf_counter() - benchmark_start_time

    # Analyze results
    metrics = calculate_metrics(
        timings, params["requests"], benchmark_duration, selected_percentiles
    )

    # Add parameters and metadata to results
    result = {
        "duration": benchmark_duration,
        "total_requests": params["requests"],
        "completed": metrics["completed"],
        "request_throughput": metrics["request_throughput"],
        "model": args.model,
        "concurrency": params["max_concurrent"],
        "batch_size": params["batch_size"],
        "request_rate": params["request_rate"],
        "num_tokens": params["num_tokens"],
    }

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics["completed"]))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print(
        "{:<40} {:<10}".format(
            "Total input tokens:",
            result["total_requests"] * result["num_tokens"] * result["batch_size"],
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics["request_throughput"]
        )
    )

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
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                metrics.get(f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                metrics.get(f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = metrics.get(
            f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = metrics.get(
            f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = metrics.get(
            f"std_{metric_attribute_name}_ms"
        )
        for p, value in metrics.get(f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


def calculate_metrics(
    timings: List[float],
    total_requests: int,
    duration: float,
    selected_percentiles: List[float],
) -> Dict[str, Any]:
    """Analyze benchmark results and return metrics."""
    if not timings:
        return {
            "completed": 0,
            "request_throughput": 0,
            "mean_e2el_ms": 0,
            "median_e2el_ms": 0,
            "std_e2el_ms": 0,
            "percentiles_e2el_ms": [],
        }

    return {
        "completed": len(timings),
        "request_throughput": total_requests / duration,
        "mean_e2el_ms": np.mean(timings or 0) * 1000,
        "std_e2el_ms": np.std(timings or 0) * 1000,
        "median_e2el_ms": np.median(timings or 0) * 1000,
        "percentiles_e2el_ms": [
            (p, np.percentile(timings or 0, p) * 1000) for p in selected_percentiles
        ],
    }


def get_args():
    parser = argparse.ArgumentParser(description="Benchmark an embedding server.")
    parser.add_argument(
        "--url", type=str, default="http://localhost:8989/v1/embeddings"
    )
    parser.add_argument("--model", type=str, default="Alibaba-NLP/gte-large-en-v1.5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument("--max-concurrent", type=int, nargs="+", default=[5])
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[10])
    parser.add_argument("--warmup-requests", type=int, nargs="+", default=[5])
    parser.add_argument("--num-requests", type=int, nargs="+", default=[50])
    parser.add_argument(
        "--request-rate",
        type=float,
        nargs="+",
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1])
    parser.add_argument("--engine", type=str, default="torch")
    parser.add_argument("--save-path", type=str, default="benchmark_results.xlsx")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--input-column",
        type=str,
        default="input",
        help="provide the valid input column of the dataset",
    )
    parser.add_argument(
        "--output-column",
        type=str,
        default=None,
        help="provide the valid Output column of the dataset",
    )

    args = parser.parse_args()

    return args


def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.num_tokens > tokenizer.model_max_length:
        raise ValueError(
            f"The number of tokens is greater than the model max length: {tokenizer.model_max_length}"
        )

    # Run benchmarks
    return asyncio.run(
        benchmark(
            args,
            tokenizer,
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
        )
    )


def run_benchmark(model, input_len, output_len, num_prompts, base_url):
    # args = get_args()
    dataset = os.path.join(
        os.path.expanduser("~"), "ShareGPT_V3_unfiltered_cleaned_split.json"
    )
    if not os.path.exists(dataset):
        print(
            "ShareGPT_V3_unfiltered_cleaned_split.json not found in home directory, using random dataset"
        )
        dataset = "random"

    class BenchmarkArgs:
        def __init__(self, model, input_len, output_len, num_prompts, base_url):
            self.model = model
            self.tokenizer = model
            self.num_requests = num_prompts
            self.seed = 42
            self.disable_tqdm = False
            self.backend = "budlatent"
            self.percentile_metrics = "e2el"
            self.metric_percentiles = "95"
            self.url = base_url
            self.dataset = dataset
            self.dataset_name = dataset
            self.dataset_path = None
            self.input_column = "input"
            self.output_column = None
            self.mean_input_len = 200
            self.std_input_len = None
            self.num_tokens = input_len
            self.random_output_len = output_len
            self.random_range_ratio = 1.0
            self.request_rate = float("inf")
            self.trust_remote_code = True
            self.profile = False
            self.save_result = False
            self.metadata = None
            self.result_dir = "./results"
            self.result_filename = None

            self.max_concurrent = num_prompts
            self.batch_size = 1
            self.warmup_requests = 0

    args = BenchmarkArgs(model, input_len, output_len, num_prompts, base_url+"/embeddings")

    return main(args)
