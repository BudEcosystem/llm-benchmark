import json
import math
import pathlib
import random
import subprocess
import time
import numpy as np
from typing import Any, Dict, Tuple

from llm_benchmark.utils.logger import logger

from transformers import LlamaTokenizerFast


RESULTS_VERSION = "2023-08-31"


class LLMPerfResults:
    def __init__(
        self,
        name: str,
        metadata: Dict[str, Any] = None,
    ):
        self.name = name
        self.metadata = metadata or {}
        self.timestamp = int(time.time())
        self.metadata["timestamp"] = self.timestamp
        self.version = RESULTS_VERSION

    def to_dict(self):
        data = {
            "version": self.version,
            "name": self.name,
        }
        data.update(self.metadata)
        data = flatten_dict(data)
        return data

    def json(self):
        data = self.to_dict()
        return json.dumps(data)


def upload_to_s3(results_path: str, s3_path: str) -> None:
    """Upload the results to s3.

    Args:
        results_path: The path to the results file.
        s3_path: The s3 path to upload the results to.

    """

    command = ["aws", "s3", "sync", results_path, f"{s3_path}/"]
    result = subprocess.run(command)
    if result.returncode == 0:
        print("Files uploaded successfully!")
    else:
        print("An error occurred:")
        print(result.stderr)


def randomly_sample_sonnet_lines_prompt(
    prompt_tokens_mean: int = 550,
    prompt_tokens_stddev: int = 250,
    expect_output_tokens: int = 150,
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer")
) -> Tuple[str, int]:
    """Generate a prompt that randomly samples lines from a the shakespeare sonnet at sonnet.txt.

    Args:
        prompt_length_mean: The mean length of the prompt to generate.
        prompt_len_stddev: The standard deviation of the length of the prompt to generate.
        expect_output_tokens: The number of tokens to expect in the output. This is used to
        determine the length of the prompt. The prompt will be generated such that the output
        will be approximately this many tokens.

    Note:
        tokens will be counted from the sonnet using the Llama tokenizer. Using one tokenizer
        ensures a fairer comparison across different LLMs. For example, if gpt 3.5 tokenizes
        a prompt in less tokens than Llama2, then this will be reflected in the results since
        they will be fed identical prompts.

    Returns:
        A tuple of the prompt and the length of the prompt.
    """

    get_token_length = lambda text: len(tokenizer.encode(text))

    prompt = (
        "Randomly stream lines from the following text "
        f"with {expect_output_tokens} output tokens. "
        "Don't generate eos tokens:\n\n"
    )
    # get a prompt length that is at least as long as the base
    num_prompt_tokens = sample_random_positive_int(
        prompt_tokens_mean, prompt_tokens_stddev
    )
    while num_prompt_tokens < get_token_length(prompt):
        num_prompt_tokens = sample_random_positive_int(
            prompt_tokens_mean, prompt_tokens_stddev
        )
    remaining_prompt_tokens = num_prompt_tokens - get_token_length(prompt)
    sonnet_path = pathlib.Path(__file__).parent.resolve() / "sonnet.txt"
    with open(sonnet_path, "r") as f:
        sonnet_lines = f.readlines()
    random.shuffle(sonnet_lines)
    sampling_lines = True
    while sampling_lines:
        for line in sonnet_lines:
            line_to_add = line
            if remaining_prompt_tokens - get_token_length(line_to_add) < 0:
                # This will cut off a line in the middle of a word, but that's ok since an
                # llm should be able to handle that.
                line_to_add = line_to_add[: int(math.ceil(remaining_prompt_tokens))]
                sampling_lines = False
                prompt += line_to_add
                break
            prompt += line_to_add
            remaining_prompt_tokens -= get_token_length(line_to_add)
    return (prompt, num_prompt_tokens, None)


def sample_random_positive_int(mean: int, stddev: int) -> int:
    """Sample random numbers from a gaussian distribution until a positive number is sampled.

    Args:
        mean: The mean of the gaussian distribution to sample from.
        stddev: The standard deviation of the gaussian distribution to sample from.

    Returns:
        A random positive integer sampled from the gaussian distribution.
    """
    ret = -1
    while ret <= 0:
        ret = int(random.gauss(mean, stddev))
    return ret


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def compute_latency_factors(model: str, request_metadata: Dict[str, Any], llm_api: str = "litellm_proxy") -> Dict[str, float]:
    """Compute the latency factors for a model."""
    from llm_benchmark.benchmark.litellm_proxy.token_benchmark_ray import (
        run_token_benchmark as litellm_run_benchmark,
)
    from llm_benchmark.benchmark.tools import format_llmperf_result
    num_completed_requests = 3
    mean_input_token = 100
    stddev_input_token = 20
    mean_output_token = 100
    stddev_output_token = 20
    N_input = []
    N_output = []
    T_total = []
    for _ in range(num_completed_requests):
        result_output, individual_responses = litellm_run_benchmark(
            model, 1, 1, mean_input_token, stddev_input_token, mean_output_token, stddev_output_token, llm_api=llm_api, request_metadata=request_metadata
        )
        result_output, _ = format_llmperf_result(result_output, individual_responses)
        if result_output["error_messages"] and result_output["completed"] == 0:
            raise Exception(f"Error messages: {', '.join(result_output['error_messages'])}")
        N_input.append(result_output["input_tokens"])
        N_output.append(result_output["output_tokens"])
        T_total.append(float(result_output["duration"]))
    # Solve for T_base, T_input, T_output
    # T_total = T_base + T_input * N_input + T_output * N_output
    
    # Build matrices
    A = np.column_stack(([1] * len(N_input), N_input, N_output))
    b = np.array(T_total)

    # Solve for T_base, T_input, T_output
    T_base, T_input, T_output = np.linalg.lstsq(A, b, rcond=None)[0]

    print(f"T_base: {T_base} ms, T_input: {T_input} ms/token, T_output: {T_output} ms/token")
    return {"T_base": float(T_base), "T_input": float(T_input), "T_output": float(T_output)}

def calculate_mock_delay(num_input_tokens, num_output_tokens, latency_factors):
    return latency_factors["T_base"] + latency_factors["T_input"] * num_input_tokens + latency_factors["T_output"] * num_output_tokens


def sample_requests(
    concurrent_requests,
    mean_input_tokens,
    stddev_input_tokens,
    mean_output_tokens,
    stddev_output_tokens,
    tokenizer,
):
    num_output_tokens_list = []
    prompts = []
    for i in range(concurrent_requests):
        num_output_tokens = (sample_random_positive_int(
            mean_output_tokens, stddev_output_tokens
        ))
        num_output_tokens_list.append(num_output_tokens)

        prompts.append(randomly_sample_sonnet_lines_prompt(
            prompt_tokens_mean=mean_input_tokens,
            prompt_tokens_stddev=stddev_input_tokens,
            expect_output_tokens=num_output_tokens,
            tokenizer=tokenizer
        ))
    return prompts, num_output_tokens_list


def transform_sampled_requests(sampled_prompts: dict, tokenizer) -> list[Tuple[str, int]]:
    get_token_length = lambda text: len(tokenizer.encode(text))
    sampled_requests = []
    num_output_token_list = []
    for dataset_id, prompts in sampled_prompts.items():
        for each in prompts:
            prompt = each["prompt"]
            sampled_requests.append((prompt, get_token_length(prompt), dataset_id))
            num_output_token_list.append(get_token_length(each["response"]))
    return sampled_requests, num_output_token_list

