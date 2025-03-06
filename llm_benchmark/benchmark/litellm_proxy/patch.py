import json
import os
import time
from typing import Any, Dict

import ray
import requests
from typing import List

from llmperf.ray_clients.litellm_client import LiteLLMClient
from llmperf.ray_clients.sagemaker_client import SageMakerClient
from llmperf.ray_clients.vertexai_client import VertexAIClient
from llmperf.ray_llm_client import LLMClient

from llmperf.models import RequestConfig
from llmperf import common_metrics


SUPPORTED_APIS = ["openai", "anthropic", "litellm"]


@ray.remote
class OpenAIChatCompletionsClient(LLMClient):
    """Client for OpenAI Chat Completions API."""

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        request_metadata = request_config.metadata if hasattr(request_config, 'metadata') else {}
        prompt = request_config.prompt
        prompt, prompt_len = prompt

        message = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]
        model = request_config.model
        body = {
            "model": model,
            "messages": message,
            "stream": True,
            # REMOVED: because this is not supported by litellm proxy
            # "ignore_eos":True
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})
        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0

        metrics = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()
        address = os.environ.get("OPENAI_API_BASE") or request_metadata.get("api_base")
        if not address:
            raise ValueError("the environment variable OPENAI_API_BASE must be set.")
        key = os.environ.get("OPENAI_API_KEY") or request_metadata.get("api_key")
        if not key:
            raise ValueError(f"the environment variable OPENAI_API_KEY must be set. {request_metadata}")
        headers = {"Authorization": f"Bearer {key}"}
        if not address:
            raise ValueError("No host provided.")
        if not address.endswith("/"):
            address = address + "/"
        address += "chat/completions"
        try:
            with requests.post(
                address,
                json=body,
                stream=True,
                timeout=600,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    response.raise_for_status()
                for chunk in response.iter_lines(chunk_size=None):
                    chunk = chunk.strip()

                    if not chunk:
                        continue
                    stem = "data: "
                    chunk = chunk[len(stem) :]
                    if chunk == b"[DONE]":
                        continue
                    tokens_received += 1
                    data = json.loads(chunk)

                    if "error" in data:
                        error_msg = data["error"]["message"]
                        error_response_code = data["error"]["code"]
                        raise RuntimeError(data["error"]["message"])
                        
                    delta = data["choices"][0]["delta"]
                    if delta.get("content", None):
                        if not ttft:
                            ttft = time.monotonic() - start_time
                            time_to_next_token.append(ttft)
                        else:
                            time_to_next_token.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                        most_recent_received_token_time = time.monotonic()
                        generated_text += delta["content"]

            total_request_time = time.monotonic() - start_time
            output_throughput = tokens_received / total_request_time

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(error_response_code)

        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token) #This should be same as metrics[common_metrics.E2E_LAT]. Leave it here for now
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len

        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config

@ray.remote
class MockLiteLLMChatCompletionsClient(LLMClient):
    """Client for LiteLLM Chat Completions API."""

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        api_key = request_config.metadata.get("api_key", "fake-api-key") if hasattr(request_config, 'metadata') else None
        if not api_key:
            raise ValueError("the provider api_key must be set.")

        prompt = request_config.prompt
        prompt, prompt_len = prompt

        message = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]

        if not hasattr(request_config, "metadata"):
            raise ValueError("the request config's metadata must be set for mock litellm client.")
        assert (
            request_config.llm_api is not None
        ), "the request config's llm_api must be set."
        assert (
            request_config.metadata.get("mock_delay") is not None
        ), "the request config's mock_delay must be set."

        model = request_config.model
        body = {
            # TODO: uncomment this if running benchmark on local with mock litellm proxy server
            # "model": "openai-test-model",
            "model": model,
            "messages": message,
            "stream": True,
            "mock_response": "This is a mock response for chat completions.",
            "mock_delay": request_config.metadata.get("mock_delay"),
            # TODO: uncomment this if running benchmark on local with mock litellm proxy server
            # "user_config": {
            #     "model_list": [{
            #         "model_name": "openai-test-model",
            #         "litellm_params": {
            #             "model": model,
            #             "api_key": api_key,
            #         }
            #     }]
            # }
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0

        metrics = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()

        address = request_config.metadata.get("litellm_proxy_url", "http://localhost:4000/v1/") if hasattr(request_config, 'metadata') else None
        if not address:
            raise ValueError("No host provided.")

        litellm_master_key = request_config.metadata.get("litellm_master_key", "sk-1234") if hasattr(request_config, 'metadata') else None
        if not litellm_master_key:
            raise ValueError("the environment variable LITELLM_MASTER_KEY must be set.")
        
        headers = {"Authorization": f"Bearer {litellm_master_key}"}

        if not address.endswith("/"):
            address = address + "/"
        address += "chat/completions"
        try:
            with requests.post(
                address,
                json=body,
                stream=True,
                timeout=600,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    response.raise_for_status()
                for chunk in response.iter_lines(chunk_size=None):
                    chunk = chunk.strip()

                    if not chunk:
                        continue
                    stem = "data: "
                    chunk = chunk[len(stem) :]
                    if chunk == b"[DONE]":
                        continue
                    tokens_received += 1
                    data = json.loads(chunk)

                    if "error" in data:
                        error_msg = data["error"]["message"]
                        error_response_code = data["error"]["code"]
                        raise RuntimeError(data["error"]["message"])
                        
                    delta = data["choices"][0]["delta"]
                    if delta.get("content", None):
                        if not ttft:
                            ttft = time.monotonic() - start_time
                            time_to_next_token.append(ttft)
                        else:
                            time_to_next_token.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                        most_recent_received_token_time = time.monotonic()
                        generated_text += delta["content"]

            total_request_time = time.monotonic() - start_time
            output_throughput = tokens_received / total_request_time

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(error_response_code)

        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token) #This should be same as metrics[common_metrics.E2E_LAT]. Leave it here for now
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config

@ray.remote
class LiteLLMChatCompletionsClient(LLMClient):
    """Client for LiteLLM Chat Completions API."""

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        api_key = request_config.metadata.get("api_key", "fake-api-key") if hasattr(request_config, 'metadata') else None
        if not api_key:
            raise ValueError("the provider api_key must be set.")

        prompt = request_config.prompt
        prompt, prompt_len = prompt

        message = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]

        model = request_config.model
        body = {
            "model": model,
            "messages": message,
            "stream": True,
            # "user_config": {
            #     "model_list": [{
            #         "model_name": "test-model",
            #         "litellm_params": {
            #             "model": model,
            #             "api_key": api_key,
            #         }
            #     }]
            # }
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0

        metrics = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()

        address = request_config.metadata.get("litellm_proxy_url", "http://localhost:4000/v1/") if hasattr(request_config, 'metadata') else None
        if not address:
            raise ValueError("No host provided.")

        litellm_master_key = request_config.metadata.get("litellm_master_key", "sk-1234") if hasattr(request_config, 'metadata') else None
        if not litellm_master_key:
            raise ValueError("the environment variable LITELLM_MASTER_KEY must be set.")
        
        headers = {"Authorization": f"Bearer {litellm_master_key}"}

        if not address.endswith("/"):
            address = address + "/"
        address += "chat/completions"
        try:
            with requests.post(
                address,
                json=body,
                stream=True,
                timeout=600,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    response.raise_for_status()
                for chunk in response.iter_lines(chunk_size=None):
                    chunk = chunk.strip()

                    if not chunk:
                        continue
                    stem = "data: "
                    chunk = chunk[len(stem) :]
                    if chunk == b"[DONE]":
                        continue
                    tokens_received += 1
                    data = json.loads(chunk)

                    if "error" in data:
                        error_msg = data["error"]["message"]
                        error_response_code = data["error"]["code"]
                        raise RuntimeError(data["error"]["message"])
                        
                    delta = data["choices"][0]["delta"]
                    if delta.get("content", None):
                        if not ttft:
                            ttft = time.monotonic() - start_time
                            time_to_next_token.append(ttft)
                        else:
                            time_to_next_token.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                        most_recent_received_token_time = time.monotonic()
                        generated_text += delta["content"]

            total_request_time = time.monotonic() - start_time
            output_throughput = tokens_received / total_request_time

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(error_response_code)

        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token) #This should be same as metrics[common_metrics.E2E_LAT]. Leave it here for now
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config



def construct_clients(llm_api: str, num_clients: int) -> List[LLMClient]:
    """Construct LLMClients that will be used to make requests to the LLM API.

    Args:
        llm_api: The name of the LLM API to use.
        num_clients: The number of concurrent requests to make.

    Returns:
        The constructed LLMCLients

    """
    if llm_api == "openai":
        clients = [OpenAIChatCompletionsClient.remote() for _ in range(num_clients)]
    elif llm_api == "sagemaker":
        clients = [SageMakerClient.remote() for _ in range(num_clients)]
    elif llm_api == "vertexai":
        clients = [VertexAIClient.remote() for _ in range(num_clients)]
    elif llm_api == "litellm_proxy":
        clients = [LiteLLMChatCompletionsClient.remote() for _ in range(num_clients)]
    elif llm_api in SUPPORTED_APIS:
        clients = [LiteLLMClient.remote() for _ in range(num_clients)]
    elif llm_api in "mock_litellm_proxy":
        clients = [MockLiteLLMChatCompletionsClient.remote() for _ in range(num_clients)]
    else:
        raise ValueError(
            f"llm_api must be one of the supported LLM APIs: {SUPPORTED_APIS}"
        )

    return clients