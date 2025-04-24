from threading import Lock
from typing import List, Optional, Tuple
import threading
import os
import json
import time

from locust import HttpUser, task, between, events

def sample_sharegpt_requests(
        dataset_path: str,
        num_requests: int,
) -> List[Tuple[str, int, int]]:
    # Load the dataset
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset = [(data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset]
    
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break
        prompt = dataset[i][0]
        completion = dataset[i][1]
        filtered_dataset.append((prompt, completion))

    return filtered_dataset


class BenchmarkUser(HttpUser):
    prompts: List[str]
    model: str
    routing_strategy: str
    num_requests: int
    index_lock: Lock
    request_index: int
    output_file: str
    all_results = []
    results_lock = Lock()

    wait_time = between(1, 3)  # Adjust as needed

    def on_start(self):
        self.model = os.getenv("MODEL", "bud-tiny-gpu-b733b3ca")
        self.routing_strategy = os.getenv("ROUTING_STRATEGY", "")
        self.output_file = os.getenv("OUTPUT_FILE", "result.jsonl")
        self.num_requests = int(os.getenv("NUM_REQUESTS", 20))
        self.request_index = 0  # Initialize request index
        self.index_lock = threading.Lock()  # Initialize a lock for thread safety
        input_requests = sample_sharegpt_requests(
            dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json',
            num_requests=self.num_requests,
        )
        self.prompts = list(map(lambda x: x[0], input_requests))

    @task
    def send_request(self):
        try:
            # Lock access to request_index to make it thread-safe
            with self.index_lock:
                prompt = self.prompts[self.request_index % len(self.prompts)]
                self.request_index += 1  # Increment request index safely

            st = time.perf_counter()
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-1234",
            }

            if self.routing_strategy:
                headers["routing-strategy"] = self.routing_strategy

            response = self.client.post("/v1/chat/completions", json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "stream": True,
                "stream_options": {
                    "include_usage": True,
                }
            }, headers=headers, stream=True)

            # Initialize variables for streaming
            generated_text = ""
            ttft = 0.0
            
            most_recent_timestamp = time.perf_counter()
            itl = []
            usage_data = None

            for line in response.iter_lines():
                if line:
                    # Remove "data: " prefix and handle [DONE]
                    # chunk = line.decode('utf-8').removeprefix("data: ")
                    chunk = line.decode('utf-8')
                    if chunk.startswith("data: "):
                        chunk = chunk[len("data: "):]
                    if chunk == "[DONE]":
                        continue

                    timestamp = time.perf_counter()
                    data = json.loads(chunk)

                    if choices := data.get("choices"):
                        content = choices[0]["delta"].get("content")
                        # First token
                        if ttft == 0.0 and content:
                            ttft = timestamp - st

                        # Decoding phase
                        elif content:
                            itl.append(timestamp - most_recent_timestamp)

                        generated_text += content or ""
                    elif usage := data.get("usage"):
                        usage_data = usage

                    most_recent_timestamp = timestamp

            latency = time.perf_counter() - st
            
            # Prepare result to write to file
            result = {
                "model": self.model,
                "prompt": prompt,
                "output": generated_text,
                "prompt_tokens": usage_data["prompt_tokens"] if usage_data else None,
                "output_tokens": usage_data["completion_tokens"] if usage_data else None,
                "total_tokens": usage_data["total_tokens"] if usage_data else None,
                "latency": latency,
                "ttft": ttft * 1000,
                "itl_mean": sum(itl) / len(itl) * 1000 if itl else None,
                "itl_std": (sum((x - (sum(itl) / len(itl))) ** 2 for x in itl) / len(itl)) ** 0.5 * 1000 if itl else None,
                "throughput": usage_data["completion_tokens"] / latency if usage_data else None,
            }

            # Add the result to all_results with thread safety
            with self.results_lock:
                self.all_results.append(result)

            with open(self.output_file, "a") as outfile:
                json.dump(result, outfile)
                outfile.write("\n")

        except Exception as e:
            print(f"Error: {str(e)}")

    @classmethod
    def generate_summary(cls):
        """Generate and save summary statistics from all results"""
        if not cls.all_results:
            print("No results to summarize")
            return

        total_time = max(r["latency"] for r in cls.all_results)
        total_output_tokens = sum(r["output_tokens"] for r in cls.all_results if r["output_tokens"])

        def calculate_percentiles(values):
            sorted_values = sorted(values)
            return {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "p50": sorted_values[int(len(values) * 0.50)],
                "p75": sorted_values[int(len(values) * 0.75)],
                "p90": sorted_values[int(len(values) * 0.90)],
                "p95": sorted_values[int(len(values) * 0.95)],
                "p99": sorted_values[int(len(values) * 0.99)],
            }

        summary = {
            "total_requests": len(cls.all_results),
            "successful_requests": sum(1 for r in cls.all_results if r.get("output")),
            "tokens": {
                "input": calculate_percentiles([r["prompt_tokens"] for r in cls.all_results if r["prompt_tokens"]]),
                "output": calculate_percentiles([r["output_tokens"] for r in cls.all_results if r["output_tokens"]]),
            },
            "latency": calculate_percentiles([r["latency"] for r in cls.all_results]),
            "ttft": calculate_percentiles([r["ttft"] for r in cls.all_results]),
            "itl": calculate_percentiles([r["itl_mean"] for r in cls.all_results if r["itl_mean"]]),
            "throughput": {
                **calculate_percentiles([r["throughput"] for r in cls.all_results if r["throughput"]]),
                "total_tokens": sum(r["total_tokens"] for r in cls.all_results if r["total_tokens"]),
                "total_output_tokens": total_output_tokens,
                "total_time_seconds": total_time,
                "overall_tokens_per_second": total_output_tokens / total_time if total_time > 0 else 0,
            }
        }

        # Save summary to a file
        summary_file = "benchmark_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        def print_metric_stats(name, stats, unit=""):
            print(f"\n{name}:")
            print(f"  Mean: {stats['mean']:.2f}{unit}")
            print(f"  Min: {stats['min']:.2f}{unit}")
            print(f"  Max: {stats['max']:.2f}{unit}")
            print(f"  P50: {stats['p50']:.2f}{unit}")
            print(f"  P75: {stats['p75']:.2f}{unit}")
            print(f"  P90: {stats['p90']:.2f}{unit}")
            print(f"  P95: {stats['p95']:.2f}{unit}")
            print(f"  P99: {stats['p99']:.2f}{unit}")

        # Print summary to console
        print("\n=== Benchmark Summary ===")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Successful Requests: {summary['successful_requests']}")

        print("\nToken Statistics:")
        print_metric_stats("Input Tokens", summary['tokens']['input'])
        print_metric_stats("Output Tokens", summary['tokens']['output'])
        print_metric_stats("End to End Latency", summary['latency'], " s")
        print_metric_stats("Time to First Token", summary['ttft'], " ms")
        print_metric_stats("Inter-token Latency", summary['itl'], " ms")
        print_metric_stats("Per-Request Throughput", summary['throughput'], " tokens/sec")

        print("\nOverall Throughput:")
        print(f"  Total Tokens: {summary['throughput']['total_tokens']}")
        print(f"  Total Output Tokens: {summary['throughput']['total_output_tokens']}")
        print(f"  Total Time (s): {summary['throughput']['total_time_seconds']:.2f}")
        print(f"  Overall Throughput (tokens/sec): {summary['throughput']['overall_tokens_per_second']:.2f}")

@events.test_stop.add_listener
def on_test_stop(**kwargs):
    """Generate summary when the test stops"""
    BenchmarkUser.generate_summary()
