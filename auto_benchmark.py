import os
import yaml
import uuid
import json
import time
import hashlib
import argparse
import threading
import itertools
from argparse import Namespace

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime

from llm_benchmark.controller import single_node as single_node_controller
from llm_benchmark.benchmark import tools as benchmark_tools
from llm_benchmark.profiler import tools as profiler_tools
from llm_benchmark.hardware import tools as hardware_tools
from llm_benchmark.hardware import monitor as hw_monitor
from llm_benchmark.engine import tools as engine_tools
from llm_benchmark.model import analysis as model_tools
from llm_benchmark.benchmark.litellm_proxy.utils import compute_latency_factors


def create_config(run_config):
    configs = []
    concurrencies = (
        [int(x) for x in run_config["num_concurrent_requests"]]
        if isinstance(run_config["num_concurrent_requests"], list)
        else [run_config["num_concurrent_requests"]]
    )

    if "token_pairs" in run_config:
        for token_pair in run_config["token_pairs"]:
            input_token, output_token = map(int, token_pair.split(","))
            for concurrency in concurrencies:
                config = {
                    "input_tokens": input_token,
                    "output_tokens": output_token,
                    "concurrency": concurrency,
                }
                configs.append(config)
        return configs
    
    input_tokens = (
        [int(x) for x in run_config["mean_input_tokens"]]
        if isinstance(run_config["mean_input_tokens"], list)
        else [run_config["mean_input_tokens"]]
    )
    output_tokens = (
        [int(x) for x in run_config["mean_output_tokens"]]
        if isinstance(run_config["mean_output_tokens"], list)
        else [run_config["mean_output_tokens"]]
    )
    

    for input_token in input_tokens:
        if input_token < 20:
            print("Skipping input token: ", input_token, " because it is less than 20")
            continue
        for output_token in output_tokens:
            for concurrency in concurrencies:
                config = {
                    "input_tokens": input_token,
                    "output_tokens": output_token,
                    "concurrency": concurrency,
                }
                configs.append(config)
    return configs


def load_checkpoint(ckpt_path: str):
    if os.path.isdir(ckpt_path):
        filepaths = sorted(Path(ckpt_path).iterdir(), key=lambda t: t.stat().st_mtime)
        ckpt_path = filepaths[-1] if len(filepaths) else None

    if not os.path.isfile(ckpt_path):
        print(f"No checkpoints found in {ckpt_path} for resuming.")
        return

    print(f"Resuming benchmarking from checkpoint {ckpt_path}.")
    with open(ckpt_path, "r") as fp:
        return json.load(fp)


def save_checkpoint(checkpoint, savepath):
    Path(savepath).parent.mkdir(exist_ok=True, parents=True)

    with open(savepath, "w") as fp:
        json.dump(checkpoint, fp, indent=4)


def warmup_benchmark(model, base_url, benchmark_script, env_values=None, latency_factors=None):
    print("Running warmup benchmark")
    _ = benchmark_tools.run_benchmark(
        model,
        base_url,
        250,
        250,
        10,
        benchmark_script,
        os.environ["PROFILER_RESULT_DIR"],
        "warmup",
        env_values=env_values,
        latency_factors=latency_factors,
    )


# Function to process and create combinations
def generate_combinations(config_section):
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


def create_engine_config(engine_config_file):
    with open(engine_config_file, "r") as f:
        engine_config = yaml.safe_load(f)

    # Separate the fixed parameters and the parameters with arrays
    # Process the 'args' section
    fixed_args, array_args, arg_combinations, arg_keys = generate_combinations(
        engine_config["args"]
    )

    # Process the 'envs' section
    fixed_envs, array_envs, env_combinations, env_keys = generate_combinations(
        engine_config["envs"]
    )

    # Create a list of configuration dictionaries with all combinations
    configs = []
    for arg_comb in arg_combinations:
        for env_comb in env_combinations:
            # Create new config dict for each combination
            new_config = {
                "args": fixed_args.copy(),  # Copy fixed args
                "envs": fixed_envs.copy(),  # Copy fixed envs
            }
            # Update with current combination of 'args'
            if arg_comb:
                new_config["args"].update(dict(zip(arg_keys, arg_comb)))

            # Update with current combination of 'envs'
            if env_comb:
                new_config["envs"].update(dict(zip(env_keys, env_comb)))

            # Append the complete config to the list
            configs.append(new_config)

    return configs, engine_config["run_config"]


def run_benchmark(args, engine_config, run_config, checkpoint=None, checkpoint_path=None):
    checkpoint = checkpoint or {}
    base_url = f"http://localhost:{engine_config['args']['port']}/v1"
    model = engine_config["args"].get("model") or engine_config["args"].get(
        "model-path"
    )

    engine_kwargs = {
        "engine": args.engine,
        "docker_image": args.docker_image,
        "env_values": engine_config["envs"] if engine_config else {},
        "result_dir": os.environ["PROFILER_RESULT_DIR"],
        "extra_args": engine_config["args"] if engine_config else [],
        "device": args.device,
        "profile_model": args.profile_model,
        "use_podman": args.use_podman,
        "image_start_command": args.image_start_command
    }
    engine_config_hash = hashlib.sha1(
        json.dumps(engine_kwargs, sort_keys=True).encode()
    ).hexdigest()

    if checkpoint.get(engine_config_hash):
        engine_config_id = checkpoint[engine_config_hash]["engine_config_id"]
    else:
        engine_config_id = str(uuid.uuid4())[:8]
        checkpoint[engine_config_hash] = {
            "engine_config_id": engine_config_id,
            "status": "pending",
            "runs": {},
        }

    engine_tools.save_engine_config(engine_config_id, Namespace(**engine_config["args"]))
    engine_tools.save_engine_envs(engine_config_id, engine_config["envs"] if engine_config else {})

    if args.docker_image:
        try:
            container_id = single_node_controller.deploy_model(
                engine_config_id=engine_config_id,
                port=engine_config["args"]["port"],
                **engine_kwargs,
            )
        except Exception as e:
            print(f"Error during {engine_config_id} deployment: {e}")
            checkpoint[engine_config_hash]["status"] = "deploy_failed"
            return checkpoint
    else:
        container_id = None

    if args.engine_config_id or container_id:
        if args.collect_engine_config:
            try:
                engine_tools.create_engine_summary(args.engine, engine_config_id, model)
            except Exception as e:
                print(f"Error during {engine_config_id} summary creation: {e}")
                checkpoint[engine_config_hash]["status"] = "engine_summary_failed"
                return checkpoint

    try:
        latency_factors = None
        if args.engine == "litellm_proxy":
            latency_factors = {"T_base": 0.5, "T_input": 0.005, "T_output": 0.005}
        warmup_benchmark(model, base_url, args.benchmark_script, env_values=engine_config["envs"] if engine_config else None, latency_factors=latency_factors)
    except Exception as e:
        print(f"Error during {engine_config_id} warm up: {e}")
        if container_id:
            single_node_controller.remove_container(container_id, args.use_podman)
        checkpoint[engine_config_hash]["status"] = "warmup_failed"
        if container_id:
            single_node_controller.remove_container(container_id, args.use_podman)

        return checkpoint

    log_metrics_task = None
    stop_event = None
    results = []

    device_config, _ = hardware_tools.create_device_config(args.device)

    try:
        configs = create_config(run_config)
        latency_factors = None
        if args.engine == "litellm_proxy":
            litellm_master_key = engine_config["envs"].get("LITELLM_MASTER_KEY")
            request_metadata = {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "litellm_proxy_url": base_url,
                "litellm_master_key": litellm_master_key
            }
            latency_factors = compute_latency_factors(model, request_metadata=request_metadata)
        for config in tqdm(configs, desc="Running benchmarks"):
            print(config)
            run_config_hash = hashlib.sha1(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest()

            run_ckpt = (
                checkpoint.get(engine_config_hash, {})
                .get("runs", {})
                .get(run_config_hash)
            )
            if run_ckpt is not None:
                run_id = run_ckpt["run_id"]
                if run_ckpt.get("status", "") == "success":
                    print(f"Skipping run for {engine_config_id}:{run_id}")
                    continue
            else:
                run_id = str(uuid.uuid4())[:8]
                checkpoint[engine_config_hash]["runs"][run_config_hash] = {
                    "run_id": run_id,
                    "status": "pending",
                }

            stop_event = threading.Event()
            log_metrics_task = threading.Thread(
                target=hw_monitor.log_system_metrics,
                kwargs={
                    "output_dir": os.path.join(
                        os.environ["PROFILER_RESULT_DIR"], model.replace("/", "--")
                    ),
                    "pid": single_node_controller.get_container_pid(container_id, args.use_podman)
                    if container_id is not None
                    else None,
                    "interval": 3,
                    "stop_event": stop_event,
                    "metadata": {
                        "run_id": run_id,
                        "engine_config_id": engine_config_id,
                    },
                },
            )
            log_metrics_task.start()

            result = benchmark_tools.run_benchmark(
                model,
                base_url,
                config["input_tokens"],
                config["output_tokens"],
                config["concurrency"],
                args.benchmark_script,
                os.environ["PROFILER_RESULT_DIR"],
                run_id,
                env_values=engine_config["envs"] if engine_config else None,
                latency_factors=latency_factors,
            )

            result["engine"] = args.engine
            result["engine_config_id"] = engine_config_id
            result["run_id"] = run_id
            result["input_tokens"] = config["input_tokens"]
            result["output_tokens"] = config["output_tokens"]
            result["concurrency"] = config["concurrency"]

            if args.engine != "litellm_proxy":
                _ = model_tools.infer(
                    model_name=model,
                    device_config=device_config,
                    seq_len=config["input_tokens"],
                    num_tokens_to_generate=config["output_tokens"],
                    batch_size_per_gpu=config["concurrency"],
                    tp_size=engine_config["args"].get("tensor-parallel-size", 1),
                    output_dir=os.environ["PROFILER_RESULT_DIR"],
                    run_id=run_id,
                    log_level="ERROR",
                )

            results.append(result)

            time.sleep(1)

            stop_event.set()
            log_metrics_task.join()
            log_metrics_task = None
            stop_event = None
            
            benchmark_tools.create_summary([result], os.environ["PROFILER_RESULT_DIR"], profiler_result=args.profile_model)
            print(result)
            checkpoint[engine_config_hash]["runs"][run_config_hash]["status"] = (
                "success"
            )
            if checkpoint_path:
                save_checkpoint(checkpoint, checkpoint_path)
    except Exception as e:
        print(f"Error during {engine_config_id} benchmark: {e}")
        checkpoint[engine_config_hash]["runs"][run_config_hash]["status"] = (
            "benchmark_failed"
        )
    finally:
        # if container_id:
        #     single_node_controller.remove_container(container_id, args.use_podman)
        if log_metrics_task is not None and stop_event is not None:
            stop_event.set()
            log_metrics_task.join()

        checkpoint[engine_config_hash]["status"] = "done"
        return checkpoint


def main(args):
    os.makedirs(os.environ["PROFILER_RESULT_DIR"], exist_ok=True)

    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(
            args.checkpoint
            or os.path.join(os.environ["PROFILER_RESULT_DIR"], "checkpoints")
        )

    new_checkpoint = deepcopy(checkpoint) if checkpoint is not None else {}
    new_ckpt_path = os.path.join(
        os.environ["PROFILER_RESULT_DIR"],
        "checkpoints",
        datetime.now().strftime("%Y%m%d-%H%M%S") + ".json",
    )

    if args.run_benchmark:
        if args.engine_config_file:
            engine_configs, run_config = create_engine_config(args.engine_config_file)
        else:
            raise ValueError("Engine config file is required")

        for engine_config in tqdm(engine_configs, desc="Running engine configs"):
            new_checkpoint = run_benchmark(
                args, engine_config, run_config, new_checkpoint, new_ckpt_path
            )
            save_checkpoint(new_checkpoint, new_ckpt_path)
            # break

    if args.profile_collectives:
        profiler_tools.profile_collectives(
            max_collective_size=4096 * 8192,
            output_dir=os.environ["PROFILER_RESULT_DIR"],
        )

    if args.profile_hardware:
        hardware_info = hardware_tools.get_hardware_info(
            output_dir=os.environ["PROFILER_RESULT_DIR"]
        )


if __name__ == "__main__":
    """
    python benchmark/auto_benchmark.py --model <model> --docker-image <docker-image> --port <port> --input-tokens <input-tokens> --output-tokens <output-tokens> --concurrency <concurrency>
    """
    args = argparse.ArgumentParser(
        description="Run a token throughput and latency benchmark."
    )

    args.add_argument(
        "--docker-image",
        type=str,
        default=None,
        help="The engine image to be used for the testing.",
    )
    args.add_argument(
        "--engine",
        type=str,
        default="bud",
        choices=["vllm", "sglang", "bud", "litellm_proxy"],
        help="The engine to be used for the testing.",
    )
    args.add_argument(
        "--engine-config-file",
        type=str,
        default=None,
        help="The engine config file to be used for the testing.",
    )
    args.add_argument(
        "--engine-config-id",
        type=str,
        default=None,
        help="The engine config id to be used for the testing.",
    )
    args.add_argument(
        "--run-benchmark",
        action="store_true",
        help="Whether to run the benchmark.",
    )
    args.add_argument(
        "--benchmark-script",
        type=str,
        default="llmperf",
        help="The benchmark script to be used for the testing.",
    )
    args.add_argument(
        "--collect-engine-config",
        action="store_true",
        help="The collect engine config to be used for the testing.",
    )
    args.add_argument(
        "--profile-collectives",
        action="store_true",
        help="Whether to profile the collectives.",
    )
    args.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu", "hpu", "rocm"],
        help="Whether to profile on gpu or cpu.",
    )
    args.add_argument(
        "--cpu-only",
        action="store_true",
        help="Whether to profile only on cpu.",
    )
    args.add_argument(
        "--profile-hardware",
        action="store_true",
        help="Whether to profile the hardware.",
    )
    args.add_argument(
        "--profile-model",
        action="store_true",
        help="Whether to profile the model.",
    )
    args.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume the benchmark from the last checkpoint",
    )
    args.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file path to resume from, if not set and  resume=True, fallbacks to the latest",
    )
    args.add_argument(
        "--use-podman",
        action="store_true",
        help="Whether to use podman instead of docker.",
    )
    args.add_argument(
        "--image-start-command",
        type=str,
        default='',
        help="The command used to start the image.",
    )
    args = args.parse_args()
    main(args)
