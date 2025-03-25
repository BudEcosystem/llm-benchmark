import os
import time
import requests
import subprocess
from typing import Optional


def build_docker_run_command(
    docker_image: str,
    env_values: list,
    result_dir: str,
    extra_args: list,
    engine_config_id: str,
    device: str = "cpu",
    profile_model: bool = False,
) -> list:
    """Constructs the docker run command."""

    env_vars = [f"-e {k}='{v}'" for k, v in env_values.items()] if env_values else []
    env_vars.append(f"-e ENGINE_CONFIG_ID={engine_config_id}")
    env_vars.append("-e PROFILER_RESULT_DIR=/root/results/")
    if profile_model:
        env_vars.append("-e ENABLE_PROFILER=True")

    arg_vars = (
        [
            f"--{k}={v}" if not isinstance(v, bool) else f"--{k}"
            for k, v in extra_args.items()
            if not isinstance(v, bool) or v is True
        ]
        if extra_args
        else []
    )

    volumes = [
        f"-v={os.path.expanduser('~')}/.cache:/root/.cache",
        f"-v={result_dir}:/root/results",
    ]

    docker_command = [
        "docker",
        "run",
        "-d",
        "-it",
        "--rm",
        "--shm-size=8G",
        "--privileged",
        "--network=host",
    ]

    if device == "gpu":
        docker_command.append("--gpus=all")
    elif device == "hpu":
        docker_command.append("--runtime=habana")

    docker_command.extend(
        [
            *env_vars,
            *volumes,
            docker_image,
            *arg_vars,
        ]
    )

    return docker_command


def deploy_model(
    engine: str,
    docker_image: str,
    env_values: dict,
    result_dir: str,
    extra_args: list,
    engine_config_id: str,
    port: int,
    warmup_sec: int = 30,
    device: str = "cpu",
    profile_model: bool = False,
) -> str:
    container_id = None
    try:
        if engine == "litellm_proxy":
            extra_args.pop('model')
        docker_command = build_docker_run_command(
            docker_image,
            env_values,
            result_dir,
            extra_args,
            engine_config_id,
            device,
            profile_model,
        )
        print(f"Deploying with Docker image {docker_image}...")
        print("Executing Docker command: " + " ".join(docker_command))

        engine_config_dir = f"{result_dir}/engine/{engine_config_id}"
        os.makedirs(engine_config_dir, exist_ok=True)

        with open(f"{engine_config_dir}/run_docker.sh", "w") as bash_file:
            bash_file.write(" ".join(docker_command))
        os.chmod(f"{engine_config_dir}/run_docker.sh", 0o755)
        print("Docker command saved to run_docker.sh for execution.")

        container = subprocess.run(
            f"bash {engine_config_dir}/run_docker.sh",
            capture_output=True,
            text=True,
            check=True,
            shell=True,
        )
        container_id = container.stdout.strip()
        print(f"Container ID: {container_id}")
        # Wait for the container to initialize
        time.sleep(warmup_sec)

        if not verify_server_status(engine, container_id, f"http://localhost:{port}/v1", env_values=env_values):
            raise RuntimeError("Server failed to start after maximum retries.")

        print(f"Container {container_id} is now running.")
        return container_id
    except subprocess.CalledProcessError as e:
        print(f"Failed to deploy model. Docker error: {e.stderr}")
        raise
    except Exception as e:
        print(f"Error deploying model: {e}")
        if container_id:
            remove_container(container_id)
        raise


def remove_container(container_id: str):
    try:
        subprocess.run(["docker", "rm", "-f", container_id], check=True)
        print(f"Container {container_id} removed.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to remove container {container_id}. Docker error: {e.stderr}")
        raise


def verify_container_status(container_id: str):
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format='{{.State.Running}}'", container_id],
            capture_output=True,
            text=True,
            check=True,
        )
        is_running = result.stdout.strip().strip("'").strip('"') == "true"
        if is_running:
            print(f"Container {container_id} is running.")
        else:
            print(f"Container {container_id} is not running.")
        return is_running
    except subprocess.CalledProcessError as e:
        print(f"Failed to check status of container {container_id}. Docker error: {e.stderr}")
        raise


def verify_server_status(
    engine: str, container_id: str, base_url: str, max_retries: int = 100, retry_interval: int = 60, env_values: Optional[dict] = None
) -> bool:
    """Verifies if the server is up and running by checking the API status."""
    url = f"{base_url}/models"
    headers = None
    
    if engine == "litellm_proxy":
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        url = f"{base_url}/health/readiness"
        litellm_master_key = env_values.get("LITELLM_MASTER_KEY")
        headers = {"Authorization": f"Bearer {litellm_master_key}"}

    for attempt in range(max_retries):
        try:
            if not verify_container_status(container_id):
                print(f"Container {container_id} is not running. Exiting.")
                return False
        except Exception as e:
            print(f"Error verifying container status: {e}")
            return False
        try:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                print("Server is up and running.")
                return True
            else:
                print(f"Server not ready. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error connecting to server: {e}")

        if attempt < max_retries - 1:
            print(
                f"Retrying in {retry_interval} seconds... (Attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(retry_interval)

    print(f"Server failed to start after {max_retries} retries.")
    return False


def get_container_pid(container_id: str):
    pid = None
    try:
        output = subprocess.run(
            ["docker", "inspect", "--format='{{.State.Pid}}'", container_id],
            capture_output=True,
            text=True,
            check=True,
        )
        pid = output.stdout.strip().strip("'").strip('"')
        print(f"Container PID is {pid}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to get container {container_id} pid. Docker error: {e.stderr}")
    finally:
        return pid
