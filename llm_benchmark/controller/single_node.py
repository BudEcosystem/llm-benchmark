import os
import time
import requests
import subprocess
from typing import Optional


docker_flags_mapping = {"DOCKER_CPUSET_CPUS": "--cpuset-cpus"}


def build_docker_run_command(
    docker_image: str,
    run_command: Optional[str],
    env_values: list,
    result_dir: str,
    extra_args: list,
    engine_config_id: str,
    device: str = "cpu",
    profile_model: bool = False,
) -> list:
    """Constructs the docker run command."""

    env_vars = (
        [f"-e {k}='{v}'" for k, v in env_values.items() if not k.startswith("DOCKER_")]
        if env_values
        else []
    )
    env_vars.append(f"-e ENGINE_CONFIG_ID={engine_config_id}")
    env_vars.append("-e PROFILER_RESULT_DIR=/root/results/")

    docker_flags = []
    for flag, mapping in docker_flags_mapping.items():
        if flag in env_values:
            docker_flags.append(f"{mapping}={env_values[flag]}")

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
        *docker_flags,
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
        ]
    )

    if run_command:
        docker_command.append(run_command)

    docker_command.extend(arg_vars)

    return docker_command

def build_podman_run_command(
    docker_image: str,
    run_command: Optional[str],
    env_values: list,
    result_dir: str,
    extra_args: list,
    engine_config_id: str,
    device: str = "cpu",
    profile_model: bool = False,
) -> list:
    """Constructs the podman run command."""

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

    podman_command = [
        "podman",
        "run",
        "-d",
        "-it",
        "--rm",
        "--network=host",
        "--ipc=host",
        "--group-add keep-groups",
        "--cap-add=SYS_PTRACE",
        "--security-opt seccomp=unconfined",
    ]

    if device in ["gpu","rocm"]:
        podman_command.append("--device=/dev/kfd")
        podman_command.append("--device=/dev/dri")
    elif device == "hpu":
        podman_command.append("--runtime=habana")

    podman_command.extend(
        [
            *env_vars,
            *volumes,
            docker_image,
        ]
    )
    if run_command:
        docker_command.append(run_command)

    docker_command.extend(arg_vars)

    return podman_command


def build_engine_run_command(
    docker_image: str,
    run_command: Optional[str],
    env_values: list,
    result_dir: str,
    extra_args: list,
    engine_config_id: str,
    device: str = "cpu",
    profile_model: bool = False,
    use_podman: bool = False,
):
    if use_podman:
        return build_podman_run_command(
            docker_image,
            run_command,
            env_values,
            result_dir,
            extra_args,
            engine_config_id,
            device,
            profile_model,
        )
    return build_docker_run_command(
            docker_image,
            run_command,
            env_values,
            result_dir,
            extra_args,
            engine_config_id,
            device,
            profile_model,
        )

def deploy_model(
    engine: str,
    docker_image: str,
    run_command: Optional[str],
    env_values: dict,
    result_dir: str,
    extra_args: list,
    engine_config_id: str,
    port: int,
    warmup_sec: int = 30,
    device: str = "cpu",
    profile_model: bool = False,
    health_check_endpoint: Optional[str] = None,
    use_podman: bool = False,
) -> str:
    container_id = None
    try:
        if engine == "litellm_proxy":
            extra_args.pop('model')
        run_command = build_engine_run_command(
            docker_image,
            run_command,
            env_values,
            result_dir,
            extra_args,
            engine_config_id,
            device,
            profile_model,
            use_podman,
        )
        print(f"Deploying with Docker image {docker_image}...")
        print("Executing command: " + " ".join(run_command))

        engine_config_dir = f"{result_dir}/engine/{engine_config_id}"
        os.makedirs(engine_config_dir, exist_ok=True)

        with open(f"{engine_config_dir}/run_engine.sh", "w") as bash_file:
            bash_file.write(" ".join(run_command))
        os.chmod(f"{engine_config_dir}/run_engine.sh", 0o755)
        print("Run command saved to run_engine.sh for execution.")

        container = subprocess.run(
            f"bash {engine_config_dir}/run_engine.sh",
            capture_output=True,
            text=True,
            check=True,
            shell=True,
        )
        container_id = container.stdout.strip()
        print(f"Container ID: {container_id}")
        # Wait for the container to initialize
        time.sleep(warmup_sec)

        if not verify_server_status(
            engine,
            container_id,
            f"http://localhost:{port}",
            health_check_endpoint,
            env_values=env_values,
            use_podman=use_podman
        ):
            raise RuntimeError("Server failed to start after maximum retries.")

        print(f"Container {container_id} is now running.")
        return container_id
    except subprocess.CalledProcessError as e:
        print(f"Failed to deploy model. Orchestrator error: {e.stderr}")
        raise
    except Exception as e:
        print(f"Error deploying model: {e}")
        if container_id:
            remove_container(container_id, use_podman)
        raise


def remove_container(container_id: str, use_podman: bool = False):
    try:
        rm_command = ["docker", "rm", "-f", container_id]
        if use_podman:
            rm_command = ["podman", "rm", "-f", container_id]
        subprocess.run(rm_command, check=True)
        print(f"Container {container_id} removed.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to remove container {container_id}. Orchestrator error: {e.stderr}")
        raise


def verify_container_status(container_id: str, use_podman:bool = False):
    try:
        inspect_command = ["docker", "inspect", "--format='{{.State.Running}}'", container_id]
        if use_podman:
            inspect_command = ["podman", "inspect", "--format='{{.State.Running}}'", container_id]
        result = subprocess.run(
            inspect_command,
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
        print(f"Failed to check status of container {container_id}. Orchestrator error: {e.stderr}")
        raise


def verify_server_status(
    engine: str,
    container_id: str,
    base_url: str,
    health_check_endpoint: Optional[str] = None,
    max_retries: int = 100,
    retry_interval: int = 60,
    env_values: Optional[dict] = None,
    use_podman = False
) -> bool:
    """Verifies if the server is up and running by checking the API status."""
    url = (
        f"{base_url}/v1/models"
        if not health_check_endpoint
        else f"{base_url}{health_check_endpoint}"
    )
    headers = None

    if engine == "litellm_proxy":
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        url = f"{base_url}/health/readiness"
        litellm_master_key = env_values.get("LITELLM_MASTER_KEY")
        headers = {"Authorization": f"Bearer {litellm_master_key}"}

    for attempt in range(max_retries):
        try:
            if not verify_container_status(container_id, use_podman):
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


def get_container_pid(container_id: str, use_podman:bool = False):
    pid = None
    try:
        inspect_command = ["docker", "inspect", "--format='{{.State.Pid}}'", container_id]
        if use_podman:
            inspect_command = ["podman", "inspect", "--format='{{.State.Pid}}'", container_id]
        output = subprocess.run(
            inspect_command,
            capture_output=True,
            text=True,
            check=True,
        )
        pid = output.stdout.strip().strip("'").strip('"')
        print(f"Container PID is {pid}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to get container {container_id} pid. Orchestrator error: {e.stderr}")
    finally:
        return pid
