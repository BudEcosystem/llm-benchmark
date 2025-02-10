import os
import sys
import uuid

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(project_root)
sys.path.insert(0, project_root)

# from argparse import Namespace
from llm_benchmark.utils.logger import logger
from llm_benchmark.controller import single_node as single_node_controller
# from llm_benchmark.engine import tools as engine_tools
from llm_benchmark.benchmark.litellm_proxy.utils import compute_latency_factors

# get engine config
engine_config = {
    "args": {"model": "openai/gpt-4-turbo", "port": 4000, "debug": True},
    "envs": {
        "LITELLM_MASTER_KEY": "sk-1234",
    },
}

model = engine_config["args"]["model"]

# deploy litellm proxy
engine_kwargs = {
    "engine": "litellm_proxy",
    "docker_image": "ghcr.io/berriai/litellm:main-latest",
    "env_values": {
        "LITELLM_MASTER_KEY": "sk-1234",
    },
    "result_dir": "./results",
    "extra_args": {"model": "openai/gpt-4-turbo", "port": 4000, "debug": True},
    "device": "cpu",
    "profile_model": False,
}

engine_config_id = str(uuid.uuid4())[:8]
logger.info(f"Deploying model with engine config id: {engine_config_id}")

# engine_tools.save_engine_config(Namespace(**engine_config["args"]))
# engine_tools.save_engine_envs(engine_config["envs"] if engine_config else {})

container_id = single_node_controller.deploy_model(
    engine_config_id=engine_config_id,
    port=engine_config["args"]["port"],
    **engine_kwargs,
)

# engine_tools.create_engine_summary("litellm_proxy", engine_config_id, model)

# compute latency factors
base_url = f"http://localhost:{engine_config['args']['port']}/v1"
litellm_master_key = engine_config["envs"].get("LITELLM_MASTER_KEY")
request_metadata = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "litellm_proxy_url": base_url,
    "litellm_master_key": litellm_master_key
}
import pdb; pdb.set_trace()
latency_factors = compute_latency_factors(model, request_metadata=request_metadata)
print(latency_factors)