import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import docker
from docker.errors import DockerException, NotFound
from docker.models.containers import Container
from requests.exceptions import ConnectionError as RequestsConnectionError

IMAGE_NAME = "redpurple:latest"
BUILD_CONTEXT = Path(__file__).parents[1]  # repo root
SOURCE_DIR = Path(__file__).parent         # source/ dir — where Dockerfile lives


def _rewrite_localhost(url: str) -> str:
    """Replace localhost/127.0.0.1 with host.docker.internal so the container can reach the host."""
    parsed = urlparse(url)
    if parsed.hostname in ("localhost", "127.0.0.1", "::1"):
        netloc = parsed.netloc.replace(parsed.hostname, "host.docker.internal", 1)
        url = urlunparse(parsed._replace(netloc=netloc))
    return url


class DockerRuntime:
    def __init__(self) -> None:
        try:
            self.client = docker.from_env(timeout=60)
            self.client.ping()
        except (DockerException, RequestsConnectionError) as e:
            raise RuntimeError(f"Docker is not available: {e}") from e

        self._containers: dict[str, Container] = {}

    def build_image(self, force: bool = False) -> None:
        try:
            self.client.images.get(IMAGE_NAME)
            if not force:
                return
        except docker.errors.ImageNotFound:
            pass

        print(f"Building image {IMAGE_NAME}...")
        _, logs = self.client.images.build(
            path=str(BUILD_CONTEXT),
            dockerfile=str(SOURCE_DIR / "Dockerfile"),
            tag=IMAGE_NAME,
            rm=True,
        )
        for entry in logs:
            line = entry.get("stream", "").rstrip()
            if line:
                print(f"  {line}")
        print("Image built.")

    def create_sandbox(self, name: str, target: str = "", max_iter: int = 100, task: str = "") -> str:
        container_name = f"redpurple-{name}"

        try:
            existing = self.client.containers.get(container_name)
            existing.remove(force=True)
        except NotFound:
            pass

        runs_dir = BUILD_CONTEXT / "runs"
        runs_dir.mkdir(exist_ok=True)

        container = self.client.containers.run(
            IMAGE_NAME,
            detach=True,
            name=container_name,
            cap_add=["NET_ADMIN", "NET_RAW"],
            extra_hosts={"host.docker.internal": "host-gateway"},
            volumes={
                str(BUILD_CONTEXT / "source"): {"bind": "/app/source", "mode": "ro"},
                str(runs_dir): {"bind": "/app/runs", "mode": "rw"},
            },
            environment={
                "REDPURPLE_LLM": os.environ.get("REDPURPLE_LLM", ""),
                "LLM_API_KEY": os.environ.get("LLM_API_KEY", ""),
                "LLM_API_BASE": os.environ.get("LLM_API_BASE", ""),
                "TARGET": _rewrite_localhost(target),
                "MAX_ITER": str(max_iter),
                "TASK": task,
                "LANGFUSE_PUBLIC_KEY": os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
                "LANGFUSE_SECRET_KEY": os.environ.get("LANGFUSE_SECRET_KEY", ""),
                "LANGFUSE_HOST": os.environ.get("LANGFUSE_HOST", ""),
            },
        )

        self._containers[container.id] = container
        return container.id

    def stream_logs(self, container_id: str) -> int:
        container = self.client.containers.get(container_id)
        for chunk in container.logs(stream=True, follow=True):
            print(chunk.decode(), end="", flush=True)
        result = container.wait()
        return result["StatusCode"]

    def destroy_sandbox(self, container_id: str) -> None:
        self._containers.pop(container_id, None)
        try:
            container = self.client.containers.get(container_id)
            container.remove(force=True)
        except NotFound:
            pass
