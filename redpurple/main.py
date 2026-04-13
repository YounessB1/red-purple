#!/usr/bin/env python3
"""Red-Purple — entry point (host side)"""

import argparse
import sys

from dotenv import load_dotenv
load_dotenv()

from redpurple.docker_runtime import DockerRuntime


def main() -> None:
    parser = argparse.ArgumentParser(prog="red-purple")
    parser.add_argument("-t", "--target", required=True, help="Target URL to pentest")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the Docker image")
    parser.add_argument("--max-iter", type=int, default=100, help="Max agent iterations (default: 100)")
    args = parser.parse_args()

    try:
        runtime = DockerRuntime()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    runtime.build_image(force=args.rebuild)

    print(f"Starting agent against {args.target}...")
    container_id = runtime.create_sandbox(name="agent", target=args.target, max_iter=args.max_iter)

    exit_code = runtime.stream_logs(container_id)

    runtime.destroy_sandbox(container_id)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
