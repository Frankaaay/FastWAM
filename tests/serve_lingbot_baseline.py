"""Launch the official LingBot server with an explicit benchmark and assembly."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("libero", "robotwin"), required=True)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    assert subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip() == "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
    for component in ("vae", "text_encoder", "tokenizer", "transformer"):
        assert (args.assembly / component).is_dir(), component
    sys.path.insert(0, str(args.repo / "wan_va"))
    import wan_va_server as server
    cfg = copy.deepcopy(server.VA_CONFIGS[args.benchmark])
    cfg.wan22_pretrained_model_name_or_path = str(args.assembly)
    cfg.host = "127.0.0.1"
    server.VA_CONFIGS[args.benchmark] = cfg
    server.init_logger()
    server.run(SimpleNamespace(config_name=args.benchmark, port=args.port,
                               save_root=str(args.output)))


if __name__ == "__main__":
    main()
