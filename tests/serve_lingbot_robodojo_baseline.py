"""Launch the pinned RoboDojo-team LingBot backend with its published recipe."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xpolicy", type=Path, required=True)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    assert subprocess.check_output(
        ["git", "-C", str(args.xpolicy), "rev-parse", "HEAD"], text=True
    ).strip() == "53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2"
    assert not subprocess.check_output(
        ["git", "-C", str(args.xpolicy), "diff", "HEAD", "--name-only"], text=True
    ).strip()
    for component in ("vae", "text_encoder", "tokenizer", "transformer"):
        assert (args.assembly / component).is_dir(), component
    sys.path.insert(0, str(args.xpolicy / "policy/LingBot_VA/lingbot_va/wan_va"))
    import wan_va_server as server
    cfg = copy.deepcopy(server.VA_CONFIGS["robotwin30_train"])
    assert cfg.action_dim == 30 and cfg.action_per_frame == 12
    cfg.wan22_pretrained_model_name_or_path = str(args.assembly)
    cfg.host = "127.0.0.1"
    server.VA_CONFIGS["robotwin30_train"] = cfg
    server.init_logger()
    server.run(SimpleNamespace(config_name="robotwin30_train", port=args.port,
                               save_root=str(args.output)))


if __name__ == "__main__":
    main()
