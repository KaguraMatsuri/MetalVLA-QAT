"""Portable command entry for Qwen/MPS + MLX MetalVLA."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("launch", "run"):
        # The waiting supervisor must not initialize MLX/Metal or load a model.
        import argparse
        from metalvla_qat.mac.automation import launch, run
        parser = argparse.ArgumentParser()
        parser.add_argument("command", choices=("launch", "run"))
        parser.add_argument("--config", type=Path, required=True)
        args = parser.parse_args()
        (launch if args.command == "launch" else run)(args.config)
    else:
        from metalvla_qat.mac.cli import main
        main()
