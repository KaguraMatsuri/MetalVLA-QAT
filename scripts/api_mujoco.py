"""Open the local vision API input panel without loading model weights."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metalvla_qat.api_panel import PanelServer
from metalvla_qat.api_policy import completion_url


def main():
    parser = argparse.ArgumentParser(description="model api")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        raise ValueError({"invalid_port": args.port})
    base = completion_url(args.base_url) if args.base_url else ""
    with PanelServer(("127.0.0.1", args.port), default_base_url=base) as server:
        def stop(*_):
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        print(json.dumps({"url": server.url, "pid": os.getpid()}), flush=True)
        if not args.no_browser:
            webbrowser.open(server.url)
        server.serve_forever(poll_interval=.2)


if __name__ == "__main__":
    main()
