from __future__ import annotations

import json
import os
import sys
import urllib.request


def main() -> int:
    port = os.environ.get("YERELLM_PORT", "8000")
    url = os.environ.get("YERELLM_HEALTH_URL", f"http://127.0.0.1:{port}/api/health")
    try:
        with urllib.request.urlopen(url, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"healthcheck failed: {exc}", file=sys.stderr)
        return 1
    return 0 if payload.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
