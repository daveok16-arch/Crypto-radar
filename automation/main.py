"""Deterministic entrypoint for the Dormant Radar cloud automation.

No LLM. This task is fixed: poll recent blocks, detect dormant wake-ups, send
alerts. Wrapping a deterministic pipeline in an agent would add token cost,
latency, and — worse — a stochastic step between detection and delivery that
could paraphrase or drop alert content.

Two platform details this handles that are easy to get wrong:

1. **Custom secrets are not injected into the environment.** They must be
   fetched from the agent server's API and passed to the subprocess explicitly.
   Reading `os.environ["TELEGRAM_BOT_TOKEN"]` here would come back empty and the
   run would silently deliver nothing.

2. **A cloud run may land on a fresh pod with an empty filesystem.** The local
   SQLite file cannot be trusted for the two facts that make repeated runs
   correct: which outpoints were already alerted (otherwise every run
   re-alerts) and the scan cursor (otherwise every run rescans). The project's
   `cloud-run` command moves both through the automation KV store when
   available, falling back to a local file when it is not.
"""

import json
import os
import subprocess
import sys
import urllib.request

# Secrets the automation needs, fetched from the agent server and forwarded.
NEEDED_SECRETS = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")


def get_secret(name):
    """Fetch a named secret from the agent server.

    Returns "" when unavailable so a missing secret degrades to "no alerts
    delivered" rather than crashing the run.
    """
    url = os.environ.get("AGENT_SERVER_URL", "").rstrip("/")
    key = os.environ.get("SESSION_API_KEY") or os.environ.get("OH_SESSION_API_KEYS_0", "")
    if not url or not key:
        return ""
    try:
        request = urllib.request.Request(
            f"{url}/api/settings/secrets/{name}",
            headers={"X-Session-API-Key": key},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read().decode().strip()
    except Exception as exc:
        print(f"could not load secret {name}: {exc}", file=sys.stderr)
        return ""


def fire_callback(status="COMPLETED", error=None):
    """Signal run completion. Must run on every exit path."""
    url = os.environ.get("AUTOMATION_CALLBACK_URL", "")
    if not url:
        return
    body = {"status": status, "run_id": os.environ.get("AUTOMATION_RUN_ID", "")}
    if error:
        body["error"] = error
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ.get('AUTOMATION_CALLBACK_API_KEY', '')}",
                },
            )
        )
    except Exception as exc:  # a failed callback must not mask the real result
        print(f"callback error: {exc}")


def main() -> int:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "src")

    # Custom secrets are not ambient; forward them explicitly.
    for name in NEEDED_SECRETS:
        if not env.get(name):
            value = get_secret(name)
            if value:
                env[name] = value
            else:
                print(f"warning: secret {name} unavailable", file=sys.stderr)

    # Tuned for a scheduled run: a small window keeps the request count low,
    # because the public mempool.space endpoint rate-limits aggressively.
    env.setdefault("SCAN_WINDOW_BLOCKS", "3")
    env.setdefault("MAX_TXS_PER_BLOCK", "150")
    env.setdefault("MIN_SPENT_SATS", "500000000")     # 5 BTC floor
    env.setdefault("DORMANT_AFTER_BLOCKS", "105120")  # ~2 years
    env.setdefault("ALERTS_ENABLED", "1")
    env.setdefault("ALERT_MIN_VALUE_SATS", "500000000")
    env.setdefault("ALERT_MIN_DORMANT_YEARS", "2")
    env.setdefault("ALERT_DIGEST_MODE", "0")
    env.setdefault("USE_CLUSTERING", "1")
    env.setdefault("BACKFILL_ENABLED", "0")           # too slow for a timed run
    env.setdefault("DORMANT_RADAR_DB", "/tmp/radar.db")
    env.setdefault("DORMANT_RADAR_CLUSTERS", "/tmp/clusters.db")
    env.setdefault("DORMANT_RADAR_STATE", "/tmp/radar_state.json")

    print("running: dormant-radar cloud-run")
    result = subprocess.run(
        [sys.executable, "-m", "dormant_radar.cli", "cloud-run"],
        env=env,
        capture_output=True,
        text=True,
        timeout=480,
    )
    print(result.stdout)
    if result.stderr:
        # stderr may carry retry warnings; the project redacts secrets there.
        print(result.stderr[-2000:], file=sys.stderr)

    if result.returncode != 0:
        fire_callback("FAILED", f"exit {result.returncode}")
        return result.returncode

    fire_callback("COMPLETED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - must always signal completion
        print(f"fatal: {exc}", file=sys.stderr)
        fire_callback("FAILED", str(exc))
        sys.exit(1)