import os
import subprocess
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, jsonify, request


app = Flask(__name__)

LLM_URL = os.getenv(
    "LLM_URL",
    os.getenv("VLLM_URL", "http://localhost:11434/v1/chat/completions"),
)
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("VLLM_MODEL", "qwen2.5:1.5b"))
LOG_TAIL_LINES = int(os.getenv("LOG_TAIL_LINES", "200"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "16000"))


def run_kubectl(args: List[str]) -> str:
    completed = subprocess.run(
        ["kubectl", *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return f"kubectl {' '.join(args)} failed:\n{completed.stderr.strip()}"
    return completed.stdout.strip()


def first_present(*values: Optional[str], default: str = "") -> str:
    for value in values:
        if value:
            return value
    return default


def truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text

    half = limit // 2
    omitted = len(text) - limit
    return (
        text[:half]
        + f"\n\n... TRUNCATED {omitted} CHARACTERS TO PROTECT THE LLM CONTEXT WINDOW ...\n\n"
        + text[-half:]
    )


def extract_target(alert: Dict[str, Any]) -> Dict[str, str]:
    labels = alert.get("labels", {})
    return {
        "namespace": first_present(
            labels.get("namespace"),
            labels.get("kubernetes_namespace"),
            labels.get("exported_namespace"),
            default="default",
        ),
        "pod": first_present(
            labels.get("pod"),
            labels.get("pod_name"),
            labels.get("kubernetes_pod_name"),
            labels.get("exported_pod"),
        ),
        "container": first_present(labels.get("container"), labels.get("container_name")),
        "alertname": first_present(labels.get("alertname"), default="UnknownAlert"),
        "severity": first_present(labels.get("severity"), default="unknown"),
    }


def gather_context(namespace: str, pod: str, container: str = "") -> Dict[str, str]:
    log_args = ["-n", namespace, "logs", pod, "--tail", str(LOG_TAIL_LINES)]
    previous_log_args = ["-n", namespace, "logs", pod, "--previous", "--tail", str(LOG_TAIL_LINES)]
    if container:
        log_args.extend(["-c", container])
        previous_log_args.extend(["-c", container])

    return {
        "pod_json": run_kubectl(["-n", namespace, "get", "pod", pod, "-o", "json"]),
        "current_logs": run_kubectl(log_args),
        "previous_logs": run_kubectl(previous_log_args),
        "events": run_kubectl(
            [
                "-n",
                namespace,
                "get",
                "events",
                "--field-selector",
                f"involvedObject.name={pod}",
                "--sort-by=.lastTimestamp",
            ]
        ),
        "describe": run_kubectl(["-n", namespace, "describe", "pod", pod]),
    }


def build_messages(alert: Dict[str, Any], target: Dict[str, str], context: Dict[str, str]) -> List[Dict[str, str]]:
    compact_context = truncate_middle(
        f"""
Alert metadata:
{alert}

Pod JSON:
{context["pod_json"]}

Current logs:
{context["current_logs"]}

Previous logs:
{context["previous_logs"]}

Kubernetes events:
{context["events"]}

kubectl describe pod:
{context["describe"]}
""".strip(),
        MAX_CONTEXT_CHARS,
    )

    system_prompt = (
        "You are a DevOps Site Reliability Engineer. "
        "A Kubernetes pod just crashed. Analyze the alert metadata, pod status, logs, "
        "and Kubernetes events. Output exactly three sentences of Root Cause Analysis, "
        "then output exactly one suggested CLI command to fix or further validate it. "
        "Do not include markdown tables."
    )

    user_prompt = (
        f"Alert target: namespace={target['namespace']} pod={target['pod']} "
        f"container={target.get('container') or 'unknown'}\n\n"
        f"{compact_context}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_llm(messages: List[Dict[str, str]]) -> str:
    response = requests.post(
        LLM_URL,
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 300,
        },
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


@app.post("/webhook")
def webhook():
    payload = request.get_json(force=True)
    alerts = payload.get("alerts", [])

    if not alerts:
        return jsonify({"error": "No alerts found in Alertmanager payload"}), 400

    results = []
    for alert in alerts:
        target = extract_target(alert)
        if not target["pod"]:
            results.append({"error": "Could not find pod label in alert", "alert": alert})
            continue

        context = gather_context(target["namespace"], target["pod"], target["container"])
        messages = build_messages(alert, target, context)
        rca = call_llm(messages)

        print("\n========== AI GENERATED RCA ==========")
        print(f"Alert: {target['alertname']} severity={target['severity']}")
        print(f"Target: {target['namespace']}/{target['pod']}")
        print(rca)
        print("======================================\n")

        results.append({"target": target, "rca": rca})

    return jsonify({"processed": len(results), "results": results})


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
