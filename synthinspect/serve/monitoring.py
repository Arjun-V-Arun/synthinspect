"""Minimal production monitoring with no extra dependencies: counters, latency histogram, score drift.

Drift is the Population Stability Index between the score distribution the
model saw at release (stored in config.json) and a rolling window of live
scores. Rule of thumb: PSI under 0.1 is stable, 0.1 to 0.25 is worth a look,
over 0.25 means the inputs have changed (new lighting, new supplier, dirty lens)
and the thresholds should not be trusted until someone checks.
"""

from __future__ import annotations

import threading
from collections import Counter, deque

import numpy as np

BUCKETS_MS = (10, 25, 50, 100, 250, 500, 1000, 2500)


def psi(reference: np.ndarray, live: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    ref = np.asarray(reference, float)
    cur = np.asarray(live, float)
    if len(ref) < 2 or len(cur) < 2:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + eps
    c = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((c - r) * np.log(c / r)))


class Monitor:
    def __init__(self, reference_scores: list[float] | None = None, window: int = 500):
        self.lock = threading.Lock()
        self.decisions: Counter[str] = Counter()
        self.latency = Counter()
        self.latency_sum = 0.0
        self.count = 0
        self.scores: deque[float] = deque(maxlen=window)
        self.reference = np.asarray(reference_scores or [], float)

    def record(self, decision: str, score: float, latency_ms: float) -> None:
        with self.lock:
            self.decisions[decision] += 1
            self.count += 1
            self.latency_sum += latency_ms
            for b in BUCKETS_MS:
                if latency_ms <= b:
                    self.latency[b] += 1
            self.scores.append(score)

    def drift(self) -> float:
        with self.lock:
            return psi(self.reference, np.array(self.scores)) if len(self.scores) >= 50 else float("nan")

    def prometheus(self, model_version: str) -> str:
        lines = [
            "# HELP inspect_decisions_total Parts by decision.",
            "# TYPE inspect_decisions_total counter",
        ]
        with self.lock:
            for d in ("pass", "review", "reject"):
                lines.append(f'inspect_decisions_total{{decision="{d}",model="{model_version}"}} {self.decisions[d]}')
            lines += ["# HELP inspect_latency_ms End-to-end inference latency.", "# TYPE inspect_latency_ms histogram"]
            for b in BUCKETS_MS:
                lines.append(f'inspect_latency_ms_bucket{{le="{b}"}} {self.latency[b]}')
            lines += [f'inspect_latency_ms_bucket{{le="+Inf"}} {self.count}', f"inspect_latency_ms_sum {self.latency_sum:.3f}", f"inspect_latency_ms_count {self.count}"]
            review_rate = self.decisions["review"] / self.count if self.count else 0.0
        d = self.drift()
        lines += [
            "# HELP inspect_review_rate Share of parts sent to a person.",
            "# TYPE inspect_review_rate gauge",
            f"inspect_review_rate {review_rate:.4f}",
            "# HELP inspect_score_psi Drift of live scores vs release (NaN until 50 parts).",
            "# TYPE inspect_score_psi gauge",
            f"inspect_score_psi {d if np.isfinite(d) else 'NaN'}",
        ]
        return "\n".join(lines) + "\n"
