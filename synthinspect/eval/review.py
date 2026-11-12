"""The review band: auto-pass, auto-reject, and send the unsure parts to a person.

Bosch describes the same pattern for its inspection AI: the model decides on
its own and passes only doubtful cases to human inspectors. The band is two
thresholds on the part score:

    score <  t_pass   -> pass   (chosen so at most `max_escape` of defects slip through)
    score >  t_reject -> reject (chosen so at most `max_false_reject` of good parts are scrapped)
    otherwise         -> review

Thresholds are picked on one half of the test set and applied to the other
half, then the halves swap (two-fold cross-fitting), repeated over several
random splits. No part is ever scored with thresholds it helped choose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Band:
    t_pass: float
    t_reject: float

    def decide(self, scores: np.ndarray) -> np.ndarray:
        """0 = pass, 1 = review, 2 = reject."""
        s = np.asarray(scores)
        d = np.ones(len(s), dtype=int)
        d[s < self.t_pass] = 0
        d[s > self.t_reject] = 2
        return d

    def to_dict(self) -> dict:
        return {"t_pass": float(self.t_pass), "t_reject": float(self.t_reject)}


def fit_band(scores: np.ndarray, labels: np.ndarray, max_escape: float = 0.0, max_false_reject: float = 0.01) -> Band:
    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    defects, good = np.sort(s[y]), np.sort(s[~y])[::-1]
    if len(defects) == 0 or len(good) == 0:
        raise ValueError("Calibration needs both good and defective parts")
    k = int(np.floor(max_escape * len(defects)))  # defects allowed to pass
    t_pass = float(defects[min(k, len(defects) - 1)])
    m = int(np.floor(max_false_reject * len(good)))  # good parts allowed to be rejected
    t_reject = float(good[min(m, len(good) - 1)])
    if t_pass > t_reject:
        # Calibration data is separable: put one boundary in the middle of the gap,
        # rather than hugging the highest good score, so new parts have margin on both sides.
        mid = (t_pass + t_reject) / 2
        return Band(mid, mid)
    return Band(t_pass, t_reject)


def outcome(band: Band, scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    d = band.decide(scores)
    y = np.asarray(labels).astype(bool)
    n_def, n_good = max(1, y.sum()), max(1, (~y).sum())
    return {
        "automation": float((d != 1).mean()),
        "review": float((d == 1).mean()),
        "escape_rate": float(((d == 0) & y).sum() / n_def),  # defective parts auto-passed
        "false_reject_rate": float(((d == 2) & ~y).sum() / n_good),  # good parts auto-rejected
        "n_escapes": int(((d == 0) & y).sum()),
        "n_false_rejects": int(((d == 2) & ~y).sum()),
    }


def cross_fit(
    scores: np.ndarray,
    labels: np.ndarray,
    max_escape: float = 0.0,
    max_false_reject: float = 0.01,
    repeats: int = 20,
    seed: int = 0,
    strata: list[str] | None = None,
    gate: np.ndarray | None = None,
    gate_tau: float = 0.0,
) -> dict[str, float]:
    """Average out-of-fold outcome over ``repeats`` random 2-fold splits (stratified by label/defect type).

    ``gate`` (optional, one value per part) is the explanation check: a part the band would
    auto-reject goes to review instead when its gate value is below ``gate_tau`` (for example,
    when the top detection's own explanation mostly lies outside its box).
    """
    s, y = np.asarray(scores, float), np.asarray(labels).astype(int)
    g = None if gate is None else np.asarray(gate, float)
    keys = np.array(strata) if strata is not None else y.astype(str)
    rng = np.random.default_rng(seed)
    totals: dict[str, list[float]] = {}
    for _ in range(repeats):
        fold = np.zeros(len(s), int)
        for k in np.unique(keys):
            idx = np.where(keys == k)[0]
            rng.shuffle(idx)
            fold[idx[: len(idx) // 2]] = 1
        pooled = {"n_escapes": 0, "n_false_rejects": 0, "decided": 0, "n": 0}
        for f in (0, 1):
            cal, app = fold == f, fold != f
            band = fit_band(s[cal], y[cal], max_escape, max_false_reject)
            d = band.decide(s[app])
            if g is not None:
                d[(d == 2) & (g[app] < gate_tau)] = 1
            pooled["n_escapes"] += int(((d == 0) & (y[app] == 1)).sum())
            pooled["n_false_rejects"] += int(((d == 2) & (y[app] == 0)).sum())
            pooled["decided"] += int((d != 1).sum())
            pooled["n"] += int(app.sum())
        res = {
            "automation": pooled["decided"] / pooled["n"],
            "escape_rate": pooled["n_escapes"] / max(1, (y == 1).sum()),
            "false_reject_rate": pooled["n_false_rejects"] / max(1, (y == 0).sum()),
        }
        for k, v in res.items():
            totals.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in totals.items()} | {f"{k}_sd": float(np.std(v)) for k, v in totals.items()}
