"""Entry-only binary MLP (PyTorch) on RELIANCE 1-minute bars; every lever in mlp_model.yaml. Writes only the crosstab Excel.
Decision point = every 1-minute bar t (first_decision..last_decision). Inputs: bars t-(look_back-1)..t.
Features: per-minute ret/vol, 10 derived, then the 50 phase-2 features (features_phase2.py); the on/off list is the
YAML `features:` (written by build_feature_config.py).
Entry = open(t+1). Forward = close(t+hold). Label 1 if forward - entry > target.move_bar rupees, else 0. No exit, no cost.
Run from modeling/:  uv run python mlp_model.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from features_phase2 import PHASE2, day_arrays, phase2_features, settings

HERE = Path(__file__).resolve().parent
DERIVED = ["Price_Change_High", "Price_Change_Low", "Price_Change_High_Close", "Price_Change_Low_Close",
           "Bouncing_back", "Not_Holding_the_price", "Status_Day_High", "Status_Day_Low",
           "Status_Local_High", "Status_Local_Low"]


def feature_names(lb: int) -> list[str]:
    """All feature names in code order: ret_t-(lb-1)..ret_t, vol_t-(lb-1)..vol_t, the 10 derived, the phase-2 list."""
    lags = [f"t-{k}" for k in range(lb - 1, 0, -1)] + ["t"]
    return [f"ret_{s}" for s in lags] + [f"vol_{s}" for s in lags] + DERIVED + PHASE2


def hm(text: str) -> int:
    hours, minutes = text.split(":")
    return int(hours) * 60 + int(minutes)


def in_ranges(days: pd.DatetimeIndex, ranges: list) -> np.ndarray:
    mask = np.zeros(len(days), dtype=bool)
    for r in ranges:
        mask |= (days >= pd.Timestamp(r["start"])) & (days <= pd.Timestamp(r["end"]))
    return mask


def check_ranges(split: dict) -> None:
    spans = sorted((pd.Timestamp(r["start"]), pd.Timestamp(r["end"]), name)
                   for name in ("train", "test") for r in split[name])
    for a, b in zip(spans, spans[1:]):
        if a[0] > a[1] or b[0] > b[1] or b[0] <= a[1]:
            raise SystemExit(f"split ranges overlap or run backwards: {a[2]} {a[0].date()}..{a[1].date()} "
                             f"and {b[2]} {b[0].date()}..{b[1].date()}")


def window_features(o, h, l, c, v, t: int, lb: int, day_open: int, gap: float) -> np.ndarray:
    """All features for decision minute t, in feature_names(lb) order. Window w = bars t-(lb-1)..t (all present)."""
    w = slice(t - lb + 1, t + 1)
    ret = np.log(c[w] / c[t])                                 # ret_t-29..ret_t; ret_t is always 0
    vol = np.log1p(v[w]) - np.log1p(v[w].mean())               # vol_t-29..vol_t
    counts = [(h[w][1:] > h[w][:-1]).sum(),                    # Price_Change_High: lb-1 = 29 vs-previous-bar compares
              (l[w][1:] < l[w][:-1]).sum(),                    # Price_Change_Low: same 29 comparisons
              (np.round(h[w] - c[w], 2) < gap).sum(),          # Price_Change_High_Close: all lb bars, rounded to paise
              (np.round(c[w] - l[w], 2) < gap).sum()]          # Price_Change_Low_Close
    x = c[t]                                                   # close of the decision bar
    day = slice(day_open, t + 1)                               # session open..t only: causal, no bar after t
    levels = [np.mean(o[w] - l[w].mean()),                     # Bouncing_back: mean(open_i - avg_low)
              np.mean(o[w] - h[w].mean()),                     # Not_Holding_the_price: mean(open_i - avg_high)
              np.nanmax(h[day]) - x, x - np.nanmin(l[day]),    # Status_Day_High, Status_Day_Low
              h[w].max() - x, x - l[w].min()]                  # Status_Local_High, Status_Local_Low
    return np.r_[ret, vol, counts, levels]


def selected_columns(cfg: dict) -> list[int]:
    """Indices (into feature_names order) of the YAML features switched on, in YAML order."""
    names, listed = feature_names(cfg["decision_points"]["look_back_minutes"]), cfg.get("features") or []
    if sorted(str(f.get("name")) for f in listed) != sorted(names) or any(
            not isinstance(f.get("on"), bool) for f in listed):
        raise SystemExit("features: the YAML list must hold every feature name for this look_back exactly once with "
                         "on true/false; rerun make_feature_excel.py and build_feature_config.py")
    keep = [names.index(f["name"]) for f in listed if f["on"]]
    if not keep:
        raise SystemExit("features: at least one feature must be on")
    return keep


# ----------------------------------------------------------------------------- block 1: data ingestion
def build_samples(cfg: dict):
    """X (n, all features), y (n,), move (n,) rupees, exc (n, 2) rupees [high - entry, entry - low over t+1..t+hold],
    is_train (n,), skipped {train, test}, days {train, test}."""
    d, split = cfg["decision_points"], cfg["split"]
    first, last = hm(d["first_decision"]), hm(d["last_decision"])
    lb, hold, day_open = d["look_back_minutes"], d["hold_minutes"], hm(cfg["data"]["session_open"])
    gap = cfg["derived_features"]["close_gap_rupees"]
    if not 2 <= lb <= first - day_open + 1:
        raise SystemExit(f"decision_points.look_back_minutes must be >= 2 and not reach before data.session_open "
                         f"on the first decision (30 bars need first_decision >= 09:44); got {lb}")
    if hold < 1 or first > last:
        raise SystemExit("decision_points: hold_minutes must be >= 1 and first_decision <= last_decision")
    check_ranges(split)
    p = settings(cfg)

    df = pd.read_csv(HERE / cfg["data"]["minute_csv"])
    ts = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Asia/Kolkata")
    df = df.assign(day=ts.dt.tz_localize(None).dt.normalize(), minute=ts.dt.hour * 60 + ts.dt.minute)
    df = df.drop_duplicates(["day", "minute"])
    # previous data day's last close (Gap_pct), from every day in the file, before the weekday and range filters
    prev_close_by_day = df.sort_values(["day", "minute"]).groupby("day")["close"].last().shift(1)
    df = df[df["day"].dt.dayofweek < 5]
    days = pd.DatetimeIndex(sorted(df["day"].unique()))
    train_day, test_day = in_ranges(days, split["train"]), in_ranges(days, split["test"])
    days, train_day = days[train_day | test_day], train_day[train_day | test_day]
    df = df[df["day"].isin(days)]
    prev_close = prev_close_by_day.reindex(days).to_numpy()
    opening = slice(day_open, day_open + p["opening_range_minutes"])

    grid = np.full((len(days), 5, 24 * 60), np.nan)  # open, close, volume, high, low by minute of day; NaN = no bar
    di, mi = days.get_indexer(df["day"]), df["minute"].to_numpy()
    for k, col in enumerate(["open", "close", "volume", "high", "low"]):
        grid[di, k, mi] = df[col].to_numpy()

    X, y, move, exc, is_train, skipped = [], [], [], [], [], {"train": 0, "test": 0}
    for i in range(len(days)):
        o, c, v, h, l = grid[i]
        day_ok = not np.isnan(prev_close[i]) and not np.isnan(grid[i, :, opening]).any()  # opening range + prev close
        day = day_arrays(o, h, l, c, v, day_open, prev_close[i], p) if day_ok else None
        for t in range(first, last + 1):  # t = decision minute (minute of day)
            if not day_ok or np.isnan(grid[i, :, t - lb + 1:t + hold + 1]).any():  # look-back, t+1..t+hold must exist
                skipped["train" if train_day[i] else "test"] += 1
                continue
            m = round(c[t + hold] - o[t + 1], 2)  # forward close - entry open, rounded to paise so ticks compare exactly
            base = window_features(o, h, l, c, v, t, lb, day_open, gap)
            X.append(np.r_[base, phase2_features(o, h, l, c, v, t, lb, day_open, base[-6:], day, p)])
            y.append(int(m > cfg["target"]["move_bar"]))
            move.append(m)
            held = slice(t + 1, t + hold + 1)  # hold bars t+1..t+hold, same bars as the forward close
            exc.append((h[held].max() - o[t + 1], o[t + 1] - l[held].min()))
            is_train.append(train_day[i])
    n_days = {"train": int(train_day.sum()), "test": int((~train_day).sum())}
    return (np.array(X, dtype=np.float32), np.array(y), np.array(move), np.array(exc).reshape(-1, 2),
            np.array(is_train), skipped, n_days)


# ----------------------------------------------------------------------------- block 2: training
def make_model(n_in: int, a: dict) -> nn.Sequential:
    layers, prev = [], n_in
    for units in a["hidden_layers"]:
        layers += [nn.Linear(prev, units), getattr(nn, a["activation"])(), nn.Dropout(a["dropout"])]
        prev = units
    return nn.Sequential(*layers, nn.Linear(prev, 2))


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, t: dict) -> None:
    opt = getattr(torch.optim, t["optimizer"])(model.parameters(), lr=t["learning_rate"],
                                               weight_decay=t["weight_decay"])
    w1 = t["label1_loss_weight"]  # class weights [1, w1]; None at 1.0 keeps the unweighted loss bit-for-bit
    weight = None if w1 == 1 else torch.tensor([1.0, float(w1)], device=X.device)
    gen = torch.Generator().manual_seed(t["seed"])
    model.train()
    for _ in range(t["epochs"]):
        for idx in torch.randperm(len(X), generator=gen).split(t["batch_size"]):
            opt.zero_grad()
            nn.functional.cross_entropy(model(X[idx]), y[idx], weight=weight).backward()
            opt.step()


def predict(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    """Predicted label: 1 if the softmax probability of label 1 > decision_threshold, else 0."""
    return (logits.softmax(1)[:, 1] > threshold).long()


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, threshold: float) -> tuple[float, float, np.ndarray]:
    model.eval()
    logits = model(X)
    pred = predict(logits, threshold)  # loss stays unweighted so it is comparable across label1_loss_weight
    return nn.functional.cross_entropy(logits, y).item(), (pred == y).float().mean().item(), pred.cpu().numpy()


# ----------------------------------------------------------------------------- block 3: output
def crosstab(y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    m = np.zeros((3, 3), dtype=int)
    np.add.at(m, (y, pred), 1)
    m[2, :2] = m[:2, :2].sum(axis=0)
    m[:, 2] = m[:, :2].sum(axis=1)
    return pd.DataFrame(m, index=pd.Index(["0", "1", "total"], name="true \\ predicted"), columns=["0", "1", "total"])


def rates(tab: pd.DataFrame) -> tuple[float, float, float]:
    """Base rate (share label 1), always-0 accuracy (share label 0), balanced accuracy (mean recall of 0 and 1)."""
    m = tab.to_numpy()
    recall = [m[k, k] / m[k, 2] if m[k, 2] else float("nan") for k in (0, 1)]
    return m[1, 2] / m[2, 2], m[0, 2] / m[2, 2], (recall[0] + recall[1]) / 2


def entries(y: np.ndarray, move: np.ndarray, exc: np.ndarray, pred: np.ndarray, n_days: int) -> tuple:
    """Entries per day, hit rate (share of predicted-1 with label 1), mean forward move, mean high - entry,
    mean entry - low (rupees, over predicted-1)."""
    on = pred == 1
    if not on.any():
        return 0.0, float("nan"), float("nan"), float("nan"), float("nan")
    return on.sum() / n_days, y[on].mean(), move[on].mean(), exc[on, 0].mean(), exc[on, 1].mean()


def check_number(value, name: str, low: float, high: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low < value < high:
        raise ValueError(f"{name} must be a number strictly between {low} and {high}; got {value!r}")


def main() -> None:
    cfg = yaml.safe_load((HERE / "mlp_model.yaml").read_text())
    t, a, d = cfg["training"], cfg["architecture"], cfg["decision_points"]
    thr = cfg["decision"]["decision_threshold"]
    check_number(thr, "decision.decision_threshold", 0, 1)
    check_number(t["label1_loss_weight"], "training.label1_loss_weight", 0, float("inf"))
    keep = selected_columns(cfg)
    X, y, move, exc, is_train, skipped, n_days = build_samples(cfg)
    if not is_train.any() or is_train.all():
        raise SystemExit("train and test must both hold at least one complete decision point")
    X = X[:, keep]  # switched-on features only, in YAML order

    mean, std = X[is_train].mean(axis=0), X[is_train].std(axis=0)  # train rows only
    std[std == 0] = 1.0  # constant column (e.g. ret_t, always 0) gets std 1
    X = (X - mean) / std
    dev = torch.device(t["device"])
    sets = {name: (torch.tensor(X[m], device=dev), torch.tensor(y[m], device=dev), y[m], move[m], exc[m])
            for name, m in (("train", is_train), ("test", ~is_train))}

    torch.manual_seed(t["seed"])
    model = make_model(X.shape[1], a).to(dev)
    train(model, *sets["train"][:2], t)

    print(f"model: MLP in={X.shape[1]} hidden={a['hidden_layers']} {a['activation']} dropout={a['dropout']} out=2 | "
          f"{t['optimizer']} lr={t['learning_rate']} weight_decay={t['weight_decay']} batch={t['batch_size']} "
          f"epochs={t['epochs']} seed={t['seed']} device={t['device']} "
          f"label1_loss_weight={t['label1_loss_weight']} decision_threshold={thr}")
    print(f"data: decisions {d['first_decision']}-{d['last_decision']} every 1m, look_back {d['look_back_minutes']}m "
          f"(t-{d['look_back_minutes'] - 1}..t), entry open(t+1), forward close(t+{d['hold_minutes']}), "
          f"label = forward - entry > {cfg['target']['move_bar']}")
    tabs = {}
    for name, (xt, yt, y_np, move_np, exc_np) in sets.items():
        loss, acc, pred = evaluate(model, xt, yt, thr)
        tabs[name] = crosstab(y_np, pred)
        base, always0, balanced = rates(tabs[name])
        per_day, hit, avg_move, avg_high, avg_low = entries(y_np, move_np, exc_np, pred, n_days[name])
        print(f"{name + ':':<6} windows={len(y_np)} skipped={skipped[name]} loss={loss:.4f} accuracy={acc:.4f} "
              f"base_rate={base:.4f} always0_accuracy={always0:.4f} balanced_accuracy={balanced:.4f}")
        print(f"{'':<6} days={n_days[name]} entries_per_day={per_day:.4f} hit_rate={hit:.4f} "
              f"avg_forward_move={avg_move:.4f} avg_high_excursion={avg_high:.4f} avg_low_excursion={avg_low:.4f}")
    with pd.ExcelWriter(HERE / cfg["output"]["crosstab_xlsx"], engine="openpyxl", mode="w") as xw:
        for name, tab in tabs.items():
            tab.to_excel(xw, sheet_name=name)


if __name__ == "__main__":
    main()
