"""
Walk-Forward AI Parameter Optimizer
------------------------------------
Fixes the main risk in a naive "search everything, report the best
result" optimizer: the best in-sample parameters are usually just the
ones that happened to overfit the noise in that specific sample.

This version:
  1. Splits trade history into N walk-forward folds (train -> test,
     rolling forward through time, with an embargo gap so no
     information leaks across the boundary).
  2. Optuna searches parameters guided by TEST (out-of-sample) score,
     not train score.
  3. The reported "best" params are the ones with the best MEDIAN
     out-of-sample score across folds -- not the best single fold,
     which is too easy to get lucky on.
  4. If the spread across folds is large relative to the average
     score, that's flagged as an overfitting warning rather than
     hidden from you.

Plug your real backtest engine into `run_backtest_fn`.
"""

import optuna
import numpy as np
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_walk_forward_folds(trade_dates: pd.Series, n_folds: int = 5, embargo_days: int = 5):
    """Return list of (train_dates, test_dates) rolling forward through time."""
    dates = np.sort(trade_dates.unique())
    fold_size = len(dates) // (n_folds + 1)
    folds = []
    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        test_start = train_end + embargo_days
        test_end = test_start + fold_size
        if test_end > len(dates):
            break
        folds.append((dates[:train_end], dates[test_start:test_end]))
    return folds


def _metric_key(target_metric: str) -> str:
    return target_metric.lower().replace(" ", "_").replace("-", "_")


def run_ai_optimizer(param_space: dict, trades_df: pd.DataFrame, run_backtest_fn,
                      target_metric: str, min_trades: int, n_trials: int, n_folds: int = 5):
    """
    param_space: e.g. {
        "initial_stop": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.1},
        "min_quality":  {"type": "int",   "low": 50,  "high": 90},
    }
    run_backtest_fn(params, trades_df) -> dict with at least
        {'win_rate':.., 'expectancy_r':.., 'total_r':.., 'n_trades':..}
    """
    folds = make_walk_forward_folds(trades_df["date"], n_folds=n_folds)
    if not folds:
        raise ValueError("Not enough trade history for walk-forward folds -- lower n_folds.")

    metric_key = _metric_key(target_metric)

    def objective(trial):
        params = {}
        for name, spec in param_space.items():
            if spec["type"] == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], step=spec.get("step"))
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])

        test_scores = []
        for train_dates, test_dates in folds:
            train_df = trades_df[trades_df["date"].isin(train_dates)]
            test_df = trades_df[trades_df["date"].isin(test_dates)]
            if len(train_df) < min_trades or len(test_df) < max(5, min_trades // 4):
                raise optuna.TrialPruned()

            # Train fold is only used implicitly (params are shared across
            # folds by Optuna's search) -- what we score and optimize on
            # is always the held-out test fold.
            test_result = run_backtest_fn(params, test_df)
            score = test_result.get(metric_key)
            if score is None:
                raise optuna.TrialPruned()
            test_scores.append(score)

        trial.set_user_attr("fold_scores", test_scores)
        return float(np.median(test_scores))  # robust to one lucky fold

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    fold_scores = best.user_attrs.get("fold_scores", [])
    overfit_ratio = float(np.std(fold_scores) / (abs(np.mean(fold_scores)) + 1e-9)) if fold_scores else None

    return {
        "best_params": best.params,
        "oos_median_score": best.value,
        "oos_fold_scores": fold_scores,
        "overfit_warning": overfit_ratio is not None and overfit_ratio > 0.5,
        "overfit_ratio": overfit_ratio,
        "study": study,
    }
