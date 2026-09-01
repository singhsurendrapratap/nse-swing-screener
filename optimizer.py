import numpy as np
import pandas as pd
import optuna

# Suppress verbose Optuna logging output
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_ai_optimizer(
    param_space: dict,
    trades_df: pd.DataFrame,
    backtest_fn,
    target_metric: str = "Expectancy R",
    min_trades_per_fold: int = 20,
    n_trials: int = 50,
    n_folds: int = 5,
) -> dict:
    """Runs an Optuna optimization search across chronological walk-forward folds.

    Returns the best parameters and cross-fold stability metrics.
    """
    if trades_df is None or trades_df.empty or len(trades_df) < (min_trades_per_fold * 2):
        return {
            "best_params": {},
            "oos_median_score": 0.0,
            "overfit_warning": False,
            "overfit_ratio": 1.0,
            "oos_fold_scores": [],
        }

    # Sort historical trades chronologically
    df_sorted = trades_df.sort_values("entry_date").reset_index(drop=True)
    n = len(df_sorted)
    fold_size = n // (n_folds + 1)

    folds = []
    for i in range(n_folds):
        train_end = (i + 1) * fold_size
        val_end = train_end + fold_size if i < n_folds - 1 else n
        train_trades = df_sorted.iloc[:train_end]
        val_trades = df_sorted.iloc[train_end:val_end]
        folds.append((train_trades, val_trades))

    def objective(trial: optuna.Trial) -> float:
        # Sample candidate parameters from defined parameter space
        sampled_params = {}
        for param, cfg in param_space.items():
            p_type = cfg.get("type", "float")
            if p_type == "float":
                sampled_params[param] = trial.suggest_float(
                    param, cfg["low"], cfg["high"], step=cfg.get("step")
                )
            elif p_type == "int":
                sampled_params[param] = trial.suggest_int(
                    param, cfg["low"], cfg["high"], step=cfg.get("step", 1)
                )

        val_scores = []
        for train_trades, val_trades in folds:
            if len(train_trades) < min_trades_per_fold or len(val_trades) < 5:
                continue

            # Run light evaluation function provided by engine
            res = backtest_fn(val_trades, sampled_params)

            if target_metric == "Expectancy R":
                score = res.get("expectancy_r", -999.0)
            elif target_metric == "Total R":
                score = res.get("total_r", -999.0)
            elif target_metric == "Win Rate":
                score = res.get("win_rate", 0.0)
            else:
                score = res.get("expectancy_r", -999.0)

            val_scores.append(score)

        return float(np.median(val_scores)) if val_scores else -999.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value

    # Compute Out-Of-Sample (OOS) performance per fold using best params
    oos_scores = []
    is_scores = []
    for train_trades, val_trades in folds:
        res_is = backtest_fn(train_trades, best_params)
        res_oos = backtest_fn(val_trades, best_params)

        key = "win_rate" if target_metric == "Win Rate" else (
            "total_r" if target_metric == "Total R" else "expectancy_r"
        )
        is_scores.append(res_is.get(key, 0.0))
        oos_scores.append(res_oos.get(key, 0.0))

    avg_is = float(np.mean(is_scores)) if is_scores else 1.0
    avg_oos = float(np.mean(oos_scores)) if oos_scores else 0.0
    overfit_ratio = avg_is / (avg_oos + 1e-6)
    overfit_warning = overfit_ratio > 2.0 or avg_oos < 0

    return {
        "best_params": best_params,
        "oos_median_score": float(best_value),
        "overfit_warning": overfit_warning,
        "overfit_ratio": overfit_ratio,
        "oos_fold_scores": oos_scores,
    }
