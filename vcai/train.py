"""Train virality models, cross-validated by creator, and write an explanation report.

Two feature sets are compared:
  content - audio/visual/pacing/duration/format/game only: what our clip generator can control
  full    - content + title, posting time, creator size, age
If "content" scores close to "full", the video itself drives virality and auto-clipping can
learn it. The gap between them shows how much packaging and distribution add.
"""
import json
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from . import config
from .dataset import add_game_category, build_dataset, feature_columns

PARAMS = dict(objective="regression", learning_rate=0.03, n_estimators=500, num_leaves=31,
              min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, verbose=-1)

FEATURE_DOCS = {
    "duration_s": "clip length (seconds)",
    "log_duration": "clip length (log)",
    "aspect_ratio": "width / height",
    "is_vertical": "vertical (9:16-style) video",
    "is_youtube": "posted on YouTube vs Twitch",
    "game": "game / category",
    "a_has_audio": "has an audio track",
    "a_loud_mean_db": "average loudness",
    "a_loud_std_db": "loudness variability",
    "a_loud_p95_db": "loudest moments level",
    "a_dynamic_range_db": "quiet-to-loud range",
    "a_silence_ratio": "share of near-silent time",
    "a_loud_peak_time_s": "when the loudest moment happens (s)",
    "a_loud_peak_time_rel": "where the loudest moment is (0=start, 1=end)",
    "a_hook_loud_delta_db": "first-3s loudness vs clip average (audio hook)",
    "a_end_loud_delta_db": "last-3s loudness vs clip average",
    "a_loud_slope": "loudness build-up over the clip",
    "a_max_loud_jump_db": "biggest sudden volume jump in 0.5s (yell / explosion / reaction)",
    "a_loud_burst_count": "number of loud bursts",
    "a_loud_burst_rate": "loud bursts per second",
    "a_onset_rate": "sound events per second (shots, hits, callouts)",
    "a_flux_mean": "audio activity (spectral flux)",
    "a_flux_peak_z": "how much the biggest audio event stands out",
    "a_centroid_mean_hz": "audio brightness (spectral centroid)",
    "a_speech_band_ratio": "share of energy in the voice band (talking / commentary)",
    "a_high_band_ratio": "share of high-frequency energy (screams, gunfire, crowd)",
    "a_hook_onset_rate": "sound events per second in the first 3s",
    "v_motion_mean": "average on-screen motion",
    "v_motion_std": "motion variability",
    "v_motion_p95": "peak motion level",
    "v_motion_burstiness": "peak motion relative to average (spiky action)",
    "v_motion_peak_time_rel": "where peak motion is (0=start, 1=end)",
    "v_hook_motion_ratio": "first-3s motion vs clip average (visual hook)",
    "v_motion_slope": "motion build-up over the clip",
    "v_center_motion_ratio": "motion concentrated in center of frame",
    "v_cut_count": "number of scene cuts",
    "v_cut_rate": "scene cuts per second (editing pace)",
    "v_first_cut_s": "time to first cut (s)",
    "v_hook_cut_count": "cuts in the first 3s",
    "v_brightness_mean": "average brightness",
    "v_brightness_std": "brightness variability",
    "v_flash_count": "sudden brightness flashes",
    "v_saturation_mean": "color saturation",
    "v_colorfulness_mean": "colorfulness",
    "v_edge_density": "visual detail / clutter",
    "v_static_pixel_ratio": "share of frame that never changes (HUD, overlays, bars)",
    "v_black_border_ratio": "letterbox / black borders",
    "x_loud_motion_corr": "audio and motion peaks line up",
    "x_climax_time_rel": "where the combined audio+motion climax is (0=start, 1=end)",
    "x_joint_peak_seconds": "seconds where audio AND motion both spike",
    "post_hour_utc": "posting hour (UTC)",
    "post_weekday": "posting weekday",
    "lang_en": "English-language",
    "log_creator_baseline_views": "creator's typical views (log)",
    "log_age_days": "clip age (log days)",
    "title_len": "title length (chars)",
    "title_words": "title word count",
    "title_caps_ratio": "share of CAPS in title",
    "title_exclaims": "exclamation marks in title",
    "title_question": "title asks a question",
    "title_emojis": "emojis in title",
    "title_has_number": "title contains a number",
    "title_hashtags": "hashtags in title",
    "title_hype_words": "hype words in title (clutch, insane, wtf...)",
    "title_1vX": "title mentions a 1vX",
}


def cross_validate(df: pd.DataFrame, cols: list[str], n_splits: int = 5, seed: int = 0) -> dict:
    groups = df["source"] + ":" + df["creator_id"].astype(str)
    k = min(n_splits, groups.nunique())
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=k).split(df, groups=groups):
        m = lgb.LGBMRegressor(**PARAMS, random_state=seed)
        m.fit(df.iloc[tr][cols], df.iloc[tr]["viral_score"])
        oof[te] = m.predict(df.iloc[te][cols])
    y, br = df["viral_score"].to_numpy(), df["breakout"].to_numpy()
    top = oof >= np.quantile(oof, 0.9)
    return {
        "folds": k,
        "spearman": float(spearmanr(y, oof).statistic),
        "r2": float(r2_score(y, oof)),
        "breakout_auc": float(roc_auc_score(br, oof)) if len(set(br)) > 1 else float("nan"),
        "top10_breakout_rate": float(br[top].mean()),
        "base_breakout_rate": float(br.mean()),
    }


def explain(model: lgb.LGBMRegressor, X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    contrib = model.predict(X, pred_contrib=True)[:, :-1]
    rows = []
    for i, c in enumerate(X.columns):
        direction = ""
        if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 1:
            rho = spearmanr(X[c], contrib[:, i], nan_policy="omit").statistic
            if np.isfinite(rho) and abs(rho) >= 0.2:
                direction = "higher = more viral" if rho > 0 else "lower = more viral"
            else:
                direction = "non-linear / mixed"
        rows.append({"feature": c, "mean_abs_shap": float(np.abs(contrib[:, i]).mean()),
                     "direction": direction, "meaning": FEATURE_DOCS.get(c, "")})
    imp = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    extra = {}
    if "game" in X.columns:
        gi = list(X.columns).index("game")
        g = pd.DataFrame({"game": X["game"].astype(str), "shap": contrib[:, gi]})
        by = g.groupby("game")["shap"].agg(["mean", "count"]).query("count >= 20").sort_values("mean")
        extra["game_effect"] = by
    return imp, extra


def quintile_table(df: pd.DataFrame, feature: str) -> pd.DataFrame | None:
    x = df[feature]
    if not pd.api.types.is_numeric_dtype(x) or x.nunique() < 5:
        return None
    bins = pd.qcut(x, 5, duplicates="drop")
    return df.groupby(bins, observed=True).agg(
        clips=("viral_score", "size"),
        median_views_multiple=("views_multiple", "median"),
        breakout_rate=("breakout", "mean"),
    )


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    lines = [head, sep]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def train(conn, min_rows: int = 200, log=print, now=None) -> dict:
    df = build_dataset(conn, now=now)
    if len(df) < min_rows:
        raise RuntimeError(f"only {len(df)} labelled clips with features (need {min_rows}); "
                           "collect + process more first")
    df = add_game_category(df)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    runs = [("all", "content"), ("all", "full")]
    runs += [(s, "content") for s, n in df["source"].value_counts().items()
             if n >= min_rows and df["source"].nunique() > 1]

    results = {}
    for scope, fset in runs:
        d = df if scope == "all" else df[df["source"] == scope].reset_index(drop=True)
        cols = feature_columns(d, fset)
        metrics = cross_validate(d, cols)
        model = lgb.LGBMRegressor(**PARAMS, random_state=0).fit(d[cols], d["viral_score"])
        imp, extra = explain(model, d[cols])
        name = f"{scope}_{fset}"
        model.booster_.save_model(str(config.MODEL_DIR / f"{name}_{stamp}.txt"))
        imp.to_csv(config.REPORT_DIR / f"importance_{name}_{stamp}.csv", index=False)
        results[name] = {"metrics": metrics, "importance": imp, "extra": extra, "data": d, "n": len(d)}
        log(f"{name:>16}: n={len(d)} spearman={metrics['spearman']:.3f} "
            f"breakout_auc={metrics['breakout_auc']:.3f} top10 hit-rate={metrics['top10_breakout_rate']:.2f}")

    report = write_report(df, results, stamp)
    (config.REPORT_DIR / f"metrics_{stamp}.json").write_text(
        json.dumps({k: {"n": v["n"], **v["metrics"]} for k, v in results.items()}, indent=2))
    log(f"report: {report}")
    return {k: v["metrics"] for k, v in results.items()}


def write_report(df: pd.DataFrame, results: dict, stamp: str):
    L = [f"# What makes a game clip go viral - report {stamp}", ""]
    L += ["## Dataset", ""]
    summ = df.groupby("source").agg(
        clips=("clip_id", "size"), creators=("creator_id", "nunique"), games=("game_name", "nunique"),
        median_views=("views", "median"),
        breakout_multiple=("views_multiple", lambda s: s.quantile(config.BREAKOUT_QUANTILE)))
    L += [_md_table(summ, "{:.1f}"), "",
          "*viral_score* = views relative to the same creator's typical clip (leave-one-out median), "
          "age-adjusted. *breakout* = top "
          f"{100 - int(config.BREAKOUT_QUANTILE * 100)}% of viral_score per platform; "
          "`breakout_multiple` is roughly how many times the creator's normal views that takes.", ""]

    L += ["## How predictable is virality? (cross-validated, unseen creators)", "",
          "| model | clips | spearman | R² | breakout AUC | breakout rate in model's top 10% | base rate |",
          "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        m = r["metrics"]
        L.append(f"| {name} | {r['n']} | {m['spearman']:.3f} | {m['r2']:.3f} | {m['breakout_auc']:.3f} | "
                 f"{m['top10_breakout_rate']:.1%} | {m['base_breakout_rate']:.1%} |")
    L += ["", "AUC 0.5 = coin flip. Lift = top-10% breakout rate / base rate.", ""]

    for name, r in results.items():
        imp = r["importance"]
        L += [f"## Drivers - {name}", "", _md_table(imp.head(20)[["feature", "meaning", "mean_abs_shap", "direction"]]), ""]
        if "game_effect" in r["extra"] and len(r["extra"]["game_effect"]):
            ge = r["extra"]["game_effect"]
            L += ["**Games that help / hurt (mean SHAP, ≥20 clips):**", "",
                  _md_table(pd.concat([ge.tail(5)[::-1], ge.head(5)]).rename(columns={"mean": "shap_mean"})), ""]
        if name.endswith("content"):
            for feat in [f for f in imp["feature"].head(6) if f != "game"]:
                qt = quintile_table(r["data"], feat)
                if qt is not None:
                    L += [f"**{feat}** ({FEATURE_DOCS.get(feat, '')}) by quintile:", "", _md_table(qt), ""]
    path = config.REPORT_DIR / f"report_{stamp}.md"
    path.write_text("\n".join(L), encoding="utf-8")
    (config.REPORT_DIR / "latest.md").write_text("\n".join(L), encoding="utf-8")
    return path
