"""
Statistical tests and hypothesis testing routines.
"""

from typing import Dict, Any, Tuple
import os
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.configs.config import TABLES_DIR


def run_friedman_and_wilcoxon_tests(
    rl_files: Dict[str, str],
    out_file: str,
    out_policy: str = "S50",
    cost_col: str = "SC_Network_Total_Cost"
) -> Dict[str, Any]:
    """
    Performs Friedman and post-hoc paired Wilcoxon signed-rank tests across policies
    using results stored in Excel spreadsheet sheets.

    Args:
        rl_files: Dictionary mapping RL labels to excel files.
        out_file: Order-up-to baseline results file name.
        out_policy: Baseline target policy seed identifier sheet.
        cost_col: Target metric column name for comparison (default is SC_Network_Total_Cost).

    Returns:
        A dictionary containing test results and p-values.
    """
    results = {}

    # Read data
    data_by_policy = {}
    for p_name, f_name in rl_files.items():
        f_path = os.path.join(TABLES_DIR, f_name)
        df = pd.read_excel(f_path, sheet_name="Raw_Episode_Data")
        # Average across the 5 models/seeds per Eval_Seed
        grouped = df.groupby("Eval_Seed")[cost_col].mean().reset_index()
        data_by_policy[p_name] = grouped.sort_values("Eval_Seed")[cost_col].values

    # Add Order-up-to baseline S50 policy
    out_path = os.path.join(TABLES_DIR, out_file)
    df_out = pd.read_excel(out_path, sheet_name="Raw_Episode_Data")
    df_s50 = df_out[df_out["Train_Seed"] == out_policy]
    grouped_s50 = df_s50.groupby("Eval_Seed")[cost_col].mean().reset_index()
    data_by_policy["OUT"] = grouped_s50.sort_values("Eval_Seed")[cost_col].values

    # Run Friedman test
    p_mappo = data_by_policy["MAPPO"]
    p_sat = data_by_policy["SA-T"]
    p_sant = data_by_policy["SA-NT"]
    p_out = data_by_policy["OUT"]

    stat_f, p_f = stats.friedmanchisquare(p_mappo, p_sat, p_sant, p_out)
    results["friedman"] = {"statistic": stat_f, "p_value": p_f}

    # Run pairwise post-hoc tests with Holm correction
    comparisons = [
        ("MAPPO", "SA-T"),
        ("MAPPO", "SA-NT"),
        ("SA-T", "SA-NT"),
    ]
    p_vals = []
    wilcoxon_stats = []
    for c1, c2 in comparisons:
        stat_w, p_w = stats.wilcoxon(data_by_policy[c1], data_by_policy[c2])
        p_vals.append(p_w)
        wilcoxon_stats.append(stat_w)

    reject, corrected_p, _, _ = multipletests(p_vals, alpha=0.05, method="holm")

    pairwise_results = []
    for idx, (c1, c2) in enumerate(comparisons):
        pairwise_results.append({
            "comparison": f"{c1} vs {c2}",
            "wilcoxon_stat": wilcoxon_stats[idx],
            "raw_p": p_vals[idx],
            "corrected_p": corrected_p[idx],
            "significant": reject[idx]
        })
    results["pairwise"] = pairwise_results

    # Additional model-level checks (n=5 independent training seeds)
    # MAPPO vs SA-T, MAPPO vs SA-NT, SA-T vs SA-NT
    # Using average cost over 50 evaluation seeds per training seed
    avg_costs = {}
    for p_name, f_name in rl_files.items():
        f_path = os.path.join(TABLES_DIR, f_name)
        df = pd.read_excel(f_path, sheet_name="Raw_Episode_Data")
        grouped = df.groupby("Train_Seed")[cost_col].mean().reset_index()
        avg_costs[p_name] = grouped.sort_values("Train_Seed")[cost_col].values

    model_level_results = []
    for c1, c2 in comparisons:
        val1 = avg_costs[c1]
        val2 = avg_costs[c2]
        t_stat, p_t = stats.ttest_rel(val1, val2)
        w_stat, p_w = stats.wilcoxon(val1, val2)
        model_level_results.append({
            "comparison": f"{c1} vs {c2}",
            "t_stat": t_stat,
            "t_p": p_t,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": p_w
        })
    results["model_level"] = model_level_results

    return results
