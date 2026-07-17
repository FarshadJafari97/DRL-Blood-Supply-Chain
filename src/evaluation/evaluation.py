"""
Evaluation module for running episodes, capturing metrics,
aggregating results, and exporting to Excel reports.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from src.configs.config import MODELS_DIR, TABLES_DIR
from src.env.platelet_env import PlateletSupplyChainEnv
from src.utils.seeding import set_seed

# Standard columns matching original notebooks
H1_COLS = ['Total_Demand', 'Fulfilled_Demand', 'Unfulfilled_Demand', 'Wastage_Units',
           'Shortage_Units', 'Wastage_Rate_Pct', 'Shortage_Rate_Pct', 'Wastage_Cost',
           'Shortage_Cost', 'Holding_Cost', 'Order_Cost', 'Total_Cost']
H2_COLS = H1_COLS
SC_COLS = ['Total_Wastage', 'Total_Shortage', 'Trans_H1_to_H2', 'Trans_H2_to_H1',
           'Trans_Cost_H1_to_H2', 'Trans_Cost_H2_to_H1', 'Network_Total_Cost']


def compute_episode_summary(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    """
    Computes summary dictionary for a given dataframe of episode logs,
    summing up specific columns and averaging others as done in the original notebook.
    """
    s = {}
    for c in cols:
        if 'Cost' in c or 'Units' in c or 'Demand' in c or 'Trans' in c:
            s[c] = float(df[c].sum())
        else:
            s[c] = float(df[c].mean())
    return s


def aggregate_seed_summaries(seed_summaries: List[Dict[str, float]]) -> pd.DataFrame:
    """Aggregates summaries across multiple evaluation seeds, calculating mean and std."""
    df = pd.DataFrame(seed_summaries)
    result = {}
    for c in df.columns:
        result[f'{c}_Mean'] = df[c].mean()
        result[f'{c}_Std'] = df[c].std()
    return pd.DataFrame([result])


def create_excel_report(
    all_h1_summaries: List[pd.DataFrame],
    all_h2_summaries: List[pd.DataFrame],
    all_sc_summaries: List[pd.DataFrame],
    raw_rows: List[Dict[str, Any]],
    train_seeds: List[Any],
    excel_filename: str,
    key_name: str = "Train_Seed"
) -> str:
    """
    Creates structured evaluation reports saved as Excel files mirroring original spreadsheets.

    Args:
        all_h1_summaries: List of Hospital 1 stats per training seed.
        all_h2_summaries: List of Hospital 2 stats per training seed.
        all_sc_summaries: List of Whole Supply Chain stats per training seed.
        raw_rows: List of step-level logs for raw episode data.
        train_seeds: Identifiers for each evaluated model/seed (e.g. TRAIN_SEEDS or policy names).
        excel_filename: Target excel spreadsheet name.
        key_name: Header name for the identifier column (e.g. "Train_Seed" or "Policy").

    Returns:
        The absolute path to the generated Excel report.
    """
    os.makedirs(TABLES_DIR, exist_ok=True)
    excel_path = os.path.join(TABLES_DIR, excel_filename)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        def write_summary_sheet(dfs: List[pd.DataFrame], sheet_name: str) -> pd.DataFrame:
            combined = pd.concat(dfs, ignore_index=True)
            mean_cols = [c for c in combined.columns if c.endswith('_Mean')]
            result = pd.DataFrame({
                'Metric': [c[:-5] for c in mean_cols],  # strip '_Mean'
                'Mean': [combined[c].mean() for c in mean_cols],
                'Std': [combined[c].std() for c in mean_cols],
                'Min': [combined[c].min() for c in mean_cols],
                'Max': [combined[c].max() for c in mean_cols]
            })
            result.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"💾 Sheet '{sheet_name}': {len(result)} metrics")
            return result

        h1_summary_df = write_summary_sheet(all_h1_summaries, 'H1_Summary')
        h2_summary_df = write_summary_sheet(all_h2_summaries, 'H2_Summary')
        sc_summary_df = write_summary_sheet(all_sc_summaries, 'SC_Summary')

        # Master sheet
        master_rows = []
        for _, row in h1_summary_df.iterrows():
            master_rows.append({'Table': 'H1', 'Metric': row['Metric'], 'Mean': row['Mean'],
                                'Std': row['Std'], 'Min': row['Min'], 'Max': row['Max']})
        for _, row in h2_summary_df.iterrows():
            master_rows.append({'Table': 'H2', 'Metric': row['Metric'], 'Mean': row['Mean'],
                                'Std': row['Std'], 'Min': row['Min'], 'Max': row['Max']})
        for _, row in sc_summary_df.iterrows():
            master_rows.append({'Table': 'SC', 'Metric': row['Metric'], 'Mean': row['Mean'],
                                'Std': row['Std'], 'Min': row['Min'], 'Max': row['Max']})
        master_df = pd.DataFrame(master_rows)
        master_df.to_excel(writer, sheet_name='Master', index=False)
        print(f"💾 Sheet 'Master': {len(master_df)} rows")

        # Model_Comparison sheet
        comparison_rows = []
        for seed, h1, h2, sc in zip(train_seeds, all_h1_summaries, all_h2_summaries, all_sc_summaries):
            comparison_rows.append({
                key_name: seed,
                'H1_Total_Cost': h1['Total_Cost_Mean'].values[0] if 'Total_Cost_Mean' in h1.columns else np.nan,
                'H2_Total_Cost': h2['Total_Cost_Mean'].values[0] if 'Total_Cost_Mean' in h2.columns else np.nan,
                'Network_Total_Cost': sc['Network_Total_Cost_Mean'].values[0] if 'Network_Total_Cost_Mean' in sc.columns else np.nan,
                'H1_Wastage_Rate': h1['Wastage_Rate_Pct_Mean'].values[0] if 'Wastage_Rate_Pct_Mean' in h1.columns else np.nan,
                'H2_Wastage_Rate': h2['Wastage_Rate_Pct_Mean'].values[0] if 'Wastage_Rate_Pct_Mean' in h2.columns else np.nan,
                'H1_Shortage_Rate': h1['Shortage_Rate_Pct_Mean'].values[0] if 'Shortage_Rate_Pct_Mean' in h1.columns else np.nan,
                'H2_Shortage_Rate': h2['Shortage_Rate_Pct_Mean'].values[0] if 'Shortage_Rate_Pct_Mean' in h2.columns else np.nan,
            })
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
        print(f"💾 Sheet 'Model_Comparison': {len(comparison_df)} models")

        # Raw_Episode_Data sheet
        raw_df = pd.DataFrame(raw_rows)
        raw_df.to_excel(writer, sheet_name='Raw_Episode_Data', index=False)
        print(f"💾 Sheet 'Raw_Episode_Data': {len(raw_df)} rows")

    print(f"✅ Excel saved: {excel_path}")
    return excel_path


def run_evaluation_episode_trace(
    env: PlateletSupplyChainEnv,
    agent: Any,
    seed: int,
    is_mappo: bool = False
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Runs a single episode evaluation, generating daily step traces for H1, H2 and SC."""
    set_seed(seed)
    state = env.reset()

    h1_daily = []
    h2_daily = []
    sc_daily = []

    for day in range(env.episode_length):
        if is_mappo:
            action = agent.get_deterministic_joint_action(state)
        else:
            if hasattr(agent, "get_deterministic_action"):
                action = agent.get_deterministic_action(state)
            else:
                action = agent.get_action(env, state)

        next_state, reward, done, info = env.step(action)

        # hospital 1 variables
        d1 = info['demand_h1']
        s1 = info['shortage_h1']
        w1 = info['wastage_h1']
        fulfilled1 = d1 - s1
        inv_h1_before = info['inv_h1_before']
        inv_h2_before = info['inv_h2_before']
        total_inv_before = inv_h1_before + inv_h2_before

        h1_hold = (inv_h1_before / total_inv_before) * info['inventory_holding_cost'] if total_inv_before > 0 else 0.0

        h1_daily.append({
            'Day': day + 1,
            'Total_Demand': d1,
            'Fulfilled_Demand': fulfilled1,
            'Unfulfilled_Demand': s1,
            'Wastage_Units': w1,
            'Shortage_Units': s1,
            'Wastage_Rate_Pct': (w1 / d1 * 100) if d1 > 0 else 0.0,
            'Shortage_Rate_Pct': (s1 / d1 * 100) if d1 > 0 else 0.0,
            'Wastage_Cost': w1 * env.wastage_cost,
            'Shortage_Cost': s1 * env.shortage_cost,
            'Holding_Cost': h1_hold,
            'Order_Cost': info['actual_order_h1'] * env.order_cost,
            'Total_Cost': (w1 * env.wastage_cost) + (s1 * env.shortage_cost) + h1_hold + (info['actual_order_h1'] * env.order_cost),
        })

        # hospital 2 variables
        d2 = info['demand_h2']
        s2 = info['shortage_h2']
        w2 = info['wastage_h2']
        fulfilled2 = d2 - s2

        h2_hold = (inv_h2_before / total_inv_before) * info['inventory_holding_cost'] if total_inv_before > 0 else 0.0

        h2_daily.append({
            'Day': day + 1,
            'Total_Demand': d2,
            'Fulfilled_Demand': fulfilled2,
            'Unfulfilled_Demand': s2,
            'Wastage_Units': w2,
            'Shortage_Units': s2,
            'Wastage_Rate_Pct': (w2 / d2 * 100) if d2 > 0 else 0.0,
            'Shortage_Rate_Pct': (s2 / d2 * 100) if d2 > 0 else 0.0,
            'Wastage_Cost': w2 * env.wastage_cost,
            'Shortage_Cost': s2 * env.shortage_cost,
            'Holding_Cost': h2_hold,
            'Order_Cost': info['actual_order_h2'] * env.order_cost,
            'Total_Cost': (w2 * env.wastage_cost) + (s2 * env.shortage_cost) + h2_hold + (info['actual_order_h2'] * env.order_cost),
        })

        # Supply Chain Network variables
        total_trans = info['transferred_h1_to_h2'] + info['transferred_h2_to_1' if 'transferred_h2_to_1' in info else 'transferred_h2_to_h1']
        trans_cost_total = info['transshipment_cost']

        if total_trans > 0:
            trans_cost_h1_to_h2 = (info['transferred_h1_to_h2'] / total_trans) * trans_cost_total
            trans_cost_h2_to_h1 = (info['transferred_h2_to_h1'] / total_trans) * trans_cost_total
        else:
            trans_cost_h1_to_h2 = 0.0
            trans_cost_h2_to_h1 = 0.0

        sc_daily.append({
            'Day': day + 1,
            'Total_Wastage': w1 + w2,
            'Total_Shortage': s1 + s2,
            'Trans_H1_to_H2': info['transferred_h1_to_h2'],
            'Trans_H2_to_H1': info['transferred_h2_to_h1'],
            'Trans_Cost_H1_to_H2': trans_cost_h1_to_h2,
            'Trans_Cost_H2_to_H1': trans_cost_h2_to_h1,
            'Network_Total_Cost': info['total_cost'],
        })

        state = next_state
        if done:
            break

    return h1_daily, h2_daily, sc_daily


def evaluate_multi_seed(
    env: PlateletSupplyChainEnv,
    agent: Any,
    train_seed: Any,
    is_mappo: bool = False,
    eval_seeds: range = range(1, 51)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """
    Evaluates an agent over a range of evaluation seeds and returns summary stats.

    Args:
        env: PlateletSupplyChainEnv instance.
        agent: Evaluated agent policy.
        train_seed: Seed of model being evaluated (or label).
        is_mappo: True if evaluating MAPPO model, False otherwise.
        eval_seeds: Range of evaluation seeds.

    Returns:
        H1 Summary stats, H2 Summary stats, SC Network Summary stats, raw rows.
    """
    h1_records_all = []
    h2_records_all = []
    sc_records_all = []

    for eseed in eval_seeds:
        h1_d, h2_d, sc_d = run_evaluation_episode_trace(env, agent, eseed, is_mappo=is_mappo)
        h1_records_all.append(pd.DataFrame(h1_d))
        h2_records_all.append(pd.DataFrame(h2_d))
        sc_records_all.append(pd.DataFrame(sc_d))

    h1_seed_summaries = [compute_episode_summary(df, H1_COLS) for df in h1_records_all]
    h2_seed_summaries = [compute_episode_summary(df, H2_COLS) for df in h2_records_all]
    sc_seed_summaries = [compute_episode_summary(df, SC_COLS) for df in sc_records_all]

    h1_model_summary = aggregate_seed_summaries(h1_seed_summaries)
    h2_model_summary = aggregate_seed_summaries(h2_seed_summaries)
    sc_model_summary = aggregate_seed_summaries(sc_seed_summaries)

    raw_rows = []
    for i, seed in enumerate(eval_seeds):
        row = {
            'Train_Seed': train_seed,
            'Eval_Seed': seed
        }
        for k, v in h1_seed_summaries[i].items():
            row[f'H1_{k}'] = v
        for k, v in h2_seed_summaries[i].items():
            row[f'H2_{k}'] = v
        for k, v in sc_seed_summaries[i].items():
            row[f'SC_{k}'] = v
        raw_rows.append(row)

    return h1_model_summary, h2_model_summary, sc_model_summary, raw_rows
