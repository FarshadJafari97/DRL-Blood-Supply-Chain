# Platelet Supply Chain Inventory Management using Reinforcement Learning

This repository contains a clean, modular, research-grade Python framework for training and evaluating reinforcement learning policies and baseline heuristic algorithms for blood platelet inventory management in hospital networks. Blood platelets have a short shelf-life and highly variable demand, presenting a challenging operations research and supply chain problem under potential system disruptions.

This codebase is reorganized from monolithic notebooks into a structured, reproducible repository suitable for publication alongside an academic paper and open-source distribution.

---

## 1. Project Description
Blood supply chains are prone to disruptions and high demand uncertainty. This project simulates a 2-hospital supply chain managing blood platelets with finite shelf-lives. The hospitals can procure platelets, and laterally transship them under potential demand disruptions.

### Policies Tracked:
1. **Multi-Agent MAPPO (MAPPO)**: Multi-agent cooperative policy with a procurement actor head and a logistics/transshipment actor head.
2. **SA-PPO (With Transshipment)**: Single-Agent PPO operating on a joint state and action space, including lateral transshipment.
3. **SA-PPO (No Transshipment)**: Single-Agent PPO without lateral transshipment capability.
4. **Order-Up-To Baseline (OUT)**: Traditional operational heuristic policy maintaining target safety inventory levels.

---

## 2. Repository Structure
The project is organized as follows:

```text
project_root/
│
├── README.md                           # Comprehensive user guide and reproduction instructions
├── requirements.txt                    # Python environment dependencies
├── LICENSE                             # MIT Open-Source License
│
├── src/                                # Core library source files
│   ├── configs/
│   │   └── config.py                   # Centralized configuration and hyperparameters
│   ├── demand/
│   │   └── demand_model.py             # Hospital demand generation (ZINB sampling)
│   ├── env/
│   │   └── platelet_env.py             # Unified Platelet Supply Chain simulation environment
│   ├── agents/
│   │   ├── models.py                   # Actor neural networks and RunningMeanStd trackers
│   │   ├── mappo.py                    # MAPPO policy training and action selection
│   │   └── ppo.py                      # Single-Agent PPO (SA-T, SA-NT) policies
│   ├── baselines/
│   │   └── order_up_to.py              # Operational Order-Up-To baseline policy
│   ├── evaluation/
│   │   └── evaluation.py               # Evaluation runners and structured Excel report building
│   └── utils/
│       ├── seeding.py                  # Centralized random seed handling
│       └── plotting.py                 # Scientific plotting style defaults
│
├── notebooks/                          # Lightweight orchestration notebooks
│   ├── 01_training.ipynb               # Trains all RL policies and generates baseline records
│   ├── 02_sensitivity_analysis.ipynb   # Runs policy evaluation under varying demand/cost scenarios
│   ├── 03_statistical_tests.ipynb      # Global Friedman and planned pairwise Wilcoxon signed-rank tests
│   └── 04_figures.ipynb                # Generates paper figures and plot visualizations
│
├── results/                            # Saved artifacts
│   ├── models/                         # Serialized PyTorch model checkpoints (.pt)
│   ├── figures/                        # Generated scientific figures and visualizations
│   ├── tables/                         # Output evaluation sheets (.xlsx)
│   └── logs/                           # Training output summaries and log tracks
│
└── tests/                              # Unit testing suite
    └── test_all.py                     # Pytest suite verifying environment and demand sampling
```

---

## 3. Installation
This project supports **Python 3.11** or **Python 3.12**.

To set up the project environment:

1. Clone this repository to your local machine.
2. Ensure you have PyEnv or Virtualenv set up.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

To run unit tests to verify your installation:
```bash
python3 -m pytest tests/
```

---

## 4. Training Instructions
All reinforcement learning training runs are parameterized in `src/configs/config.py` and orchestrated in `notebooks/01_training.ipynb`.

To train all the policies (MAPPO, SA-PPO with Transshipment, SA-PPO No Transshipment) on 5 independent training seeds and perform evaluations:

1. Launch Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook
   ```
2. Open and run all cells in `notebooks/01_training.ipynb`. This will:
   - Train the three RL agents across 5 training seeds (`42, 100, 2023, 888, 55`) for `2000` episodes each.
   - Save trained model checkpoints in `results/models/`.
   - Run a multi-seed evaluation on 50 evaluation seeds.
   - Generate evaluation result tables in `results/tables/`:
     - `All Result - MAPPO.xlsx`
     - `All Result-SA-T.xlsx`
     - `All Result-SA-NT.xlsx`
     - `All Result-OUT.xlsx`

---

## 5. Evaluation Instructions
Policy evaluations are automatically triggered after training in `01_training.ipynb`. If you have pre-saved model checkpoints in `results/models/` and want to re-evaluate them without re-training:

You can use the helper function `evaluate_multi_seed` inside `src/evaluation/evaluation.py`:
```python
from src.env.platelet_env import PlateletSupplyChainEnv
from src.agents.mappo import MultiAgentMAPPO
from src.evaluation.evaluation import evaluate_multi_seed

env = PlateletSupplyChainEnv(enable_transshipment=True)
agent = MultiAgentMAPPO()
agent.load("results/models/mappo_seed_42.pt")

h1_sum, h2_sum, sc_sum, raw_rows = evaluate_multi_seed(env, agent, train_seed=42, is_mappo=True)
```

---

## 6. Sensitivity Analysis Instructions
Sensitivity analysis assesses model robustness under different operational parameters (e.g. reduced shelf-lives, increased/reduced wastage costs, heightened demand variance, and disruption events).

1. Open `notebooks/02_sensitivity_analysis.ipynb`.
2. Run all cells. This executes the evaluation of trained models against modified parameters and outputs summary spreadsheets to `results/tables/`:
   - `Sensitivity_MAPPO.xlsx`
   - `Sensitivity_SA_T.xlsx`
   - `Sensitivity_SA_NT.xlsx`

---

## 7. Statistical Test Instructions
The framework performs repeated-measures statistical analysis to verify cost improvements' significance.
- A **Friedman test** is used for a global 4-policy comparison (MAPPO vs SA-T vs SA-NT vs Heuristic).
- Pairwise **Wilcoxon signed-rank tests** with **Holm-Bonferroni correction** are used for planned pairwise comparisons.

To execute the statistical testing pipeline:
1. Open and run `notebooks/03_statistical_tests.ipynb`.
2. Review the printed Chi-Squared statistics, raw p-values, corrected p-values, and model-level t-test robustness checks.

---

## 8. Figure Generation Instructions
Figures are generated using the scientific plotting guidelines from the paper (matching DejaVu Sans or serif fonts, color schemes, grid alpha values, and proper DPI).

1. Open `notebooks/04_figures.ipynb`.
2. Run all cells to process the excel summary sheets from the tables folder and generate figures saved to `results/figures/`:
   - `network_cost_boxplot.png` (Boxplot distribution of network costs)
   - `cost_components_stacked_bar.png` (Stacked bar chart of operating cost components)
   - `sensitivity_trajectories.png` (Sensitivity trajectories for disruption and penalties)
   - `policy_simulation_dynamics.png` (Simulated trace mapping of system inventory states under disruption)

---

## 9. Citation Information
If you use this codebase or the associated research results, please cite:

```bibtex
@article{platelet_rl_2026,
  author    = {Academic Platelet Supply Chain Research Group},
  title     = {Reinforcement Learning for Hospital Network Platelet Inventory Management under Disruptions},
  journal   = {Operations Research & Scientific Computing},
  year      = {2026},
  doi       = {10.1000/xyz123}
}
```
