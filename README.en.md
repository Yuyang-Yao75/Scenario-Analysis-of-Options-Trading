# Scenario Analysis: Option Portfolio Risk Assessment

[![EN](https://img.shields.io/badge/lang-English-blue.svg)](./README.en.md)  
[![CN](https://img.shields.io/badge/lang-中文-red.svg)](./README.md)

This project is designed for **scenario analysis of option portfolios**, evaluating profit and loss (P&L), margin requirements, and Greeks exposures under changes in underlying prices and volatilities.  
It is built on **iFinD** data interfaces, leveraging libraries such as `pandas`, `numpy`, `scipy`, and `xlsxwriter` to enable automated risk measurement and report generation.  

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

> **License**  
> This repository is released under the **PolyForm Noncommercial License 1.0.0**.  
> Noncommercial use only (learning, research, evaluation).  
> **Any commercial use requires a separate license.** For commercial licensing, contact: yyyao75@163.com.  
> See [LICENSE](./LICENSE) and [COMMERCIAL.md](./COMMERCIAL.md) for details.


---

## Features

* **One-click position retrieval and validation (iFinD + fault-tolerant backup)**  
  Log in via `THS_iFinDLogin` and use `THS_BD/THS_DR` to batch fetch required fields (underlying, close price, strike, maturity, remaining trading days, contract multiplier, etc.).  
  Automatically merge to local positions with `check_and_backup_out` integrity check: any missing values trigger **local backup** and a **readable missing column report** (missing ratio, repair suggestions), ensuring data quality and traceability.

* **Trading-session awareness and unified date semantics**  
  Supports “pre-market / intraday / post-market” modes. Automatically adjusts **date parameters and remaining trading days** (iFinD includes the current day; script applies -1/244 adjustments where appropriate), avoiding systematic biases caused by inconsistent definitions.

* **Intelligent estimation of at-the-money (ATM) underlying price (noise-reduced close)**  
  For European contracts, compute ATM price **once per underlying**:
  - Select the 5 strikes closest to spot and match calls/puts;  
  - Use **put–call parity** to infer implied spot, averaging across strikes to reduce noise;  
  - Warnings are issued when data is abnormal/incomplete, defaulting to `NaN` without breaking the workflow.

* **Dual-engine implied volatility solver (European Newton + American Bisection)**  
  - **European**: Newton iteration with Black–Scholes, enhanced with **no-arbitrage bounds** and **parity repair**. Handles cases with small Vega or convergence failure via fallback strategies.  
  - **American**: Barone–Adesi–Whaley (BAW) model with outer bisection. Includes safety fallbacks for extreme cases.

* **Unified pricing core with natural support for 2D scenario broadcasting**  
  - `bs_price` and `baw_american_option_price` vectorize over an 11×11 grid of **ds (spot change) × dv (vol change)**.  
  - `critical_price` iterates only over non-converged elements with parallelized relaxation, ensuring both efficiency and robustness.

* **Two-dimensional P&L heatmaps (contract → underlying → portfolio)**  
  - Compute option prices and P&L under **S × IV** scenarios, scaling by position and multiplier.  
  - Aggregate results to underlying and portfolio levels. Output as multi-sheet Excel: individual underlyings + `all_scenario`.

* **Margin scenario analysis (exchange rule built-in)**  
  - **A-share options**: Shanghai/Shenzhen/CFFEX rules.  
  - **Futures exchange options**: Uses iFinD field `ths_contract_short_deposit_future`, integrates with out-of-money and premium values. Takes the maximum of dual constraints, closer to risk-control practice.  
  - Supports scenario-based margin curves by underlying and portfolio.

* **European/American Greeks calculation (analytic + finite difference)**  
  - **European**: Delta/Gamma/Vega/Theta via closed forms, annualized with 244 trading days, Vega standardized to 1% vol change.  
  - **American**: Based on BAW, uses central differences and 1-day step for Greeks, more aligned with practical risk control.  
  - Aggregated to underlying and portfolio, output as ds × dv heatmaps.

* **Reports-as-documents: structured Excel outputs (ready for traders/risk)**  
  - Auto-generate dated folder + Excel with sheets:  
    - `Option Position Table (Full)` – cleaned and enriched data;  
    - Per-underlying: **P&L matrix + margin curve + Greeks heatmap**;  
    - `all_scenario`: portfolio-level aggregation.  
  - Polished formatting: diagonal headers, unified tags (`+5%`, `-10vol`), reducing misinterpretation risk.

* **Engineering details for robustness**  
  - Session validity check, exchange suffix mapping, selective iteration on non-converged elements, parity repair for out-of-bound prices, safeguards for tiny Vega/negative vol, error logging without breaking workflow.  
  - Deduplicated calculations (ATM spot per underlying) to reduce redundant calls.  
  - Type annotations, modular function boundaries for easy testing and maintenance.

* **Clear assumptions and scope**  
  - Annualized trading days `ANN_TRADING_DAYS = 244`;  
  - European default `q=0`; American futures options default `q=r`;  
  - Applicable to **SSE/SZSE ETF options** and **domestic futures exchange American options**, extensible to more markets.

* **Out-of-the-box but extensible**  
  - Configurable grid, scenario ranges, risk-free rate, margin parameters.  
  - Pricing core decoupled from risk core, enabling later integration with volatility surfaces, jump-diffusion models, or dashboards without rewriting the pipeline.

---

## Input / Output (Simplified)

**Input:**
- Local position file `Option_Position.csv`  
  - Required columns: `Code`, `Quantity` (negative for short, positive for long).  
- iFinD fetches underlyings, option attributes, market data, and margin parameters.  
- Configurable parameters:  
  - `SESSION ∈ {Pre, Intra, Post}`  
  - `r` (risk-free rate, default 0.02)  
  - Grid steps `generate_grid(ds_steps=11, dv_steps=11)` etc.

**Output:**
- Directory: `YYYY-MM-DD/`  
- File: `YYYY-MM-DD_ScenarioAnalysis.xlsx`, containing sheets:  
  1. **Option Position Table (Full)**  
  2. **Per Underlying**:  
     - P&L scenario matrix (S: -10%~+10%, IV: -10vol~+10vol)  
     - Margin scenarios  
     - Greeks heatmaps  
  3. **all_scenario**:  
     - Portfolio P&L matrix  
     - Portfolio margin table  
     - Total Greeks heatmaps.

---

## Requirements
- Python 3.8+  
- Dependencies: `pandas`, `numpy`, `scipy`, `matplotlib`, `xlsxwriter`, `iFinDPy`  
- Valid iFinD account & access permission  

---

## Installation
```bash
pip install pandas numpy scipy matplotlib xlsxwriter iFinDPy
````

---

## Usage

1. Prepare `Option_Position.csv` with option codes and quantities.
2. Set `SESSION` variable to select analysis time (pre, intra, post).
3. Run the main script:

   ```bash
   python ScenarioAnalysis.py
   ```
4. Reports are generated in `YYYY-MM-DD/` with Excel outputs.

---

## File Structure

```bash
ScenarioAnalysis.py     # Main script
Option_Position.csv     # Input positions
YYYY-MM-DD/             # Daily output directory
└── YYYY-MM-DD_ScenarioAnalysis.xlsx
README.md               # Project documentation
```

---

## Notes

* After updating the input file, delete auto-generated `Option_Position_Backup.csv` to avoid stale reads.
* For pre/intra-market analysis, ensure data sources are ready.
* Contains sensitive information (positions & Greeks); handle with confidentiality.

>Note: This project is currently in a stage of partial completion, and there will be no updates in the short term. Honestly, I am well aware that the code still has many shortcomings in terms of structure, efficiency, and complexity. If you notice any issues while reading or using it, I sincerely welcome your feedback and criticism. I firmly believe that open exchange and genuine sharing are the foundation of mutual growth, and that is the true value of this work.