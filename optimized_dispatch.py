"""
CAPELCO Optimized Power Dispatch (CLI Script)
Mixed-Integer Linear Programming Formulation with PuLP.
"""

import os
import pandas as pd
import pulp

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "data", "dispatch_input.xlsx")
    if not os.path.exists(input_file):
        input_file = os.path.join(os.path.dirname(base_dir), "dispatch_input.xlsx")

    output_file = os.path.join(base_dir, "dispatch_output.xlsx")

    print(f"Loading input data from: {input_file}")
    xls = pd.ExcelFile(input_file)

    demand_df = pd.read_excel(xls, "Demand")
    gen_df = pd.read_excel(xls, "Generators").set_index("Generator")
    spot_df = pd.read_excel(xls, "SpotMarket")

    # Format Data
    demand = {(str(row.Date)[:10], int(row.Hour)): float(row.Demand) for row in demand_df.itertuples()}
    spot_price = {(str(row.Date)[:10], int(row.Hour)): float(row.SpotPrice) for row in spot_df.itertuples()}
    dates = sorted(list(set(d for (d, h) in demand.keys())))
    all_keys = sorted(list(demand.keys()))

    # -----------------------------
    # Model Setup
    # -----------------------------
    model = pulp.LpProblem("CAPELCO_Dispatch_Optimization", pulp.LpMinimize)

    # Decision Variables
    u_pedc = pulp.LpVariable.dicts("u_PEDC", all_keys, cat=pulp.LpBinary)
    disp_spi = pulp.LpVariable.dicts("disp_SPI", all_keys, lowBound=0, upBound=8.0)
    u_dg1 = pulp.LpVariable.dicts("u_DG1", all_keys, cat=pulp.LpBinary)
    u_dg2 = pulp.LpVariable.dicts("u_DG2", all_keys, cat=pulp.LpBinary)
    disp_spot = pulp.LpVariable.dicts("disp_Spot", all_keys, lowBound=0)

    # Baseload constants
    GCGI_MW = 15.0  # Must-run 100%
    FDC_MW = 4.0    # Fixed output 4 MW

    # Objective Function: Total Procurement Cost in kPHP
    model += (
        pulp.lpSum([
            GCGI_MW * gen_df.loc["GCGI", "Price"] +
            FDC_MW * gen_df.loc["FDC", "Price"] +
            (4.0 + 4.0 * u_pedc[(d, h)]) * gen_df.loc["PEDC", "Price"] +
            disp_spi[(d, h)] * gen_df.loc["SPI", "Price"] +
            (3.1 * u_dg1[(d, h)]) * gen_df.loc["DG1", "Price"] +
            (3.8 * u_dg2[(d, h)]) * gen_df.loc["DG2", "Price"] +
            disp_spot[(d, h)] * spot_price[(d, h)]
            for (d, h) in all_keys
        ]),
        "Total_Cost_Objective"
    )

    # Constraints
    for d in dates:
        day_keys = [(dt, h) for (dt, h) in all_keys if dt == d]
        
        # 1. Daily minimum energy commitments
        # PEDC: 8 MW * 24h * 0.8 CUF = 153.6 MWh
        model += (
            pulp.lpSum([4.0 + 4.0 * u_pedc[(dt, h)] for (dt, h) in day_keys]) >= 153.6,
            f"PEDC_Daily_Min_{d}"
        )
        # SPI: 144.0 MWh daily minimum
        model += (
            pulp.lpSum([disp_spi[(dt, h)] for (dt, h) in day_keys]) >= 144.0,
            f"SPI_Daily_Min_{d}"
        )

        # 2. Hourly demand balance
        for (dt, h) in day_keys:
            model += (
                GCGI_MW +
                FDC_MW +
                (4.0 + 4.0 * u_pedc[(dt, h)]) +
                disp_spi[(dt, h)] +
                (3.1 * u_dg1[(dt, h)]) +
                (3.8 * u_dg2[(dt, h)]) +
                disp_spot[(dt, h)] == demand[(dt, h)],
                f"Demand_Balance_{dt}_{h}"
            )

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = model.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        print(f"Solver stopped with status: {pulp.LpStatus[status]}")
        return

    # -----------------------------
    # Export Results
    # -----------------------------
    results = []
    for (d, h) in all_keys:
        pedc_val = 4.0 + 4.0 * (u_pedc[(d, h)].varValue or 0.0)
        spi_val = disp_spi[(d, h)].varValue or 0.0
        dg1_val = 3.1 * (u_dg1[(d, h)].varValue or 0.0)
        dg2_val = 3.8 * (u_dg2[(d, h)].varValue or 0.0)
        spot_val = disp_spot[(d, h)].varValue or 0.0
        
        hourly_cost = (
            GCGI_MW * gen_df.loc["GCGI", "Price"] +
            FDC_MW * gen_df.loc["FDC", "Price"] +
            pedc_val * gen_df.loc["PEDC", "Price"] +
            spi_val * gen_df.loc["SPI", "Price"] +
            dg1_val * gen_df.loc["DG1", "Price"] +
            dg2_val * gen_df.loc["DG2", "Price"] +
            spot_val * spot_price[(d, h)]
        )

        results.append({
            "Date": d,
            "Hour": h,
            "Demand": demand[(d, h)],
            "GCGI": GCGI_MW,
            "FDC": FDC_MW,
            "PEDC": pedc_val,
            "SPI": spi_val,
            "DG1": dg1_val,
            "DG2": dg2_val,
            "Spot": spot_val,
            "TotalCost": hourly_cost,
            "BlendedRate": hourly_cost / demand[(d, h)] if demand[(d, h)] > 0 else 0,
        })

    results_df = pd.DataFrame(results)
    results_df.to_excel(output_file, index=False)

    print("=" * 60)
    print("OPTIMIZATION COMPLETE (MILP Exact)")
    print(f"Total Energy Demand : {results_df['Demand'].sum():,.2f} MWh")
    print(f"Total Generation Cost: PHP {results_df['TotalCost'].sum():,.2f} kPHP")
    print(f"Weighted Blended Rate: PHP {results_df['TotalCost'].sum() / results_df['Demand'].sum():.4f} / kWh")
    print(f"Results saved to    : {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
