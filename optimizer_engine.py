"""
CAPELCO Power Dispatch Optimization Engine
Mixed-Integer Linear Programming (MILP) Solver using PuLP.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import pulp


@dataclass
class OptimizationResult:
    status: str
    is_optimal: bool
    total_cost_kphp: float
    total_demand_mwh: float
    blended_rate_php_kwh: float
    spot_energy_mwh: float
    spot_cost_kphp: float
    spot_share_pct: float
    dispatch_df: pd.DataFrame
    hourly_cost_df: pd.DataFrame
    generator_metrics_df: pd.DataFrame
    daily_summary_df: pd.DataFrame
    constraint_check_df: pd.DataFrame
    model: Optional[pulp.LpProblem] = None


class DispatchOptimizer:
    """
    MILP-based Power Dispatch Optimization for CAPELCO.
    Formulates and solves hourly dispatch minimizing total power procurement cost
    while satisfying physical generator capabilities, contract commitments, and load balance.
    """

    def __init__(
        self,
        demand_df: pd.DataFrame,
        generators_df: pd.DataFrame,
        spot_df: pd.DataFrame,
    ):
        self.demand_df = demand_df.copy()
        self.generators_df = generators_df.copy()
        self.spot_df = spot_df.copy()
        self._preprocess_data()

    @classmethod
    def from_excel(cls, file_path_or_buffer: Any) -> "DispatchOptimizer":
        """Instantiate optimizer from an Excel workbook with Demand, Generators, and SpotMarket sheets."""
        xls = pd.ExcelFile(file_path_or_buffer)
        demand_df = pd.read_excel(xls, "Demand")
        generators_df = pd.read_excel(xls, "Generators")
        spot_df = pd.read_excel(xls, "SpotMarket")
        return cls(demand_df, generators_df, spot_df)

    def _preprocess_data(self):
        """Clean and normalize input datasets."""
        # Ensure correct column formats
        if "Date" in self.demand_df.columns:
            self.demand_df["Date"] = pd.to_datetime(self.demand_df["Date"]).dt.strftime("%Y-%m-%d")
        if "Date" in self.spot_df.columns:
            self.spot_df["Date"] = pd.to_datetime(self.spot_df["Date"]).dt.strftime("%Y-%m-%d")

        self.demand_df["Hour"] = self.demand_df["Hour"].astype(int)
        self.spot_df["Hour"] = self.spot_df["Hour"].astype(int)

        # Build index lookups
        self.gen_dict = {}
        for row in self.generators_df.itertuples():
            gen_name = str(row.Generator).strip()
            self.gen_dict[gen_name] = {
                "Capacity": float(row.Capacity),
                "Price": float(row.Price),
                "Baseload": float(getattr(row, "Baseload", 0)),
                "CUF": float(getattr(row, "CUF", 1.0)),
                "Notes": str(getattr(row, "Notes", "")),
            }

    def solve(
        self,
        outages: Optional[Dict[str, bool]] = None,
        capacity_overrides: Optional[Dict[str, float]] = None,
        price_multipliers: Optional[Dict[str, float]] = None,
        spot_price_multiplier: float = 1.0,
        demand_multiplier: float = 1.0,
    ) -> OptimizationResult:
        """
        Formulate and solve the MILP optimization model.
        
        Args:
            outages: Dict mapping generator name to boolean (True = on outage / 0 MW).
            capacity_overrides: Dict mapping generator name to custom max capacity in MW.
            price_multipliers: Dict mapping generator name to price scaling factor.
            spot_price_multiplier: Scaling factor for spot market prices.
            demand_multiplier: Scaling factor for hourly demand profile.
        """
        outages = outages or {}
        capacity_overrides = capacity_overrides or {}
        price_multipliers = price_multipliers or {}

        # Merge demand and spot data
        merged = pd.merge(
            self.demand_df,
            self.spot_df,
            on=["Date", "Hour"],
            how="inner"
        )
        if merged.empty:
            raise ValueError("Merged Demand and SpotMarket data is empty. Check Date and Hour alignment.")

        # Apply demand scaling
        demand = {
            (row.Date, row.Hour): float(row.Demand) * demand_multiplier
            for row in merged.itertuples()
        }
        # Apply spot price scaling
        spot_price = {
            (row.Date, row.Hour): float(row.SpotPrice) * spot_price_multiplier
            for row in merged.itertuples()
        }

        dates = sorted(list(set(row.Date for row in merged.itertuples())))
        hours_by_date = {
            d: sorted([row.Hour for row in merged.itertuples() if row.Date == d])
            for d in dates
        }
        all_keys = sorted(list(demand.keys()))

        # Generator effective parameters
        active_gens = {}
        for g, spec in self.gen_dict.items():
            is_out = outages.get(g, False)
            cap = 0.0 if is_out else capacity_overrides.get(g, spec["Capacity"])
            mult = price_multipliers.get(g, 1.0)
            price = spec["Price"] * mult
            active_gens[g] = {
                "Capacity": cap,
                "Price": price,
                "Baseload": spec["Baseload"],
                "CUF": spec["CUF"],
                "Notes": spec["Notes"],
                "IsOutage": is_out,
            }

        # Initialize PuLP Problem
        model = pulp.LpProblem("CAPELCO_Optimal_Dispatch", pulp.LpMinimize)

        # -----------------------------
        # Decision Variables
        # -----------------------------
        # GCGI: Baseload must-run (15 MW nominal or specified capacity if not on outage)
        gcgi_cap = active_gens.get("GCGI", {}).get("Capacity", 15.0)
        
        # FDC: Fixed 4 MW (or min(4, cap) if derated/outage)
        fdc_cap = min(4.0, active_gens.get("FDC", {}).get("Capacity", 4.0))

        # PEDC: Discrete 4 MW or 8 MW (4 + 4*u_pedc)
        # If cap < 8 MW, adjust accordingly
        u_pedc = pulp.LpVariable.dicts("u_PEDC", all_keys, cat=pulp.LpBinary)
        
        # SPI: Whole-number MW nominations [0, cap].
        # Integer MW is enforced in the MILP itself; do not round after solving.
        spi_cap = active_gens.get("SPI", {}).get("Capacity", 8.0)
        disp_spi = pulp.LpVariable.dicts(
            "disp_SPI",
            all_keys,
            lowBound=0,
            upBound=int(np.floor(spi_cap)),
            cat=pulp.LpInteger,
        )

        # DG1 & DG2: Discrete Peakers (0 or Capacity)
        dg1_cap = active_gens.get("DG1", {}).get("Capacity", 3.1)
        u_dg1 = pulp.LpVariable.dicts("u_DG1", all_keys, cat=pulp.LpBinary)

        dg2_cap = active_gens.get("DG2", {}).get("Capacity", 3.8)
        u_dg2 = pulp.LpVariable.dicts("u_DG2", all_keys, cat=pulp.LpBinary)

        # GCGI/FDC are fixed whole-MW commitments (15 MW / 4 MW in the
        # standard data); PEDC is already discrete at 4 or 8 MW.
        # SPI is explicitly integer above.
        #
        # DG1/DG2 remain binary 0-or-full because their supplied physical
        # capacities are fractional (3.1 and 3.8 MW). Forcing those exact
        # capacities to integers would change the physical contract.
        #
        # Spot Market: Continuous [0, inf)
        disp_spot = pulp.LpVariable.dicts("disp_Spot", all_keys, lowBound=0)

        # Unserved Energy Penalty / Slack (to guarantee feasibility & diagnostic info if demand exceeds all resources)
        unserved = pulp.LpVariable.dicts("unserved", all_keys, lowBound=0)
        PENALTY_UNSERVED = 1000.0  # PHP/kWh VOLL (Value of Lost Load)

        # Helper for PEDC hourly output
        pedc_base = 4.0 if active_gens.get("PEDC", {}).get("Capacity", 8.0) >= 4.0 else 0.0
        pedc_step = max(0.0, active_gens.get("PEDC", {}).get("Capacity", 8.0) - pedc_base)

        # -----------------------------
        # Objective Function
        # Total generation procurement cost (kPHP or PHP, consistent across all hours)
        # -----------------------------
        model += (
            pulp.lpSum([
                # GCGI Fixed
                gcgi_cap * active_gens.get("GCGI", {}).get("Price", 0) +
                # FDC Fixed
                fdc_cap * active_gens.get("FDC", {}).get("Price", 0) +
                # PEDC (4 + 4*u)
                (pedc_base + pedc_step * u_pedc[(d, h)]) * active_gens.get("PEDC", {}).get("Price", 0) +
                # SPI
                disp_spi[(d, h)] * active_gens.get("SPI", {}).get("Price", 0) +
                # DG1
                (dg1_cap * u_dg1[(d, h)]) * active_gens.get("DG1", {}).get("Price", 0) +
                # DG2
                (dg2_cap * u_dg2[(d, h)]) * active_gens.get("DG2", {}).get("Price", 0) +
                # Spot Market
                disp_spot[(d, h)] * spot_price[(d, h)] +
                # Unserved slack
                unserved[(d, h)] * PENALTY_UNSERVED
                for (d, h) in all_keys
            ]),
            "Total_Procurement_Cost"
        )

        # -----------------------------
        # Constraints
        # -----------------------------
        for d in dates:
            d_hours = hours_by_date[d]
            num_h = len(d_hours)

            # 1. Daily Minimum Energy / CUF Contractual Commitments
            # PEDC: 8 MW * 24h * 0.8 = 153.6 MWh (scaled by available capacity)
            if not active_gens.get("PEDC", {}).get("IsOutage", False) and pedc_step > 0:
                pedc_min_daily = min(153.6 * (num_h / 24.0), active_gens["PEDC"]["Capacity"] * num_h * 0.8)
                model += (
                    pulp.lpSum([pedc_base + pedc_step * u_pedc[(d, h)] for h in d_hours]) >= pedc_min_daily,
                    f"PEDC_Daily_Min_Energy_{d}"
                )

            # SPI: 144 MWh/day minimum contractual energy
            if not active_gens.get("SPI", {}).get("IsOutage", False) and spi_cap > 0:
                spi_min_daily = min(144.0 * (num_h / 24.0), spi_cap * num_h * 0.75)
                model += (
                    pulp.lpSum([disp_spi[(d, h)] for h in d_hours]) >= spi_min_daily,
                    f"SPI_Daily_Min_Energy_{d}"
                )

            # 2. Hourly Load Balance Constraint
            for h in d_hours:
                model += (
                    gcgi_cap +
                    fdc_cap +
                    (pedc_base + pedc_step * u_pedc[(d, h)]) +
                    disp_spi[(d, h)] +
                    (dg1_cap * u_dg1[(d, h)]) +
                    (dg2_cap * u_dg2[(d, h)]) +
                    disp_spot[(d, h)] +
                    unserved[(d, h)] == demand[(d, h)],
                    f"Demand_Balance_{d}_{h}"
                )

        # -----------------------------
        # Solve
        # -----------------------------
        solver = pulp.PULP_CBC_CMD(msg=0)
        status_code = model.solve(solver)
        status_str = pulp.LpStatus[status_code]
        is_optimal = (status_str == "Optimal")

        # -----------------------------
        # Collect & Format Output
        # -----------------------------
        dispatch_records = []
        cost_records = []

        for (d, h) in all_keys:
            pedc_gen = pedc_base + pedc_step * (u_pedc[(d, h)].varValue or 0.0)
            spi_gen = disp_spi[(d, h)].varValue or 0.0
            dg1_gen = dg1_cap * (u_dg1[(d, h)].varValue or 0.0)
            dg2_gen = dg2_cap * (u_dg2[(d, h)].varValue or 0.0)
            spot_gen = disp_spot[(d, h)].varValue or 0.0
            unserved_gen = unserved[(d, h)].varValue or 0.0

            # Power MW
            disp_row = {
                "Date": d,
                "Hour": h,
                "Demand_MW": demand[(d, h)],
                "GCGI_MW": gcgi_cap,
                "FDC_MW": fdc_cap,
                "PEDC_MW": pedc_gen,
                "SPI_MW": spi_gen,
                "DG1_MW": dg1_gen,
                "DG2_MW": dg2_gen,
                "Spot_MW": spot_gen,
                "Unserved_MW": unserved_gen,
            }
            dispatch_records.append(disp_row)

            # Costs (kPHP)
            gcgi_c = gcgi_cap * active_gens.get("GCGI", {}).get("Price", 0)
            fdc_c = fdc_cap * active_gens.get("FDC", {}).get("Price", 0)
            pedc_c = pedc_gen * active_gens.get("PEDC", {}).get("Price", 0)
            spi_c = spi_gen * active_gens.get("SPI", {}).get("Price", 0)
            dg1_c = dg1_gen * active_gens.get("DG1", {}).get("Price", 0)
            dg2_c = dg2_gen * active_gens.get("DG2", {}).get("Price", 0)
            spot_c = spot_gen * spot_price[(d, h)]
            tot_c = gcgi_c + fdc_c + pedc_c + spi_c + dg1_c + dg2_c + spot_c

            cost_row = {
                "Date": d,
                "Hour": h,
                "GCGI_Cost": gcgi_c,
                "FDC_Cost": fdc_c,
                "PEDC_Cost": pedc_c,
                "SPI_Cost": spi_c,
                "DG1_Cost": dg1_c,
                "DG2_Cost": dg2_c,
                "Spot_Cost": spot_c,
                "Total_Cost_kPHP": tot_c,
                "Blended_Rate_PHP_kWh": tot_c / demand[(d, h)] if demand[(d, h)] > 0 else 0,
            }
            cost_records.append(cost_row)

        dispatch_df = pd.DataFrame(dispatch_records)
        cost_df = pd.DataFrame(cost_records)

        # Summary KPIs
        total_demand = dispatch_df["Demand_MW"].sum()
        total_cost = cost_df["Total_Cost_kPHP"].sum()
        blended_rate = total_cost / total_demand if total_demand > 0 else 0.0
        spot_energy = dispatch_df["Spot_MW"].sum()
        spot_cost = cost_df["Spot_Cost"].sum()
        spot_share = (spot_energy / total_demand * 100) if total_demand > 0 else 0.0

        # Generator Performance Metrics Table
        gen_metrics = []
        gen_cols = [
            ("GCGI", "GCGI_MW", "GCGI_Cost", active_gens.get("GCGI", {}).get("Capacity", 15.0), active_gens.get("GCGI", {}).get("Price", 0)),
            ("FDC", "FDC_MW", "FDC_Cost", active_gens.get("FDC", {}).get("Capacity", 8.0), active_gens.get("FDC", {}).get("Price", 0)),
            ("PEDC", "PEDC_MW", "PEDC_Cost", active_gens.get("PEDC", {}).get("Capacity", 8.0), active_gens.get("PEDC", {}).get("Price", 0)),
            ("SPI", "SPI_MW", "SPI_Cost", active_gens.get("SPI", {}).get("Capacity", 8.0), active_gens.get("SPI", {}).get("Price", 0)),
            ("DG1", "DG1_MW", "DG1_Cost", active_gens.get("DG1", {}).get("Capacity", 3.1), active_gens.get("DG1", {}).get("Price", 0)),
            ("DG2", "DG2_MW", "DG2_Cost", active_gens.get("DG2", {}).get("Capacity", 3.8), active_gens.get("DG2", {}).get("Price", 0)),
            ("Spot Market", "Spot_MW", "Spot_Cost", 999.0, self.spot_df["SpotPrice"].mean()),
        ]

        total_hours = len(dispatch_df)
        for name, col_mw, col_cost, cap, base_price in gen_cols:
            tot_mwh = dispatch_df[col_mw].sum()
            tot_kphp = cost_df[col_cost].sum()
            share_pct = (tot_mwh / total_demand * 100) if total_demand > 0 else 0.0
            avg_rate = (tot_kphp / tot_mwh) if tot_mwh > 0 else base_price
            cuf_pct = (tot_mwh / (cap * total_hours) * 100) if (0 < cap < 900) and total_hours > 0 else np.nan

            gen_metrics.append({
                "Generator": name,
                "Capacity (MW)": cap if cap < 900 else "Variable",
                "Contract Price (PHP/kWh)": f"{base_price:.4f}" if cap < 900 else f"Avg: {base_price:.2f}",
                "Energy Dispatched (MWh)": tot_mwh,
                "Energy Share (%)": share_pct,
                "Total Cost (kPHP)": tot_kphp,
                "Effective Rate (PHP/kWh)": avg_rate,
                "Capacity Factor (CUF %)": cuf_pct,
            })

        gen_metrics_df = pd.DataFrame(gen_metrics)

        # Daily summary DataFrame
        daily_summary = dispatch_df.groupby("Date").agg({
            "Demand_MW": "sum",
            "GCGI_MW": "sum",
            "FDC_MW": "sum",
            "PEDC_MW": "sum",
            "SPI_MW": "sum",
            "DG1_MW": "sum",
            "DG2_MW": "sum",
            "Spot_MW": "sum",
        }).reset_index()

        daily_costs = cost_df.groupby("Date")["Total_Cost_kPHP"].sum().reset_index()
        daily_summary = pd.merge(daily_summary, daily_costs, on="Date")
        daily_summary["Blended_Rate_PHP_kWh"] = daily_summary["Total_Cost_kPHP"] / daily_summary["Demand_MW"]

        # ---------------------------------------------------------
        # Daily Generator Constraint Verification
        # ---------------------------------------------------------
        # This table reports the actual daily total for every resource
        # against the contractual/physical requirement used by the MILP.
        constraint_rows = []

        for d in dates:
            d_hours = hours_by_date[d]
            num_h = len(d_hours)

            # GCGI: must run at available capacity every hour.
            gcgi_actual = dispatch_df.loc[dispatch_df["Date"] == d, "GCGI_MW"].sum()
            gcgi_expected = gcgi_cap * num_h
            constraint_rows.append({
                "Date": d,
                "Generator": "GCGI",
                "Daily Total (MWh)": gcgi_actual,
                "Minimum (MWh)": gcgi_expected,
                "Maximum (MWh)": gcgi_expected,
                "Constraint": "Must run at available capacity",
                "Status": "PASS" if abs(gcgi_actual - gcgi_expected) <= 1e-6 else "FAIL",
            })

            # FDC: fixed 4 MW (or derated available capacity) every hour.
            fdc_actual = dispatch_df.loc[dispatch_df["Date"] == d, "FDC_MW"].sum()
            fdc_expected = fdc_cap * num_h
            constraint_rows.append({
                "Date": d,
                "Generator": "FDC",
                "Daily Total (MWh)": fdc_actual,
                "Minimum (MWh)": fdc_expected,
                "Maximum (MWh)": fdc_expected,
                "Constraint": "Fixed output",
                "Status": "PASS" if abs(fdc_actual - fdc_expected) <= 1e-6 else "FAIL",
            })

            # PEDC: each hour is 4 or 8 MW, with the daily minimum.
            pedc_actual = dispatch_df.loc[dispatch_df["Date"] == d, "PEDC_MW"].sum()
            pedc_min = 0.0
            if not active_gens.get("PEDC", {}).get("IsOutage", False) and pedc_step > 0:
                pedc_min = min(
                    153.6 * (num_h / 24.0),
                    active_gens["PEDC"]["Capacity"] * num_h * 0.8,
                )
            pedc_max = active_gens.get("PEDC", {}).get("Capacity", 8.0) * num_h
            pedc_hourly = dispatch_df.loc[dispatch_df["Date"] == d, "PEDC_MW"]
            pedc_values_ok = pedc_hourly.apply(
                lambda x: abs(x - 4.0) <= 1e-6 or abs(x - 8.0) <= 1e-6
            ).all() if pedc_hourly.notna().all() else False
            if active_gens.get("PEDC", {}).get("IsOutage", False):
                pedc_values_ok = (pedc_hourly.abs() <= 1e-6).all()

            constraint_rows.append({
                "Date": d,
                "Generator": "PEDC",
                "Daily Total (MWh)": pedc_actual,
                "Minimum (MWh)": pedc_min,
                "Maximum (MWh)": pedc_max,
                "Constraint": "Hourly 4/8 MW; daily minimum",
                "Status": "PASS" if pedc_values_ok and pedc_actual + 1e-6 >= pedc_min else "FAIL",
            })

            # SPI: integer MW, 0..capacity, plus daily minimum.
            spi_actual = dispatch_df.loc[dispatch_df["Date"] == d, "SPI_MW"].sum()
            spi_min = 0.0
            if not active_gens.get("SPI", {}).get("IsOutage", False) and spi_cap > 0:
                spi_min = min(
                    144.0 * (num_h / 24.0),
                    spi_cap * num_h * 0.75,
                )
            spi_max = int(np.floor(spi_cap)) * num_h
            spi_hourly = dispatch_df.loc[dispatch_df["Date"] == d, "SPI_MW"]
            spi_integer_ok = spi_hourly.apply(lambda x: abs(x - round(x)) <= 1e-6).all()
            constraint_rows.append({
                "Date": d,
                "Generator": "SPI",
                "Daily Total (MWh)": spi_actual,
                "Minimum (MWh)": spi_min,
                "Maximum (MWh)": spi_max,
                "Constraint": "Whole MW hourly; daily minimum",
                "Status": "PASS" if spi_integer_ok and spi_actual + 1e-6 >= spi_min and spi_actual <= spi_max + 1e-6 else "FAIL",
            })

            # DG1/DG2: exact 0-or-full physical capacity.
            for dg_name, dg_col, dg_cap in [
                ("DG1", "DG1_MW", dg1_cap),
                ("DG2", "DG2_MW", dg2_cap),
            ]:
                dg_actual = dispatch_df.loc[dispatch_df["Date"] == d, dg_col].sum()
                dg_values = dispatch_df.loc[dispatch_df["Date"] == d, dg_col]
                dg_binary_ok = dg_values.apply(
                    lambda x, cap=dg_cap: abs(x) <= 1e-6 or abs(x - cap) <= 1e-6
                ).all()
                constraint_rows.append({
                    "Date": d,
                    "Generator": dg_name,
                    "Daily Total (MWh)": dg_actual,
                    "Minimum (MWh)": 0.0,
                    "Maximum (MWh)": dg_cap * num_h,
                    "Constraint": f"0 or full capacity ({dg_cap:g} MW)",
                    "Status": "PASS" if dg_binary_ok else "FAIL",
                })

            # Spot market: continuous balancing resource; no generator contract.
            spot_actual = dispatch_df.loc[dispatch_df["Date"] == d, "Spot_MW"].sum()
            constraint_rows.append({
                "Date": d,
                "Generator": "Spot Market",
                "Daily Total (MWh)": spot_actual,
                "Minimum (MWh)": 0.0,
                "Maximum (MWh)": np.nan,
                "Constraint": "Continuous balancing purchase",
                "Status": "INFO",
            })

        constraint_check_df = pd.DataFrame(constraint_rows)

        return OptimizationResult(
            status=status_str,
            is_optimal=is_optimal,
            total_cost_kphp=total_cost,
            total_demand_mwh=total_demand,
            blended_rate_php_kwh=blended_rate,
            spot_energy_mwh=spot_energy,
            spot_cost_kphp=spot_cost,
            spot_share_pct=spot_share,
            dispatch_df=dispatch_df,
            hourly_cost_df=cost_df,
            generator_metrics_df=gen_metrics_df,
            daily_summary_df=daily_summary,
            constraint_check_df=constraint_check_df,
            model=model,
        )


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "dispatch_input.xlsx")
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(base_dir), "dispatch_input.xlsx")

    print("=" * 60)
    print("CAPELCO OPTIMAL DISPATCH ENGINE (MILP)")
    print("=" * 60)
    print(f"Loading data from: {data_path}")
    
    optimizer = DispatchOptimizer.from_excel(data_path)
    res = optimizer.solve()

    print(f"Solver Status        : {res.status}")
    print(f"Total Energy Demand  : {res.total_demand_mwh:,.2f} MWh")
    print(f"Total Procurement Cost: PHP {res.total_cost_kphp:,.2f} kPHP")
    print(f"Blended Rate         : PHP {res.blended_rate_php_kwh:.4f} / kWh")
    print(f"Spot Market Energy   : {res.spot_energy_mwh:,.2f} MWh ({res.spot_share_pct:.2f}%)")
    print("\n--- Generator Performance Metrics ---")
    print(res.generator_metrics_df.to_string(index=False))
    print("\n--- Daily Generator Constraint Checks ---")
    print(res.constraint_check_df.to_string(index=False))
    print("=" * 60)
