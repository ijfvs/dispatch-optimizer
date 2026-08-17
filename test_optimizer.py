"""
Unit and Integration Tests for CAPELCO Dispatch Optimizer
"""

import os
import unittest
import pandas as pd
import numpy as np

from optimizer_engine import DispatchOptimizer, OptimizationResult
from forecast_engine import DemandForecaster

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dispatch_input.xlsx")

class TestDispatchOptimizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.optimizer = DispatchOptimizer.from_excel(DATA_PATH)

    def test_base_solve_optimal(self):
        """Verify standard base case solve achieves optimal status."""
        result = self.optimizer.solve()
        self.assertTrue(result.is_optimal)
        self.assertEqual(result.status, "Optimal")
        self.assertGreater(result.total_cost_kphp, 0)
        self.assertGreater(result.total_demand_mwh, 0)

    def test_gcgi_must_run_15mw(self):
        """Verify GCGI operates at exactly 15 MW for all hours."""
        result = self.optimizer.solve()
        for val in result.dispatch_df["GCGI_MW"]:
            self.assertAlmostEqual(val, 15.0, places=4)

    def test_fdc_fixed_4mw(self):
        """Verify FDC operates at exactly 4 MW for all hours."""
        result = self.optimizer.solve()
        for val in result.dispatch_df["FDC_MW"]:
            self.assertAlmostEqual(val, 4.0, places=4)

    def test_pedc_discrete_and_daily_min(self):
        """Verify PEDC is either 4 MW or 8 MW, and daily sum >= 153.6 MWh."""
        result = self.optimizer.solve()
        for val in result.dispatch_df["PEDC_MW"]:
            self.assertTrue(np.isclose(val, 4.0, atol=1e-3) or np.isclose(val, 8.0, atol=1e-3))
        
        daily_pedc = result.dispatch_df.groupby("Date")["PEDC_MW"].sum()
        for d, tot in daily_pedc.items():
            self.assertGreaterEqual(tot, 153.6 - 1e-4)

    def test_spi_daily_min(self):
        """Verify SPI daily energy >= 144.0 MWh and hourly <= 8.0 MW."""
        result = self.optimizer.solve()
        for val in result.dispatch_df["SPI_MW"]:
            self.assertLessEqual(val, 8.0 + 1e-4)
            self.assertGreaterEqual(val, -1e-4)

        daily_spi = result.dispatch_df.groupby("Date")["SPI_MW"].sum()
        for d, tot in daily_spi.items():
            self.assertGreaterEqual(tot, 144.0 - 1e-4)

    def test_peakers_binary(self):
        """Verify DG1 is 0 or 3.1 MW, and DG2 is 0 or 3.8 MW."""
        result = self.optimizer.solve()
        for val in result.dispatch_df["DG1_MW"]:
            self.assertTrue(np.isclose(val, 0.0, atol=1e-3) or np.isclose(val, 3.1, atol=1e-3))
        for val in result.dispatch_df["DG2_MW"]:
            self.assertTrue(np.isclose(val, 0.0, atol=1e-3) or np.isclose(val, 3.8, atol=1e-3))

    def test_demand_balance(self):
        """Verify generation sum equals demand for every single hour."""
        result = self.optimizer.solve()
        df = result.dispatch_df
        gen_sum = (
            df["GCGI_MW"] + df["FDC_MW"] + df["PEDC_MW"] +
            df["SPI_MW"] + df["DG1_MW"] + df["DG2_MW"] +
            df["Spot_MW"] + df["Unserved_MW"]
        )
        for expected, actual in zip(df["Demand_MW"], gen_sum):
            self.assertAlmostEqual(expected, actual, places=3)

    def test_scenario_outage(self):
        """Verify generator outage forces generator dispatch to 0."""
        result = self.optimizer.solve(outages={"PEDC": True})
        self.assertTrue(result.is_optimal)
        for val in result.dispatch_df["PEDC_MW"]:
            self.assertAlmostEqual(val, 0.0, places=4)

    def test_forecaster(self):
        """Verify forecasting produces non-empty DataFrame with bounds."""
        forecaster = DemandForecaster(self.optimizer.demand_df)
        forecast_df = forecaster.forecast(periods=24)
        self.assertEqual(len(forecast_df), len(self.optimizer.demand_df) + 24)
        self.assertTrue((forecast_df["Demand_Upper"] >= forecast_df["Demand_Lower"]).all())


if __name__ == "__main__":
    unittest.main()
