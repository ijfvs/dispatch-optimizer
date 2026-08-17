"""
CAPELCO Demand Forecasting Engine
Prophet and Statistical Time-Series Forecasting for Hourly Load.
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class DemandForecaster:
    """
    Forecasting module for electric cooperative hourly demand.
    Uses Facebook Prophet with automatic fallback to statistical hourly profiling.
    """

    def __init__(self, historical_demand_df: pd.DataFrame):
        self.df = historical_demand_df.copy()
        self._prepare_data()

    def _prepare_data(self):
        """Prepare dataframe with timestamp column ds and target y."""
        df = self.df.copy()
        if "Date" in df.columns and "Hour" in df.columns:
            df["ds"] = pd.to_datetime(df["Date"].astype(str)) + pd.to_timedelta(df["Hour"].astype(int), unit="h")
        elif "ds" in df.columns:
            df["ds"] = pd.to_datetime(df["ds"])
        else:
            raise ValueError("Input data must contain 'Date' and 'Hour' or 'ds' timestamp.")

        if "Demand" in df.columns:
            df["y"] = df["Demand"].astype(float)
        elif "y" in df.columns:
            df["y"] = df["y"].astype(float)
        else:
            raise ValueError("Input data must contain 'Demand' or 'y' column.")

        self.prepared_df = df.sort_values("ds").reset_index(drop=True)

    def forecast(
        self,
        periods: int = 24,
        growth: str = "linear",
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
        confidence_interval: float = 0.95,
        growth_rate_pct: float = 0.0,
    ) -> pd.DataFrame:
        """
        Generate hourly demand forecast.
        
        Args:
            periods: Number of future hours to forecast (default 24).
            growth: 'linear' or 'flat'.
            daily_seasonality: Model 24-hour diurnal patterns.
            weekly_seasonality: Model day-of-week patterns.
            confidence_interval: Width of uncertainty intervals (default 0.95).
            growth_rate_pct: Manual adjustment percentage (+% demand growth).
        
        Returns:
            DataFrame with columns: ['ds', 'Date', 'Hour', 'Demand', 'Demand_Lower', 'Demand_Upper', 'Is_Forecast']
        """
        num_records = len(self.prepared_df)

        if PROPHET_AVAILABLE and num_records >= 48:
            # Full Prophet model fitting
            try:
                model = Prophet(
                    interval_width=confidence_interval,
                    daily_seasonality=daily_seasonality,
                    weekly_seasonality=weekly_seasonality if num_records >= 168 else False,
                    growth=growth,
                )
                model.fit(self.prepared_df[["ds", "y"]])
                future = model.make_future_dataframe(periods=periods, freq="h")
                forecast = model.predict(future)

                result_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                result_df.columns = ["ds", "Demand", "Demand_Lower", "Demand_Upper"]
                result_df["Demand"] = result_df["Demand"].clip(lower=0.0) * (1.0 + growth_rate_pct / 100.0)
                result_df["Demand_Lower"] = result_df["Demand_Lower"].clip(lower=0.0) * (1.0 + growth_rate_pct / 100.0)
                result_df["Demand_Upper"] = result_df["Demand_Upper"].clip(lower=0.0) * (1.0 + growth_rate_pct / 100.0)
            except Exception as e:
                # Fallback to statistical profile
                result_df = self._statistical_forecast(periods, confidence_interval, growth_rate_pct)
        else:
            # Statistical hourly persistence & seasonal pattern fallback
            result_df = self._statistical_forecast(periods, confidence_interval, growth_rate_pct)

        result_df["Date"] = result_df["ds"].dt.strftime("%Y-%m-%d")
        result_df["Hour"] = result_df["ds"].dt.hour
        result_df["Is_Forecast"] = result_df["ds"] > self.prepared_df["ds"].max()

        return result_df

    def _statistical_forecast(
        self,
        periods: int,
        confidence_interval: float,
        growth_rate_pct: float
    ) -> pd.DataFrame:
        """Statistical profile extrapolation when historical window is small."""
        last_dt = self.prepared_df["ds"].max()
        future_dts = [last_dt + pd.Timedelta(hours=i + 1) for i in range(periods)]

        # Calculate average demand profile per hour of day
        df = self.prepared_df.copy()
        df["hour"] = df["ds"].dt.hour
        hourly_stats = df.groupby("hour")["y"].agg(["mean", "std"]).fillna(0.0).to_dict("index")

        records = []
        # Historical slice
        for _, row in self.prepared_df.iterrows():
            records.append({
                "ds": row["ds"],
                "Demand": row["y"],
                "Demand_Lower": row["y"],
                "Demand_Upper": row["y"],
            })

        z = 1.96 if confidence_interval >= 0.95 else 1.645
        for fdt in future_dts:
            h = fdt.hour
            stat = hourly_stats.get(h, {"mean": self.prepared_df["y"].mean(), "std": 1.0})
            mean_val = stat["mean"] * (1.0 + growth_rate_pct / 100.0)
            std_val = max(stat["std"], 0.5)

            records.append({
                "ds": fdt,
                "Demand": max(0.0, mean_val),
                "Demand_Lower": max(0.0, mean_val - z * std_val),
                "Demand_Upper": max(0.0, mean_val + z * std_val),
            })

        return pd.DataFrame(records)


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "dispatch_input.xlsx")
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(base_dir), "dispatch_input.xlsx")

    print("=" * 60)
    print("CAPELCO DEMAND FORECASTING ENGINE")
    print("=" * 60)
    print(f"Loading demand profile from: {data_path}")
    
    demand_df = pd.read_excel(data_path, sheet_name="Demand")
    forecaster = DemandForecaster(demand_df)
    
    print(f"Historical records loaded: {len(demand_df)} hours")
    print("Running 24-hour ahead demand forecast (95% CI)...")
    
    forecast_df = forecaster.forecast(periods=24, confidence_interval=0.95)
    future_forecast = forecast_df[forecast_df["Is_Forecast"]].head(24)
    
    print("\n--- 24-Hour Ahead Forecast ---")
    print(future_forecast[["Date", "Hour", "Demand", "Demand_Lower", "Demand_Upper"]].to_string(index=False))
    print("=" * 60)
    print("Forecasting run complete.")
