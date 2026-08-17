"""
CAPELCO Power Dispatch Optimizer & Web Dashboard
Interactive Streamlit Application with MILP Optimization, Forecasting, and Visual Analytics.
"""

import io
import base64
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from optimizer_engine import DispatchOptimizer, OptimizationResult
from forecast_engine import DemandForecaster, PROPHET_AVAILABLE

# ---------------------------------------------------------
# Streamlit Page Configuration & Premium Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="CAPELCO Dispatch Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern Custom CSS
st.markdown("""
<style>
    :root {
        --capelco-bg:#0b0d13; --capelco-panel:#11151f; --capelco-border:#2a3040;
        --capelco-purple:#a855f7; --capelco-text:#f4f4f5; --capelco-muted:#a1a1aa;
    }
    .stApp { background:linear-gradient(135deg,#090b10 0%,#10131b 55%,#0b0d13 100%); color:var(--capelco-text); }
    [data-testid="stHeader"] { background:rgba(9,11,16,.96); }
    .block-container { padding-top:1rem; max-width:1500px; }
    .capelco-header {
        display:flex; align-items:center; gap:18px; padding:12px 24px 18px;
        margin:-1rem -1rem 1rem; border-bottom:1px solid #7c3aed;
        background:linear-gradient(90deg,#090b10,#11151f);
    }
    .capelco-logo { width:92px; height:92px; object-fit:contain; border-radius:50%; }
    .capelco-title { font-size:2.35rem; font-weight:800; letter-spacing:.04em; line-height:1; color:#fff; }
    .capelco-subtitle { color:#c4c4cc; font-size:1.05rem; margin-top:7px; }
    section[data-testid="stSidebar"] { background:linear-gradient(180deg,#11141d,#0c0f16); border-right:1px solid #2a3040; }
    section[data-testid="stSidebar"] .stMarkdown { color:#e4e4e7; }
    div[data-testid="stMetric"] { background:linear-gradient(145deg,#151923,#10131b); border:1px solid #2a3040; border-radius:10px; padding:16px; }
    div[data-testid="stMetric"] label { color:#a1a1aa !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#fafafa !important; }
    .stButton > button, .stDownloadButton > button {
        border:1px solid #9333ea !important; border-radius:7px !important;
        background:linear-gradient(90deg,#7c3aed,#a855f7) !important;
        color:white !important; font-weight:700 !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { background:linear-gradient(90deg,#8b5cf6,#c084fc) !important; }
    button[data-baseweb="tab"] { color:#d4d4d8 !important; background:#11151f !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color:#c084fc !important; border-bottom-color:#a855f7 !important; }
    [data-testid="stFileUploader"] { border:1px dashed #a855f7; border-radius:10px; background:rgba(168,85,247,.04); padding:6px; }
    [data-testid="stDataFrame"] { border:1px solid #2a3040; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Default Data Loader & Session State
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "dispatch_input.xlsx")
ALT_DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "dispatch_input.xlsx")

def load_default_data():
    path = DEFAULT_DATA_PATH if os.path.exists(DEFAULT_DATA_PATH) else ALT_DATA_PATH
    if os.path.exists(path):
        xls = pd.ExcelFile(path)
        return {
            "demand": pd.read_excel(xls, "Demand"),
            "generators": pd.read_excel(xls, "Generators"),
            "spot": pd.read_excel(xls, "SpotMarket"),
        }
    else:
        # Fallback synthetic demo data
        dates = ["2026-08-08"] * 24
        hours = list(range(24))
        demand = [39.4, 38.1, 36.8, 35.6, 34.9, 27.4, 28.4, 35.3, 38.4, 39.8, 41.7, 42.9,
                  40.9, 39.9, 40.1, 40.0, 42.8, 42.5, 46.5, 45.8, 47.8, 43.5, 42.5, 39.4]
        spot = [28.6, 26.2, 25.5, 24.4, 28.7, 26.5, 26.2, 27.2, 28.8, 30.5, 31.0, 31.9,
                30.4, 29.5, 29.8, 30.0, 31.5, 32.1, 36.8, 35.4, 37.9, 32.5, 31.2, 29.1]
        
        gen_data = {
            "Generator": ["GCGI", "FDC", "PEDC", "SPI", "DG1", "DG2"],
            "Capacity": [15.0, 8.0, 8.0, 8.0, 3.1, 3.8],
            "Price": [6.5406, 5.8010, 5.65815, 7.7807, 19.70, 19.70],
            "Baseload": [15, 0, 8, 4, 0, 0],
            "CUF": [1.0, 1.0, 0.8, 0.7, 1.0, 1.0],
            "Notes": [
                "Must run at 100% (GCGI: GCGI == capacity)",
                "Fixed output 4 MW",
                "PEDC: hourly 4 MW or 8 MW, daily total >=153.6 MWh",
                "SPI: hourly <=8 MW, daily >=144 MWh",
                "DG1 peaking (0-or-full)",
                "DG2 peaking (0-or-full)",
            ]
        }
        return {
            "demand": pd.DataFrame({"Date": dates, "Hour": hours, "Demand": demand}),
            "generators": pd.DataFrame(gen_data),
            "spot": pd.DataFrame({"Date": dates, "Hour": hours, "SpotPrice": spot}),
        }

def normalize_hourly_data(df):
    """Standardize Date and Hour columns for reliable merges."""
    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        ).dt.normalize()

    if "Hour" in df.columns:
        df["Hour"] = pd.to_numeric(
            df["Hour"],
            errors="coerce"
        ).astype("Int64")

    return df


if "data" not in st.session_state:
    st.session_state["data"] = load_default_data()

st.session_state["data"]["demand"] = normalize_hourly_data(
    st.session_state["data"]["demand"]
)

st.session_state["data"]["spot"] = normalize_hourly_data(
    st.session_state["data"]["spot"]
)

if "opt_result" not in st.session_state:
    st.session_state["opt_result"] = None

# Color Palette for Generators
GEN_COLORS = {
    "GCGI_MW": "#10b981",    # Emerald Green (Geothermal Must-run)
    "FDC_MW": "#06b6d4",     # Cyan (Fixed Coal)
    "PEDC_MW": "#3b82f6",    # Blue (Flexible Coal)
    "SPI_MW": "#8b5cf6",     # Purple (Base/Intermediate)
    "DG1_MW": "#f59e0b",     # Amber (Diesel Peaker 1)
    "DG2_MW": "#ea580c",     # Orange (Diesel Peaker 2)
    "Spot_MW": "#ef4444",    # Crimson Red (WESM Spot Market)
    "Unserved_MW": "#64748b",# Slate (Unserved)
}

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=64)
    st.title("CAPELCO Optimizer")
    st.caption("Power Supply & Dispatch Management System")
    st.markdown("---")

    st.subheader("⚙️ Quick Settings")
    spot_mult = st.slider("WESM Spot Price Multiplier", 0.5, 3.0, 1.0, 0.1, help="Scale all hourly spot prices")
    demand_mult = st.slider("Demand Scaling Factor", 0.7, 1.5, 1.0, 0.05, help="Scale hourly demand profile")

    st.markdown("---")
    st.subheader("🚨 Generator Outages")
    outages = {}
    for gen in ["GCGI", "FDC", "PEDC", "SPI", "DG1", "DG2"]:
        outages[gen] = st.checkbox(f"Outage: {gen}", value=False)

    st.markdown("---")
    if st.button("🚀 Run Full Optimization", type="primary", use_container_width=True):
        try:
            optimizer = DispatchOptimizer(
                st.session_state["data"]["demand"],
                st.session_state["data"]["generators"],
                st.session_state["data"]["spot"],
            )
            result = optimizer.solve(
                outages=outages,
                spot_price_multiplier=spot_mult,
                demand_multiplier=demand_mult,
            )
            st.session_state["opt_result"] = result
            st.success("Optimization solved successfully!")
        except Exception as e:
            st.error(f"Optimization error: {e}")

# =========================================================
# CAPELCO BRAND HEADER
# =========================================================
if os.path.exists(os.path.join(SCRIPT_DIR, "Logo-CAPELCO.png")):
    with open(os.path.join(SCRIPT_DIR, "Logo-CAPELCO.png"), "rb") as _logo_file:
        _logo_b64 = base64.b64encode(_logo_file.read()).decode("utf-8")
    _logo_html = f'<img class="capelco-logo" src="data:image/png;base64,{_logo_b64}" alt="CAPELCO Logo">'
else:
    _logo_html = '<div style="font-size:4rem;">⚡</div>'

st.markdown(
    f"""
    <div class="capelco-header">
        {_logo_html}
        <div>
            <div class="capelco-title">CAPELCO</div>
            <div class="capelco-subtitle">Power Dispatch Optimizer</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚡ CAPELCO")
    st.caption("Power Dispatch Optimizer")
    st.markdown("---")
    st.markdown("**NAVIGATION**")
    st.markdown("⚡ Dashboard")
    st.markdown("▣ Data Management")
    st.markdown("↗ Forecasting")
    st.markdown("◉ Dispatch Optimization")
    st.markdown("▥ Results & Analysis")
    st.markdown("▤ Reports")
    st.markdown("⚙ Settings")
    st.markdown("ⓘ About")
    st.markdown("---")
    st.caption("MILP Dispatch Engine")

# ---------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------
tab_data, tab_forecast, tab_dispatch, tab_scenario, tab_analytics, tab_reports = st.tabs([
    "📁 Data Management",
    "📈 Demand Forecasting",
    "⚡ Optimal Dispatch",
    "🔬 Scenario Analysis",
    "📊 Visual Analytics",
    "📑 Reports & Export",
])

# CAPELCO overview cards
_today_label = pd.Timestamp.now().strftime("%B %d, %Y")
_ov1, _ov2, _ov3, _ov4, _ov5 = st.columns(5)
with _ov1:
    st.metric("📅 Today", _today_label)
with _ov2:
    st.metric("🕐 Current Hour", pd.Timestamp.now().strftime("%H:%M"))
with _ov3:
    _d = st.session_state["data"]["demand"]
    _dv = float(_d["Demand"].sum()) if "Demand" in _d.columns and not _d.empty else 0.0
    st.metric("⚡ System Demand", f"{_dv:,.2f} MW")
with _ov4:
    _g = st.session_state["data"]["generators"]
    _cv = float(_g["Capacity"].sum()) if "Capacity" in _g.columns and not _g.empty else 0.0
    st.metric("🔌 Total Capacity", f"{_cv:,.2f} MW")
with _ov5:
    _s = st.session_state["data"]["spot"]
    _sv = float(_s["SpotPrice"].iloc[-1]) if "SpotPrice" in _s.columns and not _s.empty else 0.0
    st.metric("📊 Spot Price (Latest)", f"₱{_sv:,.2f}/MWh")

st.markdown("---")

# =========================================================
# TAB 1: DATA MANAGEMENT
# =========================================================
with tab_data:
    st.subheader("Data Sources & Configuration")
    st.markdown("Download the standard input template, fill the three worksheets, then upload the single Excel workbook.")

    st.info(
        "📌 Use one Excel workbook containing exactly three worksheets: "
        "**Demand**, **Generators**, and **SpotMarket**."
    )

    # The template download belongs ONLY to the Data Management tab.
    template_path = os.path.join(SCRIPT_DIR, "CAPELCO_Dispatch_Input_Template.xlsx")
    if os.path.exists(template_path):
        with open(template_path, "rb") as f:
            template_bytes = f.read()
        st.download_button(
            label="⬇️ Download Excel Input Template",
            data=template_bytes,
            file_name="CAPELCO_Dispatch_Input_Template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
            key="download_input_template",
            help="Downloads one workbook containing Demand, Generators, and SpotMarket worksheets."
        )

    st.markdown("### 📤 Upload Your Completed Excel Workbook")
    st.caption(
        "After downloading the template, fill in the three worksheets and upload "
        "the completed .xlsx file below."
    )
    workbook_file = st.file_uploader(
        "Upload CAPELCO Input Workbook",
        type=["xlsx"],
        key="u_workbook",
        help="The workbook must contain Demand, Generators, and SpotMarket worksheets."
    )

    if workbook_file:
        try:
            xls = pd.ExcelFile(workbook_file)
            required_sheets = ["Demand", "Generators", "SpotMarket"]
            missing_sheets = [s for s in required_sheets if s not in xls.sheet_names]

            if missing_sheets:
                st.error(
                    "Invalid workbook. Missing worksheet(s): "
                    + ", ".join(missing_sheets)
                    + ". Please download and use the CAPELCO Excel template."
                )
            else:
                uploaded_demand = pd.read_excel(xls, "Demand")
                uploaded_generators = pd.read_excel(xls, "Generators")
                uploaded_spot = pd.read_excel(xls, "SpotMarket")

                required_columns = {
                    "Demand": ["Date", "Hour", "Demand"],
                    "Generators": ["Generator", "Capacity", "Price", "Baseload", "CUF", "Notes"],
                    "SpotMarket": ["Date", "Hour", "SpotPrice"],
                }

                uploaded_tables = {
                    "Demand": uploaded_demand,
                    "Generators": uploaded_generators,
                    "SpotMarket": uploaded_spot,
                }

                column_errors = []
                for sheet, cols in required_columns.items():
                    missing_cols = [c for c in cols if c not in uploaded_tables[sheet].columns]
                    if missing_cols:
                        column_errors.append(
                            f"{sheet}: missing {', '.join(missing_cols)}"
                        )

                if column_errors:
                    st.error(
                        "Invalid workbook structure: "
                        + " | ".join(column_errors)
                        + ". Please use the downloadable template."
                    )
                elif uploaded_demand.empty or uploaded_generators.empty or uploaded_spot.empty:
                    st.error("All three worksheets must contain data.")
                else:
                    # Update all three datasets together only after validation succeeds.
                    st.session_state["data"]["demand"] = normalize_hourly_data(uploaded_demand)
                    st.session_state["data"]["generators"] = uploaded_generators.copy()
                    st.session_state["data"]["spot"] = normalize_hourly_data(uploaded_spot)

                    # Prevent an old optimization result from being displayed with new inputs.
                    st.session_state["opt_result"] = None

                    st.success(
                        f"Workbook loaded successfully: "
                        f"{len(uploaded_demand):,} demand rows, "
                        f"{len(uploaded_generators):,} generators, "
                        f"{len(uploaded_spot):,} spot-price rows."
                    )

                    with st.expander("📋 Workbook contents"):
                        st.write("**Demand**")
                        st.dataframe(uploaded_demand.head(10), use_container_width=True)
                        st.write("**Generators**")
                        st.dataframe(uploaded_generators, use_container_width=True)
                        st.write("**SpotMarket**")
                        st.dataframe(uploaded_spot.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read the Excel workbook: {e}")


    st.markdown("---")
    st.subheader("📊 Active Data Tables")

    tab_t1, tab_t2, tab_t3 = st.tabs(["Generators & Contracts", "Hourly Demand", "WESM Spot Market Prices"])
    with tab_t1:
        st.dataframe(st.session_state["data"]["generators"], use_container_width=True)
    with tab_t2:
        st.dataframe(st.session_state["data"]["demand"], use_container_width=True)
    with tab_t3:
        st.dataframe(st.session_state["data"]["spot"], use_container_width=True)

# =========================================================
# TAB 2: DEMAND FORECASTING
# =========================================================
with tab_forecast:
    st.subheader("Hourly Load Demand Forecasting")
    st.markdown(
        "Fit machine learning / statistical time-series models "
        "to historical load profiles to project 24-hour demand."
    )

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
            forecast_horizon = st.slider(
                "Forecast Horizon (Hours)",
                min_value=12,
                max_value=72,
                value=24,
                step=12
            )

    with col_f2:
            growth_pct = st.slider(
                "Expected Load Growth (+%)",
                min_value=-20.0,
                max_value=30.0,
                value=0.0,
                step=1.0
            )

    with col_f3:
            conf_int = st.selectbox(
                "Prediction Confidence Interval",
                [0.90, 0.95, 0.99],
                index=1
            )

    if st.button("🔮 Run Demand Forecast", type="primary"):
            try:
                forecaster = DemandForecaster(
                    st.session_state["data"]["demand"]
                )

                forecast_df = forecaster.forecast(
                    periods=forecast_horizon,
                    confidence_interval=conf_int,
                    growth_rate_pct=growth_pct,
                )

                st.session_state["forecast_df"] = forecast_df

                # Plot Forecast
                fig = go.Figure()

                # Upper confidence bound
                fig.add_trace(go.Scatter(
                    x=forecast_df["ds"],
                    y=forecast_df["Demand_Upper"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    name="Upper Bound",
                ))

                # Lower confidence bound
                fig.add_trace(go.Scatter(
                    x=forecast_df["ds"],
                    y=forecast_df["Demand_Lower"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(2, 132, 199, 0.15)",
                    name="Confidence Interval",
                ))

                # Forecast line
                fig.add_trace(go.Scatter(
                    x=forecast_df["ds"],
                    y=forecast_df["Demand"],
                    mode="lines+markers",
                    line=dict(
                        color="#0284c7",
                        width=3
                    ),
                    name="Forecasted Demand (MW)",
                ))

                fig.update_layout(
                    title=(
                        f"Hourly Load Forecast "
                        f"({forecast_horizon} Hours Ahead)"
                    ),
                    xaxis_title="Timestamp",
                    yaxis_title="Demand (MW)",
                    template="plotly_white",
                    hovermode="x unified",
                    height=450,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # Apply Forecast
                col_btn1, col_btn2 = st.columns([1, 4])

                with col_btn1:
                    if st.button("⚡ Apply Forecast to Optimizer"):
                        future_slice = forecast_df[
                            forecast_df["Is_Forecast"]
                        ].head(24)

                        if not future_slice.empty:
                            new_demand = future_slice[
                                ["Date", "Hour", "Demand"]
                            ].reset_index(drop=True)

                            st.session_state["data"]["demand"] = (
                                normalize_hourly_data(new_demand)
                            )

                            st.success(
                                "Demand profile updated with forecast!"
                            )
                        else:
                            st.warning(
                                "No forecasted hours are available to apply."
                            )

            except Exception as e:
                st.error(
                    f"Forecasting failed: {e}"
                )

# =========================================================
# TAB 3: OPTIMAL DISPATCH
# =========================================================
with tab_dispatch:
    if st.session_state["opt_result"] is None:
        # Run default solve if not yet run
        optimizer = DispatchOptimizer(
            st.session_state["data"]["demand"],
            st.session_state["data"]["generators"],
            st.session_state["data"]["spot"],
        )
        st.session_state["opt_result"] = optimizer.solve(outages=outages, spot_price_multiplier=spot_mult, demand_multiplier=demand_mult)

    res: OptimizationResult = st.session_state["opt_result"]

    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Procurement Cost</div>
            <div class="metric-value">₱{res.total_cost_kphp:,.1f}k</div>
            <div class="metric-sub">Daily Generation Cost</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Blended Generation Rate</div>
            <div class="metric-value">₱{res.blended_rate_php_kwh:.4f}</div>
            <div class="metric-sub">per kWh</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Energy Served</div>
            <div class="metric-value">{res.total_demand_mwh:,.1f} MWh</div>
            <div class="metric-sub">24-Hour Cumulative</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">WESM Spot Purchase</div>
            <div class="metric-value">{res.spot_energy_mwh:,.1f} MWh</div>
            <div class="metric-sub">{res.spot_share_pct:.1f}% of Demand</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Optimization Status</div>
            <div class="metric-value" style="color: #16a34a;">{res.status}</div>
            <div class="metric-sub">MILP Exact Solution</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Generation Stack Plotly Chart
    st.subheader("⚡ Hourly Generation Dispatch Stack vs Demand")
    
    df_disp = res.dispatch_df
    fig_stack = go.Figure()

    gen_order = [
        ("GCGI_MW", "GCGI (15 MW Must-Run)"),
        ("FDC_MW", "FDC (4 MW Fixed)"),
        ("PEDC_MW", "PEDC (4/8 MW Flexible)"),
        ("SPI_MW", "SPI (0-8 MW)"),
        ("DG1_MW", "DG1 Peaker (3.1 MW)"),
        ("DG2_MW", "DG2 Peaker (3.8 MW)"),
        ("Spot_MW", "WESM Spot Market"),
    ]

    for col, name in gen_order:
        fig_stack.add_trace(go.Bar(
            x=df_disp["Hour"],
            y=df_disp[col],
            name=name,
            marker_color=GEN_COLORS[col],
        ))

    # Add Demand Line
    fig_stack.add_trace(go.Scatter(
        x=df_disp["Hour"],
        y=df_disp["Demand_MW"],
        mode="lines+markers",
        name="Hourly Demand (MW)",
        line=dict(color="#0f172a", width=3, dash="dash"),
        marker=dict(size=7, color="#0f172a"),
    ))

    fig_stack.update_layout(
        barmode="stack",
        title="24-Hour Generation Dispatch by Resource Stack",
        xaxis=dict(title="Hour of Day (0 - 23)", tickmode="linear", tick0=0, dtick=1),
        yaxis=dict(title="Power Output (MW)"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=480,
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    # Detailed Generator Contribution Table
    st.subheader("📋 Generator Dispatch Summary & Contract Verification")
    st.dataframe(
        res.generator_metrics_df.style.format({
            "Energy Dispatched (MWh)": lambda x: f"{x:,.2f}" if pd.notnull(x) and isinstance(x, (int, float)) else str(x),
            "Energy Share (%)": lambda x: f"{x:.1f}%" if pd.notnull(x) and isinstance(x, (int, float)) else str(x),
            "Total Cost (kPHP)": lambda x: f"PHP {x:,.2f}k" if pd.notnull(x) and isinstance(x, (int, float)) else str(x),
            "Effective Rate (PHP/kWh)": lambda x: f"PHP {x:.4f}" if pd.notnull(x) and isinstance(x, (int, float)) else str(x),
            "Capacity Factor (CUF %)": lambda x: f"{x:.1f}%" if pd.notnull(x) and isinstance(x, (int, float)) else "-",
        }, na_rep="-"),
        use_container_width=True,
    )

    # Daily Generator Totals & Contract Constraint Verification
    st.subheader("✅ Daily Generator Totals & Constraint Verification")
    st.caption(
        "Daily totals are calculated directly from the solved dispatch. "
        "SPI nominations are whole MW. PEDC is 4/8 MW. DG1/DG2 are 0-or-full capacity."
    )

    # Build a dashboard-side verification table so it remains visible even if
    # an older optimizer_engine.py is accidentally still being used.
    dispatch_check = res.dispatch_df.copy()
    gen_specs = [
        ("GCGI", "GCGI_MW", "Must run at available capacity"),
        ("FDC", "FDC_MW", "Fixed output"),
        ("PEDC", "PEDC_MW", "Hourly 4/8 MW; daily minimum 153.6 MWh"),
        ("SPI", "SPI_MW", "Whole MW hourly; daily minimum 144 MWh"),
        ("DG1", "DG1_MW", "0 or full capacity (3.1 MW)"),
        ("DG2", "DG2_MW", "0 or full capacity (3.8 MW)"),
        ("Spot Market", "Spot_MW", "Continuous balancing purchase"),
    ]

    check_rows = []
    for date, day_df in dispatch_check.groupby("Date", sort=True):
        hours = len(day_df)
        for name, col, rule in gen_specs:
            total = float(day_df[col].sum()) if col in day_df.columns else 0.0
            minimum = 0.0
            maximum = None
            status = "INFO"

            if name == "GCGI":
                cap = float(day_df[col].max())
                minimum = maximum = cap * hours
                status = "PASS" if abs(total - minimum) < 1e-6 else "FAIL"
            elif name == "FDC":
                cap = float(day_df[col].max())
                minimum = maximum = cap * hours
                status = "PASS" if abs(total - minimum) < 1e-6 else "FAIL"
            elif name == "PEDC":
                minimum = min(153.6 * hours / 24.0, 8.0 * hours * 0.8)
                maximum = 8.0 * hours
                hourly_ok = day_df[col].apply(lambda x: abs(float(x) - 4) < 1e-6 or abs(float(x) - 8) < 1e-6).all()
                status = "PASS" if hourly_ok and total + 1e-6 >= minimum else "FAIL"
            elif name == "SPI":
                minimum = min(144.0 * hours / 24.0, 8.0 * hours * 0.75)
                maximum = 8.0 * hours
                integer_ok = day_df[col].apply(lambda x: abs(float(x) - round(float(x))) < 1e-6).all()
                within_cap = bool((day_df[col] >= -1e-6).all() and (day_df[col] <= 8.0 + 1e-6).all())
                status = "PASS" if integer_ok and within_cap and total + 1e-6 >= minimum else "FAIL"
            elif name == "DG1":
                cap = 3.1
                maximum = cap * hours
                hourly_ok = day_df[col].apply(lambda x: abs(float(x)) < 1e-6 or abs(float(x) - cap) < 1e-6).all()
                status = "PASS" if hourly_ok else "FAIL"
            elif name == "DG2":
                cap = 3.8
                maximum = cap * hours
                hourly_ok = day_df[col].apply(lambda x: abs(float(x)) < 1e-6 or abs(float(x) - cap) < 1e-6).all()
                status = "PASS" if hourly_ok else "FAIL"

            check_rows.append({
                "Date": date,
                "Generator": name,
                "Daily Total (MWh)": total,
                "Minimum (MWh)": minimum,
                "Maximum (MWh)": maximum,
                "Constraint": rule,
                "Status": status,
            })

    constraint_view = pd.DataFrame(check_rows)

    # Readable constraint table: explicit dark text + compact columns so the
    # table remains legible in Streamlit dark mode.
    def _constraint_row_style(row):
        status = row["Status"]
        if status == "PASS":
            bg = "#dcfce7"
            fg = "#166534"
        elif status == "FAIL":
            bg = "#fee2e2"
            fg = "#991b1b"
        else:
            bg = "#f3f4f6"
            fg = "#374151"
        return [
            f"background-color: {bg}; color: {fg}; font-weight: 500;"
            for _ in row
        ]

    constraint_style = (
        constraint_view.style
        .apply(_constraint_row_style, axis=1)
        .set_properties(**{
            "color": "#111827",
            "background-color": "#ffffff",
            "font-size": "14px",
            "padding": "7px 10px",
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#e5e7eb"),
                    ("color", "#111827"),
                    ("font-weight", "700"),
                    ("font-size", "13px"),
                ],
            },
        ])
        .format({
            "Daily Total (MWh)": "{:.0f}",
            "Minimum (MWh)": lambda x: f"{x:.1f}" if pd.notnull(x) else "—",
            "Maximum (MWh)": lambda x: f"{x:.1f}" if pd.notnull(x) else "—",
        }, na_rep="—")
    )

    st.dataframe(
        constraint_style,
        use_container_width=True,
        hide_index=True,
        height=330,
        column_config={
            "Date": st.column_config.TextColumn("Date", width="small"),
            "Generator": st.column_config.TextColumn("Generator", width="small"),
            "Daily Total (MWh)": st.column_config.NumberColumn("Daily Total (MWh)", width="small"),
            "Minimum (MWh)": st.column_config.NumberColumn("Minimum (MWh)", width="small"),
            "Maximum (MWh)": st.column_config.NumberColumn("Maximum (MWh)", width="small"),
            "Constraint": st.column_config.TextColumn("Constraint / Rule", width="large"),
            "Status": st.column_config.TextColumn("Status", width="small"),
        },
    )

    failed_constraints = constraint_view[constraint_view["Status"] == "FAIL"]
    if failed_constraints.empty:
        st.success("✅ All generator contractual and physical constraints are satisfied.")
    else:
        st.error(f"❌ {len(failed_constraints)} generator constraint check(s) failed.")

    # Detailed Hourly Schedule Expandable
    with st.expander("🔍 View Full 24-Hour Dispatch & Cost Schedule"):

    # Make sure both Date columns use the same datetime type
        res.dispatch_df["Date"] = pd.to_datetime(
            res.dispatch_df["Date"],
             errors="coerce"
        )

        res.hourly_cost_df["Date"] = pd.to_datetime(
            res.hourly_cost_df["Date"],
            errors="coerce"
        )

        merged_schedule = pd.merge(
            res.dispatch_df,
            res.hourly_cost_df[
                ["Date", "Hour", "Total_Cost_kPHP", "Blended_Rate_PHP_kWh"]
            ],
            on=["Date", "Hour"],
            how="left"
        )

        num_cols = [
            col for col in merged_schedule.columns
            if col not in ["Date", "Hour"]
        ]

        # Generator nominations are whole MW in the optimization engine.
        generator_mw_cols = [
            c for c in [
                "GCGI_MW", "FDC_MW", "PEDC_MW", "SPI_MW", "DG1_MW", "DG2_MW"
            ] if c in merged_schedule.columns
        ]

        formatters = {
            col: (
                lambda x: f"{x:.0f}"
                if isinstance(x, (int, float, np.integer, np.floating))
                else str(x)
            )
            for col in generator_mw_cols
        }

        for col in num_cols:
            if col not in formatters:
                formatters[col] = (
                    lambda x: f"{x:.2f}"
                    if isinstance(x, (int, float, np.integer, np.floating))
                    else str(x)
                )

        st.dataframe(
            merged_schedule.style.format(formatters),
            use_container_width=True
        )

# =========================================================
# TAB 4: SCENARIO ANALYSIS
# =========================================================
with tab_scenario:
    st.subheader("🔬 Multi-Scenario Stress Testing & What-If Analysis")
    st.markdown("Evaluate procurement cost risks across potential grid events, contract outages, and market volatility.")

    if st.button("🧪 Run Multi-Scenario Comparison", type="primary"):
        with st.spinner("Simulating stress test scenarios..."):
            optimizer = DispatchOptimizer(
                st.session_state["data"]["demand"],
                st.session_state["data"]["generators"],
                st.session_state["data"]["spot"],
            )

            scenarios = {
                "Base Case": {},
                "GCGI Outage (15 MW)": {"outages": {"GCGI": True}},
                "PEDC Outage (8 MW)": {"outages": {"PEDC": True}},
                "Spot Price Spike (+100%)": {"spot_price_multiplier": 2.0},
                "Spot Price Crisis (+200%)": {"spot_price_multiplier": 3.0},
                "Demand Surge (+15%)": {"demand_multiplier": 1.15},
                "High Coal Price (+20%)": {"price_multipliers": {"PEDC": 1.2, "FDC": 1.2, "SPI": 1.2}},
            }

            scenario_records = []
            base_cost = None

            for sc_name, sc_params in scenarios.items():
                sc_res = optimizer.solve(**sc_params)
                if base_cost is None:
                    base_cost = sc_res.total_cost_kphp

                delta_kphp = sc_res.total_cost_kphp - base_cost
                delta_pct = (delta_kphp / base_cost * 100) if base_cost > 0 else 0.0

                scenario_records.append({
                    "Scenario": sc_name,
                    "Status": sc_res.status,
                    "Total Cost (kPHP)": sc_res.total_cost_kphp,
                    "Blended Rate (PHP/kWh)": sc_res.blended_rate_php_kwh,
                    "Spot Purchase (MWh)": sc_res.spot_energy_mwh,
                    "Spot Share (%)": sc_res.spot_share_pct,
                    "Cost Impact (kPHP)": delta_kphp,
                    "Cost Impact (%)": delta_pct,
                })

            st.session_state["scenario_df"] = pd.DataFrame(scenario_records)

    if "scenario_df" in st.session_state and st.session_state["scenario_df"] is not None:
        sc_df = st.session_state["scenario_df"]
        st.dataframe(
            sc_df.style.format({
                "Total Cost (kPHP)": lambda x: f"PHP {x:,.1f}k" if isinstance(x, (int, float)) else str(x),
                "Blended Rate (PHP/kWh)": lambda x: f"PHP {x:.4f}" if isinstance(x, (int, float)) else str(x),
                "Spot Purchase (MWh)": lambda x: f"{x:,.1f}" if isinstance(x, (int, float)) else str(x),
                "Spot Share (%)": lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else str(x),
                "Cost Impact (kPHP)": lambda x: f"+PHP {x:,.1f}k" if (isinstance(x, (int, float)) and x > 0) else (f"PHP {x:,.1f}k" if isinstance(x, (int, float)) else str(x)),
                "Cost Impact (%)": lambda x: f"{x:+.1f}%" if isinstance(x, (int, float)) else str(x),
            }),
            use_container_width=True
        )

        col_sc1, col_sc2 = st.columns(2)
        with col_sc1:
            # Comparative Rate Chart
            fig_sc1 = px.bar(
                sc_df,
                x="Scenario",
                y="Blended Rate (PHP/kWh)",
                color="Blended Rate (PHP/kWh)",
                text_auto=".3f",
                title="Blended Generation Rate (PHP/kWh) Across Scenarios",
                color_continuous_scale="Reds",
            )
            fig_sc1.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_sc1, use_container_width=True)

        with col_sc2:
            # Cost Impact Delta Chart
            fig_sc2 = px.bar(
                sc_df,
                x="Scenario",
                y="Cost Impact (kPHP)",
                color="Cost Impact (kPHP)",
                text_auto=",.0f",
                title="Total Cost Impact vs Base Case (kPHP)",
                color_continuous_scale="Viridis",
            )
            fig_sc2.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_sc2, use_container_width=True)

# =========================================================
# TAB 5: VISUAL ANALYTICS
# =========================================================
with tab_analytics:
    st.subheader("📊 Visual Analytics & Economic Merit Order")

    if res is not None:
        col_an1, col_an2 = st.columns(2)
        with col_an1:
            # Generation Mix Donut
            gen_pie = res.generator_metrics_df[res.generator_metrics_df["Energy Dispatched (MWh)"] > 0]
            fig_pie = px.pie(
                gen_pie,
                values="Energy Dispatched (MWh)",
                names="Generator",
                title="Daily Generation Energy Share (MWh)",
                hole=0.45,
                color="Generator",
                color_discrete_map={
                    "GCGI": "#10b981", "FDC": "#06b6d4", "PEDC": "#3b82f6",
                    "SPI": "#8b5cf6", "DG1": "#f59e0b", "DG2": "#ea580c", "Spot Market": "#ef4444"
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_an2:
            # Cost Contribution Donut
            fig_c_pie = px.pie(
                gen_pie,
                values="Total Cost (kPHP)",
                names="Generator",
                title="Daily Procurement Cost Contribution (kPHP)",
                hole=0.45,
                color="Generator",
                color_discrete_map={
                    "GCGI": "#10b981", "FDC": "#06b6d4", "PEDC": "#3b82f6",
                    "SPI": "#8b5cf6", "DG1": "#f59e0b", "DG2": "#ea580c", "Spot Market": "#ef4444"
                }
            )
            fig_c_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_c_pie.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_c_pie, use_container_width=True)

    # Merge spot data aligned with hourly cost df
    hourly_cost_plot = normalize_hourly_data(res.hourly_cost_df)
    spot_plot = normalize_hourly_data(
        st.session_state["data"]["spot"]
    )

    spot_merged = pd.merge(
        hourly_cost_plot,
        spot_plot[["Date", "Hour", "SpotPrice"]],
        on=["Date", "Hour"],
        how="left"
    )

    spot_merged["EffectiveSpotPrice"] = spot_merged["SpotPrice"] * spot_mult

    # Hourly Spot Price vs Blended Rate Curve
    fig_rate = go.Figure()
    fig_rate.add_trace(go.Scatter(
        x=spot_merged["Hour"],
        y=spot_merged["EffectiveSpotPrice"],
        mode="lines+markers",
        name="WESM Spot Price (PHP/kWh)",
        line=dict(color="#ef4444", width=2, dash="dot"),
    ))
    fig_rate.add_trace(go.Scatter(
        x=spot_merged["Hour"],
        y=spot_merged["Blended_Rate_PHP_kWh"],
        mode="lines+markers",
        name="Blended Dispatch Rate (PHP/kWh)",
        line=dict(color="#0284c7", width=3),
    ))
    fig_rate.update_layout(
        title="Hourly WESM Spot Price vs CAPELCO Blended Rate",
        xaxis=dict(title="Hour of Day (0 - 23)", tickmode="linear", tick0=0, dtick=1),
        yaxis=dict(title="Rate (PHP/kWh)"),
        template="plotly_white",
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig_rate, use_container_width=True)

# =========================================================
# TAB 6: REPORTS & EXPORT
# =========================================================
with tab_reports:
    st.subheader("📑 Reports & Data Export")
    st.markdown("Generate board-ready executive summaries and download complete dispatch schedules.")

    if res is not None:
        col_exp1, col_exp2 = st.columns(2)
        
        # Excel Workbook Export
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
            res.dispatch_df.to_excel(writer, sheet_name="Hourly Dispatch", index=False)
            res.hourly_cost_df.to_excel(writer, sheet_name="Hourly Cost", index=False)
            res.generator_metrics_df.to_excel(writer, sheet_name="Generator Metrics", index=False)
            res.daily_summary_df.to_excel(writer, sheet_name="Daily Summary", index=False)
            res.constraint_check_df.to_excel(writer, sheet_name="Constraint Checks", index=False)

        with col_exp1:
            st.download_button(
                label="📥 Download Full Dispatch Schedule (.xlsx)",
                data=output_buffer.getvalue(),
                file_name="CAPELCO_Optimal_Dispatch_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )

        with col_exp2:
            csv_data = res.dispatch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download Hourly Dispatch (.csv)",
                data=csv_data,
                file_name="hourly_dispatch.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### Executive Summary Report Preview")
        constraint_failures = int((res.constraint_check_df["Status"] == "FAIL").sum())

        st.markdown(f"""
        #### **CAPELCO POWER SUPPLY DISPATCH REPORT**
        * **Operational Date**: `{res.dispatch_df['Date'].iloc[0]}`
        * **Total Energy Procured**: **{res.total_demand_mwh:,.2f} MWh**
        * **Total Procurement Cost**: **PHP {res.total_cost_kphp:,.2f} kPHP** (₱{res.total_cost_kphp*1000:,.2f})
        * **Weighted Blended Rate**: **PHP {res.blended_rate_php_kwh:.4f} / kWh**
        * **WESM Spot Market Share**: **{res.spot_energy_mwh:,.2f} MWh ({res.spot_share_pct:.2f}%)**
        * **Baseload & Contract Energy**: **{(res.total_demand_mwh - res.spot_energy_mwh):,.2f} MWh ({(100 - res.spot_share_pct):.2f}%)**
        * **Generator Constraint Checks**: **{"PASS" if constraint_failures == 0 else f"{constraint_failures} FAIL"}**
        """)

