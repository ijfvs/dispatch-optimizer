# ⚡ CAPELCO Power Dispatch Optimizer & Web Dashboard

An advanced Power Supply Portfolio Optimization & Analytics System for electric distribution utilities, featuring exact Mixed-Integer Linear Programming (MILP) optimization, Prophet time-series load forecasting, interactive Plotly visual analytics, and scenario stress testing.

---

## 🚀 Quick Start

### 1. Launch Interactive Web Dashboard
Run the one-click batch launcher or start with Streamlit:
```bash
# Option A: Run Batch Launcher
run_dashboard.bat

# Option B: Run Streamlit directly
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run CLI Optimizer Script
To run a headless optimization run directly from terminal:
```bash
python optimized_dispatch.py
```
This generates the full 24-hour dispatch schedule and exports it to `dispatch_output.xlsx`.

### 3. Run Automated Test Suite
```bash
python test_optimizer.py
```

---

## 📐 Mathematical Formulation & Contract Rules

The optimizer minimizes total procurement cost subject to physical grid and power supply agreement (PSA) constraints:

$$\min \sum_{d} \sum_{h=0}^{23} \left[ \sum_{g \in \mathcal{G}} P_{g, d, h} \cdot \text{Price}_g + P_{\text{Spot}, d, h} \cdot \text{Price}_{\text{Spot}, d, h} + P_{\text{Unserved}, d, h} \cdot \text{VOLL} \right]$$

### Operational & Contractual Rules

1. **GCGI (Geothermal Must-Run)**:
   $$P_{\text{GCGI}, d, h} = 15.0\text{ MW} \quad \forall h$$
2. **FDC (Fixed Output)**:
   $$P_{\text{FDC}, d, h} = 4.0\text{ MW} \quad \forall h$$
3. **PEDC (Discrete Flexible Thermal)**:
   $$P_{\text{PEDC}, d, h} = 4.0 + 4.0 \cdot u_{\text{PEDC}, d, h}, \quad u \in \{0, 1\}$$
   $$\sum_{h=0}^{23} P_{\text{PEDC}, d, h} \ge 153.6\text{ MWh/day} \quad (8.0\text{ MW} \times 24\text{h} \times 0.8\text{ CUF})$$
4. **SPI (Variable Contract)**:
   $$0 \le P_{\text{SPI}, d, h} \le 8.0\text{ MW}$$
   $$\sum_{h=0}^{23} P_{\text{SPI}, d, h} \ge 144.0\text{ MWh/day}$$
5. **DG1 & DG2 (Diesel Peaking Units)**:
   $$P_{\text{DG1}, d, h} = 3.1 \cdot u_{\text{DG1}, d, h}, \quad u \in \{0, 1\}$$
   $$P_{\text{DG2}, d, h} = 3.8 \cdot u_{\text{DG2}, d, h}, \quad u \in \{0, 1\}$$
6. **Hourly Power Balance**:
   $$\sum_{g \in \mathcal{G}} P_{g, d, h} + P_{\text{Spot}, d, h} + P_{\text{Unserved}, d, h} = \text{Demand}_{d, h} \quad \forall d, h$$

---

## 📁 Repository Structure

```
Optimizer/
├── app.py                   # Full Streamlit web application & interactive dashboard
├── optimizer_engine.py      # Core DispatchOptimizer MILP solver class
├── forecast_engine.py       # DemandForecaster (Prophet & statistical time-series)
├── optimized_dispatch.py    # Headless CLI optimization script
├── test_optimizer.py        # Complete unit & integration test suite
├── run_dashboard.bat        # One-click Windows batch launcher
├── README.md                # System documentation & technical guide
├── dispatch_output.xlsx     # Generated benchmark dispatch schedule
└── data/
    └── dispatch_input.xlsx  # Benchmark dataset (Demand, Generators, SpotMarket)
```
