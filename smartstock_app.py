import streamlit as st
import pandas as pd
import numpy as np
import anthropic
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartStock AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #F0F6FF; }
  [data-testid="block-container"] { padding-top: 1rem; }
  .metric-card {
    background: white; border-radius: 12px; padding: 1.1rem 1.3rem;
    border: 0.5px solid #D0DFF0; margin-bottom: 0;
  }
  .metric-card .val { font-size: 2rem; font-weight: 600; margin: 0; }
  .metric-card .lbl { font-size: 0.78rem; color: #607090; margin: 0 0 4px; }
  .badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600;
  }
  .badge-critical { background:#FCEBEB; color:#A32D2D; }
  .badge-medium   { background:#FAEEDA; color:#854F0B; }
  .badge-safe     { background:#E1F5EE; color:#0F6E56; }
  .header-bar {
    background: #0A1628; border-radius: 12px; padding: 1rem 1.5rem;
    margin-bottom: 1.2rem; display: flex; align-items: center; gap: 12px;
  }
  .insight-box {
    background: #F0FFF8; border: 1px solid #9FE1CB;
    border-radius: 10px; padding: 1rem 1.2rem; margin-top: 0.8rem;
  }
  .section-title { font-size: 0.85rem; font-weight: 600; color: #0F6E56;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.6rem; }
  div[data-testid="stTabs"] button { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Demo data ──────────────────────────────────────────────────────────────────
DEMO = [
    {"sku":"SKU-001","name":"Running Shoes (Size 9)","category":"Footwear",
     "current_stock":42,"avg_daily_sales":4.2,"lead_time_days":14},
    {"sku":"SKU-002","name":"Winter Jacket (M)","category":"Apparel",
     "current_stock":8,"avg_daily_sales":2.8,"lead_time_days":21},
    {"sku":"SKU-003","name":"Protein Powder 2kg","category":"Nutrition",
     "current_stock":130,"avg_daily_sales":5.1,"lead_time_days":7},
    {"sku":"SKU-004","name":"Yoga Mat Pro","category":"Fitness",
     "current_stock":19,"avg_daily_sales":1.5,"lead_time_days":10},
    {"sku":"SKU-005","name":"Bluetooth Headphones","category":"Electronics",
     "current_stock":3,"avg_daily_sales":2.1,"lead_time_days":18},
    {"sku":"SKU-006","name":"Vitamin D3 Capsules","category":"Health",
     "current_stock":210,"avg_daily_sales":8.4,"lead_time_days":5},
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_risk(row):
    days_left = row["current_stock"] / max(row["avg_daily_sales"], 0.01)
    if days_left < row["lead_time_days"]:
        return "critical"
    if days_left < row["lead_time_days"] * 1.5:
        return "medium"
    return "safe"

def days_left(row):
    return row["current_stock"] / max(row["avg_daily_sales"], 0.01)

def reorder_qty(row):
    safety = row["avg_daily_sales"] * row["lead_time_days"] * 0.5
    cycle  = row["avg_daily_sales"] * 30
    qty = cycle + safety - row["current_stock"]
    return max(0, round(qty))

def make_forecast(row, months=6):
    base = row["avg_daily_sales"] * 30
    dates, actuals, forecasts, uppers, lowers = [], [], [], [], []
    today = datetime.today().replace(day=1)
    for i in range(months):
        d = today + timedelta(days=30 * i)
        seasonal = 1 + 0.15 * np.sin((i + 2) * np.pi / 3)
        trend = 1 + i * 0.02
        fc = round(base * seasonal * trend)
        dates.append(d.strftime("%b %Y"))
        forecasts.append(fc)
        uppers.append(round(fc * 1.12))
        lowers.append(round(fc * 0.88))
        actuals.append(round(base * seasonal * (0.9 + np.random.uniform(0, 0.2))) if i < 2 else None)
    return pd.DataFrame({"month": dates, "actual": actuals, "forecast": forecasts,
                          "upper": uppers, "lower": lowers})

def risk_badge_html(risk):
    labels = {"critical":"🔴 Critical","medium":"🟡 Medium","safe":"🟢 Safe"}
    cls    = {"critical":"badge-critical","medium":"badge-medium","safe":"badge-safe"}
    return f'<span class="badge {cls[risk]}">{labels[risk]}</span>'

# ── Load / upload data ─────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(DEMO)
    st.session_state.df["risk"]        = st.session_state.df.apply(get_risk, axis=1)
    st.session_state.df["days_left"]   = st.session_state.df.apply(days_left, axis=1).round(1)
    st.session_state.df["reorder_qty"] = st.session_state.df.apply(reorder_qty, axis=1)

df = st.session_state.df

# ── Header ─────────────────────────────────────────────────────────────────────
col_logo, col_upload = st.columns([3, 1])
with col_logo:
    st.markdown("""
    <div class="header-bar">
      <span style="font-size:1.6rem">🧠</span>
      <span style="color:white; font-size:1.3rem; font-weight:700">SmartStock <span style="color:#00C9A7">AI</span></span>
      <span style="background:#0D3B6E; color:#8DA4C4; font-size:0.72rem;
        padding:3px 10px; border-radius:20px; margin-left:6px">
        AI-Powered Demand Forecasting · Supply Chain
      </span>
    </div>
    """, unsafe_allow_html=True)

with col_upload:
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        try:
            new_df = pd.read_csv(uploaded)
            # Flexible column mapping
            col_map = {}
            for c in new_df.columns:
                cl = c.lower().strip()
                if any(k in cl for k in ["sku","id","code"]):           col_map[c] = "sku"
                elif any(k in cl for k in ["name","product","item"]):   col_map[c] = "name"
                elif any(k in cl for k in ["cat","type"]):              col_map[c] = "category"
                elif any(k in cl for k in ["stock","inventory","qty","quantity"]): col_map[c] = "current_stock"
                elif any(k in cl for k in ["daily","demand","velocity","sales"]): col_map[c] = "avg_daily_sales"
                elif any(k in cl for k in ["lead"]):                    col_map[c] = "lead_time_days"
            new_df = new_df.rename(columns=col_map)
            for req in ["current_stock","avg_daily_sales","lead_time_days"]:
                if req not in new_df.columns:
                    new_df[req] = 10
            if "sku"      not in new_df.columns: new_df["sku"]      = [f"SKU-{i+1:03d}" for i in range(len(new_df))]
            if "name"     not in new_df.columns: new_df["name"]     = new_df["sku"]
            if "category" not in new_df.columns: new_df["category"] = "General"
            new_df["risk"]        = new_df.apply(get_risk, axis=1)
            new_df["days_left"]   = new_df.apply(days_left, axis=1).round(1)
            new_df["reorder_qty"] = new_df.apply(reorder_qty, axis=1)
            st.session_state.df = new_df
            df = new_df
            st.success(f"✓ Loaded {len(df)} SKUs from {uploaded.name}")
        except Exception as e:
            st.error(f"Could not parse file: {e}")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📦 Inventory", "🧠 AI Forecast", "🚚 Reorder Queue"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    critical_df = df[df["risk"] == "critical"]
    medium_df   = df[df["risk"] == "medium"]
    safe_df     = df[df["risk"] == "safe"]

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, len(df),             "Total SKUs",       "#185FA5"),
        (c2, len(critical_df),    "Critical Risk",    "#A32D2D"),
        (c3, len(medium_df),      "Needs Attention",  "#854F0B"),
        (c4, len(safe_df),        "Healthy Stock",    "#0F6E56"),
    ]
    for col, val, label, color in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <p class="lbl">{label}</p>
              <p class="val" style="color:{color}">{val}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    col_chart, col_critical = st.columns([1, 1])

    with col_chart:
        st.markdown('<p class="section-title">Risk distribution</p>', unsafe_allow_html=True)
        risk_counts = df["risk"].value_counts().reindex(["critical","medium","safe"], fill_value=0)
        fig_bar = go.Figure(go.Bar(
            x=["Critical","Medium","Safe"],
            y=risk_counts.values,
            marker_color=["#E24B4A","#EF9F27","#1D9E75"],
            text=risk_counts.values,
            textposition="outside",
        ))
        fig_bar.update_layout(
            height=240, margin=dict(t=20, b=20, l=10, r=10),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, yaxis=dict(showgrid=True, gridcolor="#E8EEF6"),
            xaxis=dict(showgrid=False),
        )
        fig_bar.update_traces(marker_line_width=0)
        st.plotly_chart(fig_bar, width="stretch")

    with col_critical:
        st.markdown('<p class="section-title">Critical items — immediate action needed</p>', unsafe_allow_html=True)
        if len(critical_df) == 0:
            st.success("✓ No critical items — all stock is healthy!")
        else:
            for _, row in critical_df.iterrows():
                with st.container(border=True):
                    cc1, cc2 = st.columns([3, 1])
                    with cc1:
                        st.markdown(f"**{row['name']}**")
                        st.caption(f"{int(row['current_stock'])} units · {row['days_left']:.0f} days left")
                    with cc2:
                        if st.button("Analyze →", key=f"dash_{row['sku']}"):
                            st.session_state["selected_sku"] = row["sku"]
                            st.session_state["run_ai"] = True

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — INVENTORY
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">All SKUs — click AI Analyze for demand forecast</p>', unsafe_allow_html=True)

    display_df = df[["sku","name","category","current_stock","avg_daily_sales","days_left","risk","reorder_qty"]].copy()
    display_df.columns = ["SKU","Product","Category","Stock","Daily Sales","Days Left","Risk","Reorder Qty"]

    # Color-code rows by risk
    def style_risk(val):
        if val == "critical": return "background-color:#FCEBEB; color:#A32D2D; font-weight:600"
        if val == "medium":   return "background-color:#FAEEDA; color:#854F0B; font-weight:600"
        return "background-color:#E1F5EE; color:#0F6E56; font-weight:600"

    styled = display_df.style.map(style_risk, subset=["Risk"])
    st.dataframe(styled, width="stretch", hide_index=True, height=280)

    st.markdown("---")
    st.markdown('<p class="section-title">Select SKU for AI-powered forecast</p>', unsafe_allow_html=True)

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        sku_options = df["sku"].tolist()
        sku_names   = {row["sku"]: f"{row['sku']} — {row['name']}" for _, row in df.iterrows()}
        default_idx = 0
        if "selected_sku" in st.session_state and st.session_state["selected_sku"] in sku_options:
            default_idx = sku_options.index(st.session_state["selected_sku"])
        chosen = st.selectbox("Select SKU", sku_options, index=default_idx,
                              format_func=lambda x: sku_names[x], label_visibility="collapsed")
        st.session_state["selected_sku"] = chosen
    with col_btn:
        if st.button("🧠 AI Analyze →", use_container_width=True):
            st.session_state["run_ai"] = True

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — AI FORECAST
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    if "selected_sku" not in st.session_state:
        st.info("👆 Go to the Inventory tab, select a SKU and click AI Analyze.")
    else:
        sku_id = st.session_state["selected_sku"]
        row = df[df["sku"] == sku_id].iloc[0]
        risk = row["risk"]

        # SKU header
        hc1, hc2, hc3, hc4 = st.columns([3, 1, 1, 1])
        with hc1:
            badge_colors = {"critical":"#A32D2D","medium":"#854F0B","safe":"#0F6E56"}
            badge_bg     = {"critical":"#FCEBEB","medium":"#FAEEDA","safe":"#E1F5EE"}
            st.markdown(f"""
            <div style="background:white; border:0.5px solid #D0DFF0; border-radius:10px; padding:0.8rem 1.1rem;">
              <div style="font-size:1rem; font-weight:600">{row['name']}</div>
              <div style="font-size:0.78rem; color:#607090">{row['sku']} · {row['category']}</div>
            </div>""", unsafe_allow_html=True)
        with hc2:
            st.metric("Stock", f"{int(row['current_stock'])} units")
        with hc3:
            st.metric("Days Left", f"{row['days_left']:.0f}d")
        with hc4:
            st.metric("Daily Sales", f"{row['avg_daily_sales']}/day")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # Forecast chart
        np.random.seed(int(hash(sku_id)) % 100)
        fc_df = make_forecast(row)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fc_df["month"], y=fc_df["upper"], name="Upper bound",
            line=dict(color="#9FE1CB", width=0), showlegend=False,
            hovertemplate="%{y} units<extra>Upper</extra>"
        ))
        fig.add_trace(go.Scatter(
            x=fc_df["month"], y=fc_df["lower"], name="Confidence band",
            line=dict(color="#9FE1CB", width=0),
            fill="tonexty", fillcolor="rgba(159,225,203,0.25)",
            hovertemplate="%{y} units<extra>Lower</extra>"
        ))
        fig.add_trace(go.Scatter(
            x=fc_df["month"], y=fc_df["forecast"], name="AI Forecast",
            line=dict(color="#1D9E75", width=2.5, dash="dash"),
            mode="lines+markers", marker=dict(size=6),
            hovertemplate="%{y} units<extra>Forecast</extra>"
        ))
        fig.add_trace(go.Scatter(
            x=fc_df["month"][:2], y=fc_df["actual"][:2], name="Actual",
            line=dict(color="#378ADD", width=2.5),
            mode="lines+markers", marker=dict(size=8),
            hovertemplate="%{y} units<extra>Actual</extra>"
        ))
        fig.update_layout(
            height=280, margin=dict(t=20, b=20, l=10, r=10),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#E8EEF6", title="Units / month"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

        # AI Insight
        st.markdown('<p class="section-title">🧠 Claude AI analyst insight</p>', unsafe_allow_html=True)

        run_now = st.session_state.get("run_ai", False)
        btn_col, _ = st.columns([1, 3])
        with btn_col:
            clicked = st.button("Generate AI Insight", type="primary", use_container_width=True)

        if run_now or clicked:
            st.session_state["run_ai"] = False
            prompt = f"""You are a supply chain AI analyst. Analyze this inventory item concisely.

SKU: {row['sku']} — {row['name']}
Category: {row['category']}
Current Stock: {int(row['current_stock'])} units
Avg Daily Sales: {row['avg_daily_sales']} units/day
Days of Stock Left: {row['days_left']:.1f} days
Supplier Lead Time: {row['lead_time_days']} days
Risk Level: {risk.upper()}
Suggested Reorder Qty: {int(row['reorder_qty'])} units

Give exactly 3 bullet points using • symbol:
1. Current situation assessment
2. Key risk or opportunity  
3. Specific recommended action with numbers

Be direct, data-driven, and actionable. Max 2 sentences each."""

            with st.spinner("Claude AI is analyzing this SKU..."):
                try:
                    client = anthropic.Anthropic()
                    message = client.messages.create(
                        model="claude-opus-4-6",
                        max_tokens=512,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    insight = message.content[0].text
                    st.session_state[f"insight_{sku_id}"] = insight
                except Exception as e:
                    st.session_state[f"insight_{sku_id}"] = (
                        f"• Stock status: {int(row['current_stock'])} units remain with {row['days_left']:.0f} days of cover "
                        f"against a {row['lead_time_days']}-day lead time — risk is {risk.upper()}.\n"
                        f"• Key risk: {'Stockout imminent — current stock will run out before replenishment arrives.' if risk == 'critical' else 'Buffer is thin; any demand spike could trigger a stockout.'}\n"
                        f"• Action: Raise a purchase order for {int(row['reorder_qty'])} units immediately "
                        f"to restore a 45-day safety buffer. (API error: {e})"
                    )

        if f"insight_{sku_id}" in st.session_state:
            insight_text = st.session_state[f"insight_{sku_id}"]
            st.markdown(f"""
            <div class="insight-box">
              <div style="font-size:0.78rem; color:#0F6E56; font-weight:700;
                letter-spacing:0.8px; margin-bottom:8px">AI ANALYST OUTPUT</div>
              <div style="font-size:0.9rem; line-height:1.75; white-space:pre-line;
                color:#1a2a1a">{insight_text}</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — REORDER QUEUE
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    reorder_df = df[df["reorder_qty"] > 0].copy()
    risk_order  = {"critical": 0, "medium": 1, "safe": 2}
    reorder_df["_sort"] = reorder_df["risk"].map(risk_order)
    reorder_df = reorder_df.sort_values("_sort").drop(columns="_sort")

    if len(reorder_df) == 0:
        st.success("✅ All inventory levels are optimal. No reorders needed.")
    else:
        st.markdown(f'<p class="section-title">{len(reorder_df)} SKUs need replenishment</p>', unsafe_allow_html=True)

        priority_map = {"critical": "P1 — Urgent", "medium": "P2 — Soon", "safe": "P3 — Plan"}

        for _, row in reorder_df.iterrows():
            risk     = row["risk"]
            priority = priority_map[risk]
            # Simulated unit cost
            unit_cost = 150 + (ord(row["sku"][-1]) * 7)
            est_cost  = row["reorder_qty"] * unit_cost

            badge_cols = {"critical":"badge-critical","medium":"badge-medium","safe":"badge-safe"}

            with st.container(border=True):
                r1, r2, r3, r4, r5, r6 = st.columns([2.5, 1, 1, 1.2, 1.5, 1.2])
                with r1:
                    st.markdown(f"**{row['name']}**")
                    st.caption(f"{row['sku']} · {row['category']}")
                with r2:
                    st.markdown(f"<span class='badge {badge_cols[risk]}'>{priority}</span>",
                                unsafe_allow_html=True)
                with r3:
                    st.metric("In Stock", f"{int(row['current_stock'])}")
                with r4:
                    st.metric("Order Qty", f"{int(row['reorder_qty'])} units",
                              delta=f"{'⚠️' if risk=='critical' else ''}",
                              delta_color="off")
                with r5:
                    st.metric("Est. Cost", f"₹{est_cost:,.0f}")
                with r6:
                    if st.button("🚚 Raise PO", key=f"po_{row['sku']}", use_container_width=True):
                        st.toast(f"PO raised for {int(row['reorder_qty'])} units of {row['name']}!", icon="✅")

        # Summary footer
        total_cost = sum(
            row["reorder_qty"] * (150 + ord(row["sku"][-1]) * 7)
            for _, row in reorder_df.iterrows()
        )
        st.markdown("---")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Total SKUs to Reorder", len(reorder_df))
        sc2.metric("Total Units to Order",  f"{int(reorder_df['reorder_qty'].sum()):,}")
        sc3.metric("Total Est. Spend",      f"₹{total_cost:,.0f}")

        st.caption("Reorder qty = (Avg Daily Sales × Lead Time × 1.5) + 30-day cycle stock − current stock")

        # Export
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        export_df = reorder_df[["sku","name","category","current_stock","reorder_qty","days_left"]].copy()
        export_df.columns = ["SKU","Product","Category","Current Stock","Reorder Qty","Days Left"]
        csv_bytes = export_df.to_csv(index=False).encode()
        st.download_button("⬇ Export Reorder List as CSV", data=csv_bytes,
                           file_name="smartstock_reorder_list.csv", mime="text/csv")
