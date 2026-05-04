# """
# TeleChurn AI — app.py (v6 · Final · Error-Free)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# pip install streamlit scikit-learn xgboost pandas numpy plotly
# Place PKL files in ./models/ and dataset.csv alongside app.py
# streamlit run app.py

# Bugs fixed vs v5:
#  ✅ st.stop() inside tabs blocks all later tabs — replaced with if/else
#  ✅ st.columns(0) crash when n_cols=0
#  ✅ Heatmap shows "nan" text — masked properly
#  ✅ SeniorCitizen (int) plotted as string on x-axis
#  ✅ select_dtypes deprecation warning fixed
#  ✅ All f-strings validated, no nested quote conflicts
#  ✅ Risk-summary safe for any combination of inputs
# """

# import pickle
# import warnings
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import streamlit as st
# from sklearn.preprocessing import LabelEncoder

# warnings.filterwarnings("ignore")
# #----------------------------------------------
# # ── FORCE DARK TABLE THEME (STREAMLIT CLOUD FIX) ──
# st.markdown("""
# <style>

# /* dataframe container */
# [data-testid="stDataFrame"] {
#     background-color: #0f172a;
#     border-radius: 10px;
# }

# /* table header */
# [data-testid="stDataFrame"] thead tr th {
#     background-color: #111827 !important;
#     color: #e5e7eb !important;
#     font-weight: 600;
# }

# /* table cells */
# [data-testid="stDataFrame"] tbody tr td {
#     background-color: #0f172a !important;
#     color: #e5e7eb !important;
# }

# /* row hover */
# [data-testid="stDataFrame"] tbody tr:hover td {
#     background-color: #1f2937 !important;
# }

# /* expander dataframe fix */
# [data-testid="stExpander"] [data-testid="stDataFrame"] {
#     background-color: #0f172a;
# }

# </style>
# """, unsafe_allow_html=True)
# #----------------------------------------------

# # ─────────────────────────────────────────────
# # PAGE CONFIG
# # ─────────────────────────────────────────────
# st.set_page_config(
#     page_title="TeleChurn AI",
#     page_icon="📡",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ─────────────────────────────────────────────
# # COLOR SYSTEM
# # ─────────────────────────────────────────────
# BG    = "#0F1117"
# CARD  = "#161B27"
# CARD2 = "#1C2333"
# BDR   = "#2A3347"
# BDR2  = "#374357"
# TEXT  = "#E8EDF5"
# SUB   = "#8B97B0"
# MUTED = "#4A5568"

# BLUE   = "#4F8EF7"
# INDIGO = "#7C6FF7"
# CYAN   = "#22D3EE"
# GREEN  = "#22C55E"
# AMBER  = "#F59E0B"
# RED    = "#F43F5E"
# PURPLE = "#A855F7"
# TEAL   = "#14B8A6"

# # ─────────────────────────────────────────────
# # CSS
# # ─────────────────────────────────────────────
# st.markdown(f"""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
# *,*::before,*::after{{box-sizing:border-box;margin:0;}}
# html,body,.stApp{{background:{BG}!important;color:{TEXT}!important;font-family:'Inter',sans-serif!important;}}
# #MainMenu,footer,header,[data-testid="stToolbar"]{{display:none!important;}}
# .block-container{{padding:0 2rem 5rem!important;max-width:1420px!important;}}

# [data-testid="stSidebar"]{{background:{CARD}!important;border-right:1px solid {BDR}!important;}}
# [data-testid="stSidebar"]>div:first-child{{padding-top:0!important;}}
# [data-testid="stSidebarNav"]{{display:none!important;}}

# .stTabs [data-baseweb="tab-list"]{{background:{CARD};border:1px solid {BDR};border-radius:12px;padding:4px;gap:2px;margin-bottom:24px;}}
# .stTabs [data-baseweb="tab"]{{background:transparent!important;color:{MUTED}!important;border-radius:9px!important;padding:9px 24px!important;font-family:'Space Grotesk',sans-serif!important;font-size:.84rem!important;font-weight:600!important;border:none!important;transition:all .18s!important;}}
# .stTabs [aria-selected="true"]{{background:{BLUE}!important;color:#fff!important;box-shadow:0 2px 12px {BLUE}30!important;}}

# div[data-baseweb="select"]>div{{background:{CARD2}!important;border:1px solid {BDR}!important;border-radius:8px!important;color:{TEXT}!important;}}
# div[data-baseweb="select"] *{{color:{TEXT}!important;}}
# div[data-baseweb="popover"]{{background:{CARD2}!important;border:1px solid {BDR2}!important;border-radius:10px!important;}}
# div[data-baseweb="popover"] li:hover{{background:{BDR}!important;}}
# .stNumberInput input,.stTextInput input,.stTextArea textarea{{background:{CARD2}!important;border:1px solid {BDR}!important;border-radius:8px!important;color:{TEXT}!important;font-size:.88rem!important;}}
# .stSlider>div>div>div>div{{background:{BLUE}!important;}}

# div.stButton>button{{background:{BLUE}!important;color:#fff!important;border:none!important;border-radius:10px!important;padding:14px 0!important;font-family:'Space Grotesk',sans-serif!important;font-weight:700!important;font-size:1rem!important;width:100%!important;letter-spacing:.3px!important;box-shadow:0 4px 20px {BLUE}40!important;transition:all .2s ease!important;}}
# div.stButton>button:hover{{background:#6BA3F9!important;box-shadow:0 8px 30px {BLUE}55!important;transform:translateY(-1px)!important;}}

# [data-testid="stSuccess"]{{background:#0A1F14!important;border:1px solid {GREEN}40!important;border-radius:10px!important;}}
# [data-testid="stError"]{{background:#1F0A11!important;border:1px solid {RED}40!important;border-radius:10px!important;}}
# [data-testid="stWarning"]{{background:#1A1305!important;border:1px solid {AMBER}40!important;border-radius:10px!important;}}
# [data-testid="stInfo"]{{background:#080F1F!important;border:1px solid {BLUE}40!important;border-radius:10px!important;}}
# [data-testid="stExpander"]{{background:{CARD}!important;border:1px solid {BDR}!important;border-radius:10px!important;}}
# details summary{{color:{TEXT}!important;}}

# ::-webkit-scrollbar{{width:4px;height:4px;}}
# ::-webkit-scrollbar-track{{background:{BG};}}
# ::-webkit-scrollbar-thumb{{background:{BDR2};border-radius:4px;}}
# hr{{border-color:{BDR}!important;margin:1.4rem 0!important;}}

# @keyframes pdot{{0%,100%{{opacity:1;}}50%{{opacity:.2;}}}}
# .pdot{{display:inline-block;width:7px;height:7px;border-radius:50%;background:{GREEN};animation:pdot 2s infinite;box-shadow:0 0 6px {GREEN};margin-right:6px;vertical-align:middle;}}
# @keyframes sup{{from{{opacity:0;transform:translateY(14px);}}to{{opacity:1;transform:translateY(0);}}}}
# .sup{{animation:sup .4s cubic-bezier(.22,1,.36,1) both;}}
# </style>
# """, unsafe_allow_html=True)


# # ─────────────────────────────────────────────
# # PLOTLY BASE LAYOUT
# # ─────────────────────────────────────────────
# def plotly_layout(title="", height=360):
#     return dict(
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         font=dict(family="Inter", color=SUB, size=11),
#         margin=dict(l=8, r=8, t=44, b=8),
#         height=height,
#         title=dict(
#             text=title,
#             font=dict(family="Space Grotesk", size=14, color=TEXT),
#             x=0, xanchor="left",
#         ),
#         xaxis=dict(gridcolor=BDR, linecolor=BDR, zerolinecolor=BDR),
#         yaxis=dict(gridcolor=BDR, linecolor=BDR, zerolinecolor=BDR),
#         legend=dict(
#             bgcolor="rgba(22,27,39,0.85)", bordercolor=BDR,
#             borderwidth=1, font=dict(size=11, color=SUB),
#         ),
#         colorway=[BLUE, CYAN, GREEN, AMBER, RED, PURPLE, TEAL],
#     )


# # ─────────────────────────────────────────────
# # HTML HELPERS  (no nested quote conflicts)
# # ─────────────────────────────────────────────
# def md(html):
#     st.markdown(html, unsafe_allow_html=True)

# def divider():
#     md("<hr/>")

# def sec_hdr(icon, title, sub=""):
#     sub_part = (f'<div style="font-size:.77rem;color:{SUB};margin-top:3px;">{sub}</div>'
#                 if sub else "")
#     md(f"""
#     <div style="display:flex;align-items:flex-start;gap:12px;margin:30px 0 16px;">
#       <div style="width:37px;height:37px;flex-shrink:0;border-radius:10px;
#         background:linear-gradient(135deg,{BLUE}22,{INDIGO}22);
#         border:1px solid {BLUE}35;display:flex;align-items:center;
#         justify-content:center;font-size:1.05rem;">{icon}</div>
#       <div>
#         <div style="font-family:'Space Grotesk',sans-serif;font-size:.98rem;
#           font-weight:700;color:{TEXT};">{title}</div>
#         {sub_part}
#       </div>
#     </div>""")

# def kpi(icon, label, value, accent, note=""):
#     note_part = (f'<div style="font-size:.69rem;color:{SUB};margin-top:3px;">{note}</div>'
#                  if note else "")
#     return f"""<div style="background:{CARD};border:1px solid {BDR};border-radius:12px;
#       padding:17px;position:relative;overflow:hidden;">
#       <div style="position:absolute;top:0;right:0;width:70px;height:70px;
#         background:radial-gradient(circle,{accent}18,transparent 70%);pointer-events:none;"></div>
#       <div style="font-size:1.2rem;margin-bottom:7px;">{icon}</div>
#       <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;
#         font-weight:700;color:{accent};line-height:1;margin-bottom:4px;">{value}</div>
#       <div style="font-size:.63rem;color:{MUTED};text-transform:uppercase;
#         letter-spacing:1px;font-weight:600;">{label}</div>
#       {note_part}</div>"""

# def badge(text, color):
#     return (f'<span style="background:{color}18;border:1px solid {color}35;'
#             f'color:{color};border-radius:6px;padding:3px 12px;font-size:.72rem;'
#             f'font-family:JetBrains Mono,monospace;font-weight:500;">{text}</span>')

# def insight_card(icon, title, desc, accent):
#     return f"""<div style="background:{CARD};border:1px solid {accent}30;
#       border-left:3px solid {accent};border-radius:10px;padding:18px;height:100%;">
#       <div style="font-size:1.2rem;margin-bottom:9px;">{icon}</div>
#       <div style="font-family:'Space Grotesk',sans-serif;font-size:.87rem;
#         font-weight:700;color:{accent};margin-bottom:6px;">{title}</div>
#       <div style="font-size:.79rem;color:{SUB};line-height:1.7;">{desc}</div>
#     </div>"""

# def signal_row(label, rate, accent):
#     return f"""<div style="display:flex;justify-content:space-between;align-items:center;
#       background:{CARD};border:1px solid {BDR};border-left:3px solid {accent};
#       border-radius:8px;padding:9px 14px;margin-bottom:7px;">
#       <span style="font-size:.82rem;color:{SUB};">{label}</span>
#       <span style="font-size:.82rem;font-weight:700;color:{accent};">{rate}</span>
#     </div>"""

# def risk_card_html(col_c, col_lbl, title_f, desc_f, note_f):
#     """Build risk factor card HTML without nested f-string quote issues."""
#     return (
#         f'<div style="background:{col_c}0C;border:1px solid {col_c}30;'
#         f'border-left:3px solid {col_c};border-radius:10px;'
#         f'padding:14px 16px;margin-bottom:10px;">'
#         f'<div style="font-size:.63rem;font-weight:700;color:{col_c};'
#         f'text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">'
#         f'{col_lbl}</div>'
#         f'<div style="font-size:.82rem;font-weight:600;color:{TEXT};margin-bottom:4px;">'
#         f'{title_f}</div>'
#         f'<div style="font-size:.76rem;color:{SUB};margin-bottom:3px;">{desc_f}</div>'
#         f'<div style="font-size:.7rem;color:{MUTED};font-style:italic;">{note_f}</div>'
#         f'</div>'
#     )


# # ─────────────────────────────────────────────
# # DATA LOADING
# # ─────────────────────────────────────────────
# MODELS_DIR = Path("models")

# @st.cache_resource(show_spinner="Loading model…")
# def load_artifacts():
#     needed = {
#         "model":  MODELS_DIR / "model.pkl",
#         "scaler": MODELS_DIR / "scaler.pkl",
#         "enc":    MODELS_DIR / "label_encoders.pkl",
#         "cols":   MODELS_DIR / "feature_columns.pkl",
#         "info":   MODELS_DIR / "model_info.pkl",
#     }
#     miss = [p.name for p in needed.values() if not p.exists()]
#     if miss:
#         return None, miss
#     out = {}
#     for k, p in needed.items():
#         with open(p, "rb") as f:
#             out[k] = pickle.load(f)
#     return out, []

# @st.cache_data(show_spinner="Loading dataset…")
# def load_data():
#     p = Path("dataset.csv")
#     if not p.exists():
#         return None
#     df = pd.read_csv(p)
#     df["TotalCharges"] = pd.to_numeric(
#         df["TotalCharges"].astype(str).str.strip().replace({"": "0", " ": "0"}),
#         errors="coerce",
#     ).fillna(0.0)
#     df["ChurnBin"] = (df["Churn"] == "Yes").astype(int)
#     # bins include 0 explicitly so no NaN in TenureGroup
#     df["TenureGroup"] = pd.cut(
#         df["tenure"],
#         bins=[-1, 12, 24, 48, 72],
#         labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"],
#     )
#     return df

# arts, miss_files = load_artifacts()
# df   = load_data()
# ok_m = arts is not None
# ok_d = df is not None

# m_info    = arts["info"] if ok_m else {}
# model_lbl = m_info.get("model_name", "XGBoost") if ok_m else "XGBoost"
# threshold = float(m_info.get("threshold", 0.4828)) if ok_m else 0.4828

# # Verified final metrics (from notebook output)
# ACC  = m_info.get("accuracy",  0.7757) if ok_m else 0.7757
# PREC = m_info.get("precision", 0.5612) if ok_m else 0.5612
# REC  = m_info.get("recall",    0.7112) if ok_m else 0.7112
# F1   = m_info.get("f1_score",  0.6274) if ok_m else 0.6274
# AUC  = m_info.get("roc_auc",   0.8434) if ok_m else 0.8434


# # ─────────────────────────────────────────────
# # SIDEBAR
# # ─────────────────────────────────────────────
# with st.sidebar:
#     md(f"""<div style="padding:20px 16px 0;">
#       <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
#         <div style="width:33px;height:33px;border-radius:9px;background:{BLUE};
#           display:flex;align-items:center;justify-content:center;
#           font-size:.95rem;flex-shrink:0;">📡</div>
#         <div>
#           <div style="font-family:'Space Grotesk',sans-serif;font-size:.93rem;
#             font-weight:700;color:{TEXT};">TeleChurn AI</div>
#           <div style="font-size:.62rem;color:{MUTED};letter-spacing:.5px;">
#             ANALYTICS PLATFORM</div>
#         </div>
#       </div>
#       <div style="height:1px;background:{BDR};margin-bottom:16px;"></div>
#     </div>""")

#     # Status
#     mc = GREEN if ok_m else RED
#     dc = GREEN if ok_d else AMBER
#     md(f"""<div style="padding:0 16px;">
#       <div style="font-size:.62rem;font-weight:700;color:{MUTED};letter-spacing:1.1px;
#         text-transform:uppercase;margin-bottom:8px;">System Status</div>
#       <div style="background:{CARD2};border:1px solid {mc}30;border-radius:8px;
#         padding:9px 12px;margin-bottom:5px;display:flex;align-items:center;gap:7px;">
#         <span class="pdot" style="background:{mc};box-shadow:0 0 6px {mc};"></span>
#         <span style="font-size:.77rem;color:{mc};font-weight:600;">
#           {"Model Ready" if ok_m else "Model Missing — add models/"}</span>
#       </div>
#       <div style="background:{CARD2};border:1px solid {dc}30;border-radius:8px;
#         padding:9px 12px;margin-bottom:16px;display:flex;align-items:center;gap:7px;">
#         <span style="width:7px;height:7px;border-radius:50%;
#           background:{dc};flex-shrink:0;display:inline-block;"></span>
#         <span style="font-size:.77rem;color:{dc};font-weight:600;">
#           {"Dataset Loaded" if ok_d else "No dataset.csv found"}</span>
#       </div>
#     </div>""")

#     # Model info
#     if ok_m:
#         md(f"""<div style="padding:0 16px;">
#           <div style="font-size:.62rem;font-weight:700;color:{MUTED};letter-spacing:1.1px;
#             text-transform:uppercase;margin-bottom:8px;">Best Model</div>
#           <div style="background:{CARD2};border:1px solid {BDR};border-radius:8px;
#             padding:12px;margin-bottom:16px;">
#             <div style="display:flex;justify-content:space-between;margin-bottom:7px;">
#               <span style="font-size:.71rem;color:{MUTED};">Algorithm</span>
#               <span style="font-family:'JetBrains Mono',monospace;font-size:.79rem;
#                 color:{BLUE};font-weight:600;">{model_lbl}</span>
#             </div>
#             <div style="height:1px;background:{BDR};margin:7px 0;"></div>
#             <div style="display:flex;justify-content:space-between;">
#               <span style="font-size:.71rem;color:{MUTED};">Threshold</span>
#               <span style="font-family:'JetBrains Mono',monospace;font-size:.79rem;
#                 color:{AMBER};font-weight:600;">{threshold:.4f}</span>
#             </div>
#           </div>
#         </div>""")

#     # Stats
#     _n  = f"{len(df):,}" if ok_d else "7,043"
#     _ch = f"{df['ChurnBin'].sum():,}" if ok_d else "1,869"
#     _cr = f"{df['ChurnBin'].mean()*100:.1f}%" if ok_d else "26.5%"
#     stats = [
#         ("Customers", _n, CYAN),
#         ("Churned", _ch, RED),
#         ("Churn Rate", _cr, AMBER),
#         ("Best AUC-ROC", f"{AUC:.4f}", GREEN),
#         ("Best F1", f"{F1:.4f}", TEAL),
#     ]
#     md(f"""<div style="padding:0 16px;">
#       <div style="font-size:.62rem;font-weight:700;color:{MUTED};letter-spacing:1.1px;
#         text-transform:uppercase;margin-bottom:8px;">Dataset Stats</div>""")
#     for lb, vl, ac in stats:
#         md(f"""<div style="background:{CARD2};border-radius:7px;padding:8px 10px;
#           display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
#           <span style="font-size:.73rem;color:{MUTED};">{lb}</span>
#           <span style="font-family:'JetBrains Mono',monospace;font-size:.77rem;
#             color:{ac};font-weight:600;">{vl}</span>
#         </div>""")
#     md("</div>")

#     md(f"""<div style="padding:14px 16px;">
#       <div style="height:1px;background:{BDR};margin-bottom:12px;"></div>
#       <div style="font-size:.69rem;color:{MUTED};line-height:1.8;text-align:center;">
#         XGBoost · SMOTE · GridSearchCV<br>Scikit-learn · Plotly · Streamlit
#       </div>
#     </div>""")


# # ─────────────────────────────────────────────
# # HERO
# # ─────────────────────────────────────────────
# _nc = f"{len(df):,}" if ok_d else "7,043"
# badges_html = "  ".join([
#     badge("SMOTE Balanced", GREEN),
#     badge("3 Models Tuned", INDIGO),
#     badge(_nc + " Customers", CYAN),
#     badge(f"Threshold {threshold:.4f}", AMBER),
#     badge("AUC 0.8434", BLUE),
# ])
# md(f"""<div style="padding:36px 0 22px;">
#   <div style="margin-bottom:14px;">
#     <span style="display:inline-flex;align-items:center;gap:5px;
#       background:{CARD};border:1px solid {BDR};border-radius:50px;
#       padding:4px 14px;font-size:.69rem;font-family:'JetBrains Mono',monospace;color:{MUTED};">
#       <span class="pdot"></span>LIVE ANALYTICS ENGINE
#     </span>
#   </div>
#   <h1 style="font-family:'Space Grotesk',sans-serif;font-size:2.4rem;
#     font-weight:700;color:{TEXT};line-height:1.12;margin-bottom:8px;">
#     Telco Customer
#     <span style="color:{BLUE};">Churn Intelligence</span>
#   </h1>
#   <p style="font-size:.9rem;color:{SUB};line-height:1.7;max-width:500px;margin-bottom:18px;">
#     AI-powered churn prediction · XGBoost + SMOTE + GridSearchCV · Optimal threshold calibration
#   </p>
#   <div style="display:flex;gap:8px;flex-wrap:wrap;">{badges_html}</div>
# </div>""")

# # KPI strip
# _nr  = f"{len(df):,}" if ok_d else "7,043"
# _nch = f"{df['ChurnBin'].sum():,}" if ok_d else "1,869"
# kc   = st.columns(5, gap="small")
# for col, (ic, lb, vl, ac, nt) in zip(kc, [
#     ("📡", "Model",     "Ready" if ok_m else "Missing", BLUE,  model_lbl if ok_m else "Add models/"),
#     ("🗄️", "Dataset",  "Loaded" if ok_d else "Optional", CYAN, f"{_nr} rows" if ok_d else "Add dataset.csv"),
#     ("👥", "Customers", _nr,                             GREEN, "total records"),
#     ("📉", "Churned",   _nch,                            RED,   "26.5% of total"),
#     ("🎯", "Threshold", f"{threshold:.4f}",              AMBER, "optimal F1 cut-off"),
# ]):
#     col.markdown(kpi(ic, lb, vl, ac, nt), unsafe_allow_html=True)

# md("<br>")

# # ─────────────────────────────────────────────
# # TABS
# # ─────────────────────────────────────────────
# TAB1, TAB2, TAB3 = st.tabs([
#     "🔮  Predict Churn",
#     "📊  EDA Dashboard",
#     "🤖  Model Insights",
# ])


# # ═══════════════════════════════════════════
# # TAB 1 — PREDICT
# # ═══════════════════════════════════════════
# with TAB1:
#     if not ok_m:
#         st.error(
#             f"**Model files not found.**  Missing: `{'`, `'.join(miss_files)}`  \n"
#             "Run the notebook end-to-end to generate PKL files, "
#             "then place them in a `models/` subfolder next to app.py."
#         )
#         md(f"""<div style="background:{CARD};border:1px dashed {BDR2};border-radius:14px;
#           padding:48px 24px;text-align:center;margin-top:12px;">
#           <div style="font-size:2.5rem;margin-bottom:12px;">📁</div>
#           <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;
#             font-weight:700;color:{TEXT};margin-bottom:8px;">Models Not Found</div>
#           <div style="font-size:.84rem;color:{SUB};line-height:1.7;">
#             Required: model.pkl · scaler.pkl · label_encoders.pkl
#             · feature_columns.pkl · model_info.pkl
#           </div>
#         </div>""")
#     else:
#         sec_hdr("👤", "Customer Profile",
#                 "Fill in all fields below, then click Run Prediction")

#         # CA, CB, CC = st.columns(3, gap="large")

#         # with CA:
#         #     md(f'<div style="font-size:.68rem;font-weight:700;color:{CYAN};'
#         #        f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#         #        f'Demographics</div>')
#         #     gender     = st.selectbox("Gender",         ["Male", "Female"])
#         #     senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
#         #     partner    = st.selectbox("Partner",        ["No", "Yes"])
#         #     dependents = st.selectbox("Dependents",     ["No", "Yes"])
#         #     tenure     = st.slider("Tenure (months)", 0, 72, 12)

#         # with CB:
#         #     md(f'<div style="font-size:.68rem;font-weight:700;color:{PURPLE};'
#         #        f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#         #        f'Services</div>')
#         #     phone_svc = st.selectbox("Phone Service", ["Yes", "No"])
#         #     if phone_svc == "No":
#         #         multi_lines = "No phone service"
#         #         st.selectbox("Multiple Lines", ["No phone service"], disabled=True)
#         #     else:
#         #         multi_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

#         #     internet_svc = st.selectbox("Internet Service",
#         #                                 ["DSL", "Fiber optic", "No"])
#         #     _io  = ["No internet service"] if internet_svc == "No" else ["No", "Yes"]
#         #     _dis = internet_svc == "No"

#         #     online_sec   = st.selectbox("Online Security",   _io, disabled=_dis)
#         #     online_bkp   = st.selectbox("Online Backup",     _io, disabled=_dis)
#         #     dev_protect  = st.selectbox("Device Protection", _io, disabled=_dis)
#         #     tech_support = st.selectbox("Tech Support",      _io, disabled=_dis)
#         #     stream_tv    = st.selectbox("Streaming TV",      _io, disabled=_dis)
#         #     stream_mov   = st.selectbox("Streaming Movies",  _io, disabled=_dis)

#         # with CC:
#         #     md(f'<div style="font-size:.68rem;font-weight:700;color:{TEAL};'
#         #        f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#         #        f'Billing &amp; Contract</div>')
#         #     contract  = st.selectbox("Contract Type",
#         #                              ["Month-to-month", "One year", "Two year"])
#         #     paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
#         #     payment   = st.selectbox("Payment Method", [
#         #         "Electronic check", "Mailed check",
#         #         "Bank transfer (automatic)", "Credit card (automatic)",
#         #     ])
#         #     monthly  = st.number_input("Monthly Charges ($)",
#         #                                18.0, 120.0, 65.0, 0.5, format="%.2f")
#         #     def_tot  = min(float(monthly) * max(int(tenure), 1), 8684.8)
#         #     total    = st.number_input("Total Charges ($)",
#         #                                0.0, 8684.8, def_tot, 10.0, format="%.2f")

#         #==================================
#         CA, CB, CC = st.columns(3, gap="large")

#         with CA:
#             md(f'<div style="font-size:.68rem;font-weight:700;color:{CYAN};'
#                f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#                f'Demographics</div>')
#             gender     = st.selectbox("Gender",         ["Male", "Female"], index=0) # Default: Male
#             senior     = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)      # Default: No (0)
#             partner    = st.selectbox("Partner",        ["No", "Yes"], index=1)      # Default: Yes
#             dependents = st.selectbox("Dependents",     ["No", "Yes"], index=0)      # Default: No
#             tenure     = st.slider("Tenure (months)", 0, 72, 12)                     # Default: 12

#         with CB:
#             md(f'<div style="font-size:.68rem;font-weight:700;color:{PURPLE};'
#                f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#                f'Services</div>')
#             phone_svc = st.selectbox("Phone Service", ["No", "Yes"], index=1)        # Default: Yes
#             if phone_svc == "No":
#                 multi_lines = "No phone service"
#                 st.selectbox("Multiple Lines", ["No phone service"], disabled=True)
#             else:
#                 multi_lines = st.selectbox("Multiple Lines", ["No", "Yes"], index=0) # Default: No

#             internet_svc = st.selectbox("Internet Service",
#                                         ["DSL", "Fiber optic", "No"], index=1)       # Default: Fiber optic
#             _io  = ["No internet service"] if internet_svc == "No" else ["No", "Yes"]
#             _dis = internet_svc == "No"

#             online_sec   = st.selectbox("Online Security",   _io, index=0, disabled=_dis) # Default: No
#             online_bkp   = st.selectbox("Online Backup",     _io, index=0, disabled=_dis) # Default: No
#             dev_protect  = st.selectbox("Device Protection", _io, index=0, disabled=_dis) # Default: No
#             tech_support = st.selectbox("Tech Support",      _io, index=0, disabled=_dis) # Default: No
#             stream_tv    = st.selectbox("Streaming TV",      _io, index=1, disabled=_dis) # Default: Yes
#             stream_mov   = st.selectbox("Streaming Movies",  _io, index=1, disabled=_dis) # Default: Yes

#         with CC:
#             md(f'<div style="font-size:.68rem;font-weight:700;color:{TEAL};'
#                f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#                f'Billing &amp; Contract</div>')
#             contract  = st.selectbox("Contract Type",
#                                      ["Month-to-month", "One year", "Two year"], index=0) # Default: Month-to-month
#             paperless = st.selectbox("Paperless Billing", ["No", "Yes"], index=1)         # Default: Yes
#             payment   = st.selectbox("Payment Method", [
#                 "Bank transfer (automatic)", "Credit card (automatic)", 
#                 "Electronic check", "Mailed check"
#             ], index=2)                                                                   # Default: Electronic check
#             monthly  = st.number_input("Monthly Charges ($)",
#                                        18.0, 120.0, 70.35, 0.5, format="%.2f")            # Default: 70.35
#             total    = st.number_input("Total Charges ($)",
#                                        0.0, 8684.8, 843.0, 10.0, format="%.2f")           # Default: 843.0
        
#         md("<br>")
#         b1, b2, b3 = st.columns([1, 3, 1])
#         with b2:
#             clicked = st.button("🔮  RUN CHURN PREDICTION", use_container_width=True)

#         # ── Inference ──────────────────────────────────────────────
#         if clicked:
#             raw = {
#                 "gender":           gender,
#                 "SeniorCitizen":    1 if senior == "Yes" else 0,
#                 "Partner":          partner,
#                 "Dependents":       dependents,
#                 "tenure":           int(tenure),
#                 "PhoneService":     phone_svc,
#                 "MultipleLines":    multi_lines,
#                 "InternetService":  internet_svc,
#                 "OnlineSecurity":   online_sec   if internet_svc != "No" else "No internet service",
#                 "OnlineBackup":     online_bkp   if internet_svc != "No" else "No internet service",
#                 "DeviceProtection": dev_protect  if internet_svc != "No" else "No internet service",
#                 "TechSupport":      tech_support if internet_svc != "No" else "No internet service",
#                 "StreamingTV":      stream_tv    if internet_svc != "No" else "No internet service",
#                 "StreamingMovies":  stream_mov   if internet_svc != "No" else "No internet service",
#                 "Contract":         contract,
#                 "PaperlessBilling": paperless,
#                 "PaymentMethod":    payment,
#                 "MonthlyCharges":   float(monthly),
#                 "TotalCharges":     float(total),
#             }
            
#             try:
#                 inp = pd.DataFrame([raw])
                
#                 # 1. Safely label encode
#                 for col_name, le in arts["enc"].items():
#                     if col_name in inp.columns:
#                         v = str(inp.at[0, col_name])
#                         if v in le.classes_:
#                             inp.at[0, col_name] = le.transform([v])[0]
#                         else:
#                             inp.at[0, col_name] = 0  # Fallback safety
                            
#                 # 2. Enforce column order and float type
#                 inp = inp[arts["cols"]].astype(float)
                
#                 # 3. Scale features (returns a NumPy array)
#                 inp_sc_array = arts["scaler"].transform(inp)
                
#                 # 4. FIXED: Re-wrap in DataFrame so XGBoost recognizes feature names!
#                 inp_sc = pd.DataFrame(inp_sc_array, columns=arts["cols"])

#                 # 5. Predict
#                 proba   = arts["model"].predict_proba(inp_sc)[0]
#                 churn_p = float(proba[1]) * 100
#                 stay_p  = float(proba[0]) * 100
#                 pred    = int((churn_p / 100) >= threshold)

#                 if churn_p >= 65:
#                     tier_col, tier_lbl = RED,   "HIGH RISK"
#                 elif churn_p >= 35:
#                     tier_col, tier_lbl = AMBER, "MEDIUM RISK"
#                 else:
#                     tier_col, tier_lbl = GREEN, "LOW RISK"

#                 vcol = RED if pred else GREEN
#                 vtxt = "WILL LIKELY CHURN" if pred else "LIKELY TO STAY"
#                 vico = "⚠️" if pred else "✅"
#                 vbg  = "#1A0810" if pred else "#071510"

#                 divider()
#                 sec_hdr("📈", "Prediction Result",
#                         f"Model: {model_lbl}  ·  Threshold: {threshold:.4f}")

#                 R1, R2, R3 = st.columns([1.1, 1, 1], gap="large")

            
#            #===================
#             # try:
#             #     inp = pd.DataFrame([raw])
#             #     for col_name, le in arts["enc"].items():
#             #         if col_name in inp.columns:
#             #             v = str(inp.at[0, col_name])
#             #             inp[col_name] = le.transform([v]) if v in le.classes_ else [0]
#             #     inp    = inp[arts["cols"]].astype(float)
#             #     inp_sc = arts["scaler"].transform(inp)

#             #     proba   = arts["model"].predict_proba(inp_sc)[0]
#             #     churn_p = float(proba[1]) * 100
#             #     stay_p  = float(proba[0]) * 100
#             #     pred    = int((churn_p / 100) >= threshold)

#             #     if churn_p >= 65:
#             #         tier_col, tier_lbl = RED,   "HIGH RISK"
#             #     elif churn_p >= 35:
#             #         tier_col, tier_lbl = AMBER, "MEDIUM RISK"
#             #     else:
#             #         tier_col, tier_lbl = GREEN, "LOW RISK"

#             #     vcol = RED if pred else GREEN
#             #     vtxt = "WILL LIKELY CHURN" if pred else "LIKELY TO STAY"
#             #     vico = "⚠️" if pred else "✅"
#             #     vbg  = "#1A0810" if pred else "#071510"

#             #     divider()
#             #     sec_hdr("📈", "Prediction Result",
#             #             f"Model: {model_lbl}  ·  Threshold: {threshold:.4f}")

#             #     R1, R2, R3 = st.columns([1.1, 1, 1], gap="large")

#                 # Verdict card
#                 with R1:
#                     md(f"""<div class="sup" style="background:{vbg};border:1px solid {vcol}30;
#                       border-radius:14px;padding:26px 20px;text-align:center;">
#                       <div style="font-size:.6rem;font-family:'JetBrains Mono',monospace;
#                         color:{vcol};letter-spacing:2.5px;margin-bottom:9px;">MODEL VERDICT</div>
#                       <div style="font-size:1.8rem;margin-bottom:6px;">{vico}</div>
#                       <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;
#                         font-weight:700;color:{TEXT};margin-bottom:16px;">{vtxt}</div>
#                       <div style="display:flex;justify-content:center;gap:22px;margin-bottom:14px;">
#                         <div>
#                           <div style="font-family:'Space Grotesk',sans-serif;font-size:2.1rem;
#                             font-weight:700;color:{RED};">{churn_p:.1f}%</div>
#                           <div style="font-size:.63rem;color:{MUTED};text-transform:uppercase;
#                             letter-spacing:.7px;margin-top:2px;">Churn Risk</div>
#                         </div>
#                         <div style="width:1px;background:{BDR};"></div>
#                         <div>
#                           <div style="font-family:'Space Grotesk',sans-serif;font-size:2.1rem;
#                             font-weight:700;color:{GREEN};">{stay_p:.1f}%</div>
#                           <div style="font-size:.63rem;color:{MUTED};text-transform:uppercase;
#                             letter-spacing:.7px;margin-top:2px;">Retention</div>
#                         </div>
#                       </div>
#                       <div style="display:inline-flex;align-items:center;gap:6px;
#                         background:{tier_col}18;border:1px solid {tier_col}40;
#                         border-radius:8px;padding:7px 14px;">
#                         <span style="font-size:.82rem;color:{tier_col};font-weight:700;">{tier_lbl}</span>
#                       </div>
#                     </div>""")

#                 # Gauge + bar
#                 # with R2:
#                 #     fig_g = go.Figure(go.Indicator(
#                 #         mode="gauge+number",
#                 #         value=churn_p,
#                 #         number=dict(suffix="%",
#                 #                     font=dict(size=26, color=tier_col,
#                 #                               family="Space Grotesk")),
#                 #         gauge=dict(
#                 #             axis=dict(range=[0, 100],
#                 #                       tickcolor=MUTED,
#                 #                       tickfont=dict(color=MUTED, size=9)),
#                 #             bar=dict(color=tier_col, thickness=0.62),
#                 #             bgcolor="rgba(0,0,0,0)",
#                 #             borderwidth=0,
#                 #             steps=[
#                 #                 dict(range=[0,  35], color="#071510"),
#                 #                 dict(range=[35, 65], color="#15110A"),
#                 #                 dict(range=[65,100], color="#150809"),
#                 #             ],
#                 #             threshold=dict(
#                 #                 line=dict(color=AMBER, width=2),
#                 #                 thickness=0.75,
#                 #                 value=threshold * 100,
#                 #             ),
#                 #         ),
#                 #     ))
#                 #     fig_g.update_layout(
#                 #         paper_bgcolor="rgba(0,0,0,0)",
#                 #         plot_bgcolor="rgba(0,0,0,0)",
#                 #         font=dict(color=SUB, family="Inter"),
#                 #         margin=dict(l=10, r=10, t=10, b=20),
#                 #         height=200,
#                 #         annotations=[dict(
#                 #             text=f"Decision threshold: {threshold*100:.1f}%",
#                 #             x=0.5, y=-0.18, showarrow=False,
#                 #             font=dict(color=MUTED, size=9),
#                 #         )],
#                 #     )
#                 #     st.plotly_chart(fig_g, use_container_width=True)

#                 #     fig_b = go.Figure()
#                 #     fig_b.add_trace(go.Bar(
#                 #         y=["Retention", "Churn Risk"],
#                 #         x=[stay_p, churn_p],
#                 #         orientation="h",
#                 #         marker_color=[GREEN, RED],
#                 #         marker_line_color="rgba(0,0,0,0)",
#                 #         text=[f"{stay_p:.1f}%", f"{churn_p:.1f}%"],
#                 #         textposition="outside",
#                 #         textfont=dict(color=TEXT, size=11),
#                 #     ))
#                 #     fig_b.add_vline(
#                 #         x=threshold * 100, line_dash="dash",
#                 #         line_color=AMBER, line_width=1.5,
#                 #         annotation_text=f"Threshold {threshold*100:.1f}%",
#                 #         annotation_font_color=AMBER,
#                 #         annotation_font_size=9,
#                 #         annotation_position="top right",
#                 #     )
#                 #     fig_b.update_layout(
#                 #         paper_bgcolor="rgba(0,0,0,0)",
#                 #         plot_bgcolor="rgba(0,0,0,0)",
#                 #         margin=dict(l=8, r=55, t=6, b=6),
#                 #         height=110,
#                 #         xaxis=dict(range=[0, 118], gridcolor=BDR,
#                 #                    tickfont=dict(color=MUTED), showticklabels=False),
#                 #         yaxis=dict(tickfont=dict(color=SUB, size=10),
#                 #                    gridcolor="rgba(0,0,0,0)"),
#                 #         showlegend=False,
#                 #     )
#                 #     st.plotly_chart(fig_b, use_container_width=True)
                
#                 #================================================
#                 # Gauge + bar
#                 with R2:
#                     fig_g = go.Figure(go.Indicator(
#                         mode="gauge+number",
#                         value=churn_p,
#                         title=dict(text="Confidence Level", font=dict(size=14, color=TEXT, family="Space Grotesk")),
#                         number=dict(suffix="%",
#                                     font=dict(size=32, color=tier_col,
#                                               family="Space Grotesk")),
#                         gauge=dict(
#                             axis=dict(range=[0, 100],
#                                       tickcolor=MUTED,
#                                       tickfont=dict(color=MUTED, size=10)),
#                             bar=dict(color=tier_col, thickness=0.7),
#                             bgcolor="rgba(0,0,0,0)",
#                             borderwidth=0,
#                             steps=[
#                                 dict(range=[0,  35], color=CARD2),
#                                 dict(range=[35, 65], color=BDR),
#                                 dict(range=[65,100], color=BDR2),
#                             ],
#                             threshold=dict(
#                                 line=dict(color=TEXT, width=2.5),
#                                 thickness=0.8,
#                                 value=threshold * 100,
#                             ),
#                         ),
#                     ))
#                     fig_g.update_layout(
#                         paper_bgcolor="rgba(0,0,0,0)",
#                         plot_bgcolor="rgba(0,0,0,0)",
#                         font=dict(color=SUB, family="Inter"),
#                         margin=dict(l=15, r=15, t=30, b=10),
#                         height=220,
#                     )
#                     st.plotly_chart(fig_g, use_container_width=True)

#                     fig_b = go.Figure()
#                     fig_b.add_trace(go.Bar(
#                         y=["Retention", "Churn Risk"],
#                         x=[stay_p, churn_p],
#                         orientation="h",
#                         marker_color=[GREEN, RED],
#                         marker_line_color="rgba(0,0,0,0)",
#                         text=[f"{stay_p:.1f}%", f"{churn_p:.1f}%"],
#                         textposition="outside",
#                         textfont=dict(color=TEXT, size=11, family="Space Grotesk"),
#                     ))
#                     fig_b.add_vline(
#                         x=threshold * 100, line_dash="dash",
#                         line_color=TEXT, line_width=1.5,
#                         annotation_text=f"Threshold {threshold*100:.1f}%",
#                         annotation_font_color=SUB,
#                         annotation_font_size=10,
#                         annotation_position="top right",
#                     )
#                     fig_b.update_layout(
#                         paper_bgcolor="rgba(0,0,0,0)",
#                         plot_bgcolor="rgba(0,0,0,0)",
#                         margin=dict(l=5, r=50, t=10, b=10),
#                         height=120,
#                         xaxis=dict(range=[0, 115], gridcolor="rgba(0,0,0,0)",
#                                    tickfont=dict(color=MUTED), showticklabels=False),
#                         yaxis=dict(tickfont=dict(color=TEXT, size=11),
#                                    gridcolor="rgba(0,0,0,0)"),
#                         showlegend=False,
#                     )
#                     st.plotly_chart(fig_b, use_container_width=True)
#                 #================================================

#                 # Actions
#                 with R3:
#                     if pred:
#                         actions = [
#                             ("📞", "Call within 24 hours",       RED),
#                             ("💰", "Offer personalised discount", AMBER),
#                             ("📋", "Propose annual contract",      BLUE),
#                             ("🛡️", "Add free Tech Support trial", PURPLE),
#                             ("🎁", "Send loyalty reward offer",   GREEN),
#                         ]
#                         hdr_col, hdr_txt = RED,   "🚨 Retention Actions"
#                     else:
#                         actions = [
#                             ("⭐", "Enrol in rewards programme",  GREEN),
#                             ("📦", "Upsell premium services",     TEAL),
#                             ("🔁", "Encourage contract upgrade",  BLUE),
#                             ("📣", "Request referral or review",  PURPLE),
#                             ("📊", "Monitor usage trends",        CYAN),
#                         ]
#                         hdr_col, hdr_txt = GREEN, "✅ Growth Actions"

#                     rows_h = "".join(
#                         f'<div style="display:flex;align-items:center;gap:9px;'
#                         f'padding:8px 10px;border-radius:7px;margin-bottom:4px;'
#                         f'background:{ac}0C;border:1px solid {ac}1A;">'
#                         f'<span style="font-size:.88rem;">{ico}</span>'
#                         f'<span style="font-size:.78rem;color:{SUB};">{txt}</span>'
#                         f'</div>'
#                         for ico, txt, ac in actions
#                     )
#                     md(f'<div style="background:{CARD};border:1px solid {BDR};'
#                        f'border-radius:13px;padding:16px;">'
#                        f'<div style="font-size:.74rem;font-weight:700;color:{hdr_col};'
#                        f'font-family:Space Grotesk,sans-serif;margin-bottom:12px;">'
#                        f'{hdr_txt}</div>{rows_h}</div>')

#                 # ── What Drives This Prediction ─────────────────────
#                 md("<br>")
#                 sec_hdr("🔍", "What Drives This Prediction?",
#                         "Risk factors identified from EDA patterns for this customer profile")

#                 # Build risk factors — simple lists, no complex unpacking
#                 high_risk, med_risk, low_risk = [], [], []

#                 if "Month-to-month" in contract:
#                     high_risk.append((
#                         "📅 Month-to-month Contract",
#                         "Historical churn rate: 42.7% — highest of all contract types",
#                         "vs only 2.8% churn on 2-year plans",
#                     ))
#                 if "Electronic" in payment:
#                     high_risk.append((
#                         "💳 Electronic Check Payment",
#                         "Historical churn rate: 45.3% — highest of all payment methods",
#                         "Customers on auto-pay churn significantly less",
#                     ))
#                 if internet_svc == "Fiber optic":
#                     high_risk.append((
#                         "🌐 Fiber Optic Internet",
#                         "Historical churn rate: 41.9% vs DSL at 18.9%",
#                         "Price-to-value perception is the key driver",
#                     ))
#                 if int(tenure) <= 12:
#                     high_risk.append((
#                         "🆕 New Customer (first 12 months)",
#                         f"First-year churn rate: 47.7% — highest risk period",
#                         f"This customer has {tenure} months tenure",
#                     ))
#                 if online_sec == "No" and internet_svc != "No":
#                     med_risk.append((
#                         "🔒 No Online Security",
#                         "Customers without it churn at 41.8%",
#                         "Adding this service reduces dissatisfaction",
#                     ))
#                 if tech_support == "No" and internet_svc != "No":
#                     med_risk.append((
#                         "🛠 No Tech Support",
#                         "Customers without it churn at 41.6%",
#                         "Proactive support prevents frustration",
#                     ))
#                 if float(monthly) > 70:
#                     med_risk.append((
#                         "💸 High Monthly Charges",
#                         f"${monthly:.0f}/month exceeds the $70 risk threshold",
#                         "High charges correlate strongly with churn",
#                     ))
#                 if senior == "Yes":
#                     med_risk.append((
#                         "👴 Senior Citizen",
#                         "Churn rate 41.7% vs 24.3% for non-seniors",
#                         "Dedicated support programmes improve retention",
#                     ))
#                 if contract == "Two year":
#                     low_risk.append((
#                         "📋 Two-Year Contract",
#                         "Churn rate only 2.8% — strongest retention signal",
#                         "Long-term commitment is the best loyalty indicator",
#                     ))
#                 if int(tenure) >= 48:
#                     low_risk.append((
#                         "🏆 Long-Tenure Customer",
#                         f"{tenure} months loyalty — churn risk drops dramatically",
#                         "Customers over 48 months churn at only 6.6%",
#                     ))
#                 if contract == "One year":
#                     low_risk.append((
#                         "📋 One-Year Contract",
#                         "Contracted customers churn far less than month-to-month",
#                         "Encourage upgrade to 2-year for even better retention",
#                     ))
#                 if online_sec == "Yes" and tech_support == "Yes":
#                     low_risk.append((
#                         "✅ Fully Supported Customer",
#                         "Both online security and tech support active",
#                         "Well-served customers have lower churn propensity",
#                     ))

#                 # Fallback if nothing detected
#                 if not high_risk and not med_risk and not low_risk:
#                     low_risk.append((
#                         "✅ No Major Risk Factors",
#                         "This customer does not match high-risk churn profiles",
#                         "Continue monitoring engagement and usage metrics",
#                     ))

#                 # Render in 3 columns — safe, no complex unpacking in f-strings
#                 all_factors = (
#                     [(RED,   "High Risk Factor",   t) for t in high_risk] +
#                     [(AMBER, "Medium Risk Factor", t) for t in med_risk]  +
#                     [(GREEN, "Positive Signal",    t) for t in low_risk]
#                 )

#                 if all_factors:
#                     ncols = min(3, len(all_factors))
#                     if ncols < 1:
#                         ncols = 1
#                     fcols = st.columns(ncols, gap="medium")
#                     for idx, factor in enumerate(all_factors):
#                         col_c   = factor[0]
#                         col_lbl = factor[1]
#                         title_f, desc_f, note_f = factor[2]
#                         card_html = risk_card_html(col_c, col_lbl, title_f, desc_f, note_f)
#                         with fcols[idx % ncols]:
#                             md(card_html)

#                 # Summary
#                 n_h = len(high_risk)
#                 n_m = len(med_risk)
#                 n_l = len(low_risk)
#                 s_col = RED if n_h >= 2 else AMBER if (n_h == 1 or n_m >= 2) else GREEN
#                 md(f'<div style="background:{s_col}0A;border:1px solid {s_col}28;'
#                    f'border-radius:10px;padding:12px 16px;margin-top:10px;">'
#                    f'<span style="font-size:.82rem;color:{TEXT};font-weight:600;">'
#                    f'Risk Summary: </span>'
#                    f'<span style="font-size:.82rem;color:{SUB};">'
#                    f'{n_h} high-risk · {n_m} medium-risk · {n_l} positive signal(s). '
#                    f'Churn probability: </span>'
#                    f'<strong style="color:{tier_col};">{churn_p:.1f}% ({tier_lbl})</strong>'
#                    f'</div>')

#             except Exception as exc:
#                 st.error(f"**Prediction error:** {exc}")
#                 st.info(
#                     "Check that all 5 PKL files in `models/` were generated by "
#                     "running the notebook end-to-end with the same dataset."
#                 )

#         else:
#             divider()
#             md(f'<div style="background:{CARD};border:1px dashed {BDR2};'
#                f'border-radius:14px;padding:48px 24px;text-align:center;margin:8px 0;">'
#                f'<div style="font-size:2.4rem;margin-bottom:12px;">🔮</div>'
#                f'<div style="font-family:Space Grotesk,sans-serif;font-size:1rem;'
#                f'font-weight:700;color:{TEXT};margin-bottom:8px;">Ready to Analyse</div>'
#                f'<div style="font-size:.84rem;color:{SUB};line-height:1.7;'
#                f'max-width:360px;margin:0 auto;">Complete the customer profile above '
#                f'and click <strong style="color:{BLUE};">Run Churn Prediction</strong> '
#                f'to see AI results with risk insights.</div></div>')

#             md("<br>")
#             sec_hdr("📌", "Key Churn Signals",
#                     "Historical EDA patterns — top factors that drive customer churn")
#             s1, s2 = st.columns(2, gap="large")
#             with s1:
#                 for lb, rt, ac in [
#                     ("Month-to-month contract",  "42.7% churn rate", RED),
#                     ("Electronic check payment", "45.3% churn rate", RED),
#                     ("Fiber optic internet",     "41.9% churn rate", RED),
#                     ("Tenure ≤ 12 months",       "47.7% churn rate", RED),
#                 ]:
#                     md(signal_row(lb, rt, ac))
#             with s2:
#                 for lb, rt, ac in [
#                     ("No online security",  "41.8% churn rate", AMBER),
#                     ("No tech support",     "41.6% churn rate", AMBER),
#                     ("Two-year contract",   " 2.8% churn rate", GREEN),
#                     ("Tenure > 48 months",  " 6.6% churn rate", GREEN),
#                 ]:
#                     md(signal_row(lb, rt, ac))


# # ═══════════════════════════════════════════
# # TAB 2 — EDA DASHBOARD
# # ═══════════════════════════════════════════
# with TAB2:
#     if not ok_d:
#         st.warning(
#             "**dataset.csv not found.**  "
#             "Place it in the same folder as app.py and restart Streamlit."
#         )
#         md(f'<div style="background:{CARD};border:1px dashed {BDR2};'
#            f'border-radius:14px;padding:48px 24px;text-align:center;margin-top:12px;">'
#            f'<div style="font-size:2.4rem;margin-bottom:12px;">📊</div>'
#            f'<div style="font-family:Space Grotesk,sans-serif;font-size:1rem;'
#            f'font-weight:700;color:{TEXT};margin-bottom:8px;">Dataset Required</div>'
#            f'<div style="font-size:.84rem;color:{SUB};line-height:1.7;">'
#            f'Add dataset.csv to the app folder to view interactive EDA charts.</div></div>')
#     else:
#         # KPIs
#         sec_hdr("📊", "Dataset Overview",
#                 "7,043 Telco customers · 21 raw features · Kaggle Telco Churn dataset")
#         kc2 = st.columns(6, gap="small")
#         for col, (ic, lb, vl, ac, nt) in zip(kc2, [
#             ("👥", "Total",       f"{len(df):,}",                              BLUE,   ""),
#             ("🔴", "Churned",     f"{df['ChurnBin'].sum():,}",                 RED,    f"{df['ChurnBin'].mean()*100:.1f}%"),
#             ("🟢", "Retained",    f"{(~df['ChurnBin'].astype(bool)).sum():,}", GREEN,  f"{(1-df['ChurnBin'].mean())*100:.1f}%"),
#             ("📅", "Avg Tenure",  f"{df['tenure'].mean():.0f} mo",             PURPLE, "months"),
#             ("💵", "Avg Monthly", f"${df['MonthlyCharges'].mean():.0f}",       AMBER,  "/month"),
#             ("🧮", "Features",    "19",                                         TEAL,   "post-encoding"),
#         ]):
#             col.markdown(kpi(ic, lb, vl, ac, nt), unsafe_allow_html=True)

#         divider()

#         # ── 1. Churn Distribution ───────────────────────────────
#         sec_hdr("📉", "1 · Churn Distribution",
#                 "73.5% No Churn vs 26.5% Churned — class imbalance handled with SMOTE on training data")

#         no_ct  = int((df["Churn"] == "No").sum())
#         yes_ct = int((df["Churn"] == "Yes").sum())

#         ca, cb = st.columns(2, gap="large")
#         with ca:
#             fig = go.Figure(go.Bar(
#                 x=["No Churn", "Churned"],
#                 y=[no_ct, yes_ct],
#                 marker_color=[GREEN, RED],
#                 marker_line_color="rgba(0,0,0,0)",
#                 text=[f"{no_ct:,}", f"{yes_ct:,}"],
#                 textposition="outside",
#                 textfont=dict(color=TEXT, size=13),
#             ))
#             fig.update_layout(**plotly_layout("Churn Count", 300))
#             fig.update_yaxes(title="Customers", gridcolor=BDR)
#             fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig, use_container_width=True)

#         with cb:
#             fig = go.Figure(go.Pie(
#                 labels=["No Churn", "Churned"],
#                 values=[no_ct, yes_ct],
#                 hole=0.44,
#                 pull=[0, 0.05],
#                 marker=dict(colors=[GREEN, RED],
#                             line=dict(color=BG, width=2)),
#                 textinfo="label+percent",
#                 textfont=dict(size=12),
#             ))
#             fig.update_layout(**plotly_layout("Churn Split (%)", 300))
#             fig.update_layout(showlegend=False)
#             st.plotly_chart(fig, use_container_width=True)

#         md(f'<div style="background:{AMBER}0A;border:1px solid {AMBER}30;'
#            f'border-radius:9px;padding:11px 16px;margin:6px 0;">'
#            f'<strong style="color:{AMBER};">⚠ Class Imbalance:</strong>'
#            f'<span style="color:{SUB};font-size:.84rem;"> 73.5% vs 26.5% — '
#            f'SMOTE applied on training data only to prevent data leakage.</span></div>')

#         # ── 2. Numerical Distributions ──────────────────────────
#         divider()
#         sec_hdr("📐", "2 · Numerical Feature Distributions",
#                 "Tenure, MonthlyCharges, TotalCharges — split by churn status")

#         num_cols_list = ["tenure", "MonthlyCharges", "TotalCharges"]
#         fig2a = make_subplots(
#             rows=1, cols=3,
#             subplot_titles=["Tenure (months)", "Monthly Charges ($)", "Total Charges ($)"],
#             horizontal_spacing=0.06,
#         )
#         for i, col_name in enumerate(num_cols_list, 1):
#             for churn_val, nm, clr in [
#                 (False, "No Churn", BLUE),
#                 (True,  "Churned",  RED),
#             ]:
#                 sub = df[df["ChurnBin"] == int(churn_val)][col_name]
#                 fig2a.add_trace(go.Histogram(
#                     x=sub, name=nm, marker_color=clr,
#                     opacity=0.7, nbinsx=30,
#                     showlegend=(i == 1),
#                     legendgroup=nm,
#                 ), row=1, col=i)
#         lyt2a = plotly_layout("Distribution by Churn Status", 320)
#         lyt2a["barmode"] = "overlay"
#         fig2a.update_layout(**lyt2a)
#         fig2a.update_xaxes(gridcolor=BDR, linecolor=BDR)
#         fig2a.update_yaxes(gridcolor=BDR, linecolor=BDR)
#         for ann in fig2a.layout.annotations:
#             ann.font.color = SUB
#             ann.font.size  = 11
#         st.plotly_chart(fig2a, use_container_width=True)

#         fig2b = make_subplots(
#             rows=1, cols=3,
#             subplot_titles=["Tenure vs Churn", "Monthly Charges vs Churn",
#                             "Total Charges vs Churn"],
#             horizontal_spacing=0.06,
#         )
#         for i, col_name in enumerate(num_cols_list, 1):
#             for churn_val, nm, clr in [
#                 ("No", "No Churn", BLUE),
#                 ("Yes", "Churned", RED),
#             ]:
#                 fig2b.add_trace(go.Box(
#                     y=df[df["Churn"] == churn_val][col_name],
#                     name=nm, marker_color=clr, line_color=clr,
#                     showlegend=(i == 1), legendgroup=nm,
#                 ), row=1, col=i)
#         lyt2b = plotly_layout("Spread by Churn Status", 320)
#         lyt2b["boxmode"] = "group"
#         fig2b.update_layout(**lyt2b)
#         fig2b.update_xaxes(gridcolor="rgba(0,0,0,0)")
#         fig2b.update_yaxes(gridcolor=BDR)
#         for ann in fig2b.layout.annotations:
#             ann.font.color = SUB
#             ann.font.size  = 11
#         st.plotly_chart(fig2b, use_container_width=True)

#         md(f'<div style="background:{CARD2};border:1px solid {BDR};'
#            f'border-radius:9px;padding:11px 16px;margin:4px 0;">'
#            f'<span style="font-size:.82rem;color:{SUB};">'
#            f'<strong style="color:{RED};">Tenure:</strong> Bimodal — many new + many long-term customers &nbsp;·&nbsp; '
#            f'<strong style="color:{AMBER};">MonthlyCharges:</strong> Churners tend to pay more &nbsp;·&nbsp; '
#            f'<strong style="color:{BLUE};">TotalCharges:</strong> Right-skewed, correlated with tenure'
#            f'</span></div>')

#         # ── 3. Categorical Churn Rates ────────────────────────
#         divider()
#         sec_hdr("📊", "3 · Churn Rate by Category",
#                 "Which service and billing categories drive churn most?")

#         cat_feats = [
#             "Contract", "InternetService", "PaymentMethod",
#             "TechSupport", "OnlineSecurity", "PaperlessBilling",
#         ]
#         fig3 = make_subplots(
#             rows=2, cols=3,
#             subplot_titles=cat_feats,
#             vertical_spacing=0.2,
#             horizontal_spacing=0.07,
#         )
#         for idx, feat in enumerate(cat_feats):
#             r, c = divmod(idx, 3)
#             cr = (df.groupby(feat)["Churn"]
#                   .apply(lambda x: (x == "Yes").mean() * 100)
#                   .sort_values(ascending=False)
#                   .reset_index())
#             cr.columns = [feat, "Pct"]
#             bclrs = [RED if v > 30 else AMBER if v > 15 else GREEN
#                      for v in cr["Pct"]]
#             fig3.add_trace(go.Bar(
#                 x=cr[feat], y=cr["Pct"],
#                 marker_color=bclrs,
#                 marker_line_color="rgba(0,0,0,0)",
#                 text=[f"{v:.1f}%" for v in cr["Pct"]],
#                 textposition="outside",
#                 textfont=dict(color=TEXT, size=9),
#                 showlegend=False,
#             ), row=r+1, col=c+1)

#         lyt3 = plotly_layout("Churn Rate (%) by Feature Category", 540)
#         fig3.update_layout(**lyt3)
#         fig3.update_xaxes(tickfont=dict(size=8, color=SUB),
#                           gridcolor="rgba(0,0,0,0)")
#         fig3.update_yaxes(gridcolor=BDR, tickfont=dict(color=MUTED))
#         for ann in fig3.layout.annotations:
#             ann.font.color = SUB
#             ann.font.size  = 11
#         st.plotly_chart(fig3, use_container_width=True)

#         md(f'<div style="background:{CARD2};border:1px solid {BDR};'
#            f'border-radius:9px;padding:11px 16px;margin:4px 0;">'
#            f'<span style="font-size:.82rem;color:{SUB};">'
#            f'<strong style="color:{RED};">Contract:</strong> Month-to-month 42.7% vs Two-year 2.8% &nbsp;·&nbsp; '
#            f'<strong style="color:{RED};">Payment:</strong> Electronic check 45.3% &nbsp;·&nbsp; '
#            f'<strong style="color:{AMBER};">Internet:</strong> Fiber optic 41.9% vs DSL 18.9% &nbsp;·&nbsp; '
#            f'<strong style="color:{AMBER};">Tech Support:</strong> Without 41.6% vs With 14.8%'
#            f'</span></div>')

#         # ── 4. Demographics ────────────────────────────────────
#         divider()
#         sec_hdr("👥", "4 · Demographics vs Churn",
#                 "Gender, Senior Citizen, Partner, Dependents breakdown")

#         demo_feats = ["gender", "SeniorCitizen", "Partner", "Dependents"]
#         d_cols = st.columns(4, gap="small")
#         for i, feat in enumerate(demo_feats):
#             with d_cols[i]:
#                 ct = (df.groupby(feat)["Churn"]
#                       .apply(lambda x: (x == "Yes").mean() * 100)
#                       .reset_index())
#                 ct.columns = [feat, "Pct"]
#                 # Convert x to string to handle int (SeniorCitizen = 0/1)
#                 x_labels = ct[feat].astype(str).tolist()
#                 clrs = [BLUE, RED] if len(ct) == 2 else [BLUE]*len(ct)
#                 fig_d = go.Figure(go.Bar(
#                     x=x_labels,
#                     y=ct["Pct"].tolist(),
#                     marker_color=clrs,
#                     marker_line_color="rgba(0,0,0,0)",
#                     text=[f"{v:.1f}%" for v in ct["Pct"]],
#                     textposition="outside",
#                     textfont=dict(color=TEXT, size=10),
#                 ))
#                 fig_d.update_layout(**plotly_layout(feat, 240))
#                 fig_d.update_layout(margin=dict(l=5, r=5, t=40, b=5))
#                 fig_d.update_yaxes(title="Churn Rate (%)", range=[0, max(ct["Pct"])*1.32])
#                 fig_d.update_xaxes(gridcolor="rgba(0,0,0,0)")
#                 st.plotly_chart(fig_d, use_container_width=True)

#         # ── 5. Tenure Groups ───────────────────────────────────
#         divider()
#         sec_hdr("⏱️", "5 · Churn Rate by Tenure Group",
#                 "0-12 months is the highest-risk window — 47.7% churn rate")

#         tg = (df.groupby("TenureGroup", observed=True)["Churn"]
#               .apply(lambda x: (x == "Yes").mean() * 100)
#               .reset_index())
#         tg.columns = ["Group", "Pct"]
#         tg["Group"] = tg["Group"].astype(str)  # safe cast from Categorical

#         tg_colors = [RED, AMBER, BLUE, TEAL]
#         tc1, tc2 = st.columns([2, 1], gap="large")
#         with tc1:
#             fig_tg = go.Figure(go.Bar(
#                 x=tg["Group"].tolist(),
#                 y=tg["Pct"].tolist(),
#                 marker_color=tg_colors[:len(tg)],
#                 marker_line_color="rgba(0,0,0,0)",
#                 text=[f"{v:.1f}%" for v in tg["Pct"]],
#                 textposition="outside",
#                 textfont=dict(color=TEXT, size=12),
#             ))
#             fig_tg.update_layout(**plotly_layout("Churn Rate by Tenure Group", 300))
#             fig_tg.update_yaxes(title="Churn Rate (%)", gridcolor=BDR)
#             fig_tg.update_xaxes(title="Tenure Group", gridcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig_tg, use_container_width=True)
#         with tc2:
#             for row_idx, row in tg.iterrows():
#                 ac = tg_colors[row_idx] if row_idx < len(tg_colors) else BLUE
#                 md(f'<div style="background:{CARD};border:1px solid {ac}30;'
#                    f'border-left:3px solid {ac};border-radius:8px;'
#                    f'padding:12px 14px;margin-bottom:7px;">'
#                    f'<div style="font-size:.69rem;color:{MUTED};margin-bottom:2px;">'
#                    f'{row["Group"]}</div>'
#                    f'<div style="font-family:Space Grotesk,sans-serif;font-size:1.4rem;'
#                    f'font-weight:700;color:{ac};">{row["Pct"]:.1f}%</div>'
#                    f'</div>')

#         # ── 6. Correlation Heatmap ─────────────────────────────
#         divider()
#         sec_hdr("🔥", "6 · Feature Correlation Heatmap",
#                 "Label-encoded · lower triangle only · key correlations with Churn")

#         df_corr = df.drop(
#             columns=["ChurnBin", "TenureGroup", "customerID"], errors="ignore"
#         ).copy()
#         le_tmp = LabelEncoder()
#         # Fix: use include="object" with str handling for newer pandas
#         obj_cols = [c for c in df_corr.columns
#                     if df_corr[c].dtype == object or df_corr[c].dtype.name == "string"]
#         for col_name in obj_cols:
#             df_corr[col_name] = le_tmp.fit_transform(df_corr[col_name].astype(str))

#         corr = df_corr.corr().round(2)
#         # Mask upper triangle — replace with NaN
#         mask = np.triu(np.ones_like(corr.values, dtype=bool), k=1)
#         z_vals = corr.values.copy().astype(float)
#         z_vals[mask] = np.nan

#         # Build text array — show value or empty string (no "nan")
#         text_vals = []
#         for row_arr in z_vals:
#             row_text = []
#             for v in row_arr:
#                 row_text.append("" if np.isnan(v) else str(round(v, 2)))
#             text_vals.append(row_text)

#         # fig_hm = go.Figure(go.Heatmap(
#         #     z=z_vals,
#         #     x=corr.columns.tolist(),
#         #     y=corr.index.tolist(),
#         #     colorscale="RdBu_r",
#         #     zmid=0,
#         #     zmin=-1, zmax=1,
#         #     text=text_vals,
#         #     texttemplate="%{text}",
#         #     textfont=dict(size=7, color="white"),
#         #     hoverongaps=False,
#         #  colorbar=dict(
#         #         title=dict(
#         #             text="r", 
#         #             font=dict(color=SUB, size=10)
#         #         ),
#         #         tickfont=dict(color=SUB, size=9),
#         #     ),
#         # ))
#         # hm_layout = plotly_layout("Feature Correlation Matrix", 520)
#         # hm_layout["xaxis"] = dict(
#         #     tickfont=dict(size=8, color=SUB), tickangle=45,
#         #     gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
#         # )
#         # hm_layout["yaxis"] = dict(
#         #     tickfont=dict(size=8, color=SUB),
#         #     gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
#         # )
#         # hm_layout["margin"] = dict(l=120, r=20, t=50, b=120)
#         # fig_hm.update_layout(**hm_layout)
#         # st.plotly_chart(fig_hm, use_container_width=True)

#         # md(f'<div style="background:{CARD2};border:1px solid {BDR};'
#         #    f'border-radius:9px;padding:11px 16px;margin:4px 0;">'
#         #    f'<span style="font-size:.82rem;color:{SUB};">'
#         #    f'<strong style="color:{TEAL};">tenure</strong> negatively correlated with Churn — '
#         #    f'long-term customers stay &nbsp;·&nbsp; '
#         #    f'<strong style="color:{RED};">MonthlyCharges</strong> positively correlated — '
#         #    f'expensive plans drive churn &nbsp;·&nbsp; '
#         #    f'TotalCharges highly correlated with tenure (expected)'
#         #    f'</span></div>')
        
#         #===========================
#         fig_hm = go.Figure(go.Heatmap(
#             z=z_vals,
#             x=corr.columns.tolist(),
#             y=corr.index.tolist(),
#             colorscale=[
#                 [0.00, '#3b4cc0'], [0.15, '#6788ee'], [0.35, '#9abbff'], 
#                 [0.50, '#e2e2e2'], [0.65, '#f1a88d'], [0.85, '#d35c4e'], [1.00, '#b40426']
#             ],
#             zmid=0,
#             zmin=-1, zmax=1,
#             text=text_vals,
#             texttemplate="<b>%{text}</b>",
#             textfont=dict(size=11),  # <-- FIXED: Plotly will now auto-contrast the text!
#             hoverongaps=False,
#             colorbar=dict(
#                 title=dict(
#                     text="r", 
#                     font=dict(color=SUB, size=12)
#                 ),
#                 tickfont=dict(color=SUB, size=11),
#             ),
#         ))
       
        
#         # Increased height from 520 to 750 so it doesn't look shrunk
#         hm_layout = plotly_layout("Feature Correlation Matrix", 750) 
        
#         hm_layout["xaxis"] = dict(
#             tickfont=dict(size=10, color=SUB), tickangle=45,  # Increased axis font size
#             gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
#         )
#         hm_layout["yaxis"] = dict(
#             tickfont=dict(size=10, color=SUB),                # Increased axis font size
#             gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
#         )
#         hm_layout["margin"] = dict(l=120, r=20, t=50, b=120)
        
#         fig_hm.update_layout(**hm_layout)
#         st.plotly_chart(fig_hm, use_container_width=True)

#         md(f'<div style="background:{CARD2};border:1px solid {BDR};'
#            f'border-radius:9px;padding:11px 16px;margin:4px 0;">'
#            f'<span style="font-size:.82rem;color:{SUB};">'
#            f'<strong style="color:{TEAL};">tenure</strong> negatively correlated with Churn — '
#            f'long-term customers stay &nbsp;·&nbsp; '
#            f'<strong style="color:{RED};">MonthlyCharges</strong> positively correlated — '
#            f'expensive plans drive churn &nbsp;·&nbsp; '
#            f'TotalCharges highly correlated with tenure (expected)'
#            f'</span></div>')
#         #===========================

#         # Raw data browser
#         with st.expander("📋  Browse Raw Dataset"):
#             flt = st.text_input("Filter by Churn (Yes / No / blank = all):", "")
#             vdf = (df[df["Churn"].str.contains(flt, case=False, na=False)]
#                    if flt else df)
#             st.dataframe(
#                 vdf.drop(columns=["ChurnBin", "TenureGroup"], errors="ignore").head(300),
#                 use_container_width=True,
#             )
#             st.caption(f"Showing {min(300, len(vdf))} of {len(vdf):,} rows")


# # ═══════════════════════════════════════════
# # TAB 3 — MODEL INSIGHTS
# # ═══════════════════════════════════════════
# with TAB3:

#     # Pipeline
#     sec_hdr("🏆", "Model Training Pipeline",
#             "End-to-end: raw data → SMOTE → tuning → best model → PKL artifacts")
#     steps_html = "  ".join(
#         f'<div style="background:{cl}12;border:1px solid {cl}30;border-radius:7px;'
#         f'padding:7px 13px;font-size:.77rem;font-family:JetBrains Mono,monospace;'
#         f'color:{cl};white-space:nowrap;">{s}</div>'
#         for s, cl in [
#             ("1. Data Cleaning",       BLUE),
#             ("2. Feature Encoding",    CYAN),
#             ("3. Train/Test Split",    TEAL),
#             ("4. SMOTE Balancing",     GREEN),
#             ("5. Baseline ×3 Models",  PURPLE),
#             ("6. GridSearchCV Tuning", AMBER),
#             ("7. Select by AUC-ROC",   RED),
#             ("8. Save PKL Artifacts",  BLUE),
#         ]
#     )
#     md(f'<div style="background:{CARD};border:1px solid {BDR};border-radius:11px;'
#        f'padding:15px 17px;margin-bottom:20px;">'
#        f'<div style="display:flex;gap:8px;flex-wrap:wrap;">{steps_html}</div></div>')

#     # Performance table
#     sec_hdr("📊", "Model Performance — Baseline vs Tuned",
#             "Correct verified values from notebook execution")

#     perf_df = pd.DataFrame({
#         "Model":           ["XGBoost  ★ Best", "Random Forest", "Logistic Regression"],
#         "Base Accuracy":   [0.7779, 0.7771, 0.7424],
#         "Tuned Accuracy":  [0.7757, 0.7622, 0.7410],
#         "Base F1":         [0.5865, 0.5890, 0.6191],
#         "Tuned F1":        [0.6274, 0.6223, 0.6146],
#         "Base AUC-ROC":    [0.8146, 0.8210, 0.8391],
#         "Tuned AUC-ROC":   [0.8434, 0.8397, 0.8382],
#         "Speed":           ["Medium", "Slow", "Fast"],
#     })
#     st.dataframe(perf_df, use_container_width=True, hide_index=True)

#     # Metric bar charts
#     md("<br>")
#     mc_cols = st.columns(3, gap="medium")
#     mnames  = ["XGBoost", "Random Forest", "Logistic Reg."]
#     for i, (metric, vals, ac) in enumerate([
#         ("Tuned Accuracy", [0.7757, 0.7622, 0.7410], BLUE),
#         ("Tuned AUC-ROC",  [0.8434, 0.8397, 0.8382], AMBER),
#         ("Tuned F1-Score", [0.6274, 0.6223, 0.6146], GREEN),
#     ]):
#         with mc_cols[i]:
#             best   = max(vals)
#             bclrs  = [ac if abs(v - best) < 1e-5 else BDR2 for v in vals]
#             texts  = [f"{v:.4f} ★" if abs(v-best)<1e-5 else f"{v:.4f}" for v in vals]
#             fig_mc = go.Figure(go.Bar(
#                 y=mnames, x=vals, orientation="h",
#                 marker_color=bclrs,
#                 marker_line_color="rgba(0,0,0,0)",
#                 text=texts, textposition="outside",
#                 textfont=dict(color=TEXT, size=10),
#             ))
#             fig_mc.update_layout(**plotly_layout(metric, 210))
#             fig_mc.update_layout(margin=dict(l=8, r=65, t=42, b=8))
#             fig_mc.update_xaxes(range=[min(vals)-0.05, max(vals)+0.07], gridcolor=BDR)
#             fig_mc.update_yaxes(gridcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig_mc, use_container_width=True)

#     # Baseline vs Tuned comparison
#     sec_hdr("📈", "Baseline vs Tuned Improvement",
#             "F1-Score and AUC-ROC gain after GridSearchCV hyperparameter tuning")

#     fig_bt = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=["F1-Score: Baseline vs Tuned", "AUC-ROC: Baseline vs Tuned"],
#         horizontal_spacing=0.1,
#     )
#     for cidx, (bv, tv, mn_col) in enumerate([
#         ([0.5865, 0.5890, 0.6191], [0.6274, 0.6223, 0.6146], "F1"),
#         ([0.8146, 0.8210, 0.8391], [0.8434, 0.8397, 0.8382], "AUC"),
#     ], 1):
#         all_v = bv + tv
#         fig_bt.add_trace(go.Bar(
#             name="Baseline", x=mnames, y=bv,
#             marker_color=BDR2,
#             marker_line_color="rgba(0,0,0,0)",
#             text=[f"{v:.4f}" for v in bv], textposition="outside",
#             textfont=dict(color=SUB, size=9),
#             showlegend=(cidx == 1),
#         ), row=1, col=cidx)
#         fig_bt.add_trace(go.Bar(
#             name="Tuned", x=mnames, y=tv,
#             marker_color=BLUE,
#             marker_line_color="rgba(0,0,0,0)",
#             text=[f"{v:.4f}" for v in tv], textposition="outside",
#             textfont=dict(color=TEXT, size=9),
#             showlegend=(cidx == 1),
#         ), row=1, col=cidx)
#         fig_bt.update_yaxes(range=[min(all_v)-0.04, max(all_v)+0.06],
#                             gridcolor=BDR, row=1, col=cidx)
#         fig_bt.update_xaxes(gridcolor="rgba(0,0,0,0)", row=1, col=cidx)

#     lyt_bt = plotly_layout("Baseline vs Tuned — All 3 Models", 340)
#     lyt_bt["barmode"] = "group"
#     fig_bt.update_layout(**lyt_bt)
#     for ann in fig_bt.layout.annotations:
#         ann.font.color = SUB
#         ann.font.size  = 11
#     st.plotly_chart(fig_bt, use_container_width=True)

#     # Feature Importance
#     if ok_m:
#         sec_hdr("📌", "Feature Importance",
#                 f"Top 15 features driving {model_lbl} predictions")
#         try:
#             actual_m = arts["model"]
#             if hasattr(actual_m, "named_steps"):
#                 actual_m = actual_m.named_steps.get("model", actual_m)
#             if hasattr(actual_m, "feature_importances_"):
#                 imps = actual_m.feature_importances_
#             elif hasattr(actual_m, "coef_"):
#                 imps = np.abs(actual_m.coef_[0])
#             else:
#                 raise ValueError("No importances or coefficients on model object")

#             fi = (pd.Series(imps, index=arts["cols"])
#                   .sort_values(ascending=True)
#                   .tail(15))
#             top3 = sorted(fi.values)[-3]
#             fi_clrs = [AMBER if v >= top3 else BLUE for v in fi.values]
#             fi_txt  = [f"{v:.4f}" for v in fi.values]

#             fig_fi = go.Figure(go.Bar(
#                 y=fi.index.tolist(),
#                 x=fi.values.tolist(),
#                 orientation="h",
#                 marker_color=fi_clrs,
#                 marker_line_color="rgba(0,0,0,0)",
#                 text=fi_txt, textposition="outside",
#                 textfont=dict(color=TEXT, size=9),
#             ))
#             fig_fi.update_layout(**plotly_layout(f"Feature Importances — {model_lbl}", 440))
#             fig_fi.update_layout(margin=dict(l=8, r=70, t=50, b=8))
#             fig_fi.update_xaxes(gridcolor=BDR, title="Importance Score")
#             fig_fi.update_yaxes(gridcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig_fi, use_container_width=True)
#             md(f'<div style="font-size:.78rem;color:{MUTED};margin-top:4px;">'
#                f'<span style="color:{AMBER};">■</span> Top 3 drivers &nbsp;'
#                f'<span style="color:{BLUE};">■</span> Other features</div>')
#         except Exception as e:
#             st.info(f"Feature importance unavailable: {e}")
#     else:
#         st.info("Load model PKL files to see feature importance chart.")

#     # Final metrics
#     sec_hdr("📋", "Final Model Metrics — XGBoost",
#             "Held-out test set · 1,409 samples · no data leakage")

#     m5c = st.columns(5, gap="small")
#     for col, (ic, lb, vl, ac) in zip(m5c, [
#         ("🎯", "Accuracy",  f"{ACC*100:.2f}%", BLUE),
#         ("⚡", "Precision", f"{PREC*100:.2f}%", PURPLE),
#         ("🔍", "Recall",    f"{REC*100:.2f}%",  TEAL),
#         ("📊", "F1 Score",  f"{F1:.4f}",        GREEN),
#         ("📈", "AUC-ROC",   f"{AUC:.4f}",       AMBER),
#     ]):
#         col.markdown(kpi(ic, lb, vl, ac), unsafe_allow_html=True)

#     # Classification report + hyperparams
#     md("<br>")
#     cr1, cr2 = st.columns(2, gap="large")

#     with cr1:
#         sec_hdr("📄", "Classification Report",
#                 "Per-class breakdown on test set — No Churn vs Churn")
#         report_data = pd.DataFrame({
#             "Class":     ["No Churn", "Churn", "Macro Avg", "Weighted Avg"],
#             "Precision": [0.88, 0.56, 0.72, 0.80],
#             "Recall":    [0.80, 0.71, 0.76, 0.78],
#             "F1-Score":  [0.84, 0.63, 0.73, 0.78],
#             "Support":   [1035, 374, 1409, 1409],
#         })
#         st.dataframe(report_data, use_container_width=True, hide_index=True)

#     with cr2:
#         sec_hdr("⚙️", "Best Hyperparameters",
#                 "XGBoost — GridSearchCV with 5-fold StratifiedKFold · scoring=F1")
#         params = {}
#         if ok_m:
#             raw_params = m_info.get("best_params", {})
#             if raw_params:
#                 params = {k.replace("model__", ""): v for k, v in raw_params.items()}
#         if not params:
#             params = {
#                 "colsample_bytree": 0.8,
#                 "gamma":            0,
#                 "learning_rate":    0.05,
#                 "max_depth":        3,
#                 "n_estimators":     200,
#                 "reg_lambda":       1,
#                 "subsample":        0.8,
#             }
#         rows_p = "".join(
#             f'<div style="display:flex;justify-content:space-between;align-items:center;'
#             f'padding:8px 0;border-bottom:1px solid {BDR}38;">'
#             f'<span style="font-family:JetBrains Mono,monospace;font-size:.79rem;'
#             f'color:{CYAN};">{k}</span>'
#             f'<span style="font-size:.79rem;color:{TEXT};font-weight:600;">{v}</span>'
#             f'</div>'
#             for k, v in params.items()
#         )
#         md(f'<div style="background:{CARD};border:1px solid {BDR};'
#            f'border-radius:10px;padding:13px 15px;">{rows_p}</div>')

#     # Business Insights
#     divider()
#     sec_hdr("💡", "Key Business Insights",
#             "Actionable retention strategies from EDA and model feature importance")

#     insights_data = [
#         ("📅", "Contract Type",
#          "Month-to-month customers churn at 42.7% vs only 2.8% on 2-year plans. "
#          "Offer discounts to upgrade contract length.",
#          RED),
#         ("🆕", "New Customers",
#          "First 12 months = 47.7% churn rate — highest-risk window. "
#          "Invest heavily in onboarding and early engagement.",
#          AMBER),
#         ("💸", "Monthly Charges",
#          "Bills over $70/month strongly correlate with churn. "
#          "Introduce loyalty pricing and value bundles for high-payers.",
#          AMBER),
#         ("👴", "Senior Citizens",
#          "Senior customers churn at 41.7% vs 24.3% for non-seniors. "
#          "Dedicated support and simplified plans improve retention.",
#          RED),
#         ("💳", "Payment Method",
#          "Electronic check = 45.3% churn — highest of all methods. "
#          "Incentivise migration to automatic payment.",
#          RED),
#         ("🌐", "Fiber Optic",
#          "Fiber optic churn: 41.9% vs DSL 18.9%. "
#          "Price-to-value perception drives dissatisfaction — target with loyalty offers.",
#          AMBER),
#     ]
#     for row_g in [insights_data[:3], insights_data[3:]]:
#         rc = st.columns(3, gap="medium")
#         for col, (ic, ti, de, ac) in zip(rc, row_g):
#             col.markdown(insight_card(ic, ti, de, ac), unsafe_allow_html=True)
#         md("<br>")


# # ─────────────────────────────────────────────
# # FOOTER
# # ─────────────────────────────────────────────
# md(f'<div style="margin-top:52px;padding:18px 0;border-top:1px solid {BDR};'
#    f'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">'
#    f'<div style="display:flex;align-items:center;gap:9px;">'
#    f'<div style="width:28px;height:28px;border-radius:8px;background:{BLUE};'
#    f'display:flex;align-items:center;justify-content:center;font-size:.88rem;">📡</div>'
#    f'<span style="font-family:Space Grotesk,sans-serif;font-size:.88rem;'
#    f'font-weight:700;color:{TEXT};">TeleChurn AI</span>'
#    f'<span style="font-size:.76rem;color:{MUTED};">· Telco Customer Intelligence</span>'
#    f'</div>'
#    f'<div style="font-size:.7rem;color:{MUTED};">'
#    f'XGBoost · SMOTE · GridSearchCV · Scikit-learn · Plotly · Streamlit · Python'
#    f'</div></div>')



# -------------------new----------------------------
# """
# TeleChurn AI — app.py (v6 · Final · Error-Free)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# pip install streamlit scikit-learn xgboost pandas numpy plotly
# Place PKL files in ./models/ and dataset.csv alongside app.py
# streamlit run app.py

# Bugs fixed vs v5:
#  ✅ st.stop() inside tabs blocks all later tabs — replaced with if/else
#  ✅ st.columns(0) crash when n_cols=0
#  ✅ Heatmap shows "nan" text — masked properly
#  ✅ SeniorCitizen (int) plotted as string on x-axis
#  ✅ select_dtypes deprecation warning fixed
#  ✅ All f-strings validated, no nested quote conflicts
#  ✅ Risk-summary safe for any combination of inputs
# """

# import pickle
# import warnings
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import streamlit as st
# from sklearn.preprocessing import LabelEncoder

# warnings.filterwarnings("ignore")
# #----------------------------------------------
# # ── FORCE DARK TABLE THEME (STREAMLIT CLOUD FIX) ──
# st.markdown("""
# <style>

# /* dataframe container */
# [data-testid="stDataFrame"] {
#     background-color: #0f172a;
#     border-radius: 10px;
# }

# /* table header */
# [data-testid="stDataFrame"] thead tr th {
#     background-color: #111827 !important;
#     color: #e5e7eb !important;
#     font-weight: 600;
# }

# /* table cells */
# [data-testid="stDataFrame"] tbody tr td {
#     background-color: #0f172a !important;
#     color: #e5e7eb !important;
# }

# /* row hover */
# [data-testid="stDataFrame"] tbody tr:hover td {
#     background-color: #1f2937 !important;
# }

# /* expander dataframe fix */
# [data-testid="stExpander"] [data-testid="stDataFrame"] {
#     background-color: #0f172a;
# }

# </style>
# """, unsafe_allow_html=True)
# #----------------------------------------------

# # ─────────────────────────────────────────────
# # PAGE CONFIG
# # ─────────────────────────────────────────────
# st.set_page_config(
#     page_title="TeleChurn AI",
#     page_icon="📡",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ─────────────────────────────────────────────
# # COLOR SYSTEM
# # ─────────────────────────────────────────────
# BG    = "#0F1117"
# CARD  = "#161B27"
# CARD2 = "#1C2333"
# BDR   = "#2A3347"
# BDR2  = "#374357"
# TEXT  = "#E8EDF5"
# SUB   = "#8B97B0"
# MUTED = "#4A5568"

# BLUE   = "#4F8EF7"
# INDIGO = "#7C6FF7"
# CYAN   = "#22D3EE"
# GREEN  = "#22C55E"
# AMBER  = "#F59E0B"
# RED    = "#F43F5E"
# PURPLE = "#A855F7"
# TEAL   = "#14B8A6"

# # ─────────────────────────────────────────────
# # CSS
# # ─────────────────────────────────────────────
# st.markdown(f"""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
# *,*::before,*::after{{box-sizing:border-box;margin:0;}}
# html,body,.stApp{{background:{BG}!important;color:{TEXT}!important;font-family:'Inter',sans-serif!important;}}
# #MainMenu,footer,header,[data-testid="stToolbar"]{{display:none!important;}}
# .block-container{{padding:0 2rem 5rem!important;max-width:1420px!important;}}

# [data-testid="stSidebar"]{{background:{CARD}!important;border-right:1px solid {BDR}!important;}}
# [data-testid="stSidebar"]>div:first-child{{padding-top:0!important;}}
# [data-testid="stSidebarNav"]{{display:none!important;}}

# .stTabs [data-baseweb="tab-list"]{{background:{CARD};border:1px solid {BDR};border-radius:12px;padding:4px;gap:2px;margin-bottom:24px;}}
# .stTabs [data-baseweb="tab"]{{background:transparent!important;color:{MUTED}!important;border-radius:9px!important;padding:9px 24px!important;font-family:'Space Grotesk',sans-serif!important;font-size:.84rem!important;font-weight:600!important;border:none!important;transition:all .18s!important;}}
# .stTabs [aria-selected="true"]{{background:{BLUE}!important;color:#fff!important;box-shadow:0 2px 12px {BLUE}30!important;}}

# div[data-baseweb="select"]>div{{background:{CARD2}!important;border:1px solid {BDR}!important;border-radius:8px!important;color:{TEXT}!important;}}
# div[data-baseweb="select"] *{{color:{TEXT}!important;}}
# div[data-baseweb="popover"]{{background:{CARD2}!important;border:1px solid {BDR2}!important;border-radius:10px!important;}}
# div[data-baseweb="popover"] li:hover{{background:{BDR}!important;}}
# .stNumberInput input,.stTextInput input,.stTextArea textarea{{background:{CARD2}!important;border:1px solid {BDR}!important;border-radius:8px!important;color:{TEXT}!important;font-size:.88rem!important;}}
# .stSlider>div>div>div>div{{background:{BLUE}!important;}}

# div.stButton>button{{background:{BLUE}!important;color:#fff!important;border:none!important;border-radius:10px!important;padding:14px 0!important;font-family:'Space Grotesk',sans-serif!important;font-weight:700!important;font-size:1rem!important;width:100%!important;letter-spacing:.3px!important;box-shadow:0 4px 20px {BLUE}40!important;transition:all .2s ease!important;}}
# div.stButton>button:hover{{background:#6BA3F9!important;box-shadow:0 8px 30px {BLUE}55!important;transform:translateY(-1px)!important;}}

# [data-testid="stSuccess"]{{background:#0A1F14!important;border:1px solid {GREEN}40!important;border-radius:10px!important;}}
# [data-testid="stError"]{{background:#1F0A11!important;border:1px solid {RED}40!important;border-radius:10px!important;}}
# [data-testid="stWarning"]{{background:#1A1305!important;border:1px solid {AMBER}40!important;border-radius:10px!important;}}
# [data-testid="stInfo"]{{background:#080F1F!important;border:1px solid {BLUE}40!important;border-radius:10px!important;}}
# [data-testid="stExpander"]{{background:{CARD}!important;border:1px solid {BDR}!important;border-radius:10px!important;}}
# details summary{{color:{TEXT}!important;}}

# ::-webkit-scrollbar{{width:4px;height:4px;}}
# ::-webkit-scrollbar-track{{background:{BG};}}
# ::-webkit-scrollbar-thumb{{background:{BDR2};border-radius:4px;}}
# hr{{border-color:{BDR}!important;margin:1.4rem 0!important;}}

# @keyframes pdot{{0%,100%{{opacity:1;}}50%{{opacity:.2;}}}}
# .pdot{{display:inline-block;width:7px;height:7px;border-radius:50%;background:{GREEN};animation:pdot 2s infinite;box-shadow:0 0 6px {GREEN};margin-right:6px;vertical-align:middle;}}
# @keyframes sup{{from{{opacity:0;transform:translateY(14px);}}to{{opacity:1;transform:translateY(0);}}}}
# .sup{{animation:sup .4s cubic-bezier(.22,1,.36,1) both;}}
# </style>
# """, unsafe_allow_html=True)


# # ─────────────────────────────────────────────
# # PLOTLY BASE LAYOUT
# # ─────────────────────────────────────────────
# def plotly_layout(title="", height=360):
#     return dict(
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         font=dict(family="Inter", color=SUB, size=11),
#         margin=dict(l=8, r=8, t=44, b=8),
#         height=height,
#         title=dict(
#             text=title,
#             font=dict(family="Space Grotesk", size=14, color=TEXT),
#             x=0, xanchor="left",
#         ),
#         xaxis=dict(gridcolor=BDR, linecolor=BDR, zerolinecolor=BDR),
#         yaxis=dict(gridcolor=BDR, linecolor=BDR, zerolinecolor=BDR),
#         legend=dict(
#             bgcolor="rgba(22,27,39,0.85)", bordercolor=BDR,
#             borderwidth=1, font=dict(size=11, color=SUB),
#         ),
#         colorway=[BLUE, CYAN, GREEN, AMBER, RED, PURPLE, TEAL],
#     )


# # ─────────────────────────────────────────────
# # HTML HELPERS  (no nested quote conflicts)
# # ─────────────────────────────────────────────
# def md(html):
#     st.markdown(html, unsafe_allow_html=True)

# def divider():
#     md("<hr/>")

# def sec_hdr(icon, title, sub=""):
#     sub_part = (f'<div style="font-size:.77rem;color:{SUB};margin-top:3px;">{sub}</div>'
#                 if sub else "")
#     md(f"""
#     <div style="display:flex;align-items:flex-start;gap:12px;margin:30px 0 16px;">
#       <div style="width:37px;height:37px;flex-shrink:0;border-radius:10px;
#         background:linear-gradient(135deg,{BLUE}22,{INDIGO}22);
#         border:1px solid {BLUE}35;display:flex;align-items:center;
#         justify-content:center;font-size:1.05rem;">{icon}</div>
#       <div>
#         <div style="font-family:'Space Grotesk',sans-serif;font-size:.98rem;
#           font-weight:700;color:{TEXT};">{title}</div>
#         {sub_part}
#       </div>
#     </div>""")

# def kpi(icon, label, value, accent, note=""):
#     note_part = (f'<div style="font-size:.69rem;color:{SUB};margin-top:3px;">{note}</div>'
#                  if note else "")
#     return f"""<div style="background:{CARD};border:1px solid {BDR};border-radius:12px;
#       padding:17px;position:relative;overflow:hidden;">
#       <div style="position:absolute;top:0;right:0;width:70px;height:70px;
#         background:radial-gradient(circle,{accent}18,transparent 70%);pointer-events:none;"></div>
#       <div style="font-size:1.2rem;margin-bottom:7px;">{icon}</div>
#       <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;
#         font-weight:700;color:{accent};line-height:1;margin-bottom:4px;">{value}</div>
#       <div style="font-size:.63rem;color:{MUTED};text-transform:uppercase;
#         letter-spacing:1px;font-weight:600;">{label}</div>
#       {note_part}</div>"""

# def badge(text, color):
#     return (f'<span style="background:{color}18;border:1px solid {color}35;'
#             f'color:{color};border-radius:6px;padding:3px 12px;font-size:.72rem;'
#             f'font-family:JetBrains Mono,monospace;font-weight:500;">{text}</span>')

# def insight_card(icon, title, desc, accent):
#     return f"""<div style="background:{CARD};border:1px solid {accent}30;
#       border-left:3px solid {accent};border-radius:10px;padding:18px;height:100%;">
#       <div style="font-size:1.2rem;margin-bottom:9px;">{icon}</div>
#       <div style="font-family:'Space Grotesk',sans-serif;font-size:.87rem;
#         font-weight:700;color:{accent};margin-bottom:6px;">{title}</div>
#       <div style="font-size:.79rem;color:{SUB};line-height:1.7;">{desc}</div>
#     </div>"""

# def signal_row(label, rate, accent):
#     return f"""<div style="display:flex;justify-content:space-between;align-items:center;
#       background:{CARD};border:1px solid {BDR};border-left:3px solid {accent};
#       border-radius:8px;padding:9px 14px;margin-bottom:7px;">
#       <span style="font-size:.82rem;color:{SUB};">{label}</span>
#       <span style="font-size:.82rem;font-weight:700;color:{accent};">{rate}</span>
#     </div>"""

# def risk_card_html(col_c, col_lbl, title_f, desc_f, note_f):
#     """Build risk factor card HTML without nested f-string quote issues."""
#     return (
#         f'<div style="background:{col_c}0C;border:1px solid {col_c}30;'
#         f'border-left:3px solid {col_c};border-radius:10px;'
#         f'padding:14px 16px;margin-bottom:10px;">'
#         f'<div style="font-size:.63rem;font-weight:700;color:{col_c};'
#         f'text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">'
#         f'{col_lbl}</div>'
#         f'<div style="font-size:.82rem;font-weight:600;color:{TEXT};margin-bottom:4px;">'
#         f'{title_f}</div>'
#         f'<div style="font-size:.76rem;color:{SUB};margin-bottom:3px;">{desc_f}</div>'
#         f'<div style="font-size:.7rem;color:{MUTED};font-style:italic;">{note_f}</div>'
#         f'</div>'
#     )


# # ─────────────────────────────────────────────
# # DATA LOADING
# # ─────────────────────────────────────────────
# MODELS_DIR = Path("models")

# @st.cache_resource(show_spinner="Loading model…")
# def load_artifacts():
#     needed = {
#         "model":  MODELS_DIR / "model.pkl",
#         "scaler": MODELS_DIR / "scaler.pkl",
#         "enc":    MODELS_DIR / "label_encoders.pkl",
#         "cols":   MODELS_DIR / "feature_columns.pkl",
#         "info":   MODELS_DIR / "model_info.pkl",
#     }
#     miss = [p.name for p in needed.values() if not p.exists()]
#     if miss:
#         return None, miss
#     out = {}
#     for k, p in needed.items():
#         with open(p, "rb") as f:
#             out[k] = pickle.load(f)
#     return out, []

# @st.cache_data(show_spinner="Loading dataset…")
# def load_data():
#     p = Path("dataset.csv")
#     if not p.exists():
#         return None
#     df = pd.read_csv(p)
#     df["TotalCharges"] = pd.to_numeric(
#         df["TotalCharges"].astype(str).str.strip().replace({"": "0", " ": "0"}),
#         errors="coerce",
#     ).fillna(0.0)
#     df["ChurnBin"] = (df["Churn"] == "Yes").astype(int)
#     # bins include 0 explicitly so no NaN in TenureGroup
#     df["TenureGroup"] = pd.cut(
#         df["tenure"],
#         bins=[-1, 12, 24, 48, 72],
#         labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"],
#     )
#     return df

# arts, miss_files = load_artifacts()
# df   = load_data()
# ok_m = arts is not None
# ok_d = df is not None

# m_info    = arts["info"] if ok_m else {}
# model_lbl = m_info.get("model_name", "XGBoost") if ok_m else "XGBoost"
# threshold = float(m_info.get("threshold", 0.4828)) if ok_m else 0.4828

# # Verified final metrics (from notebook output)
# ACC  = m_info.get("accuracy",  0.7757) if ok_m else 0.7757
# PREC = m_info.get("precision", 0.5612) if ok_m else 0.5612
# REC  = m_info.get("recall",    0.7112) if ok_m else 0.7112
# F1   = m_info.get("f1_score",  0.6274) if ok_m else 0.6274
# AUC  = m_info.get("roc_auc",   0.8434) if ok_m else 0.8434


# # ─────────────────────────────────────────────
# # SIDEBAR
# # ─────────────────────────────────────────────
# with st.sidebar:
#     md(f"""<div style="padding:20px 16px 0;">
#       <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
#         <div style="width:33px;height:33px;border-radius:9px;background:{BLUE};
#           display:flex;align-items:center;justify-content:center;
#           font-size:.95rem;flex-shrink:0;">📡</div>
#         <div>
#           <div style="font-family:'Space Grotesk',sans-serif;font-size:.93rem;
#             font-weight:700;color:{TEXT};">TeleChurn AI</div>
#           <div style="font-size:.62rem;color:{MUTED};letter-spacing:.5px;">
#             ANALYTICS PLATFORM</div>
#         </div>
#       </div>
#       <div style="height:1px;background:{BDR};margin-bottom:16px;"></div>
#     </div>""")

#     # Status
#     mc = GREEN if ok_m else RED
#     dc = GREEN if ok_d else AMBER
#     md(f"""<div style="padding:0 16px;">
#       <div style="font-size:.62rem;font-weight:700;color:{MUTED};letter-spacing:1.1px;
#         text-transform:uppercase;margin-bottom:8px;">System Status</div>
#       <div style="background:{CARD2};border:1px solid {mc}30;border-radius:8px;
#         padding:9px 12px;margin-bottom:5px;display:flex;align-items:center;gap:7px;">
#         <span class="pdot" style="background:{mc};box-shadow:0 0 6px {mc};"></span>
#         <span style="font-size:.77rem;color:{mc};font-weight:600;">
#           {"Model Ready" if ok_m else "Model Missing — add models/"}</span>
#       </div>
#       <div style="background:{CARD2};border:1px solid {dc}30;border-radius:8px;
#         padding:9px 12px;margin-bottom:16px;display:flex;align-items:center;gap:7px;">
#         <span style="width:7px;height:7px;border-radius:50%;
#           background:{dc};flex-shrink:0;display:inline-block;"></span>
#         <span style="font-size:.77rem;color:{dc};font-weight:600;">
#           {"Dataset Loaded" if ok_d else "No dataset.csv found"}</span>
#       </div>
#     </div>""")

#     # Model info
#     if ok_m:
#         md(f"""<div style="padding:0 16px;">
#           <div style="font-size:.62rem;font-weight:700;color:{MUTED};letter-spacing:1.1px;
#             text-transform:uppercase;margin-bottom:8px;">Best Model</div>
#           <div style="background:{CARD2};border:1px solid {BDR};border-radius:8px;
#             padding:12px;margin-bottom:16px;">
#             <div style="display:flex;justify-content:space-between;margin-bottom:7px;">
#               <span style="font-size:.71rem;color:{MUTED};">Algorithm</span>
#               <span style="font-family:'JetBrains Mono',monospace;font-size:.79rem;
#                 color:{BLUE};font-weight:600;">{model_lbl}</span>
#             </div>
#             <div style="height:1px;background:{BDR};margin:7px 0;"></div>
#             <div style="display:flex;justify-content:space-between;">
#               <span style="font-size:.71rem;color:{MUTED};">Threshold</span>
#               <span style="font-family:'JetBrains Mono',monospace;font-size:.79rem;
#                 color:{AMBER};font-weight:600;">{threshold:.4f}</span>
#             </div>
#           </div>
#         </div>""")

#     # Stats
#     _n  = f"{len(df):,}" if ok_d else "7,043"
#     _ch = f"{df['ChurnBin'].sum():,}" if ok_d else "1,869"
#     _cr = f"{df['ChurnBin'].mean()*100:.1f}%" if ok_d else "26.5%"
#     stats = [
#         ("Customers", _n, CYAN),
#         ("Churned", _ch, RED),
#         ("Churn Rate", _cr, AMBER),
#         ("Best AUC-ROC", f"{AUC:.4f}", GREEN),
#         ("Best F1", f"{F1:.4f}", TEAL),
#     ]
#     md(f"""<div style="padding:0 16px;">
#       <div style="font-size:.62rem;font-weight:700;color:{MUTED};letter-spacing:1.1px;
#         text-transform:uppercase;margin-bottom:8px;">Dataset Stats</div>""")
#     for lb, vl, ac in stats:
#         md(f"""<div style="background:{CARD2};border-radius:7px;padding:8px 10px;
#           display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
#           <span style="font-size:.73rem;color:{MUTED};">{lb}</span>
#           <span style="font-family:'JetBrains Mono',monospace;font-size:.77rem;
#             color:{ac};font-weight:600;">{vl}</span>
#         </div>""")
#     md("</div>")

#     md(f"""<div style="padding:14px 16px;">
#       <div style="height:1px;background:{BDR};margin-bottom:12px;"></div>
#       <div style="font-size:.69rem;color:{MUTED};line-height:1.8;text-align:center;">
#         XGBoost · SMOTE · GridSearchCV<br>Scikit-learn · Plotly · Streamlit
#       </div>
#     </div>""")


# # ─────────────────────────────────────────────
# # HERO
# # ─────────────────────────────────────────────
# _nc = f"{len(df):,}" if ok_d else "7,043"
# badges_html = "  ".join([
#     badge("SMOTE Balanced", GREEN),
#     badge("3 Models Tuned", INDIGO),
#     badge(_nc + " Customers", CYAN),
#     badge(f"Threshold {threshold:.4f}", AMBER),
#     badge("AUC 0.8434", BLUE),
# ])
# md(f"""<div style="padding:36px 0 22px;">
#   <div style="margin-bottom:14px;">
#     <span style="display:inline-flex;align-items:center;gap:5px;
#       background:{CARD};border:1px solid {BDR};border-radius:50px;
#       padding:4px 14px;font-size:.69rem;font-family:'JetBrains Mono',monospace;color:{MUTED};">
#       <span class="pdot"></span>LIVE ANALYTICS ENGINE
#     </span>
#   </div>
#   <h1 style="font-family:'Space Grotesk',sans-serif;font-size:2.4rem;
#     font-weight:700;color:{TEXT};line-height:1.12;margin-bottom:8px;">
#     Telco Customer
#     <span style="color:{BLUE};">Churn Intelligence</span>
#   </h1>
#   <p style="font-size:.9rem;color:{SUB};line-height:1.7;max-width:500px;margin-bottom:18px;">
#     AI-powered churn prediction · XGBoost + SMOTE + GridSearchCV · Optimal threshold calibration
#   </p>
#   <div style="display:flex;gap:8px;flex-wrap:wrap;">{badges_html}</div>
# </div>""")

# # KPI strip
# _nr  = f"{len(df):,}" if ok_d else "7,043"
# _nch = f"{df['ChurnBin'].sum():,}" if ok_d else "1,869"
# kc   = st.columns(5, gap="small")
# for col, (ic, lb, vl, ac, nt) in zip(kc, [
#     ("📡", "Model",     "Ready" if ok_m else "Missing", BLUE,  model_lbl if ok_m else "Add models/"),
#     ("🗄️", "Dataset",  "Loaded" if ok_d else "Optional", CYAN, f"{_nr} rows" if ok_d else "Add dataset.csv"),
#     ("👥", "Customers", _nr,                             GREEN, "total records"),
#     ("📉", "Churned",   _nch,                            RED,   "26.5% of total"),
#     ("🎯", "Threshold", f"{threshold:.4f}",              AMBER, "optimal F1 cut-off"),
# ]):
#     col.markdown(kpi(ic, lb, vl, ac, nt), unsafe_allow_html=True)

# md("<br>")

# # ─────────────────────────────────────────────
# # TABS
# # ─────────────────────────────────────────────
# TAB1, TAB2, TAB3 = st.tabs([
#     "🔮  Predict Churn",
#     "📊  EDA Dashboard",
#     "🤖  Model Insights",
# ])


# # ═══════════════════════════════════════════
# # TAB 1 — PREDICT
# # ═══════════════════════════════════════════
# with TAB1:
#     if not ok_m:
#         st.error(
#             f"**Model files not found.**  Missing: `{'`, `'.join(miss_files)}`  \n"
#             "Run the notebook end-to-end to generate PKL files, "
#             "then place them in a `models/` subfolder next to app.py."
#         )
#         md(f"""<div style="background:{CARD};border:1px dashed {BDR2};border-radius:14px;
#           padding:48px 24px;text-align:center;margin-top:12px;">
#           <div style="font-size:2.5rem;margin-bottom:12px;">📁</div>
#           <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;
#             font-weight:700;color:{TEXT};margin-bottom:8px;">Models Not Found</div>
#           <div style="font-size:.84rem;color:{SUB};line-height:1.7;">
#             Required: model.pkl · scaler.pkl · label_encoders.pkl
#             · feature_columns.pkl · model_info.pkl
#           </div>
#         </div>""")
#     else:
#         sec_hdr("👤", "Customer Profile",
#                 "Fill in all fields below, then click Run Prediction")

#         # CA, CB, CC = st.columns(3, gap="large")

#         # with CA:
#         #     md(f'<div style="font-size:.68rem;font-weight:700;color:{CYAN};'
#         #        f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#         #        f'Demographics</div>')
#         #     gender     = st.selectbox("Gender",         ["Male", "Female"])
#         #     senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
#         #     partner    = st.selectbox("Partner",        ["No", "Yes"])
#         #     dependents = st.selectbox("Dependents",     ["No", "Yes"])
#         #     tenure     = st.slider("Tenure (months)", 0, 72, 12)

#         # with CB:
#         #     md(f'<div style="font-size:.68rem;font-weight:700;color:{PURPLE};'
#         #        f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#         #        f'Services</div>')
#         #     phone_svc = st.selectbox("Phone Service", ["Yes", "No"])
#         #     if phone_svc == "No":
#         #         multi_lines = "No phone service"
#         #         st.selectbox("Multiple Lines", ["No phone service"], disabled=True)
#         #     else:
#         #         multi_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

#         #     internet_svc = st.selectbox("Internet Service",
#         #                                 ["DSL", "Fiber optic", "No"])
#         #     _io  = ["No internet service"] if internet_svc == "No" else ["No", "Yes"]
#         #     _dis = internet_svc == "No"

#         #     online_sec   = st.selectbox("Online Security",   _io, disabled=_dis)
#         #     online_bkp   = st.selectbox("Online Backup",     _io, disabled=_dis)
#         #     dev_protect  = st.selectbox("Device Protection", _io, disabled=_dis)
#         #     tech_support = st.selectbox("Tech Support",      _io, disabled=_dis)
#         #     stream_tv    = st.selectbox("Streaming TV",      _io, disabled=_dis)
#         #     stream_mov   = st.selectbox("Streaming Movies",  _io, disabled=_dis)

#         # with CC:
#         #     md(f'<div style="font-size:.68rem;font-weight:700;color:{TEAL};'
#         #        f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#         #        f'Billing &amp; Contract</div>')
#         #     contract  = st.selectbox("Contract Type",
#         #                              ["Month-to-month", "One year", "Two year"])
#         #     paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
#         #     payment   = st.selectbox("Payment Method", [
#         #         "Electronic check", "Mailed check",
#         #         "Bank transfer (automatic)", "Credit card (automatic)",
#         #     ])
#         #     monthly  = st.number_input("Monthly Charges ($)",
#         #                                18.0, 120.0, 65.0, 0.5, format="%.2f")
#         #     def_tot  = min(float(monthly) * max(int(tenure), 1), 8684.8)
#         #     total    = st.number_input("Total Charges ($)",
#         #                                0.0, 8684.8, def_tot, 10.0, format="%.2f")

#         #==================================
#         CA, CB, CC = st.columns(3, gap="large")

#         with CA:
#             md(f'<div style="font-size:.68rem;font-weight:700;color:{CYAN};'
#                f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#                f'Demographics</div>')
#             gender     = st.selectbox("Gender",         ["Male", "Female"], index=0) # Default: Male
#             senior     = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)      # Default: No (0)
#             partner    = st.selectbox("Partner",        ["No", "Yes"], index=1)      # Default: Yes
#             dependents = st.selectbox("Dependents",     ["No", "Yes"], index=0)      # Default: No
#             tenure     = st.slider("Tenure (months)", 0, 72, 12)                     # Default: 12

#         with CB:
#             md(f'<div style="font-size:.68rem;font-weight:700;color:{PURPLE};'
#                f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#                f'Services</div>')
#             phone_svc = st.selectbox("Phone Service", ["No", "Yes"], index=1)        # Default: Yes
#             if phone_svc == "No":
#                 multi_lines = "No phone service"
#                 st.selectbox("Multiple Lines", ["No phone service"], disabled=True)
#             else:
#                 multi_lines = st.selectbox("Multiple Lines", ["No", "Yes"], index=0) # Default: No

#             internet_svc = st.selectbox("Internet Service",
#                                         ["DSL", "Fiber optic", "No"], index=1)       # Default: Fiber optic
#             _io  = ["No internet service"] if internet_svc == "No" else ["No", "Yes"]
#             _dis = internet_svc == "No"

#             online_sec   = st.selectbox("Online Security",   _io, index=0, disabled=_dis) # Default: No
#             online_bkp   = st.selectbox("Online Backup",     _io, index=0, disabled=_dis) # Default: No
#             dev_protect  = st.selectbox("Device Protection", _io, index=0, disabled=_dis) # Default: No
#             tech_support = st.selectbox("Tech Support",      _io, index=0, disabled=_dis) # Default: No
#             stream_tv    = st.selectbox("Streaming TV",      _io, index=1, disabled=_dis) # Default: Yes
#             stream_mov   = st.selectbox("Streaming Movies",  _io, index=1, disabled=_dis) # Default: Yes

#         with CC:
#             md(f'<div style="font-size:.68rem;font-weight:700;color:{TEAL};'
#                f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#                f'Billing &amp; Contract</div>')
#             contract  = st.selectbox("Contract Type",
#                                      ["Month-to-month", "One year", "Two year"], index=0) # Default: Month-to-month
#             paperless = st.selectbox("Paperless Billing", ["No", "Yes"], index=1)         # Default: Yes
#             payment   = st.selectbox("Payment Method", [
#                 "Bank transfer (automatic)", "Credit card (automatic)", 
#                 "Electronic check", "Mailed check"
#             ], index=2)                                                                   # Default: Electronic check
#             monthly  = st.number_input("Monthly Charges ($)",
#                                        18.0, 120.0, 70.35, 0.5, format="%.2f")            # Default: 70.35
#             total    = st.number_input("Total Charges ($)",
#                                        0.0, 8684.8, 843.0, 10.0, format="%.2f")           # Default: 843.0
        
#         md("<br>")
#         b1, b2, b3 = st.columns([1, 3, 1])
#         with b2:
#             clicked = st.button("🔮  RUN CHURN PREDICTION", use_container_width=True)

#         # ── Inference ──────────────────────────────────────────────
#         if clicked:
#             raw = {
#                 "gender":           gender,
#                 "SeniorCitizen":    1 if senior == "Yes" else 0,
#                 "Partner":          partner,
#                 "Dependents":       dependents,
#                 "tenure":           int(tenure),
#                 "PhoneService":     phone_svc,
#                 "MultipleLines":    multi_lines,
#                 "InternetService":  internet_svc,
#                 "OnlineSecurity":   online_sec   if internet_svc != "No" else "No internet service",
#                 "OnlineBackup":     online_bkp   if internet_svc != "No" else "No internet service",
#                 "DeviceProtection": dev_protect  if internet_svc != "No" else "No internet service",
#                 "TechSupport":      tech_support if internet_svc != "No" else "No internet service",
#                 "StreamingTV":      stream_tv    if internet_svc != "No" else "No internet service",
#                 "StreamingMovies":  stream_mov   if internet_svc != "No" else "No internet service",
#                 "Contract":         contract,
#                 "PaperlessBilling": paperless,
#                 "PaymentMethod":    payment,
#                 "MonthlyCharges":   float(monthly),
#                 "TotalCharges":     float(total),
#             }
            
#             try:
#                 inp = pd.DataFrame([raw])
                
#                 # 1. Safely label encode
#                 for col_name, le in arts["enc"].items():
#                     if col_name in inp.columns:
#                         v = str(inp.at[0, col_name])
#                         if v in le.classes_:
#                             inp.at[0, col_name] = le.transform([v])[0]
#                         else:
#                             inp.at[0, col_name] = 0  # Fallback safety
                            
#                 # 2. Enforce column order and float type
#                 inp = inp[arts["cols"]].astype(float)
                
#                 # 3. Scale features (returns a NumPy array)
#                 inp_sc_array = arts["scaler"].transform(inp)
                
#                 # 4. FIXED: Re-wrap in DataFrame so XGBoost recognizes feature names!
#                 inp_sc = pd.DataFrame(inp_sc_array, columns=arts["cols"])

#                 # 5. Predict
#                 proba   = arts["model"].predict_proba(inp_sc)[0]
#                 churn_p = float(proba[1]) * 100
#                 stay_p  = float(proba[0]) * 100
#                 pred    = int((churn_p / 100) >= threshold)

#                 if churn_p >= 65:
#                     tier_col, tier_lbl = RED,   "HIGH RISK"
#                 elif churn_p >= 35:
#                     tier_col, tier_lbl = AMBER, "MEDIUM RISK"
#                 else:
#                     tier_col, tier_lbl = GREEN, "LOW RISK"

#                 vcol = RED if pred else GREEN
#                 vtxt = "WILL LIKELY CHURN" if pred else "LIKELY TO STAY"
#                 vico = "⚠️" if pred else "✅"
#                 vbg  = "#1A0810" if pred else "#071510"

#                 divider()
#                 sec_hdr("📈", "Prediction Result",
#                         f"Model: {model_lbl}  ·  Threshold: {threshold:.4f}")

#                 R1, R2, R3 = st.columns([1.1, 1, 1], gap="large")

            
#            #===================
#             # try:
#             #     inp = pd.DataFrame([raw])
#             #     for col_name, le in arts["enc"].items():
#             #         if col_name in inp.columns:
#             #             v = str(inp.at[0, col_name])
#             #             inp[col_name] = le.transform([v]) if v in le.classes_ else [0]
#             #     inp    = inp[arts["cols"]].astype(float)
#             #     inp_sc = arts["scaler"].transform(inp)

#             #     proba   = arts["model"].predict_proba(inp_sc)[0]
#             #     churn_p = float(proba[1]) * 100
#             #     stay_p  = float(proba[0]) * 100
#             #     pred    = int((churn_p / 100) >= threshold)

#             #     if churn_p >= 65:
#             #         tier_col, tier_lbl = RED,   "HIGH RISK"
#             #     elif churn_p >= 35:
#             #         tier_col, tier_lbl = AMBER, "MEDIUM RISK"
#             #     else:
#             #         tier_col, tier_lbl = GREEN, "LOW RISK"

#             #     vcol = RED if pred else GREEN
#             #     vtxt = "WILL LIKELY CHURN" if pred else "LIKELY TO STAY"
#             #     vico = "⚠️" if pred else "✅"
#             #     vbg  = "#1A0810" if pred else "#071510"

#             #     divider()
#             #     sec_hdr("📈", "Prediction Result",
#             #             f"Model: {model_lbl}  ·  Threshold: {threshold:.4f}")

#             #     R1, R2, R3 = st.columns([1.1, 1, 1], gap="large")

#                 # Verdict card
#                 with R1:
#                     md(f"""<div class="sup" style="background:{vbg};border:1px solid {vcol}30;
#                       border-radius:14px;padding:26px 20px;text-align:center;">
#                       <div style="font-size:.6rem;font-family:'JetBrains Mono',monospace;
#                         color:{vcol};letter-spacing:2.5px;margin-bottom:9px;">MODEL VERDICT</div>
#                       <div style="font-size:1.8rem;margin-bottom:6px;">{vico}</div>
#                       <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;
#                         font-weight:700;color:{TEXT};margin-bottom:16px;">{vtxt}</div>
#                       <div style="display:flex;justify-content:center;gap:22px;margin-bottom:14px;">
#                         <div>
#                           <div style="font-family:'Space Grotesk',sans-serif;font-size:2.1rem;
#                             font-weight:700;color:{RED};">{churn_p:.1f}%</div>
#                           <div style="font-size:.63rem;color:{MUTED};text-transform:uppercase;
#                             letter-spacing:.7px;margin-top:2px;">Churn Risk</div>
#                         </div>
#                         <div style="width:1px;background:{BDR};"></div>
#                         <div>
#                           <div style="font-family:'Space Grotesk',sans-serif;font-size:2.1rem;
#                             font-weight:700;color:{GREEN};">{stay_p:.1f}%</div>
#                           <div style="font-size:.63rem;color:{MUTED};text-transform:uppercase;
#                             letter-spacing:.7px;margin-top:2px;">Retention</div>
#                         </div>
#                       </div>
#                       <div style="display:inline-flex;align-items:center;gap:6px;
#                         background:{tier_col}18;border:1px solid {tier_col}40;
#                         border-radius:8px;padding:7px 14px;">
#                         <span style="font-size:.82rem;color:{tier_col};font-weight:700;">{tier_lbl}</span>
#                       </div>
#                     </div>""")

#                 # Gauge + bar
#                 # with R2:
#                 #     fig_g = go.Figure(go.Indicator(
#                 #         mode="gauge+number",
#                 #         value=churn_p,
#                 #         number=dict(suffix="%",
#                 #                     font=dict(size=26, color=tier_col,
#                 #                               family="Space Grotesk")),
#                 #         gauge=dict(
#                 #             axis=dict(range=[0, 100],
#                 #                       tickcolor=MUTED,
#                 #                       tickfont=dict(color=MUTED, size=9)),
#                 #             bar=dict(color=tier_col, thickness=0.62),
#                 #             bgcolor="rgba(0,0,0,0)",
#                 #             borderwidth=0,
#                 #             steps=[
#                 #                 dict(range=[0,  35], color="#071510"),
#                 #                 dict(range=[35, 65], color="#15110A"),
#                 #                 dict(range=[65,100], color="#150809"),
#                 #             ],
#                 #             threshold=dict(
#                 #                 line=dict(color=AMBER, width=2),
#                 #                 thickness=0.75,
#                 #                 value=threshold * 100,
#                 #             ),
#                 #         ),
#                 #     ))
#                 #     fig_g.update_layout(
#                 #         paper_bgcolor="rgba(0,0,0,0)",
#                 #         plot_bgcolor="rgba(0,0,0,0)",
#                 #         font=dict(color=SUB, family="Inter"),
#                 #         margin=dict(l=10, r=10, t=10, b=20),
#                 #         height=200,
#                 #         annotations=[dict(
#                 #             text=f"Decision threshold: {threshold*100:.1f}%",
#                 #             x=0.5, y=-0.18, showarrow=False,
#                 #             font=dict(color=MUTED, size=9),
#                 #         )],
#                 #     )
#                 #     st.plotly_chart(fig_g, use_container_width=True)

#                 #     fig_b = go.Figure()
#                 #     fig_b.add_trace(go.Bar(
#                 #         y=["Retention", "Churn Risk"],
#                 #         x=[stay_p, churn_p],
#                 #         orientation="h",
#                 #         marker_color=[GREEN, RED],
#                 #         marker_line_color="rgba(0,0,0,0)",
#                 #         text=[f"{stay_p:.1f}%", f"{churn_p:.1f}%"],
#                 #         textposition="outside",
#                 #         textfont=dict(color=TEXT, size=11),
#                 #     ))
#                 #     fig_b.add_vline(
#                 #         x=threshold * 100, line_dash="dash",
#                 #         line_color=AMBER, line_width=1.5,
#                 #         annotation_text=f"Threshold {threshold*100:.1f}%",
#                 #         annotation_font_color=AMBER,
#                 #         annotation_font_size=9,
#                 #         annotation_position="top right",
#                 #     )
#                 #     fig_b.update_layout(
#                 #         paper_bgcolor="rgba(0,0,0,0)",
#                 #         plot_bgcolor="rgba(0,0,0,0)",
#                 #         margin=dict(l=8, r=55, t=6, b=6),
#                 #         height=110,
#                 #         xaxis=dict(range=[0, 118], gridcolor=BDR,
#                 #                    tickfont=dict(color=MUTED), showticklabels=False),
#                 #         yaxis=dict(tickfont=dict(color=SUB, size=10),
#                 #                    gridcolor="rgba(0,0,0,0)"),
#                 #         showlegend=False,
#                 #     )
#                 #     st.plotly_chart(fig_b, use_container_width=True)
                
#                 #================================================
#                 # Gauge + bar
#                 with R2:
#                     fig_g = go.Figure(go.Indicator(
#                         mode="gauge+number",
#                         value=churn_p,
#                         title=dict(text="Confidence Level", font=dict(size=14, color=TEXT, family="Space Grotesk")),
#                         number=dict(suffix="%",
#                                     font=dict(size=32, color=tier_col,
#                                               family="Space Grotesk")),
#                         gauge=dict(
#                             axis=dict(range=[0, 100],
#                                       tickcolor=MUTED,
#                                       tickfont=dict(color=MUTED, size=10)),
#                             bar=dict(color=tier_col, thickness=0.7),
#                             bgcolor="rgba(0,0,0,0)",
#                             borderwidth=0,
#                             steps=[
#                                 dict(range=[0,  35], color=CARD2),
#                                 dict(range=[35, 65], color=BDR),
#                                 dict(range=[65,100], color=BDR2),
#                             ],
#                             threshold=dict(
#                                 line=dict(color=TEXT, width=2.5),
#                                 thickness=0.8,
#                                 value=threshold * 100,
#                             ),
#                         ),
#                     ))
#                     fig_g.update_layout(
#                         paper_bgcolor="rgba(0,0,0,0)",
#                         plot_bgcolor="rgba(0,0,0,0)",
#                         font=dict(color=SUB, family="Inter"),
#                         margin=dict(l=15, r=15, t=30, b=10),
#                         height=220,
#                     )
#                     st.plotly_chart(fig_g, use_container_width=True)

#                     fig_b = go.Figure()
#                     fig_b.add_trace(go.Bar(
#                         y=["Retention", "Churn Risk"],
#                         x=[stay_p, churn_p],
#                         orientation="h",
#                         marker_color=[GREEN, RED],
#                         marker_line_color="rgba(0,0,0,0)",
#                         text=[f"{stay_p:.1f}%", f"{churn_p:.1f}%"],
#                         textposition="outside",
#                         textfont=dict(color=TEXT, size=11, family="Space Grotesk"),
#                     ))
#                     fig_b.add_vline(
#                         x=threshold * 100, line_dash="dash",
#                         line_color=TEXT, line_width=1.5,
#                         annotation_text=f"Threshold {threshold*100:.1f}%",
#                         annotation_font_color=SUB,
#                         annotation_font_size=10,
#                         annotation_position="top right",
#                     )
#                     fig_b.update_layout(
#                         paper_bgcolor="rgba(0,0,0,0)",
#                         plot_bgcolor="rgba(0,0,0,0)",
#                         margin=dict(l=5, r=50, t=10, b=10),
#                         height=120,
#                         xaxis=dict(range=[0, 115], gridcolor="rgba(0,0,0,0)",
#                                    tickfont=dict(color=MUTED), showticklabels=False),
#                         yaxis=dict(tickfont=dict(color=TEXT, size=11),
#                                    gridcolor="rgba(0,0,0,0)"),
#                         showlegend=False,
#                     )
#                     st.plotly_chart(fig_b, use_container_width=True)
#                 #================================================

#                 # Actions
#                 with R3:
#                     if pred:
#                         actions = [
#                             ("📞", "Call within 24 hours",       RED),
#                             ("💰", "Offer personalised discount", AMBER),
#                             ("📋", "Propose annual contract",      BLUE),
#                             ("🛡️", "Add free Tech Support trial", PURPLE),
#                             ("🎁", "Send loyalty reward offer",   GREEN),
#                         ]
#                         hdr_col, hdr_txt = RED,   "🚨 Retention Actions"
#                     else:
#                         actions = [
#                             ("⭐", "Enrol in rewards programme",  GREEN),
#                             ("📦", "Upsell premium services",     TEAL),
#                             ("🔁", "Encourage contract upgrade",  BLUE),
#                             ("📣", "Request referral or review",  PURPLE),
#                             ("📊", "Monitor usage trends",        CYAN),
#                         ]
#                         hdr_col, hdr_txt = GREEN, "✅ Growth Actions"

#                     rows_h = "".join(
#                         f'<div style="display:flex;align-items:center;gap:9px;'
#                         f'padding:8px 10px;border-radius:7px;margin-bottom:4px;'
#                         f'background:{ac}0C;border:1px solid {ac}1A;">'
#                         f'<span style="font-size:.88rem;">{ico}</span>'
#                         f'<span style="font-size:.78rem;color:{SUB};">{txt}</span>'
#                         f'</div>'
#                         for ico, txt, ac in actions
#                     )
#                     md(f'<div style="background:{CARD};border:1px solid {BDR};'
#                        f'border-radius:13px;padding:16px;">'
#                        f'<div style="font-size:.74rem;font-weight:700;color:{hdr_col};'
#                        f'font-family:Space Grotesk,sans-serif;margin-bottom:12px;">'
#                        f'{hdr_txt}</div>{rows_h}</div>')

#                 # ── What Drives This Prediction ─────────────────────
#                 md("<br>")
#                 sec_hdr("🔍", "What Drives This Prediction?",
#                         "Risk factors identified from EDA patterns for this customer profile")

#                 # Build risk factors — simple lists, no complex unpacking
#                 high_risk, med_risk, low_risk = [], [], []

#                 if "Month-to-month" in contract:
#                     high_risk.append((
#                         "📅 Month-to-month Contract",
#                         "Historical churn rate: 42.7% — highest of all contract types",
#                         "vs only 2.8% churn on 2-year plans",
#                     ))
#                 if "Electronic" in payment:
#                     high_risk.append((
#                         "💳 Electronic Check Payment",
#                         "Historical churn rate: 45.3% — highest of all payment methods",
#                         "Customers on auto-pay churn significantly less",
#                     ))
#                 if internet_svc == "Fiber optic":
#                     high_risk.append((
#                         "🌐 Fiber Optic Internet",
#                         "Historical churn rate: 41.9% vs DSL at 18.9%",
#                         "Price-to-value perception is the key driver",
#                     ))
#                 if int(tenure) <= 12:
#                     high_risk.append((
#                         "🆕 New Customer (first 12 months)",
#                         f"First-year churn rate: 47.7% — highest risk period",
#                         f"This customer has {tenure} months tenure",
#                     ))
#                 if online_sec == "No" and internet_svc != "No":
#                     med_risk.append((
#                         "🔒 No Online Security",
#                         "Customers without it churn at 41.8%",
#                         "Adding this service reduces dissatisfaction",
#                     ))
#                 if tech_support == "No" and internet_svc != "No":
#                     med_risk.append((
#                         "🛠 No Tech Support",
#                         "Customers without it churn at 41.6%",
#                         "Proactive support prevents frustration",
#                     ))
#                 if float(monthly) > 70:
#                     med_risk.append((
#                         "💸 High Monthly Charges",
#                         f"${monthly:.0f}/month exceeds the $70 risk threshold",
#                         "High charges correlate strongly with churn",
#                     ))
#                 if senior == "Yes":
#                     med_risk.append((
#                         "👴 Senior Citizen",
#                         "Churn rate 41.7% vs 24.3% for non-seniors",
#                         "Dedicated support programmes improve retention",
#                     ))
#                 if contract == "Two year":
#                     low_risk.append((
#                         "📋 Two-Year Contract",
#                         "Churn rate only 2.8% — strongest retention signal",
#                         "Long-term commitment is the best loyalty indicator",
#                     ))
#                 if int(tenure) >= 48:
#                     low_risk.append((
#                         "🏆 Long-Tenure Customer",
#                         f"{tenure} months loyalty — churn risk drops dramatically",
#                         "Customers over 48 months churn at only 6.6%",
#                     ))
#                 if contract == "One year":
#                     low_risk.append((
#                         "📋 One-Year Contract",
#                         "Contracted customers churn far less than month-to-month",
#                         "Encourage upgrade to 2-year for even better retention",
#                     ))
#                 if online_sec == "Yes" and tech_support == "Yes":
#                     low_risk.append((
#                         "✅ Fully Supported Customer",
#                         "Both online security and tech support active",
#                         "Well-served customers have lower churn propensity",
#                     ))

#                 # Fallback if nothing detected
#                 if not high_risk and not med_risk and not low_risk:
#                     low_risk.append((
#                         "✅ No Major Risk Factors",
#                         "This customer does not match high-risk churn profiles",
#                         "Continue monitoring engagement and usage metrics",
#                     ))

#                 # Render in 3 columns — safe, no complex unpacking in f-strings
#                 all_factors = (
#                     [(RED,   "High Risk Factor",   t) for t in high_risk] +
#                     [(AMBER, "Medium Risk Factor", t) for t in med_risk]  +
#                     [(GREEN, "Positive Signal",    t) for t in low_risk]
#                 )

#                 if all_factors:
#                     ncols = min(3, len(all_factors))
#                     if ncols < 1:
#                         ncols = 1
#                     fcols = st.columns(ncols, gap="medium")
#                     for idx, factor in enumerate(all_factors):
#                         col_c   = factor[0]
#                         col_lbl = factor[1]
#                         title_f, desc_f, note_f = factor[2]
#                         card_html = risk_card_html(col_c, col_lbl, title_f, desc_f, note_f)
#                         with fcols[idx % ncols]:
#                             md(card_html)

#                 # Summary
#                 n_h = len(high_risk)
#                 n_m = len(med_risk)
#                 n_l = len(low_risk)
#                 s_col = RED if n_h >= 2 else AMBER if (n_h == 1 or n_m >= 2) else GREEN
#                 md(f'<div style="background:{s_col}0A;border:1px solid {s_col}28;'
#                    f'border-radius:10px;padding:12px 16px;margin-top:10px;">'
#                    f'<span style="font-size:.82rem;color:{TEXT};font-weight:600;">'
#                    f'Risk Summary: </span>'
#                    f'<span style="font-size:.82rem;color:{SUB};">'
#                    f'{n_h} high-risk · {n_m} medium-risk · {n_l} positive signal(s). '
#                    f'Churn probability: </span>'
#                    f'<strong style="color:{tier_col};">{churn_p:.1f}% ({tier_lbl})</strong>'
#                    f'</div>')

#             except Exception as exc:
#                 st.error(f"**Prediction error:** {exc}")
#                 st.info(
#                     "Check that all 5 PKL files in `models/` were generated by "
#                     "running the notebook end-to-end with the same dataset."
#                 )

#         else:
#             divider()
#             md(f'<div style="background:{CARD};border:1px dashed {BDR2};'
#                f'border-radius:14px;padding:48px 24px;text-align:center;margin:8px 0;">'
#                f'<div style="font-size:2.4rem;margin-bottom:12px;">🔮</div>'
#                f'<div style="font-family:Space Grotesk,sans-serif;font-size:1rem;'
#                f'font-weight:700;color:{TEXT};margin-bottom:8px;">Ready to Analyse</div>'
#                f'<div style="font-size:.84rem;color:{SUB};line-height:1.7;'
#                f'max-width:360px;margin:0 auto;">Complete the customer profile above '
#                f'and click <strong style="color:{BLUE};">Run Churn Prediction</strong> '
#                f'to see AI results with risk insights.</div></div>')

#             md("<br>")
#             sec_hdr("📌", "Key Churn Signals",
#                     "Historical EDA patterns — top factors that drive customer churn")
#             s1, s2 = st.columns(2, gap="large")
#             with s1:
#                 for lb, rt, ac in [
#                     ("Month-to-month contract",  "42.7% churn rate", RED),
#                     ("Electronic check payment", "45.3% churn rate", RED),
#                     ("Fiber optic internet",     "41.9% churn rate", RED),
#                     ("Tenure ≤ 12 months",       "47.7% churn rate", RED),
#                 ]:
#                     md(signal_row(lb, rt, ac))
#             with s2:
#                 for lb, rt, ac in [
#                     ("No online security",  "41.8% churn rate", AMBER),
#                     ("No tech support",     "41.6% churn rate", AMBER),
#                     ("Two-year contract",   " 2.8% churn rate", GREEN),
#                     ("Tenure > 48 months",  " 6.6% churn rate", GREEN),
#                 ]:
#                     md(signal_row(lb, rt, ac))


# # ═══════════════════════════════════════════
# # TAB 2 — EDA DASHBOARD
# # ═══════════════════════════════════════════
# with TAB2:
#     if not ok_d:
#         st.warning(
#             "**dataset.csv not found.**  "
#             "Place it in the same folder as app.py and restart Streamlit."
#         )
#         md(f'<div style="background:{CARD};border:1px dashed {BDR2};'
#            f'border-radius:14px;padding:48px 24px;text-align:center;margin-top:12px;">'
#            f'<div style="font-size:2.4rem;margin-bottom:12px;">📊</div>'
#            f'<div style="font-family:Space Grotesk,sans-serif;font-size:1rem;'
#            f'font-weight:700;color:{TEXT};margin-bottom:8px;">Dataset Required</div>'
#            f'<div style="font-size:.84rem;color:{SUB};line-height:1.7;">'
#            f'Add dataset.csv to the app folder to view interactive EDA charts.</div></div>')
#     else:
#         # KPIs
#         sec_hdr("📊", "Dataset Overview",
#                 "7,043 Telco customers · 21 raw features · Kaggle Telco Churn dataset")
#         kc2 = st.columns(6, gap="small")
#         for col, (ic, lb, vl, ac, nt) in zip(kc2, [
#             ("👥", "Total",       f"{len(df):,}",                              BLUE,   ""),
#             ("🔴", "Churned",     f"{df['ChurnBin'].sum():,}",                 RED,    f"{df['ChurnBin'].mean()*100:.1f}%"),
#             ("🟢", "Retained",    f"{(~df['ChurnBin'].astype(bool)).sum():,}", GREEN,  f"{(1-df['ChurnBin'].mean())*100:.1f}%"),
#             ("📅", "Avg Tenure",  f"{df['tenure'].mean():.0f} mo",             PURPLE, "months"),
#             ("💵", "Avg Monthly", f"${df['MonthlyCharges'].mean():.0f}",       AMBER,  "/month"),
#             ("🧮", "Features",    "19",                                         TEAL,   "post-encoding"),
#         ]):
#             col.markdown(kpi(ic, lb, vl, ac, nt), unsafe_allow_html=True)

#         divider()

#         # ── 1. Churn Distribution ───────────────────────────────
#         sec_hdr("📉", "1 · Churn Distribution",
#                 "73.5% No Churn vs 26.5% Churned — class imbalance handled with SMOTE on training data")

#         no_ct  = int((df["Churn"] == "No").sum())
#         yes_ct = int((df["Churn"] == "Yes").sum())

#         ca, cb = st.columns(2, gap="large")
#         with ca:
#             fig = go.Figure(go.Bar(
#                 x=["No Churn", "Churned"],
#                 y=[no_ct, yes_ct],
#                 marker_color=[GREEN, RED],
#                 marker_line_color="rgba(0,0,0,0)",
#                 text=[f"{no_ct:,}", f"{yes_ct:,}"],
#                 textposition="outside",
#                 textfont=dict(color=TEXT, size=13),
#             ))
#             fig.update_layout(**plotly_layout("Churn Count", 300))
#             fig.update_yaxes(title="Customers", gridcolor=BDR)
#             fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig, use_container_width=True)

#         with cb:
#             fig = go.Figure(go.Pie(
#                 labels=["No Churn", "Churned"],
#                 values=[no_ct, yes_ct],
#                 hole=0.44,
#                 pull=[0, 0.05],
#                 marker=dict(colors=[GREEN, RED],
#                             line=dict(color=BG, width=2)),
#                 textinfo="label+percent",
#                 textfont=dict(size=12),
#             ))
#             fig.update_layout(**plotly_layout("Churn Split (%)", 300))
#             fig.update_layout(showlegend=False)
#             st.plotly_chart(fig, use_container_width=True)

#         md(f'<div style="background:{AMBER}0A;border:1px solid {AMBER}30;'
#            f'border-radius:9px;padding:11px 16px;margin:6px 0;">'
#            f'<strong style="color:{AMBER};">⚠ Class Imbalance:</strong>'
#            f'<span style="color:{SUB};font-size:.84rem;"> 73.5% vs 26.5% — '
#            f'SMOTE applied on training data only to prevent data leakage.</span></div>')

#         # ── 2. Numerical Distributions ──────────────────────────
#         divider()
#         sec_hdr("📐", "2 · Numerical Feature Distributions",
#                 "Tenure, MonthlyCharges, TotalCharges — split by churn status")

#         num_cols_list = ["tenure", "MonthlyCharges", "TotalCharges"]
#         fig2a = make_subplots(
#             rows=1, cols=3,
#             subplot_titles=["Tenure (months)", "Monthly Charges ($)", "Total Charges ($)"],
#             horizontal_spacing=0.06,
#         )
#         for i, col_name in enumerate(num_cols_list, 1):
#             for churn_val, nm, clr in [
#                 (False, "No Churn", BLUE),
#                 (True,  "Churned",  RED),
#             ]:
#                 sub = df[df["ChurnBin"] == int(churn_val)][col_name]
#                 fig2a.add_trace(go.Histogram(
#                     x=sub, name=nm, marker_color=clr,
#                     opacity=0.7, nbinsx=30,
#                     showlegend=(i == 1),
#                     legendgroup=nm,
#                 ), row=1, col=i)
#         lyt2a = plotly_layout("Distribution by Churn Status", 320)
#         lyt2a["barmode"] = "overlay"
#         fig2a.update_layout(**lyt2a)
#         fig2a.update_xaxes(gridcolor=BDR, linecolor=BDR)
#         fig2a.update_yaxes(gridcolor=BDR, linecolor=BDR)
#         for ann in fig2a.layout.annotations:
#             ann.font.color = SUB
#             ann.font.size  = 11
#         st.plotly_chart(fig2a, use_container_width=True)

#         fig2b = make_subplots(
#             rows=1, cols=3,
#             subplot_titles=["Tenure vs Churn", "Monthly Charges vs Churn",
#                             "Total Charges vs Churn"],
#             horizontal_spacing=0.06,
#         )
#         for i, col_name in enumerate(num_cols_list, 1):
#             for churn_val, nm, clr in [
#                 ("No", "No Churn", BLUE),
#                 ("Yes", "Churned", RED),
#             ]:
#                 fig2b.add_trace(go.Box(
#                     y=df[df["Churn"] == churn_val][col_name],
#                     name=nm, marker_color=clr, line_color=clr,
#                     showlegend=(i == 1), legendgroup=nm,
#                 ), row=1, col=i)
#         lyt2b = plotly_layout("Spread by Churn Status", 320)
#         lyt2b["boxmode"] = "group"
#         fig2b.update_layout(**lyt2b)
#         fig2b.update_xaxes(gridcolor="rgba(0,0,0,0)")
#         fig2b.update_yaxes(gridcolor=BDR)
#         for ann in fig2b.layout.annotations:
#             ann.font.color = SUB
#             ann.font.size  = 11
#         st.plotly_chart(fig2b, use_container_width=True)

#         md(f'<div style="background:{CARD2};border:1px solid {BDR};'
#            f'border-radius:9px;padding:11px 16px;margin:4px 0;">'
#            f'<span style="font-size:.82rem;color:{SUB};">'
#            f'<strong style="color:{RED};">Tenure:</strong> Bimodal — many new + many long-term customers &nbsp;·&nbsp; '
#            f'<strong style="color:{AMBER};">MonthlyCharges:</strong> Churners tend to pay more &nbsp;·&nbsp; '
#            f'<strong style="color:{BLUE};">TotalCharges:</strong> Right-skewed, correlated with tenure'
#            f'</span></div>')

#         # ── 3. Categorical Churn Rates ────────────────────────
#         divider()
#         sec_hdr("📊", "3 · Churn Rate by Category",
#                 "Which service and billing categories drive churn most?")

#         cat_feats = [
#             "Contract", "InternetService", "PaymentMethod",
#             "TechSupport", "OnlineSecurity", "PaperlessBilling",
#         ]
#         fig3 = make_subplots(
#             rows=2, cols=3,
#             subplot_titles=cat_feats,
#             vertical_spacing=0.2,
#             horizontal_spacing=0.07,
#         )
#         for idx, feat in enumerate(cat_feats):
#             r, c = divmod(idx, 3)
#             cr = (df.groupby(feat)["Churn"]
#                   .apply(lambda x: (x == "Yes").mean() * 100)
#                   .sort_values(ascending=False)
#                   .reset_index())
#             cr.columns = [feat, "Pct"]
#             bclrs = [RED if v > 30 else AMBER if v > 15 else GREEN
#                      for v in cr["Pct"]]
#             fig3.add_trace(go.Bar(
#                 x=cr[feat], y=cr["Pct"],
#                 marker_color=bclrs,
#                 marker_line_color="rgba(0,0,0,0)",
#                 text=[f"{v:.1f}%" for v in cr["Pct"]],
#                 textposition="outside",
#                 textfont=dict(color=TEXT, size=9),
#                 showlegend=False,
#             ), row=r+1, col=c+1)

#         lyt3 = plotly_layout("Churn Rate (%) by Feature Category", 540)
#         fig3.update_layout(**lyt3)
#         fig3.update_xaxes(tickfont=dict(size=8, color=SUB),
#                           gridcolor="rgba(0,0,0,0)")
#         fig3.update_yaxes(gridcolor=BDR, tickfont=dict(color=MUTED))
#         for ann in fig3.layout.annotations:
#             ann.font.color = SUB
#             ann.font.size  = 11
#         st.plotly_chart(fig3, use_container_width=True)

#         md(f'<div style="background:{CARD2};border:1px solid {BDR};'
#            f'border-radius:9px;padding:11px 16px;margin:4px 0;">'
#            f'<span style="font-size:.82rem;color:{SUB};">'
#            f'<strong style="color:{RED};">Contract:</strong> Month-to-month 42.7% vs Two-year 2.8% &nbsp;·&nbsp; '
#            f'<strong style="color:{RED};">Payment:</strong> Electronic check 45.3% &nbsp;·&nbsp; '
#            f'<strong style="color:{AMBER};">Internet:</strong> Fiber optic 41.9% vs DSL 18.9% &nbsp;·&nbsp; '
#            f'<strong style="color:{AMBER};">Tech Support:</strong> Without 41.6% vs With 14.8%'
#            f'</span></div>')

#         # ── 4. Demographics ────────────────────────────────────
#         divider()
#         sec_hdr("👥", "4 · Demographics vs Churn",
#                 "Gender, Senior Citizen, Partner, Dependents breakdown")

#         demo_feats = ["gender", "SeniorCitizen", "Partner", "Dependents"]
#         d_cols = st.columns(4, gap="small")
#         for i, feat in enumerate(demo_feats):
#             with d_cols[i]:
#                 ct = (df.groupby(feat)["Churn"]
#                       .apply(lambda x: (x == "Yes").mean() * 100)
#                       .reset_index())
#                 ct.columns = [feat, "Pct"]
#                 # Convert x to string to handle int (SeniorCitizen = 0/1)
#                 x_labels = ct[feat].astype(str).tolist()
#                 clrs = [BLUE, RED] if len(ct) == 2 else [BLUE]*len(ct)
#                 fig_d = go.Figure(go.Bar(
#                     x=x_labels,
#                     y=ct["Pct"].tolist(),
#                     marker_color=clrs,
#                     marker_line_color="rgba(0,0,0,0)",
#                     text=[f"{v:.1f}%" for v in ct["Pct"]],
#                     textposition="outside",
#                     textfont=dict(color=TEXT, size=10),
#                 ))
#                 fig_d.update_layout(**plotly_layout(feat, 240))
#                 fig_d.update_layout(margin=dict(l=5, r=5, t=40, b=5))
#                 fig_d.update_yaxes(title="Churn Rate (%)", range=[0, max(ct["Pct"])*1.32])
#                 fig_d.update_xaxes(gridcolor="rgba(0,0,0,0)")
#                 st.plotly_chart(fig_d, use_container_width=True)

#         # ── 5. Tenure Groups ───────────────────────────────────
#         divider()
#         sec_hdr("⏱️", "5 · Churn Rate by Tenure Group",
#                 "0-12 months is the highest-risk window — 47.7% churn rate")

#         tg = (df.groupby("TenureGroup", observed=True)["Churn"]
#               .apply(lambda x: (x == "Yes").mean() * 100)
#               .reset_index())
#         tg.columns = ["Group", "Pct"]
#         tg["Group"] = tg["Group"].astype(str)  # safe cast from Categorical

#         tg_colors = [RED, AMBER, BLUE, TEAL]
#         tc1, tc2 = st.columns([2, 1], gap="large")
#         with tc1:
#             fig_tg = go.Figure(go.Bar(
#                 x=tg["Group"].tolist(),
#                 y=tg["Pct"].tolist(),
#                 marker_color=tg_colors[:len(tg)],
#                 marker_line_color="rgba(0,0,0,0)",
#                 text=[f"{v:.1f}%" for v in tg["Pct"]],
#                 textposition="outside",
#                 textfont=dict(color=TEXT, size=12),
#             ))
#             fig_tg.update_layout(**plotly_layout("Churn Rate by Tenure Group", 300))
#             fig_tg.update_yaxes(title="Churn Rate (%)", gridcolor=BDR)
#             fig_tg.update_xaxes(title="Tenure Group", gridcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig_tg, use_container_width=True)
#         with tc2:
#             for row_idx, row in tg.iterrows():
#                 ac = tg_colors[row_idx] if row_idx < len(tg_colors) else BLUE
#                 md(f'<div style="background:{CARD};border:1px solid {ac}30;'
#                    f'border-left:3px solid {ac};border-radius:8px;'
#                    f'padding:12px 14px;margin-bottom:7px;">'
#                    f'<div style="font-size:.69rem;color:{MUTED};margin-bottom:2px;">'
#                    f'{row["Group"]}</div>'
#                    f'<div style="font-family:Space Grotesk,sans-serif;font-size:1.4rem;'
#                    f'font-weight:700;color:{ac};">{row["Pct"]:.1f}%</div>'
#                    f'</div>')

#         # ── 6. Correlation Heatmap ─────────────────────────────
#         divider()
#         sec_hdr("🔥", "6 · Feature Correlation Heatmap",
#                 "Label-encoded · lower triangle only · key correlations with Churn")

#         df_corr = df.drop(
#             columns=["ChurnBin", "TenureGroup", "customerID"], errors="ignore"
#         ).copy()
#         le_tmp = LabelEncoder()
#         # Fix: use include="object" with str handling for newer pandas
#         obj_cols = [c for c in df_corr.columns
#                     if df_corr[c].dtype == object or df_corr[c].dtype.name == "string"]
#         for col_name in obj_cols:
#             df_corr[col_name] = le_tmp.fit_transform(df_corr[col_name].astype(str))

#         corr = df_corr.corr().round(2)
#         # Mask upper triangle — replace with NaN
#         mask = np.triu(np.ones_like(corr.values, dtype=bool), k=1)
#         z_vals = corr.values.copy().astype(float)
#         z_vals[mask] = np.nan

#         # Build text array — show value or empty string (no "nan")
#         text_vals = []
#         for row_arr in z_vals:
#             row_text = []
#             for v in row_arr:
#                 row_text.append("" if np.isnan(v) else str(round(v, 2)))
#             text_vals.append(row_text)

#         # fig_hm = go.Figure(go.Heatmap(
#         #     z=z_vals,
#         #     x=corr.columns.tolist(),
#         #     y=corr.index.tolist(),
#         #     colorscale="RdBu_r",
#         #     zmid=0,
#         #     zmin=-1, zmax=1,
#         #     text=text_vals,
#         #     texttemplate="%{text}",
#         #     textfont=dict(size=7, color="white"),
#         #     hoverongaps=False,
#         #  colorbar=dict(
#         #         title=dict(
#         #             text="r", 
#         #             font=dict(color=SUB, size=10)
#         #         ),
#         #         tickfont=dict(color=SUB, size=9),
#         #     ),
#         # ))
#         # hm_layout = plotly_layout("Feature Correlation Matrix", 520)
#         # hm_layout["xaxis"] = dict(
#         #     tickfont=dict(size=8, color=SUB), tickangle=45,
#         #     gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
#         # )
#         # hm_layout["yaxis"] = dict(
#         #     tickfont=dict(size=8, color=SUB),
#         #     gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
#         # )
#         # hm_layout["margin"] = dict(l=120, r=20, t=50, b=120)
#         # fig_hm.update_layout(**hm_layout)
#         # st.plotly_chart(fig_hm, use_container_width=True)

#         # md(f'<div style="background:{CARD2};border:1px solid {BDR};'
#         #    f'border-radius:9px;padding:11px 16px;margin:4px 0;">'
#         #    f'<span style="font-size:.82rem;color:{SUB};">'
#         #    f'<strong style="color:{TEAL};">tenure</strong> negatively correlated with Churn — '
#         #    f'long-term customers stay &nbsp;·&nbsp; '
#         #    f'<strong style="color:{RED};">MonthlyCharges</strong> positively correlated — '
#         #    f'expensive plans drive churn &nbsp;·&nbsp; '
#         #    f'TotalCharges highly correlated with tenure (expected)'
#         #    f'</span></div>')
        
#         #===========================
#         fig_hm = go.Figure(go.Heatmap(
#             z=z_vals,
#             x=corr.columns.tolist(),
#             y=corr.index.tolist(),
#             colorscale=[
#                 [0.00, '#3b4cc0'], [0.15, '#6788ee'], [0.35, '#9abbff'], 
#                 [0.50, '#e2e2e2'], [0.65, '#f1a88d'], [0.85, '#d35c4e'], [1.00, '#b40426']
#             ],
#             zmid=0,
#             zmin=-1, zmax=1,
#             text=text_vals,
#             texttemplate="<b>%{text}</b>",
#             textfont=dict(size=11),  # <-- FIXED: Plotly will now auto-contrast the text!
#             hoverongaps=False,
#             colorbar=dict(
#                 title=dict(
#                     text="r", 
#                     font=dict(color=SUB, size=12)
#                 ),
#                 tickfont=dict(color=SUB, size=11),
#             ),
#         ))
       
        
#         # Increased height from 520 to 750 so it doesn't look shrunk
#         hm_layout = plotly_layout("Feature Correlation Matrix", 750) 
        
#         hm_layout["xaxis"] = dict(
#             tickfont=dict(size=10, color=SUB), tickangle=45,  # Increased axis font size
#             gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
#         )
#         hm_layout["yaxis"] = dict(
#             tickfont=dict(size=10, color=SUB),                # Increased axis font size
#             gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
#         )
#         hm_layout["margin"] = dict(l=120, r=20, t=50, b=120)
        
#         fig_hm.update_layout(**hm_layout)
#         st.plotly_chart(fig_hm, use_container_width=True)

#         md(f'<div style="background:{CARD2};border:1px solid {BDR};'
#            f'border-radius:9px;padding:11px 16px;margin:4px 0;">'
#            f'<span style="font-size:.82rem;color:{SUB};">'
#            f'<strong style="color:{TEAL};">tenure</strong> negatively correlated with Churn — '
#            f'long-term customers stay &nbsp;·&nbsp; '
#            f'<strong style="color:{RED};">MonthlyCharges</strong> positively correlated — '
#            f'expensive plans drive churn &nbsp;·&nbsp; '
#            f'TotalCharges highly correlated with tenure (expected)'
#            f'</span></div>')
#         #===========================

#         # Raw data browser
#         with st.expander("📋  Browse Raw Dataset"):
#             flt = st.text_input("Filter by Churn (Yes / No / blank = all):", "")
#             vdf = (df[df["Churn"].str.contains(flt, case=False, na=False)]
#                    if flt else df)
#             st.dataframe(
#                 vdf.drop(columns=["ChurnBin", "TenureGroup"], errors="ignore").head(300),
#                 use_container_width=True,
#             )
#             st.caption(f"Showing {min(300, len(vdf))} of {len(vdf):,} rows")


# # ═══════════════════════════════════════════
# # TAB 3 — MODEL INSIGHTS
# # ═══════════════════════════════════════════
# with TAB3:

#     # Pipeline
#     sec_hdr("🏆", "Model Training Pipeline",
#             "End-to-end: raw data → SMOTE → tuning → best model → PKL artifacts")
#     steps_html = "  ".join(
#         f'<div style="background:{cl}12;border:1px solid {cl}30;border-radius:7px;'
#         f'padding:7px 13px;font-size:.77rem;font-family:JetBrains Mono,monospace;'
#         f'color:{cl};white-space:nowrap;">{s}</div>'
#         for s, cl in [
#             ("1. Data Cleaning",       BLUE),
#             ("2. Feature Encoding",    CYAN),
#             ("3. Train/Test Split",    TEAL),
#             ("4. SMOTE Balancing",     GREEN),
#             ("5. Baseline ×3 Models",  PURPLE),
#             ("6. GridSearchCV Tuning", AMBER),
#             ("7. Select by AUC-ROC",   RED),
#             ("8. Save PKL Artifacts",  BLUE),
#         ]
#     )
#     md(f'<div style="background:{CARD};border:1px solid {BDR};border-radius:11px;'
#        f'padding:15px 17px;margin-bottom:20px;">'
#        f'<div style="display:flex;gap:8px;flex-wrap:wrap;">{steps_html}</div></div>')

#     # Performance table
#     sec_hdr("📊", "Model Performance — Baseline vs Tuned",
#             "Correct verified values from notebook execution")

#     perf_df = pd.DataFrame({
#         "Model":           ["XGBoost  ★ Best", "Random Forest", "Logistic Regression"],
#         "Base Accuracy":   [0.7779, 0.7771, 0.7424],
#         "Tuned Accuracy":  [0.7757, 0.7622, 0.7410],
#         "Base F1":         [0.5865, 0.5890, 0.6191],
#         "Tuned F1":        [0.6274, 0.6223, 0.6146],
#         "Base AUC-ROC":    [0.8146, 0.8210, 0.8391],
#         "Tuned AUC-ROC":   [0.8434, 0.8397, 0.8382],
#         "Speed":           ["Medium", "Slow", "Fast"],
#     })
#     st.dataframe(perf_df, use_container_width=True, hide_index=True)

#     # Metric bar charts
#     md("<br>")
#     mc_cols = st.columns(3, gap="medium")
#     mnames  = ["XGBoost", "Random Forest", "Logistic Reg."]
#     for i, (metric, vals, ac) in enumerate([
#         ("Tuned Accuracy", [0.7757, 0.7622, 0.7410], BLUE),
#         ("Tuned AUC-ROC",  [0.8434, 0.8397, 0.8382], AMBER),
#         ("Tuned F1-Score", [0.6274, 0.6223, 0.6146], GREEN),
#     ]):
#         with mc_cols[i]:
#             best   = max(vals)
#             bclrs  = [ac if abs(v - best) < 1e-5 else BDR2 for v in vals]
#             texts  = [f"{v:.4f} ★" if abs(v-best)<1e-5 else f"{v:.4f}" for v in vals]
#             fig_mc = go.Figure(go.Bar(
#                 y=mnames, x=vals, orientation="h",
#                 marker_color=bclrs,
#                 marker_line_color="rgba(0,0,0,0)",
#                 text=texts, textposition="outside",
#                 textfont=dict(color=TEXT, size=10),
#             ))
#             fig_mc.update_layout(**plotly_layout(metric, 210))
#             fig_mc.update_layout(margin=dict(l=8, r=65, t=42, b=8))
#             fig_mc.update_xaxes(range=[min(vals)-0.05, max(vals)+0.07], gridcolor=BDR)
#             fig_mc.update_yaxes(gridcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig_mc, use_container_width=True)

#     # Baseline vs Tuned comparison
#     sec_hdr("📈", "Baseline vs Tuned Improvement",
#             "F1-Score and AUC-ROC gain after GridSearchCV hyperparameter tuning")

#     fig_bt = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=["F1-Score: Baseline vs Tuned", "AUC-ROC: Baseline vs Tuned"],
#         horizontal_spacing=0.1,
#     )
#     for cidx, (bv, tv, mn_col) in enumerate([
#         ([0.5865, 0.5890, 0.6191], [0.6274, 0.6223, 0.6146], "F1"),
#         ([0.8146, 0.8210, 0.8391], [0.8434, 0.8397, 0.8382], "AUC"),
#     ], 1):
#         all_v = bv + tv
#         fig_bt.add_trace(go.Bar(
#             name="Baseline", x=mnames, y=bv,
#             marker_color=BDR2,
#             marker_line_color="rgba(0,0,0,0)",
#             text=[f"{v:.4f}" for v in bv], textposition="outside",
#             textfont=dict(color=SUB, size=9),
#             showlegend=(cidx == 1),
#         ), row=1, col=cidx)
#         fig_bt.add_trace(go.Bar(
#             name="Tuned", x=mnames, y=tv,
#             marker_color=BLUE,
#             marker_line_color="rgba(0,0,0,0)",
#             text=[f"{v:.4f}" for v in tv], textposition="outside",
#             textfont=dict(color=TEXT, size=9),
#             showlegend=(cidx == 1),
#         ), row=1, col=cidx)
#         fig_bt.update_yaxes(range=[min(all_v)-0.04, max(all_v)+0.06],
#                             gridcolor=BDR, row=1, col=cidx)
#         fig_bt.update_xaxes(gridcolor="rgba(0,0,0,0)", row=1, col=cidx)

#     lyt_bt = plotly_layout("Baseline vs Tuned — All 3 Models", 340)
#     lyt_bt["barmode"] = "group"
#     fig_bt.update_layout(**lyt_bt)
#     for ann in fig_bt.layout.annotations:
#         ann.font.color = SUB
#         ann.font.size  = 11
#     st.plotly_chart(fig_bt, use_container_width=True)

#     # Feature Importance
#     if ok_m:
#         sec_hdr("📌", "Feature Importance",
#                 f"Top 15 features driving {model_lbl} predictions")
#         try:
#             actual_m = arts["model"]
#             if hasattr(actual_m, "named_steps"):
#                 actual_m = actual_m.named_steps.get("model", actual_m)
#             if hasattr(actual_m, "feature_importances_"):
#                 imps = actual_m.feature_importances_
#             elif hasattr(actual_m, "coef_"):
#                 imps = np.abs(actual_m.coef_[0])
#             else:
#                 raise ValueError("No importances or coefficients on model object")

#             fi = (pd.Series(imps, index=arts["cols"])
#                   .sort_values(ascending=True)
#                   .tail(15))
#             top3 = sorted(fi.values)[-3]
#             fi_clrs = [AMBER if v >= top3 else BLUE for v in fi.values]
#             fi_txt  = [f"{v:.4f}" for v in fi.values]

#             fig_fi = go.Figure(go.Bar(
#                 y=fi.index.tolist(),
#                 x=fi.values.tolist(),
#                 orientation="h",
#                 marker_color=fi_clrs,
#                 marker_line_color="rgba(0,0,0,0)",
#                 text=fi_txt, textposition="outside",
#                 textfont=dict(color=TEXT, size=9),
#             ))
#             fig_fi.update_layout(**plotly_layout(f"Feature Importances — {model_lbl}", 440))
#             fig_fi.update_layout(margin=dict(l=8, r=70, t=50, b=8))
#             fig_fi.update_xaxes(gridcolor=BDR, title="Importance Score")
#             fig_fi.update_yaxes(gridcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig_fi, use_container_width=True)
#             md(f'<div style="font-size:.78rem;color:{MUTED};margin-top:4px;">'
#                f'<span style="color:{AMBER};">■</span> Top 3 drivers &nbsp;'
#                f'<span style="color:{BLUE};">■</span> Other features</div>')
#         except Exception as e:
#             st.info(f"Feature importance unavailable: {e}")
#     else:
#         st.info("Load model PKL files to see feature importance chart.")

#     # Final metrics
#     sec_hdr("📋", "Final Model Metrics — XGBoost",
#             "Held-out test set · 1,409 samples · no data leakage")

#     m5c = st.columns(5, gap="small")
#     for col, (ic, lb, vl, ac) in zip(m5c, [
#         ("🎯", "Accuracy",  f"{ACC*100:.2f}%", BLUE),
#         ("⚡", "Precision", f"{PREC*100:.2f}%", PURPLE),
#         ("🔍", "Recall",    f"{REC*100:.2f}%",  TEAL),
#         ("📊", "F1 Score",  f"{F1:.4f}",        GREEN),
#         ("📈", "AUC-ROC",   f"{AUC:.4f}",       AMBER),
#     ]):
#         col.markdown(kpi(ic, lb, vl, ac), unsafe_allow_html=True)

#     # Classification report + hyperparams
#     md("<br>")
#     cr1, cr2 = st.columns(2, gap="large")

#     with cr1:
#         sec_hdr("📄", "Classification Report",
#                 "Per-class breakdown on test set — No Churn vs Churn")
#         report_data = pd.DataFrame({
#             "Class":     ["No Churn", "Churn", "Macro Avg", "Weighted Avg"],
#             "Precision": [0.88, 0.56, 0.72, 0.80],
#             "Recall":    [0.80, 0.71, 0.76, 0.78],
#             "F1-Score":  [0.84, 0.63, 0.73, 0.78],
#             "Support":   [1035, 374, 1409, 1409],
#         })
#         st.dataframe(report_data, use_container_width=True, hide_index=True)

#     with cr2:
#         sec_hdr("⚙️", "Best Hyperparameters",
#                 "XGBoost — GridSearchCV with 5-fold StratifiedKFold · scoring=F1")
#         params = {}
#         if ok_m:
#             raw_params = m_info.get("best_params", {})
#             if raw_params:
#                 params = {k.replace("model__", ""): v for k, v in raw_params.items()}
#         if not params:
#             params = {
#                 "colsample_bytree": 0.8,
#                 "gamma":            0,
#                 "learning_rate":    0.05,
#                 "max_depth":        3,
#                 "n_estimators":     200,
#                 "reg_lambda":       1,
#                 "subsample":        0.8,
#             }
#         rows_p = "".join(
#             f'<div style="display:flex;justify-content:space-between;align-items:center;'
#             f'padding:8px 0;border-bottom:1px solid {BDR}38;">'
#             f'<span style="font-family:JetBrains Mono,monospace;font-size:.79rem;'
#             f'color:{CYAN};">{k}</span>'
#             f'<span style="font-size:.79rem;color:{TEXT};font-weight:600;">{v}</span>'
#             f'</div>'
#             for k, v in params.items()
#         )
#         md(f'<div style="background:{CARD};border:1px solid {BDR};'
#            f'border-radius:10px;padding:13px 15px;">{rows_p}</div>')

#     # Business Insights
#     divider()
#     sec_hdr("💡", "Key Business Insights",
#             "Actionable retention strategies from EDA and model feature importance")

#     insights_data = [
#         ("📅", "Contract Type",
#          "Month-to-month customers churn at 42.7% vs only 2.8% on 2-year plans. "
#          "Offer discounts to upgrade contract length.",
#          RED),
#         ("🆕", "New Customers",
#          "First 12 months = 47.7% churn rate — highest-risk window. "
#          "Invest heavily in onboarding and early engagement.",
#          AMBER),
#         ("💸", "Monthly Charges",
#          "Bills over $70/month strongly correlate with churn. "
#          "Introduce loyalty pricing and value bundles for high-payers.",
#          AMBER),
#         ("👴", "Senior Citizens",
#          "Senior customers churn at 41.7% vs 24.3% for non-seniors. "
#          "Dedicated support and simplified plans improve retention.",
#          RED),
#         ("💳", "Payment Method",
#          "Electronic check = 45.3% churn — highest of all methods. "
#          "Incentivise migration to automatic payment.",
#          RED),
#         ("🌐", "Fiber Optic",
#          "Fiber optic churn: 41.9% vs DSL 18.9%. "
#          "Price-to-value perception drives dissatisfaction — target with loyalty offers.",
#          AMBER),
#     ]
#     for row_g in [insights_data[:3], insights_data[3:]]:
#         rc = st.columns(3, gap="medium")
#         for col, (ic, ti, de, ac) in zip(rc, row_g):
#             col.markdown(insight_card(ic, ti, de, ac), unsafe_allow_html=True)
#         md("<br>")


# # ─────────────────────────────────────────────
# # FOOTER
# # ─────────────────────────────────────────────
# md(f'<div style="margin-top:52px;padding:18px 0;border-top:1px solid {BDR};'
#    f'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">'
#    f'<div style="display:flex;align-items:center;gap:9px;">'
#    f'<div style="width:28px;height:28px;border-radius:8px;background:{BLUE};'
#    f'display:flex;align-items:center;justify-content:center;font-size:.88rem;">📡</div>'
#    f'<span style="font-family:Space Grotesk,sans-serif;font-size:.88rem;'
#    f'font-weight:700;color:{TEXT};">TeleChurn AI</span>'
#    f'<span style="font-size:.76rem;color:{MUTED};">· Telco Customer Intelligence</span>'
#    f'</div>'
#    f'<div style="font-size:.7rem;color:{MUTED};">'
#    f'XGBoost · SMOTE · GridSearchCV · Scikit-learn · Plotly · Streamlit · Python'
#    f'</div></div>')



# -------------------new----------------------------
# """
# TeleChurn AI — app.py (v6 · Final · Error-Free)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# pip install streamlit scikit-learn xgboost pandas numpy plotly
# Place PKL files in ./models/ and dataset.csv alongside app.py
# streamlit run app.py

# Bugs fixed vs v5:
#  ✅ st.stop() inside tabs blocks all later tabs — replaced with if/else
#  ✅ st.columns(0) crash when n_cols=0
#  ✅ Heatmap shows "nan" text — masked properly
#  ✅ SeniorCitizen (int) plotted as string on x-axis
#  ✅ select_dtypes deprecation warning fixed
#  ✅ All f-strings validated, no nested quote conflicts
#  ✅ Risk-summary safe for any combination of inputs
# """

# import pickle
# import warnings
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import streamlit as st
# from sklearn.preprocessing import LabelEncoder

# warnings.filterwarnings("ignore")
# #----------------------------------------------
# # ── FORCE DARK TABLE THEME (STREAMLIT CLOUD FIX) ──
# st.markdown("""
# <style>

# /* dataframe container */
# [data-testid="stDataFrame"] {
#     background-color: #0f172a;
#     border-radius: 10px;
# }

# /* table header */
# [data-testid="stDataFrame"] thead tr th {
#     background-color: #111827 !important;
#     color: #e5e7eb !important;
#     font-weight: 600;
# }

# /* table cells */
# [data-testid="stDataFrame"] tbody tr td {
#     background-color: #0f172a !important;
#     color: #e5e7eb !important;
# }

# /* row hover */
# [data-testid="stDataFrame"] tbody tr:hover td {
#     background-color: #1f2937 !important;
# }

# /* expander dataframe fix */
# [data-testid="stExpander"] [data-testid="stDataFrame"] {
#     background-color: #0f172a;
# }

# </style>
# """, unsafe_allow_html=True)
# #----------------------------------------------

# # ─────────────────────────────────────────────
# # PAGE CONFIG
# # ─────────────────────────────────────────────
# st.set_page_config(
#     page_title="TeleChurn AI",
#     page_icon="📡",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ─────────────────────────────────────────────
# # COLOR SYSTEM
# # ─────────────────────────────────────────────
# BG    = "#0F1117"
# CARD  = "#161B27"
# CARD2 = "#1C2333"
# BDR   = "#2A3347"
# BDR2  = "#374357"
# TEXT  = "#E8EDF5"
# SUB   = "#8B97B0"
# MUTED = "#4A5568"

# BLUE   = "#4F8EF7"
# INDIGO = "#7C6FF7"
# CYAN   = "#22D3EE"
# GREEN  = "#22C55E"
# AMBER  = "#F59E0B"
# RED    = "#F43F5E"
# PURPLE = "#A855F7"
# TEAL   = "#14B8A6"

# # ─────────────────────────────────────────────
# # CSS
# # ─────────────────────────────────────────────
# st.markdown(f"""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
# *,*::before,*::after{{box-sizing:border-box;margin:0;}}
# html,body,.stApp{{background:{BG}!important;color:{TEXT}!important;font-family:'Inter',sans-serif!important;}}
# #MainMenu,footer,header,[data-testid="stToolbar"]{{display:none!important;}}
# .block-container{{padding:0 2rem 5rem!important;max-width:1420px!important;}}

# [data-testid="stSidebar"]{{background:{CARD}!important;border-right:1px solid {BDR}!important;}}
# [data-testid="stSidebar"]>div:first-child{{padding-top:0!important;}}
# [data-testid="stSidebarNav"]{{display:none!important;}}

# .stTabs [data-baseweb="tab-list"]{{background:{CARD};border:1px solid {BDR};border-radius:12px;padding:4px;gap:2px;margin-bottom:24px;}}
# .stTabs [data-baseweb="tab"]{{background:transparent!important;color:{MUTED}!important;border-radius:9px!important;padding:9px 24px!important;font-family:'Space Grotesk',sans-serif!important;font-size:.84rem!important;font-weight:600!important;border:none!important;transition:all .18s!important;}}
# .stTabs [aria-selected="true"]{{background:{BLUE}!important;color:#fff!important;box-shadow:0 2px 12px {BLUE}30!important;}}

# div[data-baseweb="select"]>div{{background:{CARD2}!important;border:1px solid {BDR}!important;border-radius:8px!important;color:{TEXT}!important;}}
# div[data-baseweb="select"] *{{color:{TEXT}!important;}}
# div[data-baseweb="popover"]{{background:{CARD2}!important;border:1px solid {BDR2}!important;border-radius:10px!important;}}
# div[data-baseweb="popover"] li:hover{{background:{BDR}!important;}}
# .stNumberInput input,.stTextInput input,.stTextArea textarea{{background:{CARD2}!important;border:1px solid {BDR}!important;border-radius:8px!important;color:{TEXT}!important;font-size:.88rem!important;}}
# .stSlider>div>div>div>div{{background:{BLUE}!important;}}

# div.stButton>button{{background:{BLUE}!important;color:#fff!important;border:none!important;border-radius:10px!important;padding:14px 0!important;font-family:'Space Grotesk',sans-serif!important;font-weight:700!important;font-size:1rem!important;width:100%!important;letter-spacing:.3px!important;box-shadow:0 4px 20px {BLUE}40!important;transition:all .2s ease!important;}}
# div.stButton>button:hover{{background:#6BA3F9!important;box-shadow:0 8px 30px {BLUE}55!important;transform:translateY(-1px)!important;}}

# [data-testid="stSuccess"]{{background:#0A1F14!important;border:1px solid {GREEN}40!important;border-radius:10px!important;}}
# [data-testid="stError"]{{background:#1F0A11!important;border:1px solid {RED}40!important;border-radius:10px!important;}}
# [data-testid="stWarning"]{{background:#1A1305!important;border:1px solid {AMBER}40!important;border-radius:10px!important;}}
# [data-testid="stInfo"]{{background:#080F1F!important;border:1px solid {BLUE}40!important;border-radius:10px!important;}}
# [data-testid="stExpander"]{{background:{CARD}!important;border:1px solid {BDR}!important;border-radius:10px!important;}}
# details summary{{color:{TEXT}!important;}}

# ::-webkit-scrollbar{{width:4px;height:4px;}}
# ::-webkit-scrollbar-track{{background:{BG};}}
# ::-webkit-scrollbar-thumb{{background:{BDR2};border-radius:4px;}}
# hr{{border-color:{BDR}!important;margin:1.4rem 0!important;}}

# @keyframes pdot{{0%,100%{{opacity:1;}}50%{{opacity:.2;}}}}
# .pdot{{display:inline-block;width:7px;height:7px;border-radius:50%;background:{GREEN};animation:pdot 2s infinite;box-shadow:0 0 6px {GREEN};margin-right:6px;vertical-align:middle;}}
# @keyframes sup{{from{{opacity:0;transform:translateY(14px);}}to{{opacity:1;transform:translateY(0);}}}}
# .sup{{animation:sup .4s cubic-bezier(.22,1,.36,1) both;}}
# </style>
# """, unsafe_allow_html=True)


# # ─────────────────────────────────────────────
# # PLOTLY BASE LAYOUT
# # ─────────────────────────────────────────────
# def plotly_layout(title="", height=360):
#     return dict(
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         font=dict(family="Inter", color=SUB, size=11),
#         margin=dict(l=8, r=8, t=44, b=8),
#         height=height,
#         title=dict(
#             text=title,
#             font=dict(family="Space Grotesk", size=14, color=TEXT),
#             x=0, xanchor="left",
#         ),
#         xaxis=dict(gridcolor=BDR, linecolor=BDR, zerolinecolor=BDR),
#         yaxis=dict(gridcolor=BDR, linecolor=BDR, zerolinecolor=BDR),
#         legend=dict(
#             bgcolor="rgba(22,27,39,0.85)", bordercolor=BDR,
#             borderwidth=1, font=dict(size=11, color=SUB),
#         ),
#         colorway=[BLUE, CYAN, GREEN, AMBER, RED, PURPLE, TEAL],
#     )


# # ─────────────────────────────────────────────
# # HTML HELPERS  (no nested quote conflicts)
# # ─────────────────────────────────────────────
# def md(html):
#     st.markdown(html, unsafe_allow_html=True)

# def divider():
#     md("<hr/>")

# def sec_hdr(icon, title, sub=""):
#     sub_part = (f'<div style="font-size:.77rem;color:{SUB};margin-top:3px;">{sub}</div>'
#                 if sub else "")
#     md(f"""
#     <div style="display:flex;align-items:flex-start;gap:12px;margin:30px 0 16px;">
#       <div style="width:37px;height:37px;flex-shrink:0;border-radius:10px;
#         background:linear-gradient(135deg,{BLUE}22,{INDIGO}22);
#         border:1px solid {BLUE}35;display:flex;align-items:center;
#         justify-content:center;font-size:1.05rem;">{icon}</div>
#       <div>
#         <div style="font-family:'Space Grotesk',sans-serif;font-size:.98rem;
#           font-weight:700;color:{TEXT};">{title}</div>
#         {sub_part}
#       </div>
#     </div>""")

# def kpi(icon, label, value, accent, note=""):
#     note_part = (f'<div style="font-size:.69rem;color:{SUB};margin-top:3px;">{note}</div>'
#                  if note else "")
#     return f"""<div style="background:{CARD};border:1px solid {BDR};border-radius:12px;
#       padding:17px;position:relative;overflow:hidden;">
#       <div style="position:absolute;top:0;right:0;width:70px;height:70px;
#         background:radial-gradient(circle,{accent}18,transparent 70%);pointer-events:none;"></div>
#       <div style="font-size:1.2rem;margin-bottom:7px;">{icon}</div>
#       <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;
#         font-weight:700;color:{accent};line-height:1;margin-bottom:4px;">{value}</div>
#       <div style="font-size:.63rem;color:{MUTED};text-transform:uppercase;
#         letter-spacing:1px;font-weight:600;">{label}</div>
#       {note_part}</div>"""

# def badge(text, color):
#     return (f'<span style="background:{color}18;border:1px solid {color}35;'
#             f'color:{color};border-radius:6px;padding:3px 12px;font-size:.72rem;'
#             f'font-family:JetBrains Mono,monospace;font-weight:500;">{text}</span>')

# def insight_card(icon, title, desc, accent):
#     return f"""<div style="background:{CARD};border:1px solid {accent}30;
#       border-left:3px solid {accent};border-radius:10px;padding:18px;height:100%;">
#       <div style="font-size:1.2rem;margin-bottom:9px;">{icon}</div>
#       <div style="font-family:'Space Grotesk',sans-serif;font-size:.87rem;
#         font-weight:700;color:{accent};margin-bottom:6px;">{title}</div>
#       <div style="font-size:.79rem;color:{SUB};line-height:1.7;">{desc}</div>
#     </div>"""

# def signal_row(label, rate, accent):
#     return f"""<div style="display:flex;justify-content:space-between;align-items:center;
#       background:{CARD};border:1px solid {BDR};border-left:3px solid {accent};
#       border-radius:8px;padding:9px 14px;margin-bottom:7px;">
#       <span style="font-size:.82rem;color:{SUB};">{label}</span>
#       <span style="font-size:.82rem;font-weight:700;color:{accent};">{rate}</span>
#     </div>"""

# def risk_card_html(col_c, col_lbl, title_f, desc_f, note_f):
#     """Build risk factor card HTML without nested f-string quote issues."""
#     return (
#         f'<div style="background:{col_c}0C;border:1px solid {col_c}30;'
#         f'border-left:3px solid {col_c};border-radius:10px;'
#         f'padding:14px 16px;margin-bottom:10px;">'
#         f'<div style="font-size:.63rem;font-weight:700;color:{col_c};'
#         f'text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">'
#         f'{col_lbl}</div>'
#         f'<div style="font-size:.82rem;font-weight:600;color:{TEXT};margin-bottom:4px;">'
#         f'{title_f}</div>'
#         f'<div style="font-size:.76rem;color:{SUB};margin-bottom:3px;">{desc_f}</div>'
#         f'<div style="font-size:.7rem;color:{MUTED};font-style:italic;">{note_f}</div>'
#         f'</div>'
#     )


# # ─────────────────────────────────────────────
# # DATA LOADING
# # ─────────────────────────────────────────────
# MODELS_DIR = Path("models")

# @st.cache_resource(show_spinner="Loading model…")
# def load_artifacts():
#     needed = {
#         "model":  MODELS_DIR / "model.pkl",
#         "scaler": MODELS_DIR / "scaler.pkl",
#         "enc":    MODELS_DIR / "label_encoders.pkl",
#         "cols":   MODELS_DIR / "feature_columns.pkl",
#         "info":   MODELS_DIR / "model_info.pkl",
#     }
#     miss = [p.name for p in needed.values() if not p.exists()]
#     if miss:
#         return None, miss
#     out = {}
#     for k, p in needed.items():
#         with open(p, "rb") as f:
#             out[k] = pickle.load(f)
#     return out, []

# @st.cache_data(show_spinner="Loading dataset…")
# def load_data():
#     p = Path("dataset.csv")
#     if not p.exists():
#         return None
#     df = pd.read_csv(p)
#     df["TotalCharges"] = pd.to_numeric(
#         df["TotalCharges"].astype(str).str.strip().replace({"": "0", " ": "0"}),
#         errors="coerce",
#     ).fillna(0.0)
#     df["ChurnBin"] = (df["Churn"] == "Yes").astype(int)
#     # bins include 0 explicitly so no NaN in TenureGroup
#     df["TenureGroup"] = pd.cut(
#         df["tenure"],
#         bins=[-1, 12, 24, 48, 72],
#         labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"],
#     )
#     return df

# arts, miss_files = load_artifacts()
# df   = load_data()
# ok_m = arts is not None
# ok_d = df is not None

# m_info    = arts["info"] if ok_m else {}
# model_lbl = m_info.get("model_name", "XGBoost") if ok_m else "XGBoost"
# threshold = float(m_info.get("threshold", 0.4828)) if ok_m else 0.4828

# # Verified final metrics (from notebook output)
# ACC  = m_info.get("accuracy",  0.7757) if ok_m else 0.7757
# PREC = m_info.get("precision", 0.5612) if ok_m else 0.5612
# REC  = m_info.get("recall",    0.7112) if ok_m else 0.7112
# F1   = m_info.get("f1_score",  0.6274) if ok_m else 0.6274
# AUC  = m_info.get("roc_auc",   0.8434) if ok_m else 0.8434


# # ─────────────────────────────────────────────
# # SIDEBAR
# # ─────────────────────────────────────────────
# with st.sidebar:
#     md(f"""<div style="padding:20px 16px 0;">
#       <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
#         <div style="width:33px;height:33px;border-radius:9px;background:{BLUE};
#           display:flex;align-items:center;justify-content:center;
#           font-size:.95rem;flex-shrink:0;">📡</div>
#         <div>
#           <div style="font-family:'Space Grotesk',sans-serif;font-size:.93rem;
#             font-weight:700;color:{TEXT};">TeleChurn AI</div>
#           <div style="font-size:.62rem;color:{MUTED};letter-spacing:.5px;">
#             ANALYTICS PLATFORM</div>
#         </div>
#       </div>
#       <div style="height:1px;background:{BDR};margin-bottom:16px;"></div>
#     </div>""")

#     # Status
#     mc = GREEN if ok_m else RED
#     dc = GREEN if ok_d else AMBER
#     md(f"""<div style="padding:0 16px;">
#       <div style="font-size:.62rem;font-weight:700;color:{MUTED};letter-spacing:1.1px;
#         text-transform:uppercase;margin-bottom:8px;">System Status</div>
#       <div style="background:{CARD2};border:1px solid {mc}30;border-radius:8px;
#         padding:9px 12px;margin-bottom:5px;display:flex;align-items:center;gap:7px;">
#         <span class="pdot" style="background:{mc};box-shadow:0 0 6px {mc};"></span>
#         <span style="font-size:.77rem;color:{mc};font-weight:600;">
#           {"Model Ready" if ok_m else "Model Missing — add models/"}</span>
#       </div>
#       <div style="background:{CARD2};border:1px solid {dc}30;border-radius:8px;
#         padding:9px 12px;margin-bottom:16px;display:flex;align-items:center;gap:7px;">
#         <span style="width:7px;height:7px;border-radius:50%;
#           background:{dc};flex-shrink:0;display:inline-block;"></span>
#         <span style="font-size:.77rem;color:{dc};font-weight:600;">
#           {"Dataset Loaded" if ok_d else "No dataset.csv found"}</span>
#       </div>
#     </div>""")

#     # Model info
#     if ok_m:
#         md(f"""<div style="padding:0 16px;">
#           <div style="font-size:.62rem;font-weight:700;color:{MUTED};letter-spacing:1.1px;
#             text-transform:uppercase;margin-bottom:8px;">Best Model</div>
#           <div style="background:{CARD2};border:1px solid {BDR};border-radius:8px;
#             padding:12px;margin-bottom:16px;">
#             <div style="display:flex;justify-content:space-between;margin-bottom:7px;">
#               <span style="font-size:.71rem;color:{MUTED};">Algorithm</span>
#               <span style="font-family:'JetBrains Mono',monospace;font-size:.79rem;
#                 color:{BLUE};font-weight:600;">{model_lbl}</span>
#             </div>
#             <div style="height:1px;background:{BDR};margin:7px 0;"></div>
#             <div style="display:flex;justify-content:space-between;">
#               <span style="font-size:.71rem;color:{MUTED};">Threshold</span>
#               <span style="font-family:'JetBrains Mono',monospace;font-size:.79rem;
#                 color:{AMBER};font-weight:600;">{threshold:.4f}</span>
#             </div>
#           </div>
#         </div>""")

#     # Stats
#     _n  = f"{len(df):,}" if ok_d else "7,043"
#     _ch = f"{df['ChurnBin'].sum():,}" if ok_d else "1,869"
#     _cr = f"{df['ChurnBin'].mean()*100:.1f}%" if ok_d else "26.5%"
#     stats = [
#         ("Customers", _n, CYAN),
#         ("Churned", _ch, RED),
#         ("Churn Rate", _cr, AMBER),
#         ("Best AUC-ROC", f"{AUC:.4f}", GREEN),
#         ("Best F1", f"{F1:.4f}", TEAL),
#     ]
#     md(f"""<div style="padding:0 16px;">
#       <div style="font-size:.62rem;font-weight:700;color:{MUTED};letter-spacing:1.1px;
#         text-transform:uppercase;margin-bottom:8px;">Dataset Stats</div>""")
#     for lb, vl, ac in stats:
#         md(f"""<div style="background:{CARD2};border-radius:7px;padding:8px 10px;
#           display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
#           <span style="font-size:.73rem;color:{MUTED};">{lb}</span>
#           <span style="font-family:'JetBrains Mono',monospace;font-size:.77rem;
#             color:{ac};font-weight:600;">{vl}</span>
#         </div>""")
#     md("</div>")

#     md(f"""<div style="padding:14px 16px;">
#       <div style="height:1px;background:{BDR};margin-bottom:12px;"></div>
#       <div style="font-size:.69rem;color:{MUTED};line-height:1.8;text-align:center;">
#         XGBoost · SMOTE · GridSearchCV<br>Scikit-learn · Plotly · Streamlit
#       </div>
#     </div>""")


# # ─────────────────────────────────────────────
# # HERO
# # ─────────────────────────────────────────────
# _nc = f"{len(df):,}" if ok_d else "7,043"
# badges_html = "  ".join([
#     badge("SMOTE Balanced", GREEN),
#     badge("3 Models Tuned", INDIGO),
#     badge(_nc + " Customers", CYAN),
#     badge(f"Threshold {threshold:.4f}", AMBER),
#     badge("AUC 0.8434", BLUE),
# ])
# md(f"""<div style="padding:36px 0 22px;">
#   <div style="margin-bottom:14px;">
#     <span style="display:inline-flex;align-items:center;gap:5px;
#       background:{CARD};border:1px solid {BDR};border-radius:50px;
#       padding:4px 14px;font-size:.69rem;font-family:'JetBrains Mono',monospace;color:{MUTED};">
#       <span class="pdot"></span>LIVE ANALYTICS ENGINE
#     </span>
#   </div>
#   <h1 style="font-family:'Space Grotesk',sans-serif;font-size:2.4rem;
#     font-weight:700;color:{TEXT};line-height:1.12;margin-bottom:8px;">
#     Telco Customer
#     <span style="color:{BLUE};">Churn Intelligence</span>
#   </h1>
#   <p style="font-size:.9rem;color:{SUB};line-height:1.7;max-width:500px;margin-bottom:18px;">
#     AI-powered churn prediction · XGBoost + SMOTE + GridSearchCV · Optimal threshold calibration
#   </p>
#   <div style="display:flex;gap:8px;flex-wrap:wrap;">{badges_html}</div>
# </div>""")

# # KPI strip
# _nr  = f"{len(df):,}" if ok_d else "7,043"
# _nch = f"{df['ChurnBin'].sum():,}" if ok_d else "1,869"
# kc   = st.columns(5, gap="small")
# for col, (ic, lb, vl, ac, nt) in zip(kc, [
#     ("📡", "Model",     "Ready" if ok_m else "Missing", BLUE,  model_lbl if ok_m else "Add models/"),
#     ("🗄️", "Dataset",  "Loaded" if ok_d else "Optional", CYAN, f"{_nr} rows" if ok_d else "Add dataset.csv"),
#     ("👥", "Customers", _nr,                             GREEN, "total records"),
#     ("📉", "Churned",   _nch,                            RED,   "26.5% of total"),
#     ("🎯", "Threshold", f"{threshold:.4f}",              AMBER, "optimal F1 cut-off"),
# ]):
#     col.markdown(kpi(ic, lb, vl, ac, nt), unsafe_allow_html=True)

# md("<br>")

# # ─────────────────────────────────────────────
# # TABS
# # ─────────────────────────────────────────────
# TAB1, TAB2, TAB3 = st.tabs([
#     "🔮  Predict Churn",
#     "📊  EDA Dashboard",
#     "🤖  Model Insights",
# ])


# # ═══════════════════════════════════════════
# # TAB 1 — PREDICT
# # ═══════════════════════════════════════════
# with TAB1:
#     if not ok_m:
#         st.error(
#             f"**Model files not found.**  Missing: `{'`, `'.join(miss_files)}`  \n"
#             "Run the notebook end-to-end to generate PKL files, "
#             "then place them in a `models/` subfolder next to app.py."
#         )
#         md(f"""<div style="background:{CARD};border:1px dashed {BDR2};border-radius:14px;
#           padding:48px 24px;text-align:center;margin-top:12px;">
#           <div style="font-size:2.5rem;margin-bottom:12px;">📁</div>
#           <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;
#             font-weight:700;color:{TEXT};margin-bottom:8px;">Models Not Found</div>
#           <div style="font-size:.84rem;color:{SUB};line-height:1.7;">
#             Required: model.pkl · scaler.pkl · label_encoders.pkl
#             · feature_columns.pkl · model_info.pkl
#           </div>
#         </div>""")
#     else:
#         sec_hdr("👤", "Customer Profile",
#                 "Fill in all fields below, then click Run Prediction")

#         # CA, CB, CC = st.columns(3, gap="large")

#         # with CA:
#         #     md(f'<div style="font-size:.68rem;font-weight:700;color:{CYAN};'
#         #        f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#         #        f'Demographics</div>')
#         #     gender     = st.selectbox("Gender",         ["Male", "Female"])
#         #     senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
#         #     partner    = st.selectbox("Partner",        ["No", "Yes"])
#         #     dependents = st.selectbox("Dependents",     ["No", "Yes"])
#         #     tenure     = st.slider("Tenure (months)", 0, 72, 12)

#         # with CB:
#         #     md(f'<div style="font-size:.68rem;font-weight:700;color:{PURPLE};'
#         #        f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#         #        f'Services</div>')
#         #     phone_svc = st.selectbox("Phone Service", ["Yes", "No"])
#         #     if phone_svc == "No":
#         #         multi_lines = "No phone service"
#         #         st.selectbox("Multiple Lines", ["No phone service"], disabled=True)
#         #     else:
#         #         multi_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

#         #     internet_svc = st.selectbox("Internet Service",
#         #                                 ["DSL", "Fiber optic", "No"])
#         #     _io  = ["No internet service"] if internet_svc == "No" else ["No", "Yes"]
#         #     _dis = internet_svc == "No"

#         #     online_sec   = st.selectbox("Online Security",   _io, disabled=_dis)
#         #     online_bkp   = st.selectbox("Online Backup",     _io, disabled=_dis)
#         #     dev_protect  = st.selectbox("Device Protection", _io, disabled=_dis)
#         #     tech_support = st.selectbox("Tech Support",      _io, disabled=_dis)
#         #     stream_tv    = st.selectbox("Streaming TV",      _io, disabled=_dis)
#         #     stream_mov   = st.selectbox("Streaming Movies",  _io, disabled=_dis)

#         # with CC:
#         #     md(f'<div style="font-size:.68rem;font-weight:700;color:{TEAL};'
#         #        f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#         #        f'Billing &amp; Contract</div>')
#         #     contract  = st.selectbox("Contract Type",
#         #                              ["Month-to-month", "One year", "Two year"])
#         #     paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
#         #     payment   = st.selectbox("Payment Method", [
#         #         "Electronic check", "Mailed check",
#         #         "Bank transfer (automatic)", "Credit card (automatic)",
#         #     ])
#         #     monthly  = st.number_input("Monthly Charges ($)",
#         #                                18.0, 120.0, 65.0, 0.5, format="%.2f")
#         #     def_tot  = min(float(monthly) * max(int(tenure), 1), 8684.8)
#         #     total    = st.number_input("Total Charges ($)",
#         #                                0.0, 8684.8, def_tot, 10.0, format="%.2f")

#         #==================================
#         CA, CB, CC = st.columns(3, gap="large")

#         with CA:
#             md(f'<div style="font-size:.68rem;font-weight:700;color:{CYAN};'
#                f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#                f'Demographics</div>')
#             gender     = st.selectbox("Gender",         ["Male", "Female"], index=0) # Default: Male
#             senior     = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)      # Default: No (0)
#             partner    = st.selectbox("Partner",        ["No", "Yes"], index=1)      # Default: Yes
#             dependents = st.selectbox("Dependents",     ["No", "Yes"], index=0)      # Default: No
#             tenure     = st.slider("Tenure (months)", 0, 72, 12)                     # Default: 12

#         with CB:
#             md(f'<div style="font-size:.68rem;font-weight:700;color:{PURPLE};'
#                f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#                f'Services</div>')
#             phone_svc = st.selectbox("Phone Service", ["No", "Yes"], index=1)        # Default: Yes
#             if phone_svc == "No":
#                 multi_lines = "No phone service"
#                 st.selectbox("Multiple Lines", ["No phone service"], disabled=True)
#             else:
#                 multi_lines = st.selectbox("Multiple Lines", ["No", "Yes"], index=0) # Default: No

#             internet_svc = st.selectbox("Internet Service",
#                                         ["DSL", "Fiber optic", "No"], index=1)       # Default: Fiber optic
#             _io  = ["No internet service"] if internet_svc == "No" else ["No", "Yes"]
#             _dis = internet_svc == "No"

#             online_sec   = st.selectbox("Online Security",   _io, index=0, disabled=_dis) # Default: No
#             online_bkp   = st.selectbox("Online Backup",     _io, index=0, disabled=_dis) # Default: No
#             dev_protect  = st.selectbox("Device Protection", _io, index=0, disabled=_dis) # Default: No
#             tech_support = st.selectbox("Tech Support",      _io, index=0, disabled=_dis) # Default: No
#             stream_tv    = st.selectbox("Streaming TV",      _io, index=1, disabled=_dis) # Default: Yes
#             stream_mov   = st.selectbox("Streaming Movies",  _io, index=1, disabled=_dis) # Default: Yes

#         with CC:
#             md(f'<div style="font-size:.68rem;font-weight:700;color:{TEAL};'
#                f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
#                f'Billing &amp; Contract</div>')
#             contract  = st.selectbox("Contract Type",
#                                      ["Month-to-month", "One year", "Two year"], index=0) # Default: Month-to-month
#             paperless = st.selectbox("Paperless Billing", ["No", "Yes"], index=1)         # Default: Yes
#             payment   = st.selectbox("Payment Method", [
#                 "Bank transfer (automatic)", "Credit card (automatic)", 
#                 "Electronic check", "Mailed check"
#             ], index=2)                                                                   # Default: Electronic check
#             monthly  = st.number_input("Monthly Charges ($)",
#                                        18.0, 120.0, 70.35, 0.5, format="%.2f")            # Default: 70.35
#             total    = st.number_input("Total Charges ($)",
#                                        0.0, 8684.8, 843.0, 10.0, format="%.2f")           # Default: 843.0
        
#         md("<br>")
#         b1, b2, b3 = st.columns([1, 3, 1])
#         with b2:
#             clicked = st.button("🔮  RUN CHURN PREDICTION", use_container_width=True)

#         # ── Inference ──────────────────────────────────────────────
#         if clicked:
#             raw = {
#                 "gender":           gender,
#                 "SeniorCitizen":    1 if senior == "Yes" else 0,
#                 "Partner":          partner,
#                 "Dependents":       dependents,
#                 "tenure":           int(tenure),
#                 "PhoneService":     phone_svc,
#                 "MultipleLines":    multi_lines,
#                 "InternetService":  internet_svc,
#                 "OnlineSecurity":   online_sec   if internet_svc != "No" else "No internet service",
#                 "OnlineBackup":     online_bkp   if internet_svc != "No" else "No internet service",
#                 "DeviceProtection": dev_protect  if internet_svc != "No" else "No internet service",
#                 "TechSupport":      tech_support if internet_svc != "No" else "No internet service",
#                 "StreamingTV":      stream_tv    if internet_svc != "No" else "No internet service",
#                 "StreamingMovies":  stream_mov   if internet_svc != "No" else "No internet service",
#                 "Contract":         contract,
#                 "PaperlessBilling": paperless,
#                 "PaymentMethod":    payment,
#                 "MonthlyCharges":   float(monthly),
#                 "TotalCharges":     float(total),
#             }
            
#             try:
#                 inp = pd.DataFrame([raw])
                
#                 # 1. Safely label encode
#                 for col_name, le in arts["enc"].items():
#                     if col_name in inp.columns:
#                         v = str(inp.at[0, col_name])
#                         if v in le.classes_:
#                             inp.at[0, col_name] = le.transform([v])[0]
#                         else:
#                             inp.at[0, col_name] = 0  # Fallback safety
                            
#                 # 2. Enforce column order and float type
#                 inp = inp[arts["cols"]].astype(float)
                
#                 # 3. Scale features (returns a NumPy array)
#                 inp_sc_array = arts["scaler"].transform(inp)
                
#                 # 4. FIXED: Re-wrap in DataFrame so XGBoost recognizes feature names!
#                 inp_sc = pd.DataFrame(inp_sc_array, columns=arts["cols"])

#                 # 5. Predict
#                 proba   = arts["model"].predict_proba(inp_sc)[0]
#                 churn_p = float(proba[1]) * 100
#                 stay_p  = float(proba[0]) * 100
#                 pred    = int((churn_p / 100) >= threshold)

#                 if churn_p >= 65:
#                     tier_col, tier_lbl = RED,   "HIGH RISK"
#                 elif churn_p >= 35:
#                     tier_col, tier_lbl = AMBER, "MEDIUM RISK"
#                 else:
#                     tier_col, tier_lbl = GREEN, "LOW RISK"

#                 vcol = RED if pred else GREEN
#                 vtxt = "WILL LIKELY CHURN" if pred else "LIKELY TO STAY"
#                 vico = "⚠️" if pred else "✅"
#                 vbg  = "#1A0810" if pred else "#071510"

#                 divider()
#                 sec_hdr("📈", "Prediction Result",
#                         f"Model: {model_lbl}  ·  Threshold: {threshold:.4f}")

#                 R1, R2, R3 = st.columns([1.1, 1, 1], gap="large")

            
#            #===================
#             # try:
#             #     inp = pd.DataFrame([raw])
#             #     for col_name, le in arts["enc"].items():
#             #         if col_name in inp.columns:
#             #             v = str(inp.at[0, col_name])
#             #             inp[col_name] = le.transform([v]) if v in le.classes_ else [0]
#             #     inp    = inp[arts["cols"]].astype(float)
#             #     inp_sc = arts["scaler"].transform(inp)

#             #     proba   = arts["model"].predict_proba(inp_sc)[0]
#             #     churn_p = float(proba[1]) * 100
#             #     stay_p  = float(proba[0]) * 100
#             #     pred    = int((churn_p / 100) >= threshold)

#             #     if churn_p >= 65:
#             #         tier_col, tier_lbl = RED,   "HIGH RISK"
#             #     elif churn_p >= 35:
#             #         tier_col, tier_lbl = AMBER, "MEDIUM RISK"
#             #     else:
#             #         tier_col, tier_lbl = GREEN, "LOW RISK"

#             #     vcol = RED if pred else GREEN
#             #     vtxt = "WILL LIKELY CHURN" if pred else "LIKELY TO STAY"
#             #     vico = "⚠️" if pred else "✅"
#             #     vbg  = "#1A0810" if pred else "#071510"

#             #     divider()
#             #     sec_hdr("📈", "Prediction Result",
#             #             f"Model: {model_lbl}  ·  Threshold: {threshold:.4f}")

#             #     R1, R2, R3 = st.columns([1.1, 1, 1], gap="large")

#                 # Verdict card
#                 with R1:
#                     md(f"""<div class="sup" style="background:{vbg};border:1px solid {vcol}30;
#                       border-radius:14px;padding:26px 20px;text-align:center;">
#                       <div style="font-size:.6rem;font-family:'JetBrains Mono',monospace;
#                         color:{vcol};letter-spacing:2.5px;margin-bottom:9px;">MODEL VERDICT</div>
#                       <div style="font-size:1.8rem;margin-bottom:6px;">{vico}</div>
#                       <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;
#                         font-weight:700;color:{TEXT};margin-bottom:16px;">{vtxt}</div>
#                       <div style="display:flex;justify-content:center;gap:22px;margin-bottom:14px;">
#                         <div>
#                           <div style="font-family:'Space Grotesk',sans-serif;font-size:2.1rem;
#                             font-weight:700;color:{RED};">{churn_p:.1f}%</div>
#                           <div style="font-size:.63rem;color:{MUTED};text-transform:uppercase;
#                             letter-spacing:.7px;margin-top:2px;">Churn Risk</div>
#                         </div>
#                         <div style="width:1px;background:{BDR};"></div>
#                         <div>
#                           <div style="font-family:'Space Grotesk',sans-serif;font-size:2.1rem;
#                             font-weight:700;color:{GREEN};">{stay_p:.1f}%</div>
#                           <div style="font-size:.63rem;color:{MUTED};text-transform:uppercase;
#                             letter-spacing:.7px;margin-top:2px;">Retention</div>
#                         </div>
#                       </div>
#                       <div style="display:inline-flex;align-items:center;gap:6px;
#                         background:{tier_col}18;border:1px solid {tier_col}40;
#                         border-radius:8px;padding:7px 14px;">
#                         <span style="font-size:.82rem;color:{tier_col};font-weight:700;">{tier_lbl}</span>
#                       </div>
#                     </div>""")

#                 # Gauge + bar
#                 # with R2:
#                 #     fig_g = go.Figure(go.Indicator(
#                 #         mode="gauge+number",
#                 #         value=churn_p,
#                 #         number=dict(suffix="%",
#                 #                     font=dict(size=26, color=tier_col,
#                 #                               family="Space Grotesk")),
#                 #         gauge=dict(
#                 #             axis=dict(range=[0, 100],
#                 #                       tickcolor=MUTED,
#                 #                       tickfont=dict(color=MUTED, size=9)),
#                 #             bar=dict(color=tier_col, thickness=0.62),
#                 #             bgcolor="rgba(0,0,0,0)",
#                 #             borderwidth=0,
#                 #             steps=[
#                 #                 dict(range=[0,  35], color="#071510"),
#                 #                 dict(range=[35, 65], color="#15110A"),
#                 #                 dict(range=[65,100], color="#150809"),
#                 #             ],
#                 #             threshold=dict(
#                 #                 line=dict(color=AMBER, width=2),
#                 #                 thickness=0.75,
#                 #                 value=threshold * 100,
#                 #             ),
#                 #         ),
#                 #     ))
#                 #     fig_g.update_layout(
#                 #         paper_bgcolor="rgba(0,0,0,0)",
#                 #         plot_bgcolor="rgba(0,0,0,0)",
#                 #         font=dict(color=SUB, family="Inter"),
#                 #         margin=dict(l=10, r=10, t=10, b=20),
#                 #         height=200,
#                 #         annotations=[dict(
#                 #             text=f"Decision threshold: {threshold*100:.1f}%",
#                 #             x=0.5, y=-0.18, showarrow=False,
#                 #             font=dict(color=MUTED, size=9),
#                 #         )],
#                 #     )
#                 #     st.plotly_chart(fig_g, use_container_width=True)

#                 #     fig_b = go.Figure()
#                 #     fig_b.add_trace(go.Bar(
#                 #         y=["Retention", "Churn Risk"],
#                 #         x=[stay_p, churn_p],
#                 #         orientation="h",
#                 #         marker_color=[GREEN, RED],
#                 #         marker_line_color="rgba(0,0,0,0)",
#                 #         text=[f"{stay_p:.1f}%", f"{churn_p:.1f}%"],
#                 #         textposition="outside",
#                 #         textfont=dict(color=TEXT, size=11),
#                 #     ))
#                 #     fig_b.add_vline(
#                 #         x=threshold * 100, line_dash="dash",
#                 #         line_color=AMBER, line_width=1.5,
#                 #         annotation_text=f"Threshold {threshold*100:.1f}%",
#                 #         annotation_font_color=AMBER,
#                 #         annotation_font_size=9,
#                 #         annotation_position="top right",
#                 #     )
#                 #     fig_b.update_layout(
#                 #         paper_bgcolor="rgba(0,0,0,0)",
#                 #         plot_bgcolor="rgba(0,0,0,0)",
#                 #         margin=dict(l=8, r=55, t=6, b=6),
#                 #         height=110,
#                 #         xaxis=dict(range=[0, 118], gridcolor=BDR,
#                 #                    tickfont=dict(color=MUTED), showticklabels=False),
#                 #         yaxis=dict(tickfont=dict(color=SUB, size=10),
#                 #                    gridcolor="rgba(0,0,0,0)"),
#                 #         showlegend=False,
#                 #     )
#                 #     st.plotly_chart(fig_b, use_container_width=True)
                
#                 #================================================
#                 # Gauge + bar
#                 with R2:
#                     fig_g = go.Figure(go.Indicator(
#                         mode="gauge+number",
#                         value=churn_p,
#                         title=dict(text="Confidence Level", font=dict(size=14, color=TEXT, family="Space Grotesk")),
#                         number=dict(suffix="%",
#                                     font=dict(size=32, color=tier_col,
#                                               family="Space Grotesk")),
#                         gauge=dict(
#                             axis=dict(range=[0, 100],
#                                       tickcolor=MUTED,
#                                       tickfont=dict(color=MUTED, size=10)),
#                             bar=dict(color=tier_col, thickness=0.7),
#                             bgcolor="rgba(0,0,0,0)",
#                             borderwidth=0,
#                             steps=[
#                                 dict(range=[0,  35], color=CARD2),
#                                 dict(range=[35, 65], color=BDR),
#                                 dict(range=[65,100], color=BDR2),
#                             ],
#                             threshold=dict(
#                                 line=dict(color=TEXT, width=2.5),
#                                 thickness=0.8,
#                                 value=threshold * 100,
#                             ),
#                         ),
#                     ))
#                     fig_g.update_layout(
#                         paper_bgcolor="rgba(0,0,0,0)",
#                         plot_bgcolor="rgba(0,0,0,0)",
#                         font=dict(color=SUB, family="Inter"),
#                         margin=dict(l=15, r=15, t=30, b=10),
#                         height=220,
#                     )
#                     st.plotly_chart(fig_g, use_container_width=True)

#                     fig_b = go.Figure()
#                     fig_b.add_trace(go.Bar(
#                         y=["Retention", "Churn Risk"],
#                         x=[stay_p, churn_p],
#                         orientation="h",
#                         marker_color=[GREEN, RED],
#                         marker_line_color="rgba(0,0,0,0)",
#                         text=[f"{stay_p:.1f}%", f"{churn_p:.1f}%"],
#                         textposition="outside",
#                         textfont=dict(color=TEXT, size=11, family="Space Grotesk"),
#                     ))
#                     fig_b.add_vline(
#                         x=threshold * 100, line_dash="dash",
#                         line_color=TEXT, line_width=1.5,
#                         annotation_text=f"Threshold {threshold*100:.1f}%",
#                         annotation_font_color=SUB,
#                         annotation_font_size=10,
#                         annotation_position="top right",
#                     )
#                     fig_b.update_layout(
#                         paper_bgcolor="rgba(0,0,0,0)",
#                         plot_bgcolor="rgba(0,0,0,0)",
#                         margin=dict(l=5, r=50, t=10, b=10),
#                         height=120,
#                         xaxis=dict(range=[0, 115], gridcolor="rgba(0,0,0,0)",
#                                    tickfont=dict(color=MUTED), showticklabels=False),
#                         yaxis=dict(tickfont=dict(color=TEXT, size=11),
#                                    gridcolor="rgba(0,0,0,0)"),
#                         showlegend=False,
#                     )
#                     st.plotly_chart(fig_b, use_container_width=True)
#                 #================================================

#                 # Actions
#                 with R3:
#                     if pred:
#                         actions = [
#                             ("📞", "Call within 24 hours",       RED),
#                             ("💰", "Offer personalised discount", AMBER),
#                             ("📋", "Propose annual contract",      BLUE),
#                             ("🛡️", "Add free Tech Support trial", PURPLE),
#                             ("🎁", "Send loyalty reward offer",   GREEN),
#                         ]
#                         hdr_col, hdr_txt = RED,   "🚨 Retention Actions"
#                     else:
#                         actions = [
#                             ("⭐", "Enrol in rewards programme",  GREEN),
#                             ("📦", "Upsell premium services",     TEAL),
#                             ("🔁", "Encourage contract upgrade",  BLUE),
#                             ("📣", "Request referral or review",  PURPLE),
#                             ("📊", "Monitor usage trends",        CYAN),
#                         ]
#                         hdr_col, hdr_txt = GREEN, "✅ Growth Actions"

#                     rows_h = "".join(
#                         f'<div style="display:flex;align-items:center;gap:9px;'
#                         f'padding:8px 10px;border-radius:7px;margin-bottom:4px;'
#                         f'background:{ac}0C;border:1px solid {ac}1A;">'
#                         f'<span style="font-size:.88rem;">{ico}</span>'
#                         f'<span style="font-size:.78rem;color:{SUB};">{txt}</span>'
#                         f'</div>'
#                         for ico, txt, ac in actions
#                     )
#                     md(f'<div style="background:{CARD};border:1px solid {BDR};'
#                        f'border-radius:13px;padding:16px;">'
#                        f'<div style="font-size:.74rem;font-weight:700;color:{hdr_col};'
#                        f'font-family:Space Grotesk,sans-serif;margin-bottom:12px;">'
#                        f'{hdr_txt}</div>{rows_h}</div>')

#                 # ── What Drives This Prediction ─────────────────────
#                 md("<br>")
#                 sec_hdr("🔍", "What Drives This Prediction?",
#                         "Risk factors identified from EDA patterns for this customer profile")

#                 # Build risk factors — simple lists, no complex unpacking
#                 high_risk, med_risk, low_risk = [], [], []

#                 if "Month-to-month" in contract:
#                     high_risk.append((
#                         "📅 Month-to-month Contract",
#                         "Historical churn rate: 42.7% — highest of all contract types",
#                         "vs only 2.8% churn on 2-year plans",
#                     ))
#                 if "Electronic" in payment:
#                     high_risk.append((
#                         "💳 Electronic Check Payment",
#                         "Historical churn rate: 45.3% — highest of all payment methods",
#                         "Customers on auto-pay churn significantly less",
#                     ))
#                 if internet_svc == "Fiber optic":
#                     high_risk.append((
#                         "🌐 Fiber Optic Internet",
#                         "Historical churn rate: 41.9% vs DSL at 18.9%",
#                         "Price-to-value perception is the key driver",
#                     ))
#                 if int(tenure) <= 12:
#                     high_risk.append((
#                         "🆕 New Customer (first 12 months)",
#                         f"First-year churn rate: 47.7% — highest risk period",
#                         f"This customer has {tenure} months tenure",
#                     ))
#                 if online_sec == "No" and internet_svc != "No":
#                     med_risk.append((
#                         "🔒 No Online Security",
#                         "Customers without it churn at 41.8%",
#                         "Adding this service reduces dissatisfaction",
#                     ))
#                 if tech_support == "No" and internet_svc != "No":
#                     med_risk.append((
#                         "🛠 No Tech Support",
#                         "Customers without it churn at 41.6%",
#                         "Proactive support prevents frustration",
#                     ))
#                 if float(monthly) > 70:
#                     med_risk.append((
#                         "💸 High Monthly Charges",
#                         f"${monthly:.0f}/month exceeds the $70 risk threshold",
#                         "High charges correlate strongly with churn",
#                     ))
#                 if senior == "Yes":
#                     med_risk.append((
#                         "👴 Senior Citizen",
#                         "Churn rate 41.7% vs 24.3% for non-seniors",
#                         "Dedicated support programmes improve retention",
#                     ))
#                 if contract == "Two year":
#                     low_risk.append((
#                         "📋 Two-Year Contract",
#                         "Churn rate only 2.8% — strongest retention signal",
#                         "Long-term commitment is the best loyalty indicator",
#                     ))
#                 if int(tenure) >= 48:
#                     low_risk.append((
#                         "🏆 Long-Tenure Customer",
#                         f"{tenure} months loyalty — churn risk drops dramatically",
#                         "Customers over 48 months churn at only 6.6%",
#                     ))
#                 if contract == "One year":
#                     low_risk.append((
#                         "📋 One-Year Contract",
#                         "Contracted customers churn far less than month-to-month",
#                         "Encourage upgrade to 2-year for even better retention",
#                     ))
#                 if online_sec == "Yes" and tech_support == "Yes":
#                     low_risk.append((
#                         "✅ Fully Supported Customer",
#                         "Both online security and tech support active",
#                         "Well-served customers have lower churn propensity",
#                     ))

#                 # Fallback if nothing detected
#                 if not high_risk and not med_risk and not low_risk:
#                     low_risk.append((
#                         "✅ No Major Risk Factors",
#                         "This customer does not match high-risk churn profiles",
#                         "Continue monitoring engagement and usage metrics",
#                     ))

#                 # Render in 3 columns — safe, no complex unpacking in f-strings
#                 all_factors = (
#                     [(RED,   "High Risk Factor",   t) for t in high_risk] +
#                     [(AMBER, "Medium Risk Factor", t) for t in med_risk]  +
#                     [(GREEN, "Positive Signal",    t) for t in low_risk]
#                 )

#                 if all_factors:
#                     ncols = min(3, len(all_factors))
#                     if ncols < 1:
#                         ncols = 1
#                     fcols = st.columns(ncols, gap="medium")
#                     for idx, factor in enumerate(all_factors):
#                         col_c   = factor[0]
#                         col_lbl = factor[1]
#                         title_f, desc_f, note_f = factor[2]
#                         card_html = risk_card_html(col_c, col_lbl, title_f, desc_f, note_f)
#                         with fcols[idx % ncols]:
#                             md(card_html)

#                 # Summary
#                 n_h = len(high_risk)
#                 n_m = len(med_risk)
#                 n_l = len(low_risk)
#                 s_col = RED if n_h >= 2 else AMBER if (n_h == 1 or n_m >= 2) else GREEN
#                 md(f'<div style="background:{s_col}0A;border:1px solid {s_col}28;'
#                    f'border-radius:10px;padding:12px 16px;margin-top:10px;">'
#                    f'<span style="font-size:.82rem;color:{TEXT};font-weight:600;">'
#                    f'Risk Summary: </span>'
#                    f'<span style="font-size:.82rem;color:{SUB};">'
#                    f'{n_h} high-risk · {n_m} medium-risk · {n_l} positive signal(s). '
#                    f'Churn probability: </span>'
#                    f'<strong style="color:{tier_col};">{churn_p:.1f}% ({tier_lbl})</strong>'
#                    f'</div>')

#             except Exception as exc:
#                 st.error(f"**Prediction error:** {exc}")
#                 st.info(
#                     "Check that all 5 PKL files in `models/` were generated by "
#                     "running the notebook end-to-end with the same dataset."
#                 )

#         else:
#             divider()
#             md(f'<div style="background:{CARD};border:1px dashed {BDR2};'
#                f'border-radius:14px;padding:48px 24px;text-align:center;margin:8px 0;">'
#                f'<div style="font-size:2.4rem;margin-bottom:12px;">🔮</div>'
#                f'<div style="font-family:Space Grotesk,sans-serif;font-size:1rem;'
#                f'font-weight:700;color:{TEXT};margin-bottom:8px;">Ready to Analyse</div>'
#                f'<div style="font-size:.84rem;color:{SUB};line-height:1.7;'
#                f'max-width:360px;margin:0 auto;">Complete the customer profile above '
#                f'and click <strong style="color:{BLUE};">Run Churn Prediction</strong> '
#                f'to see AI results with risk insights.</div></div>')

#             md("<br>")
#             sec_hdr("📌", "Key Churn Signals",
#                     "Historical EDA patterns — top factors that drive customer churn")
#             s1, s2 = st.columns(2, gap="large")
#             with s1:
#                 for lb, rt, ac in [
#                     ("Month-to-month contract",  "42.7% churn rate", RED),
#                     ("Electronic check payment", "45.3% churn rate", RED),
#                     ("Fiber optic internet",     "41.9% churn rate", RED),
#                     ("Tenure ≤ 12 months",       "47.7% churn rate", RED),
#                 ]:
#                     md(signal_row(lb, rt, ac))
#             with s2:
#                 for lb, rt, ac in [
#                     ("No online security",  "41.8% churn rate", AMBER),
#                     ("No tech support",     "41.6% churn rate", AMBER),
#                     ("Two-year contract",   " 2.8% churn rate", GREEN),
#                     ("Tenure > 48 months",  " 6.6% churn rate", GREEN),
#                 ]:
#                     md(signal_row(lb, rt, ac))


# # ═══════════════════════════════════════════
# # TAB 2 — EDA DASHBOARD
# # ═══════════════════════════════════════════
# with TAB2:
#     if not ok_d:
#         st.warning(
#             "**dataset.csv not found.**  "
#             "Place it in the same folder as app.py and restart Streamlit."
#         )
#         md(f'<div style="background:{CARD};border:1px dashed {BDR2};'
#            f'border-radius:14px;padding:48px 24px;text-align:center;margin-top:12px;">'
#            f'<div style="font-size:2.4rem;margin-bottom:12px;">📊</div>'
#            f'<div style="font-family:Space Grotesk,sans-serif;font-size:1rem;'
#            f'font-weight:700;color:{TEXT};margin-bottom:8px;">Dataset Required</div>'
#            f'<div style="font-size:.84rem;color:{SUB};line-height:1.7;">'
#            f'Add dataset.csv to the app folder to view interactive EDA charts.</div></div>')
#     else:
#         # KPIs
#         sec_hdr("📊", "Dataset Overview",
#                 "7,043 Telco customers · 21 raw features · Kaggle Telco Churn dataset")
#         kc2 = st.columns(6, gap="small")
#         for col, (ic, lb, vl, ac, nt) in zip(kc2, [
#             ("👥", "Total",       f"{len(df):,}",                              BLUE,   ""),
#             ("🔴", "Churned",     f"{df['ChurnBin'].sum():,}",                 RED,    f"{df['ChurnBin'].mean()*100:.1f}%"),
#             ("🟢", "Retained",    f"{(~df['ChurnBin'].astype(bool)).sum():,}", GREEN,  f"{(1-df['ChurnBin'].mean())*100:.1f}%"),
#             ("📅", "Avg Tenure",  f"{df['tenure'].mean():.0f} mo",             PURPLE, "months"),
#             ("💵", "Avg Monthly", f"${df['MonthlyCharges'].mean():.0f}",       AMBER,  "/month"),
#             ("🧮", "Features",    "19",                                         TEAL,   "post-encoding"),
#         ]):
#             col.markdown(kpi(ic, lb, vl, ac, nt), unsafe_allow_html=True)

#         divider()

#         # ── 1. Churn Distribution ───────────────────────────────
#         sec_hdr("📉", "1 · Churn Distribution",
#                 "73.5% No Churn vs 26.5% Churned — class imbalance handled with SMOTE on training data")

#         no_ct  = int((df["Churn"] == "No").sum())
#         yes_ct = int((df["Churn"] == "Yes").sum())

#         ca, cb = st.columns(2, gap="large")
#         with ca:
#             fig = go.Figure(go.Bar(
#                 x=["No Churn", "Churned"],
#                 y=[no_ct, yes_ct],
#                 marker_color=[GREEN, RED],
#                 marker_line_color="rgba(0,0,0,0)",
#                 text=[f"{no_ct:,}", f"{yes_ct:,}"],
#                 textposition="outside",
#                 textfont=dict(color=TEXT, size=13),
#             ))
#             fig.update_layout(**plotly_layout("Churn Count", 300))
#             fig.update_yaxes(title="Customers", gridcolor=BDR)
#             fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig, use_container_width=True)

#         with cb:
#             fig = go.Figure(go.Pie(
#                 labels=["No Churn", "Churned"],
#                 values=[no_ct, yes_ct],
#                 hole=0.44,
#                 pull=[0, 0.05],
#                 marker=dict(colors=[GREEN, RED],
#                             line=dict(color=BG, width=2)),
#                 textinfo="label+percent",
#                 textfont=dict(size=12),
#             ))
#             fig.update_layout(**plotly_layout("Churn Split (%)", 300))
#             fig.update_layout(showlegend=False)
#             st.plotly_chart(fig, use_container_width=True)

#         md(f'<div style="background:{AMBER}0A;border:1px solid {AMBER}30;'
#            f'border-radius:9px;padding:11px 16px;margin:6px 0;">'
#            f'<strong style="color:{AMBER};">⚠ Class Imbalance:</strong>'
#            f'<span style="color:{SUB};font-size:.84rem;"> 73.5% vs 26.5% — '
#            f'SMOTE applied on training data only to prevent data leakage.</span></div>')

#         # ── 2. Numerical Distributions ──────────────────────────
#         divider()
#         sec_hdr("📐", "2 · Numerical Feature Distributions",
#                 "Tenure, MonthlyCharges, TotalCharges — split by churn status")

#         num_cols_list = ["tenure", "MonthlyCharges", "TotalCharges"]
#         fig2a = make_subplots(
#             rows=1, cols=3,
#             subplot_titles=["Tenure (months)", "Monthly Charges ($)", "Total Charges ($)"],
#             horizontal_spacing=0.06,
#         )
#         for i, col_name in enumerate(num_cols_list, 1):
#             for churn_val, nm, clr in [
#                 (False, "No Churn", BLUE),
#                 (True,  "Churned",  RED),
#             ]:
#                 sub = df[df["ChurnBin"] == int(churn_val)][col_name]
#                 fig2a.add_trace(go.Histogram(
#                     x=sub, name=nm, marker_color=clr,
#                     opacity=0.7, nbinsx=30,
#                     showlegend=(i == 1),
#                     legendgroup=nm,
#                 ), row=1, col=i)
#         lyt2a = plotly_layout("Distribution by Churn Status", 320)
#         lyt2a["barmode"] = "overlay"
#         fig2a.update_layout(**lyt2a)
#         fig2a.update_xaxes(gridcolor=BDR, linecolor=BDR)
#         fig2a.update_yaxes(gridcolor=BDR, linecolor=BDR)
#         for ann in fig2a.layout.annotations:
#             ann.font.color = SUB
#             ann.font.size  = 11
#         st.plotly_chart(fig2a, use_container_width=True)

#         fig2b = make_subplots(
#             rows=1, cols=3,
#             subplot_titles=["Tenure vs Churn", "Monthly Charges vs Churn",
#                             "Total Charges vs Churn"],
#             horizontal_spacing=0.06,
#         )
#         for i, col_name in enumerate(num_cols_list, 1):
#             for churn_val, nm, clr in [
#                 ("No", "No Churn", BLUE),
#                 ("Yes", "Churned", RED),
#             ]:
#                 fig2b.add_trace(go.Box(
#                     y=df[df["Churn"] == churn_val][col_name],
#                     name=nm, marker_color=clr, line_color=clr,
#                     showlegend=(i == 1), legendgroup=nm,
#                 ), row=1, col=i)
#         lyt2b = plotly_layout("Spread by Churn Status", 320)
#         lyt2b["boxmode"] = "group"
#         fig2b.update_layout(**lyt2b)
#         fig2b.update_xaxes(gridcolor="rgba(0,0,0,0)")
#         fig2b.update_yaxes(gridcolor=BDR)
#         for ann in fig2b.layout.annotations:
#             ann.font.color = SUB
#             ann.font.size  = 11
#         st.plotly_chart(fig2b, use_container_width=True)

#         md(f'<div style="background:{CARD2};border:1px solid {BDR};'
#            f'border-radius:9px;padding:11px 16px;margin:4px 0;">'
#            f'<span style="font-size:.82rem;color:{SUB};">'
#            f'<strong style="color:{RED};">Tenure:</strong> Bimodal — many new + many long-term customers &nbsp;·&nbsp; '
#            f'<strong style="color:{AMBER};">MonthlyCharges:</strong> Churners tend to pay more &nbsp;·&nbsp; '
#            f'<strong style="color:{BLUE};">TotalCharges:</strong> Right-skewed, correlated with tenure'
#            f'</span></div>')

#         # ── 3. Categorical Churn Rates ────────────────────────
#         divider()
#         sec_hdr("📊", "3 · Churn Rate by Category",
#                 "Which service and billing categories drive churn most?")

#         cat_feats = [
#             "Contract", "InternetService", "PaymentMethod",
#             "TechSupport", "OnlineSecurity", "PaperlessBilling",
#         ]
#         fig3 = make_subplots(
#             rows=2, cols=3,
#             subplot_titles=cat_feats,
#             vertical_spacing=0.2,
#             horizontal_spacing=0.07,
#         )
#         for idx, feat in enumerate(cat_feats):
#             r, c = divmod(idx, 3)
#             cr = (df.groupby(feat)["Churn"]
#                   .apply(lambda x: (x == "Yes").mean() * 100)
#                   .sort_values(ascending=False)
#                   .reset_index())
#             cr.columns = [feat, "Pct"]
#             bclrs = [RED if v > 30 else AMBER if v > 15 else GREEN
#                      for v in cr["Pct"]]
#             fig3.add_trace(go.Bar(
#                 x=cr[feat], y=cr["Pct"],
#                 marker_color=bclrs,
#                 marker_line_color="rgba(0,0,0,0)",
#                 text=[f"{v:.1f}%" for v in cr["Pct"]],
#                 textposition="outside",
#                 textfont=dict(color=TEXT, size=9),
#                 showlegend=False,
#             ), row=r+1, col=c+1)

#         lyt3 = plotly_layout("Churn Rate (%) by Feature Category", 540)
#         fig3.update_layout(**lyt3)
#         fig3.update_xaxes(tickfont=dict(size=8, color=SUB),
#                           gridcolor="rgba(0,0,0,0)")
#         fig3.update_yaxes(gridcolor=BDR, tickfont=dict(color=MUTED))
#         for ann in fig3.layout.annotations:
#             ann.font.color = SUB
#             ann.font.size  = 11
#         st.plotly_chart(fig3, use_container_width=True)

#         md(f'<div style="background:{CARD2};border:1px solid {BDR};'
#            f'border-radius:9px;padding:11px 16px;margin:4px 0;">'
#            f'<span style="font-size:.82rem;color:{SUB};">'
#            f'<strong style="color:{RED};">Contract:</strong> Month-to-month 42.7% vs Two-year 2.8% &nbsp;·&nbsp; '
#            f'<strong style="color:{RED};">Payment:</strong> Electronic check 45.3% &nbsp;·&nbsp; '
#            f'<strong style="color:{AMBER};">Internet:</strong> Fiber optic 41.9% vs DSL 18.9% &nbsp;·&nbsp; '
#            f'<strong style="color:{AMBER};">Tech Support:</strong> Without 41.6% vs With 14.8%'
#            f'</span></div>')

#         # ── 4. Demographics ────────────────────────────────────
#         divider()
#         sec_hdr("👥", "4 · Demographics vs Churn",
#                 "Gender, Senior Citizen, Partner, Dependents breakdown")

#         demo_feats = ["gender", "SeniorCitizen", "Partner", "Dependents"]
#         d_cols = st.columns(4, gap="small")
#         for i, feat in enumerate(demo_feats):
#             with d_cols[i]:
#                 ct = (df.groupby(feat)["Churn"]
#                       .apply(lambda x: (x == "Yes").mean() * 100)
#                       .reset_index())
#                 ct.columns = [feat, "Pct"]
#                 # Convert x to string to handle int (SeniorCitizen = 0/1)
#                 x_labels = ct[feat].astype(str).tolist()
#                 clrs = [BLUE, RED] if len(ct) == 2 else [BLUE]*len(ct)
#                 fig_d = go.Figure(go.Bar(
#                     x=x_labels,
#                     y=ct["Pct"].tolist(),
#                     marker_color=clrs,
#                     marker_line_color="rgba(0,0,0,0)",
#                     text=[f"{v:.1f}%" for v in ct["Pct"]],
#                     textposition="outside",
#                     textfont=dict(color=TEXT, size=10),
#                 ))
#                 fig_d.update_layout(**plotly_layout(feat, 240))
#                 fig_d.update_layout(margin=dict(l=5, r=5, t=40, b=5))
#                 fig_d.update_yaxes(title="Churn Rate (%)", range=[0, max(ct["Pct"])*1.32])
#                 fig_d.update_xaxes(gridcolor="rgba(0,0,0,0)")
#                 st.plotly_chart(fig_d, use_container_width=True)

#         # ── 5. Tenure Groups ───────────────────────────────────
#         divider()
#         sec_hdr("⏱️", "5 · Churn Rate by Tenure Group",
#                 "0-12 months is the highest-risk window — 47.7% churn rate")

#         tg = (df.groupby("TenureGroup", observed=True)["Churn"]
#               .apply(lambda x: (x == "Yes").mean() * 100)
#               .reset_index())
#         tg.columns = ["Group", "Pct"]
#         tg["Group"] = tg["Group"].astype(str)  # safe cast from Categorical

#         tg_colors = [RED, AMBER, BLUE, TEAL]
#         tc1, tc2 = st.columns([2, 1], gap="large")
#         with tc1:
#             fig_tg = go.Figure(go.Bar(
#                 x=tg["Group"].tolist(),
#                 y=tg["Pct"].tolist(),
#                 marker_color=tg_colors[:len(tg)],
#                 marker_line_color="rgba(0,0,0,0)",
#                 text=[f"{v:.1f}%" for v in tg["Pct"]],
#                 textposition="outside",
#                 textfont=dict(color=TEXT, size=12),
#             ))
#             fig_tg.update_layout(**plotly_layout("Churn Rate by Tenure Group", 300))
#             fig_tg.update_yaxes(title="Churn Rate (%)", gridcolor=BDR)
#             fig_tg.update_xaxes(title="Tenure Group", gridcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig_tg, use_container_width=True)
#         with tc2:
#             for row_idx, row in tg.iterrows():
#                 ac = tg_colors[row_idx] if row_idx < len(tg_colors) else BLUE
#                 md(f'<div style="background:{CARD};border:1px solid {ac}30;'
#                    f'border-left:3px solid {ac};border-radius:8px;'
#                    f'padding:12px 14px;margin-bottom:7px;">'
#                    f'<div style="font-size:.69rem;color:{MUTED};margin-bottom:2px;">'
#                    f'{row["Group"]}</div>'
#                    f'<div style="font-family:Space Grotesk,sans-serif;font-size:1.4rem;'
#                    f'font-weight:700;color:{ac};">{row["Pct"]:.1f}%</div>'
#                    f'</div>')

#         # ── 6. Correlation Heatmap ─────────────────────────────
#         divider()
#         sec_hdr("🔥", "6 · Feature Correlation Heatmap",
#                 "Label-encoded · lower triangle only · key correlations with Churn")

#         df_corr = df.drop(
#             columns=["ChurnBin", "TenureGroup", "customerID"], errors="ignore"
#         ).copy()
#         le_tmp = LabelEncoder()
#         # Fix: use include="object" with str handling for newer pandas
#         obj_cols = [c for c in df_corr.columns
#                     if df_corr[c].dtype == object or df_corr[c].dtype.name == "string"]
#         for col_name in obj_cols:
#             df_corr[col_name] = le_tmp.fit_transform(df_corr[col_name].astype(str))

#         corr = df_corr.corr().round(2)
#         # Mask upper triangle — replace with NaN
#         mask = np.triu(np.ones_like(corr.values, dtype=bool), k=1)
#         z_vals = corr.values.copy().astype(float)
#         z_vals[mask] = np.nan

#         # Build text array — show value or empty string (no "nan")
#         text_vals = []
#         for row_arr in z_vals:
#             row_text = []
#             for v in row_arr:
#                 row_text.append("" if np.isnan(v) else str(round(v, 2)))
#             text_vals.append(row_text)

#         # fig_hm = go.Figure(go.Heatmap(
#         #     z=z_vals,
#         #     x=corr.columns.tolist(),
#         #     y=corr.index.tolist(),
#         #     colorscale="RdBu_r",
#         #     zmid=0,
#         #     zmin=-1, zmax=1,
#         #     text=text_vals,
#         #     texttemplate="%{text}",
#         #     textfont=dict(size=7, color="white"),
#         #     hoverongaps=False,
#         #  colorbar=dict(
#         #         title=dict(
#         #             text="r", 
#         #             font=dict(color=SUB, size=10)
#         #         ),
#         #         tickfont=dict(color=SUB, size=9),
#         #     ),
#         # ))
#         # hm_layout = plotly_layout("Feature Correlation Matrix", 520)
#         # hm_layout["xaxis"] = dict(
#         #     tickfont=dict(size=8, color=SUB), tickangle=45,
#         #     gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
#         # )
#         # hm_layout["yaxis"] = dict(
#         #     tickfont=dict(size=8, color=SUB),
#         #     gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
#         # )
#         # hm_layout["margin"] = dict(l=120, r=20, t=50, b=120)
#         # fig_hm.update_layout(**hm_layout)
#         # st.plotly_chart(fig_hm, use_container_width=True)

#         # md(f'<div style="background:{CARD2};border:1px solid {BDR};'
#         #    f'border-radius:9px;padding:11px 16px;margin:4px 0;">'
#         #    f'<span style="font-size:.82rem;color:{SUB};">'
#         #    f'<strong style="color:{TEAL};">tenure</strong> negatively correlated with Churn — '
#         #    f'long-term customers stay &nbsp;·&nbsp; '
#         #    f'<strong style="color:{RED};">MonthlyCharges</strong> positively correlated — '
#         #    f'expensive plans drive churn &nbsp;·&nbsp; '
#         #    f'TotalCharges highly correlated with tenure (expected)'
#         #    f'</span></div>')
        
#         #===========================
#         fig_hm = go.Figure(go.Heatmap(
#             z=z_vals,
#             x=corr.columns.tolist(),
#             y=corr.index.tolist(),
#             colorscale=[
#                 [0.00, '#3b4cc0'], [0.15, '#6788ee'], [0.35, '#9abbff'], 
#                 [0.50, '#e2e2e2'], [0.65, '#f1a88d'], [0.85, '#d35c4e'], [1.00, '#b40426']
#             ],
#             zmid=0,
#             zmin=-1, zmax=1,
#             text=text_vals,
#             texttemplate="<b>%{text}</b>",
#             textfont=dict(size=11),  # <-- FIXED: Plotly will now auto-contrast the text!
#             hoverongaps=False,
#             colorbar=dict(
#                 title=dict(
#                     text="r", 
#                     font=dict(color=SUB, size=12)
#                 ),
#                 tickfont=dict(color=SUB, size=11),
#             ),
#         ))
       
        
#         # Increased height from 520 to 750 so it doesn't look shrunk
#         hm_layout = plotly_layout("Feature Correlation Matrix", 750) 
        
#         hm_layout["xaxis"] = dict(
#             tickfont=dict(size=10, color=SUB), tickangle=45,  # Increased axis font size
#             gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
#         )
#         hm_layout["yaxis"] = dict(
#             tickfont=dict(size=10, color=SUB),                # Increased axis font size
#             gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
#         )
#         hm_layout["margin"] = dict(l=120, r=20, t=50, b=120)
        
#         fig_hm.update_layout(**hm_layout)
#         st.plotly_chart(fig_hm, use_container_width=True)

#         md(f'<div style="background:{CARD2};border:1px solid {BDR};'
#            f'border-radius:9px;padding:11px 16px;margin:4px 0;">'
#            f'<span style="font-size:.82rem;color:{SUB};">'
#            f'<strong style="color:{TEAL};">tenure</strong> negatively correlated with Churn — '
#            f'long-term customers stay &nbsp;·&nbsp; '
#            f'<strong style="color:{RED};">MonthlyCharges</strong> positively correlated — '
#            f'expensive plans drive churn &nbsp;·&nbsp; '
#            f'TotalCharges highly correlated with tenure (expected)'
#            f'</span></div>')
#         #===========================

#         # Raw data browser
#         with st.expander("📋  Browse Raw Dataset"):
#             flt = st.text_input("Filter by Churn (Yes / No / blank = all):", "")
#             vdf = (df[df["Churn"].str.contains(flt, case=False, na=False)]
#                    if flt else df)
#             st.dataframe(
#                 vdf.drop(columns=["ChurnBin", "TenureGroup"], errors="ignore").head(300),
#                 use_container_width=True,
#             )
#             st.caption(f"Showing {min(300, len(vdf))} of {len(vdf):,} rows")


# # ═══════════════════════════════════════════
# # TAB 3 — MODEL INSIGHTS
# # ═══════════════════════════════════════════
# with TAB3:

#     # Pipeline
#     sec_hdr("🏆", "Model Training Pipeline",
#             "End-to-end: raw data → SMOTE → tuning → best model → PKL artifacts")
#     steps_html = "  ".join(
#         f'<div style="background:{cl}12;border:1px solid {cl}30;border-radius:7px;'
#         f'padding:7px 13px;font-size:.77rem;font-family:JetBrains Mono,monospace;'
#         f'color:{cl};white-space:nowrap;">{s}</div>'
#         for s, cl in [
#             ("1. Data Cleaning",       BLUE),
#             ("2. Feature Encoding",    CYAN),
#             ("3. Train/Test Split",    TEAL),
#             ("4. SMOTE Balancing",     GREEN),
#             ("5. Baseline ×3 Models",  PURPLE),
#             ("6. GridSearchCV Tuning", AMBER),
#             ("7. Select by AUC-ROC",   RED),
#             ("8. Save PKL Artifacts",  BLUE),
#         ]
#     )
#     md(f'<div style="background:{CARD};border:1px solid {BDR};border-radius:11px;'
#        f'padding:15px 17px;margin-bottom:20px;">'
#        f'<div style="display:flex;gap:8px;flex-wrap:wrap;">{steps_html}</div></div>')

#     # Performance table
#     sec_hdr("📊", "Model Performance — Baseline vs Tuned",
#             "Correct verified values from notebook execution")

#     perf_df = pd.DataFrame({
#         "Model":           ["XGBoost  ★ Best", "Random Forest", "Logistic Regression"],
#         "Base Accuracy":   [0.7779, 0.7771, 0.7424],
#         "Tuned Accuracy":  [0.7757, 0.7622, 0.7410],
#         "Base F1":         [0.5865, 0.5890, 0.6191],
#         "Tuned F1":        [0.6274, 0.6223, 0.6146],
#         "Base AUC-ROC":    [0.8146, 0.8210, 0.8391],
#         "Tuned AUC-ROC":   [0.8434, 0.8397, 0.8382],
#         "Speed":           ["Medium", "Slow", "Fast"],
#     })
#     st.dataframe(perf_df, use_container_width=True, hide_index=True)

#     # Metric bar charts
#     md("<br>")
#     mc_cols = st.columns(3, gap="medium")
#     mnames  = ["XGBoost", "Random Forest", "Logistic Reg."]
#     for i, (metric, vals, ac) in enumerate([
#         ("Tuned Accuracy", [0.7757, 0.7622, 0.7410], BLUE),
#         ("Tuned AUC-ROC",  [0.8434, 0.8397, 0.8382], AMBER),
#         ("Tuned F1-Score", [0.6274, 0.6223, 0.6146], GREEN),
#     ]):
#         with mc_cols[i]:
#             best   = max(vals)
#             bclrs  = [ac if abs(v - best) < 1e-5 else BDR2 for v in vals]
#             texts  = [f"{v:.4f} ★" if abs(v-best)<1e-5 else f"{v:.4f}" for v in vals]
#             fig_mc = go.Figure(go.Bar(
#                 y=mnames, x=vals, orientation="h",
#                 marker_color=bclrs,
#                 marker_line_color="rgba(0,0,0,0)",
#                 text=texts, textposition="outside",
#                 textfont=dict(color=TEXT, size=10),
#             ))
#             fig_mc.update_layout(**plotly_layout(metric, 210))
#             fig_mc.update_layout(margin=dict(l=8, r=65, t=42, b=8))
#             fig_mc.update_xaxes(range=[min(vals)-0.05, max(vals)+0.07], gridcolor=BDR)
#             fig_mc.update_yaxes(gridcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig_mc, use_container_width=True)

#     # Baseline vs Tuned comparison
#     sec_hdr("📈", "Baseline vs Tuned Improvement",
#             "F1-Score and AUC-ROC gain after GridSearchCV hyperparameter tuning")

#     fig_bt = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=["F1-Score: Baseline vs Tuned", "AUC-ROC: Baseline vs Tuned"],
#         horizontal_spacing=0.1,
#     )
#     for cidx, (bv, tv, mn_col) in enumerate([
#         ([0.5865, 0.5890, 0.6191], [0.6274, 0.6223, 0.6146], "F1"),
#         ([0.8146, 0.8210, 0.8391], [0.8434, 0.8397, 0.8382], "AUC"),
#     ], 1):
#         all_v = bv + tv
#         fig_bt.add_trace(go.Bar(
#             name="Baseline", x=mnames, y=bv,
#             marker_color=BDR2,
#             marker_line_color="rgba(0,0,0,0)",
#             text=[f"{v:.4f}" for v in bv], textposition="outside",
#             textfont=dict(color=SUB, size=9),
#             showlegend=(cidx == 1),
#         ), row=1, col=cidx)
#         fig_bt.add_trace(go.Bar(
#             name="Tuned", x=mnames, y=tv,
#             marker_color=BLUE,
#             marker_line_color="rgba(0,0,0,0)",
#             text=[f"{v:.4f}" for v in tv], textposition="outside",
#             textfont=dict(color=TEXT, size=9),
#             showlegend=(cidx == 1),
#         ), row=1, col=cidx)
#         fig_bt.update_yaxes(range=[min(all_v)-0.04, max(all_v)+0.06],
#                             gridcolor=BDR, row=1, col=cidx)
#         fig_bt.update_xaxes(gridcolor="rgba(0,0,0,0)", row=1, col=cidx)

#     lyt_bt = plotly_layout("Baseline vs Tuned — All 3 Models", 340)
#     lyt_bt["barmode"] = "group"
#     fig_bt.update_layout(**lyt_bt)
#     for ann in fig_bt.layout.annotations:
#         ann.font.color = SUB
#         ann.font.size  = 11
#     st.plotly_chart(fig_bt, use_container_width=True)

#     # Feature Importance
#     if ok_m:
#         sec_hdr("📌", "Feature Importance",
#                 f"Top 15 features driving {model_lbl} predictions")
#         try:
#             actual_m = arts["model"]
#             if hasattr(actual_m, "named_steps"):
#                 actual_m = actual_m.named_steps.get("model", actual_m)
#             if hasattr(actual_m, "feature_importances_"):
#                 imps = actual_m.feature_importances_
#             elif hasattr(actual_m, "coef_"):
#                 imps = np.abs(actual_m.coef_[0])
#             else:
#                 raise ValueError("No importances or coefficients on model object")

#             fi = (pd.Series(imps, index=arts["cols"])
#                   .sort_values(ascending=True)
#                   .tail(15))
#             top3 = sorted(fi.values)[-3]
#             fi_clrs = [AMBER if v >= top3 else BLUE for v in fi.values]
#             fi_txt  = [f"{v:.4f}" for v in fi.values]

#             fig_fi = go.Figure(go.Bar(
#                 y=fi.index.tolist(),
#                 x=fi.values.tolist(),
#                 orientation="h",
#                 marker_color=fi_clrs,
#                 marker_line_color="rgba(0,0,0,0)",
#                 text=fi_txt, textposition="outside",
#                 textfont=dict(color=TEXT, size=9),
#             ))
#             fig_fi.update_layout(**plotly_layout(f"Feature Importances — {model_lbl}", 440))
#             fig_fi.update_layout(margin=dict(l=8, r=70, t=50, b=8))
#             fig_fi.update_xaxes(gridcolor=BDR, title="Importance Score")
#             fig_fi.update_yaxes(gridcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig_fi, use_container_width=True)
#             md(f'<div style="font-size:.78rem;color:{MUTED};margin-top:4px;">'
#                f'<span style="color:{AMBER};">■</span> Top 3 drivers &nbsp;'
#                f'<span style="color:{BLUE};">■</span> Other features</div>')
#         except Exception as e:
#             st.info(f"Feature importance unavailable: {e}")
#     else:
#         st.info("Load model PKL files to see feature importance chart.")

#     # Final metrics
#     sec_hdr("📋", "Final Model Metrics — XGBoost",
#             "Held-out test set · 1,409 samples · no data leakage")

#     m5c = st.columns(5, gap="small")
#     for col, (ic, lb, vl, ac) in zip(m5c, [
#         ("🎯", "Accuracy",  f"{ACC*100:.2f}%", BLUE),
#         ("⚡", "Precision", f"{PREC*100:.2f}%", PURPLE),
#         ("🔍", "Recall",    f"{REC*100:.2f}%",  TEAL),
#         ("📊", "F1 Score",  f"{F1:.4f}",        GREEN),
#         ("📈", "AUC-ROC",   f"{AUC:.4f}",       AMBER),
#     ]):
#         col.markdown(kpi(ic, lb, vl, ac), unsafe_allow_html=True)

#     # Classification report + hyperparams
#     md("<br>")
#     cr1, cr2 = st.columns(2, gap="large")

#     with cr1:
#         sec_hdr("📄", "Classification Report",
#                 "Per-class breakdown on test set — No Churn vs Churn")
#         report_data = pd.DataFrame({
#             "Class":     ["No Churn", "Churn", "Macro Avg", "Weighted Avg"],
#             "Precision": [0.88, 0.56, 0.72, 0.80],
#             "Recall":    [0.80, 0.71, 0.76, 0.78],
#             "F1-Score":  [0.84, 0.63, 0.73, 0.78],
#             "Support":   [1035, 374, 1409, 1409],
#         })
#         st.dataframe(report_data, use_container_width=True, hide_index=True)

#     with cr2:
#         sec_hdr("⚙️", "Best Hyperparameters",
#                 "XGBoost — GridSearchCV with 5-fold StratifiedKFold · scoring=F1")
#         params = {}
#         if ok_m:
#             raw_params = m_info.get("best_params", {})
#             if raw_params:
#                 params = {k.replace("model__", ""): v for k, v in raw_params.items()}
#         if not params:
#             params = {
#                 "colsample_bytree": 0.8,
#                 "gamma":            0,
#                 "learning_rate":    0.05,
#                 "max_depth":        3,
#                 "n_estimators":     200,
#                 "reg_lambda":       1,
#                 "subsample":        0.8,
#             }
#         rows_p = "".join(
#             f'<div style="display:flex;justify-content:space-between;align-items:center;'
#             f'padding:8px 0;border-bottom:1px solid {BDR}38;">'
#             f'<span style="font-family:JetBrains Mono,monospace;font-size:.79rem;'
#             f'color:{CYAN};">{k}</span>'
#             f'<span style="font-size:.79rem;color:{TEXT};font-weight:600;">{v}</span>'
#             f'</div>'
#             for k, v in params.items()
#         )
#         md(f'<div style="background:{CARD};border:1px solid {BDR};'
#            f'border-radius:10px;padding:13px 15px;">{rows_p}</div>')

#     # Business Insights
#     divider()
#     sec_hdr("💡", "Key Business Insights",
#             "Actionable retention strategies from EDA and model feature importance")

#     insights_data = [
#         ("📅", "Contract Type",
#          "Month-to-month customers churn at 42.7% vs only 2.8% on 2-year plans. "
#          "Offer discounts to upgrade contract length.",
#          RED),
#         ("🆕", "New Customers",
#          "First 12 months = 47.7% churn rate — highest-risk window. "
#          "Invest heavily in onboarding and early engagement.",
#          AMBER),
#         ("💸", "Monthly Charges",
#          "Bills over $70/month strongly correlate with churn. "
#          "Introduce loyalty pricing and value bundles for high-payers.",
#          AMBER),
#         ("👴", "Senior Citizens",
#          "Senior customers churn at 41.7% vs 24.3% for non-seniors. "
#          "Dedicated support and simplified plans improve retention.",
#          RED),
#         ("💳", "Payment Method",
#          "Electronic check = 45.3% churn — highest of all methods. "
#          "Incentivise migration to automatic payment.",
#          RED),
#         ("🌐", "Fiber Optic",
#          "Fiber optic churn: 41.9% vs DSL 18.9%. "
#          "Price-to-value perception drives dissatisfaction — target with loyalty offers.",
#          AMBER),
#     ]
#     for row_g in [insights_data[:3], insights_data[3:]]:
#         rc = st.columns(3, gap="medium")
#         for col, (ic, ti, de, ac) in zip(rc, row_g):
#             col.markdown(insight_card(ic, ti, de, ac), unsafe_allow_html=True)
#         md("<br>")


# # ─────────────────────────────────────────────
# # FOOTER
# # ─────────────────────────────────────────────
# md(f'<div style="margin-top:52px;padding:18px 0;border-top:1px solid {BDR};'
#    f'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">'
#    f'<div style="display:flex;align-items:center;gap:9px;">'
#    f'<div style="width:28px;height:28px;border-radius:8px;background:{BLUE};'
#    f'display:flex;align-items:center;justify-content:center;font-size:.88rem;">📡</div>'
#    f'<span style="font-family:Space Grotesk,sans-serif;font-size:.88rem;'
#    f'font-weight:700;color:{TEXT};">TeleChurn AI</span>'
#    f'<span style="font-size:.76rem;color:{MUTED};">· Telco Customer Intelligence</span>'
#    f'</div>'
#    f'<div style="font-size:.7rem;color:{MUTED};">'
#    f'XGBoost · SMOTE · GridSearchCV · Scikit-learn · Plotly · Streamlit · Python'
#    f'</div></div>')



# -------------------new----------------------------
"""
TeleChurn AI — app.py (v6 · Final · Error-Free)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pip install streamlit scikit-learn xgboost pandas numpy plotly
Place PKL files in ./models/ and dataset.csv alongside app.py
streamlit run app.py

Bugs fixed vs v5:
 ✅ st.stop() inside tabs blocks all later tabs — replaced with if/else
 ✅ st.columns(0) crash when n_cols=0
 ✅ Heatmap shows "nan" text — masked properly
 ✅ SeniorCitizen (int) plotted as string on x-axis
 ✅ select_dtypes deprecation warning fixed
 ✅ All f-strings validated, no nested quote conflicts
 ✅ Risk-summary safe for any combination of inputs
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
#----------------------------------------------
# ── FORCE DARK TABLE THEME (STREAMLIT CLOUD FIX) ──
st.markdown("""
<style>

/* dataframe container */
[data-testid="stDataFrame"] {
    background-color: #0f172a;
    border-radius: 10px;
}

/* table header */
[data-testid="stDataFrame"] thead tr th {
    background-color: #111827 !important;
    color: #e5e7eb !important;
    font-weight: 600;
}

/* table cells */
[data-testid="stDataFrame"] tbody tr td {
    background-color: #0f172a !important;
    color: #e5e7eb !important;
}

/* row hover */
[data-testid="stDataFrame"] tbody tr:hover td {
    background-color: #1f2937 !important;
}

/* expander dataframe fix */
[data-testid="stExpander"] [data-testid="stDataFrame"] {
    background-color: #0f172a;
}

</style>
""", unsafe_allow_html=True)
#----------------------------------------------

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TeleChurn AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# COLOR SYSTEM
# ─────────────────────────────────────────────
BG    = "#0F1117"
CARD  = "#161B27"
CARD2 = "#1C2333"
BDR   = "#2A3347"
BDR2  = "#374357"
TEXT  = "#E8EDF5"
SUB   = "#8B97B0"
MUTED = "#4A5568"

BLUE   = "#4F8EF7"
INDIGO = "#7C6FF7"
CYAN   = "#22D3EE"
GREEN  = "#22C55E"
AMBER  = "#F59E0B"
RED    = "#F43F5E"
PURPLE = "#A855F7"
TEAL   = "#14B8A6"

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*,*::before,*::after{{box-sizing:border-box;margin:0;}}
html,body,.stApp{{background:{BG}!important;color:{TEXT}!important;font-family:'Inter',sans-serif!important;}}
#MainMenu,footer,header,[data-testid="stToolbar"]{{display:none!important;}}
.block-container{{padding:0 2rem 5rem!important;max-width:1420px!important;}}

[data-testid="stSidebar"]{{background:{CARD}!important;border-right:1px solid {BDR}!important;}}
[data-testid="stSidebar"]>div:first-child{{padding-top:0!important;}}
[data-testid="stSidebarNav"]{{display:none!important;}}

.stTabs [data-baseweb="tab-list"]{{background:{CARD};border:1px solid {BDR};border-radius:12px;padding:4px;gap:2px;margin-bottom:24px;}}
.stTabs [data-baseweb="tab"]{{background:transparent!important;color:{MUTED}!important;border-radius:9px!important;padding:9px 24px!important;font-family:'Space Grotesk',sans-serif!important;font-size:.84rem!important;font-weight:600!important;border:none!important;transition:all .18s!important;}}
.stTabs [aria-selected="true"]{{background:{BLUE}!important;color:#fff!important;box-shadow:0 2px 12px {BLUE}30!important;}}

div[data-baseweb="select"]>div{{background:{CARD2}!important;border:1px solid {BDR}!important;border-radius:8px!important;color:{TEXT}!important;}}
div[data-baseweb="select"] *{{color:{TEXT}!important;}}
div[data-baseweb="popover"]{{background:{CARD2}!important;border:1px solid {BDR2}!important;border-radius:10px!important;}}
div[data-baseweb="popover"] li:hover{{background:{BDR}!important;}}
.stNumberInput input,.stTextInput input,.stTextArea textarea{{background:{CARD2}!important;border:1px solid {BDR}!important;border-radius:8px!important;color:{TEXT}!important;font-size:.88rem!important;}}
.stSlider>div>div>div>div{{background:{BLUE}!important;}}

div.stButton>button{{background:{BLUE}!important;color:#fff!important;border:none!important;border-radius:10px!important;padding:14px 0!important;font-family:'Space Grotesk',sans-serif!important;font-weight:700!important;font-size:1rem!important;width:100%!important;letter-spacing:.3px!important;box-shadow:0 4px 20px {BLUE}40!important;transition:all .2s ease!important;}}
div.stButton>button:hover{{background:#6BA3F9!important;box-shadow:0 8px 30px {BLUE}55!important;transform:translateY(-1px)!important;}}

[data-testid="stSuccess"]{{background:#0A1F14!important;border:1px solid {GREEN}40!important;border-radius:10px!important;}}
[data-testid="stError"]{{background:#1F0A11!important;border:1px solid {RED}40!important;border-radius:10px!important;}}
[data-testid="stWarning"]{{background:#1A1305!important;border:1px solid {AMBER}40!important;border-radius:10px!important;}}
[data-testid="stInfo"]{{background:#080F1F!important;border:1px solid {BLUE}40!important;border-radius:10px!important;}}
[data-testid="stExpander"]{{background:{CARD}!important;border:1px solid {BDR}!important;border-radius:10px!important;}}
details summary{{color:{TEXT}!important;}}

::-webkit-scrollbar{{width:4px;height:4px;}}
::-webkit-scrollbar-track{{background:{BG};}}
::-webkit-scrollbar-thumb{{background:{BDR2};border-radius:4px;}}
hr{{border-color:{BDR}!important;margin:1.4rem 0!important;}}

@keyframes pdot{{0%,100%{{opacity:1;}}50%{{opacity:.2;}}}}
.pdot{{display:inline-block;width:7px;height:7px;border-radius:50%;background:{GREEN};animation:pdot 2s infinite;box-shadow:0 0 6px {GREEN};margin-right:6px;vertical-align:middle;}}
@keyframes sup{{from{{opacity:0;transform:translateY(14px);}}to{{opacity:1;transform:translateY(0);}}}}
.sup{{animation:sup .4s cubic-bezier(.22,1,.36,1) both;}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PLOTLY BASE LAYOUT
# ─────────────────────────────────────────────
def plotly_layout(title="", height=360):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=SUB, size=11),
        margin=dict(l=8, r=8, t=44, b=8),
        height=height,
        title=dict(
            text=title,
            font=dict(family="Space Grotesk", size=14, color=TEXT),
            x=0, xanchor="left",
        ),
        xaxis=dict(gridcolor=BDR, linecolor=BDR, zerolinecolor=BDR),
        yaxis=dict(gridcolor=BDR, linecolor=BDR, zerolinecolor=BDR),
        legend=dict(
            bgcolor="rgba(22,27,39,0.85)", bordercolor=BDR,
            borderwidth=1, font=dict(size=11, color=SUB),
        ),
        colorway=[BLUE, CYAN, GREEN, AMBER, RED, PURPLE, TEAL],
    )


# ─────────────────────────────────────────────
# HTML HELPERS  (no nested quote conflicts)
# ─────────────────────────────────────────────
def md(html):
    st.markdown(html, unsafe_allow_html=True)

def divider():
    md("<hr/>")

def sec_hdr(icon, title, sub=""):
    sub_part = (f'<div style="font-size:.77rem;color:{SUB};margin-top:3px;">{sub}</div>'
                if sub else "")
    md(f"""
    <div style="display:flex;align-items:flex-start;gap:12px;margin:30px 0 16px;">
      <div style="width:37px;height:37px;flex-shrink:0;border-radius:10px;
        background:linear-gradient(135deg,{BLUE}22,{INDIGO}22);
        border:1px solid {BLUE}35;display:flex;align-items:center;
        justify-content:center;font-size:1.05rem;">{icon}</div>
      <div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:.98rem;
          font-weight:700;color:{TEXT};">{title}</div>
        {sub_part}
      </div>
    </div>""")

def kpi(icon, label, value, accent, note=""):
    note_part = (f'<div style="font-size:.69rem;color:{SUB};margin-top:3px;">{note}</div>'
                 if note else "")
    return f"""<div style="background:{CARD};border:1px solid {BDR};border-radius:12px;
      padding:17px;position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;right:0;width:70px;height:70px;
        background:radial-gradient(circle,{accent}18,transparent 70%);pointer-events:none;"></div>
      <div style="font-size:1.2rem;margin-bottom:7px;">{icon}</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;
        font-weight:700;color:{accent};line-height:1;margin-bottom:4px;">{value}</div>
      <div style="font-size:.63rem;color:{MUTED};text-transform:uppercase;
        letter-spacing:1px;font-weight:600;">{label}</div>
      {note_part}</div>"""

def badge(text, color):
    return (f'<span style="background:{color}18;border:1px solid {color}35;'
            f'color:{color};border-radius:6px;padding:3px 12px;font-size:.72rem;'
            f'font-family:JetBrains Mono,monospace;font-weight:500;">{text}</span>')

def insight_card(icon, title, desc, accent):
    return f"""<div style="background:{CARD};border:1px solid {accent}30;
      border-left:3px solid {accent};border-radius:10px;padding:18px;height:100%;">
      <div style="font-size:1.2rem;margin-bottom:9px;">{icon}</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:.87rem;
        font-weight:700;color:{accent};margin-bottom:6px;">{title}</div>
      <div style="font-size:.79rem;color:{SUB};line-height:1.7;">{desc}</div>
    </div>"""

def signal_row(label, rate, accent):
    return f"""<div style="display:flex;justify-content:space-between;align-items:center;
      background:{CARD};border:1px solid {BDR};border-left:3px solid {accent};
      border-radius:8px;padding:9px 14px;margin-bottom:7px;">
      <span style="font-size:.82rem;color:{SUB};">{label}</span>
      <span style="font-size:.82rem;font-weight:700;color:{accent};">{rate}</span>
    </div>"""

def risk_card_html(col_c, col_lbl, title_f, desc_f, note_f):
    """Build risk factor card HTML without nested f-string quote issues."""
    return (
        f'<div style="background:{col_c}0C;border:1px solid {col_c}30;'
        f'border-left:3px solid {col_c};border-radius:10px;'
        f'padding:14px 16px;margin-bottom:10px;">'
        f'<div style="font-size:.63rem;font-weight:700;color:{col_c};'
        f'text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">'
        f'{col_lbl}</div>'
        f'<div style="font-size:.82rem;font-weight:600;color:{TEXT};margin-bottom:4px;">'
        f'{title_f}</div>'
        f'<div style="font-size:.76rem;color:{SUB};margin-bottom:3px;">{desc_f}</div>'
        f'<div style="font-size:.7rem;color:{MUTED};font-style:italic;">{note_f}</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
MODELS_DIR = Path("models")

@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    needed = {
        "model":  MODELS_DIR / "model.pkl",
        "scaler": MODELS_DIR / "scaler.pkl",
        "enc":    MODELS_DIR / "label_encoders.pkl",
        "cols":   MODELS_DIR / "feature_columns.pkl",
        "info":   MODELS_DIR / "model_info.pkl",
    }
    miss = [p.name for p in needed.values() if not p.exists()]
    if miss:
        return None, miss
    out = {}
    for k, p in needed.items():
        with open(p, "rb") as f:
            out[k] = pickle.load(f)
    return out, []

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    p = Path("dataset.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip().replace({"": "0", " ": "0"}),
        errors="coerce",
    ).fillna(0.0)
    df["ChurnBin"] = (df["Churn"] == "Yes").astype(int)
    # bins include 0 explicitly so no NaN in TenureGroup
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"],
    )
    return df

arts, miss_files = load_artifacts()
df   = load_data()
ok_m = arts is not None
ok_d = df is not None

m_info    = arts["info"] if ok_m else {}
model_lbl = m_info.get("model_name", "XGBoost") if ok_m else "XGBoost"
threshold = float(m_info.get("threshold", 0.4828)) if ok_m else 0.4828

# Verified final metrics (from notebook output)
ACC  = m_info.get("accuracy",  0.7757) if ok_m else 0.7757
PREC = m_info.get("precision", 0.5612) if ok_m else 0.5612
REC  = m_info.get("recall",    0.7112) if ok_m else 0.7112
F1   = m_info.get("f1_score",  0.6274) if ok_m else 0.6274
AUC  = m_info.get("roc_auc",   0.8434) if ok_m else 0.8434


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    md(f"""<div style="padding:20px 16px 0;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
        <div style="width:33px;height:33px;border-radius:9px;background:{BLUE};
          display:flex;align-items:center;justify-content:center;
          font-size:.95rem;flex-shrink:0;">📡</div>
        <div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:.93rem;
            font-weight:700;color:{TEXT};">TeleChurn AI</div>
          <div style="font-size:.62rem;color:{MUTED};letter-spacing:.5px;">
            ANALYTICS PLATFORM</div>
        </div>
      </div>
      <div style="height:1px;background:{BDR};margin-bottom:16px;"></div>
    </div>""")

    # Status
    mc = GREEN if ok_m else RED
    dc = GREEN if ok_d else AMBER
    md(f"""<div style="padding:0 16px;">
      <div style="font-size:.62rem;font-weight:700;color:{MUTED};letter-spacing:1.1px;
        text-transform:uppercase;margin-bottom:8px;">System Status</div>
      <div style="background:{CARD2};border:1px solid {mc}30;border-radius:8px;
        padding:9px 12px;margin-bottom:5px;display:flex;align-items:center;gap:7px;">
        <span class="pdot" style="background:{mc};box-shadow:0 0 6px {mc};"></span>
        <span style="font-size:.77rem;color:{mc};font-weight:600;">
          {"Model Ready" if ok_m else "Model Missing — add models/"}</span>
      </div>
      <div style="background:{CARD2};border:1px solid {dc}30;border-radius:8px;
        padding:9px 12px;margin-bottom:16px;display:flex;align-items:center;gap:7px;">
        <span style="width:7px;height:7px;border-radius:50%;
          background:{dc};flex-shrink:0;display:inline-block;"></span>
        <span style="font-size:.77rem;color:{dc};font-weight:600;">
          {"Dataset Loaded" if ok_d else "No dataset.csv found"}</span>
      </div>
    </div>""")

    # Model info
    if ok_m:
        md(f"""<div style="padding:0 16px;">
          <div style="font-size:.62rem;font-weight:700;color:{MUTED};letter-spacing:1.1px;
            text-transform:uppercase;margin-bottom:8px;">Best Model</div>
          <div style="background:{CARD2};border:1px solid {BDR};border-radius:8px;
            padding:12px;margin-bottom:16px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:7px;">
              <span style="font-size:.71rem;color:{MUTED};">Algorithm</span>
              <span style="font-family:'JetBrains Mono',monospace;font-size:.79rem;
                color:{BLUE};font-weight:600;">{model_lbl}</span>
            </div>
            <div style="height:1px;background:{BDR};margin:7px 0;"></div>
            <div style="display:flex;justify-content:space-between;">
              <span style="font-size:.71rem;color:{MUTED};">Threshold</span>
              <span style="font-family:'JetBrains Mono',monospace;font-size:.79rem;
                color:{AMBER};font-weight:600;">{threshold:.4f}</span>
            </div>
          </div>
        </div>""")

    # Stats
    _n  = f"{len(df):,}" if ok_d else "7,043"
    _ch = f"{df['ChurnBin'].sum():,}" if ok_d else "1,869"
    _cr = f"{df['ChurnBin'].mean()*100:.1f}%" if ok_d else "26.5%"
    stats = [
        ("Customers", _n, CYAN),
        ("Churned", _ch, RED),
        ("Churn Rate", _cr, AMBER),
        ("Best AUC-ROC", f"{AUC:.4f}", GREEN),
        ("Best F1", f"{F1:.4f}", TEAL),
    ]
    md(f"""<div style="padding:0 16px;">
      <div style="font-size:.62rem;font-weight:700;color:{MUTED};letter-spacing:1.1px;
        text-transform:uppercase;margin-bottom:8px;">Dataset Stats</div>""")
    for lb, vl, ac in stats:
        md(f"""<div style="background:{CARD2};border-radius:7px;padding:8px 10px;
          display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
          <span style="font-size:.73rem;color:{MUTED};">{lb}</span>
          <span style="font-family:'JetBrains Mono',monospace;font-size:.77rem;
            color:{ac};font-weight:600;">{vl}</span>
        </div>""")
    md("</div>")

    md(f"""<div style="padding:14px 16px;">
      <div style="height:1px;background:{BDR};margin-bottom:12px;"></div>
      <div style="font-size:.69rem;color:{MUTED};line-height:1.8;text-align:center;">
        XGBoost · SMOTE · GridSearchCV<br>Scikit-learn · Plotly · Streamlit
      </div>
    </div>""")


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
_nc = f"{len(df):,}" if ok_d else "7,043"
badges_html = "  ".join([
    badge("SMOTE Balanced", GREEN),
    badge("3 Models Tuned", INDIGO),
    badge(_nc + " Customers", CYAN),
    badge(f"Threshold {threshold:.4f}", AMBER),
    badge("AUC 0.8434", BLUE),
])
md(f"""<div style="padding:36px 0 22px;">
  <div style="margin-bottom:14px;">
    <span style="display:inline-flex;align-items:center;gap:5px;
      background:{CARD};border:1px solid {BDR};border-radius:50px;
      padding:4px 14px;font-size:.69rem;font-family:'JetBrains Mono',monospace;color:{MUTED};">
      <span class="pdot"></span>LIVE ANALYTICS ENGINE
    </span>
  </div>
  <h1 style="font-family:'Space Grotesk',sans-serif;font-size:2.4rem;
    font-weight:700;color:{TEXT};line-height:1.12;margin-bottom:8px;">
    Telco Customer
    <span style="color:{BLUE};">Churn Intelligence</span>
  </h1>
  <p style="font-size:.9rem;color:{SUB};line-height:1.7;max-width:500px;margin-bottom:18px;">
    AI-powered churn prediction · XGBoost + SMOTE + GridSearchCV · Optimal threshold calibration
  </p>
  <div style="display:flex;gap:8px;flex-wrap:wrap;">{badges_html}</div>
</div>""")

# KPI strip
_nr  = f"{len(df):,}" if ok_d else "7,043"
_nch = f"{df['ChurnBin'].sum():,}" if ok_d else "1,869"
kc   = st.columns(5, gap="small")
for col, (ic, lb, vl, ac, nt) in zip(kc, [
    ("📡", "Model",     "Ready" if ok_m else "Missing", BLUE,  model_lbl if ok_m else "Add models/"),
    ("🗄️", "Dataset",  "Loaded" if ok_d else "Optional", CYAN, f"{_nr} rows" if ok_d else "Add dataset.csv"),
    ("👥", "Customers", _nr,                             GREEN, "total records"),
    ("📉", "Churned",   _nch,                            RED,   "26.5% of total"),
    ("🎯", "Threshold", f"{threshold:.4f}",              AMBER, "optimal F1 cut-off"),
]):
    col.markdown(kpi(ic, lb, vl, ac, nt), unsafe_allow_html=True)

md("<br>")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
TAB1, TAB2, TAB3 = st.tabs([
    "🔮  Predict Churn",
    "📊  EDA Dashboard",
    "🤖  Model Insights",
])


# ═══════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════
with TAB1:
    if not ok_m:
        st.error(
            f"**Model files not found.**  Missing: `{'`, `'.join(miss_files)}`  \n"
            "Run the notebook end-to-end to generate PKL files, "
            "then place them in a `models/` subfolder next to app.py."
        )
        md(f"""<div style="background:{CARD};border:1px dashed {BDR2};border-radius:14px;
          padding:48px 24px;text-align:center;margin-top:12px;">
          <div style="font-size:2.5rem;margin-bottom:12px;">📁</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;
            font-weight:700;color:{TEXT};margin-bottom:8px;">Models Not Found</div>
          <div style="font-size:.84rem;color:{SUB};line-height:1.7;">
            Required: model.pkl · scaler.pkl · label_encoders.pkl
            · feature_columns.pkl · model_info.pkl
          </div>
        </div>""")
    else:
        sec_hdr("👤", "Customer Profile",
                "Fill in all fields below, then click Run Prediction")

        # CA, CB, CC = st.columns(3, gap="large")

        # with CA:
        #     md(f'<div style="font-size:.68rem;font-weight:700;color:{CYAN};'
        #        f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
        #        f'Demographics</div>')
        #     gender     = st.selectbox("Gender",         ["Male", "Female"])
        #     senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
        #     partner    = st.selectbox("Partner",        ["No", "Yes"])
        #     dependents = st.selectbox("Dependents",     ["No", "Yes"])
        #     tenure     = st.slider("Tenure (months)", 0, 72, 12)

        # with CB:
        #     md(f'<div style="font-size:.68rem;font-weight:700;color:{PURPLE};'
        #        f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
        #        f'Services</div>')
        #     phone_svc = st.selectbox("Phone Service", ["Yes", "No"])
        #     if phone_svc == "No":
        #         multi_lines = "No phone service"
        #         st.selectbox("Multiple Lines", ["No phone service"], disabled=True)
        #     else:
        #         multi_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

        #     internet_svc = st.selectbox("Internet Service",
        #                                 ["DSL", "Fiber optic", "No"])
        #     _io  = ["No internet service"] if internet_svc == "No" else ["No", "Yes"]
        #     _dis = internet_svc == "No"

        #     online_sec   = st.selectbox("Online Security",   _io, disabled=_dis)
        #     online_bkp   = st.selectbox("Online Backup",     _io, disabled=_dis)
        #     dev_protect  = st.selectbox("Device Protection", _io, disabled=_dis)
        #     tech_support = st.selectbox("Tech Support",      _io, disabled=_dis)
        #     stream_tv    = st.selectbox("Streaming TV",      _io, disabled=_dis)
        #     stream_mov   = st.selectbox("Streaming Movies",  _io, disabled=_dis)

        # with CC:
        #     md(f'<div style="font-size:.68rem;font-weight:700;color:{TEAL};'
        #        f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
        #        f'Billing &amp; Contract</div>')
        #     contract  = st.selectbox("Contract Type",
        #                              ["Month-to-month", "One year", "Two year"])
        #     paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        #     payment   = st.selectbox("Payment Method", [
        #         "Electronic check", "Mailed check",
        #         "Bank transfer (automatic)", "Credit card (automatic)",
        #     ])
        #     monthly  = st.number_input("Monthly Charges ($)",
        #                                18.0, 120.0, 65.0, 0.5, format="%.2f")
        #     def_tot  = min(float(monthly) * max(int(tenure), 1), 8684.8)
        #     total    = st.number_input("Total Charges ($)",
        #                                0.0, 8684.8, def_tot, 10.0, format="%.2f")

        #==================================
        CA, CB, CC = st.columns(3, gap="large")

        with CA:
            md(f'<div style="font-size:.68rem;font-weight:700;color:{CYAN};'
               f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
               f'Demographics</div>')
            gender     = st.selectbox("Gender",         ["Male", "Female"], index=0) # Default: Male
            senior     = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)      # Default: No (0)
            partner    = st.selectbox("Partner",        ["No", "Yes"], index=1)      # Default: Yes
            dependents = st.selectbox("Dependents",     ["No", "Yes"], index=0)      # Default: No
            tenure     = st.slider("Tenure (months)", 0, 72, 12)                     # Default: 12

        with CB:
            md(f'<div style="font-size:.68rem;font-weight:700;color:{PURPLE};'
               f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
               f'Services</div>')
            phone_svc = st.selectbox("Phone Service", ["No", "Yes"], index=1)        # Default: Yes
            if phone_svc == "No":
                multi_lines = "No phone service"
                st.selectbox("Multiple Lines", ["No phone service"], disabled=True)
            else:
                multi_lines = st.selectbox("Multiple Lines", ["No", "Yes"], index=0) # Default: No

            internet_svc = st.selectbox("Internet Service",
                                        ["DSL", "Fiber optic", "No"], index=1)       # Default: Fiber optic
            _io  = ["No internet service"] if internet_svc == "No" else ["No", "Yes"]
            _dis = internet_svc == "No"

            online_sec   = st.selectbox("Online Security",   _io, index=0, disabled=_dis) # Default: No
            online_bkp   = st.selectbox("Online Backup",     _io, index=0, disabled=_dis) # Default: No
            dev_protect  = st.selectbox("Device Protection", _io, index=0, disabled=_dis) # Default: No
            tech_support = st.selectbox("Tech Support",      _io, index=0, disabled=_dis) # Default: No
            stream_tv    = st.selectbox("Streaming TV",      _io, index=1, disabled=_dis) # Default: Yes
            stream_mov   = st.selectbox("Streaming Movies",  _io, index=1, disabled=_dis) # Default: Yes

        with CC:
            md(f'<div style="font-size:.68rem;font-weight:700;color:{TEAL};'
               f'text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">'
               f'Billing &amp; Contract</div>')
            contract  = st.selectbox("Contract Type",
                                     ["Month-to-month", "One year", "Two year"], index=0) # Default: Month-to-month
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"], index=1)         # Default: Yes
            payment   = st.selectbox("Payment Method", [
                "Bank transfer (automatic)", "Credit card (automatic)", 
                "Electronic check", "Mailed check"
            ], index=2)                                                                   # Default: Electronic check
            monthly  = st.number_input("Monthly Charges ($)",
                                       18.0, 120.0, 70.35, 0.5, format="%.2f")            # Default: 70.35
            total    = st.number_input("Total Charges ($)",
                                       0.0, 8684.8, 843.0, 10.0, format="%.2f")           # Default: 843.0
        
        md("<br>")
        b1, b2, b3 = st.columns([1, 3, 1])
        with b2:
            clicked = st.button("🔮  RUN CHURN PREDICTION", use_container_width=True)

        # ── Inference ──────────────────────────────────────────────
        if clicked:
            raw = {
                "gender":           gender,
                "SeniorCitizen":    1 if senior == "Yes" else 0,
                "Partner":          partner,
                "Dependents":       dependents,
                "tenure":           int(tenure),
                "PhoneService":     phone_svc,
                "MultipleLines":    multi_lines,
                "InternetService":  internet_svc,
                "OnlineSecurity":   online_sec   if internet_svc != "No" else "No internet service",
                "OnlineBackup":     online_bkp   if internet_svc != "No" else "No internet service",
                "DeviceProtection": dev_protect  if internet_svc != "No" else "No internet service",
                "TechSupport":      tech_support if internet_svc != "No" else "No internet service",
                "StreamingTV":      stream_tv    if internet_svc != "No" else "No internet service",
                "StreamingMovies":  stream_mov   if internet_svc != "No" else "No internet service",
                "Contract":         contract,
                "PaperlessBilling": paperless,
                "PaymentMethod":    payment,
                "MonthlyCharges":   float(monthly),
                "TotalCharges":     float(total),
            }
            
            try:
                inp = pd.DataFrame([raw])
                
                # 1. Safely label encode
                for col_name, le in arts["enc"].items():
                    if col_name in inp.columns:
                        v = str(inp.at[0, col_name])
                        if v in le.classes_:
                            inp.at[0, col_name] = le.transform([v])[0]
                        else:
                            inp.at[0, col_name] = 0  # Fallback safety
                            
                # 2. Enforce column order and float type
                inp = inp[arts["cols"]].astype(float)
                
                # 3. Scale features (returns a NumPy array)
                inp_sc_array = arts["scaler"].transform(inp)
                
                # 4. FIXED: Re-wrap in DataFrame so XGBoost recognizes feature names!
                inp_sc = pd.DataFrame(inp_sc_array, columns=arts["cols"])

                # 5. Predict
                proba   = arts["model"].predict_proba(inp_sc)[0]
                churn_p = float(proba[1]) * 100
                stay_p  = float(proba[0]) * 100
                pred    = int((churn_p / 100) >= threshold)

                if churn_p >= 65:
                    tier_col, tier_lbl = RED,   "HIGH RISK"
                elif churn_p >= 35:
                    tier_col, tier_lbl = AMBER, "MEDIUM RISK"
                else:
                    tier_col, tier_lbl = GREEN, "LOW RISK"

                vcol = RED if pred else GREEN
                vtxt = "WILL LIKELY CHURN" if pred else "LIKELY TO STAY"
                vico = "⚠️" if pred else "✅"
                vbg  = "#1A0810" if pred else "#071510"

                divider()
                sec_hdr("📈", "Prediction Result",
                        f"Model: {model_lbl}  ·  Threshold: {threshold:.4f}")

                R1, R2, R3 = st.columns([1.1, 1, 1], gap="large")

            
           #===================
            # try:
            #     inp = pd.DataFrame([raw])
            #     for col_name, le in arts["enc"].items():
            #         if col_name in inp.columns:
            #             v = str(inp.at[0, col_name])
            #             inp[col_name] = le.transform([v]) if v in le.classes_ else [0]
            #     inp    = inp[arts["cols"]].astype(float)
            #     inp_sc = arts["scaler"].transform(inp)

            #     proba   = arts["model"].predict_proba(inp_sc)[0]
            #     churn_p = float(proba[1]) * 100
            #     stay_p  = float(proba[0]) * 100
            #     pred    = int((churn_p / 100) >= threshold)

            #     if churn_p >= 65:
            #         tier_col, tier_lbl = RED,   "HIGH RISK"
            #     elif churn_p >= 35:
            #         tier_col, tier_lbl = AMBER, "MEDIUM RISK"
            #     else:
            #         tier_col, tier_lbl = GREEN, "LOW RISK"

            #     vcol = RED if pred else GREEN
            #     vtxt = "WILL LIKELY CHURN" if pred else "LIKELY TO STAY"
            #     vico = "⚠️" if pred else "✅"
            #     vbg  = "#1A0810" if pred else "#071510"

            #     divider()
            #     sec_hdr("📈", "Prediction Result",
            #             f"Model: {model_lbl}  ·  Threshold: {threshold:.4f}")

            #     R1, R2, R3 = st.columns([1.1, 1, 1], gap="large")

                # Verdict card
                with R1:
                    md(f"""<div class="sup" style="background:{vbg};border:1px solid {vcol}30;
                      border-radius:14px;padding:26px 20px;text-align:center;">
                      <div style="font-size:.6rem;font-family:'JetBrains Mono',monospace;
                        color:{vcol};letter-spacing:2.5px;margin-bottom:9px;">MODEL VERDICT</div>
                      <div style="font-size:1.8rem;margin-bottom:6px;">{vico}</div>
                      <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;
                        font-weight:700;color:{TEXT};margin-bottom:16px;">{vtxt}</div>
                      <div style="display:flex;justify-content:center;gap:22px;margin-bottom:14px;">
                        <div>
                          <div style="font-family:'Space Grotesk',sans-serif;font-size:2.1rem;
                            font-weight:700;color:{RED};">{churn_p:.1f}%</div>
                          <div style="font-size:.63rem;color:{MUTED};text-transform:uppercase;
                            letter-spacing:.7px;margin-top:2px;">Churn Risk</div>
                        </div>
                        <div style="width:1px;background:{BDR};"></div>
                        <div>
                          <div style="font-family:'Space Grotesk',sans-serif;font-size:2.1rem;
                            font-weight:700;color:{GREEN};">{stay_p:.1f}%</div>
                          <div style="font-size:.63rem;color:{MUTED};text-transform:uppercase;
                            letter-spacing:.7px;margin-top:2px;">Retention</div>
                        </div>
                      </div>
                      <div style="display:inline-flex;align-items:center;gap:6px;
                        background:{tier_col}18;border:1px solid {tier_col}40;
                        border-radius:8px;padding:7px 14px;">
                        <span style="font-size:.82rem;color:{tier_col};font-weight:700;">{tier_lbl}</span>
                      </div>
                    </div>""")

                # Gauge + bar
                # with R2:
                #     fig_g = go.Figure(go.Indicator(
                #         mode="gauge+number",
                #         value=churn_p,
                #         number=dict(suffix="%",
                #                     font=dict(size=26, color=tier_col,
                #                               family="Space Grotesk")),
                #         gauge=dict(
                #             axis=dict(range=[0, 100],
                #                       tickcolor=MUTED,
                #                       tickfont=dict(color=MUTED, size=9)),
                #             bar=dict(color=tier_col, thickness=0.62),
                #             bgcolor="rgba(0,0,0,0)",
                #             borderwidth=0,
                #             steps=[
                #                 dict(range=[0,  35], color="#071510"),
                #                 dict(range=[35, 65], color="#15110A"),
                #                 dict(range=[65,100], color="#150809"),
                #             ],
                #             threshold=dict(
                #                 line=dict(color=AMBER, width=2),
                #                 thickness=0.75,
                #                 value=threshold * 100,
                #             ),
                #         ),
                #     ))
                #     fig_g.update_layout(
                #         paper_bgcolor="rgba(0,0,0,0)",
                #         plot_bgcolor="rgba(0,0,0,0)",
                #         font=dict(color=SUB, family="Inter"),
                #         margin=dict(l=10, r=10, t=10, b=20),
                #         height=200,
                #         annotations=[dict(
                #             text=f"Decision threshold: {threshold*100:.1f}%",
                #             x=0.5, y=-0.18, showarrow=False,
                #             font=dict(color=MUTED, size=9),
                #         )],
                #     )
                #     st.plotly_chart(fig_g, use_container_width=True)

                #     fig_b = go.Figure()
                #     fig_b.add_trace(go.Bar(
                #         y=["Retention", "Churn Risk"],
                #         x=[stay_p, churn_p],
                #         orientation="h",
                #         marker_color=[GREEN, RED],
                #         marker_line_color="rgba(0,0,0,0)",
                #         text=[f"{stay_p:.1f}%", f"{churn_p:.1f}%"],
                #         textposition="outside",
                #         textfont=dict(color=TEXT, size=11),
                #     ))
                #     fig_b.add_vline(
                #         x=threshold * 100, line_dash="dash",
                #         line_color=AMBER, line_width=1.5,
                #         annotation_text=f"Threshold {threshold*100:.1f}%",
                #         annotation_font_color=AMBER,
                #         annotation_font_size=9,
                #         annotation_position="top right",
                #     )
                #     fig_b.update_layout(
                #         paper_bgcolor="rgba(0,0,0,0)",
                #         plot_bgcolor="rgba(0,0,0,0)",
                #         margin=dict(l=8, r=55, t=6, b=6),
                #         height=110,
                #         xaxis=dict(range=[0, 118], gridcolor=BDR,
                #                    tickfont=dict(color=MUTED), showticklabels=False),
                #         yaxis=dict(tickfont=dict(color=SUB, size=10),
                #                    gridcolor="rgba(0,0,0,0)"),
                #         showlegend=False,
                #     )
                #     st.plotly_chart(fig_b, use_container_width=True)
                
                #================================================
                # Gauge + bar
                with R2:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=churn_p,
                        title=dict(text="Confidence Level", font=dict(size=14, color=TEXT, family="Space Grotesk")),
                        number=dict(suffix="%",
                                    font=dict(size=32, color=tier_col,
                                              family="Space Grotesk")),
                        gauge=dict(
                            axis=dict(range=[0, 100],
                                      tickcolor=MUTED,
                                      tickfont=dict(color=MUTED, size=10)),
                            bar=dict(color=tier_col, thickness=0.7),
                            bgcolor="rgba(0,0,0,0)",
                            borderwidth=0,
                            steps=[
                                dict(range=[0,  35], color=CARD2),
                                dict(range=[35, 65], color=BDR),
                                dict(range=[65,100], color=BDR2),
                            ],
                            threshold=dict(
                                line=dict(color=TEXT, width=2.5),
                                thickness=0.8,
                                value=threshold * 100,
                            ),
                        ),
                    ))
                    fig_g.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=SUB, family="Inter"),
                        margin=dict(l=15, r=15, t=30, b=10),
                        height=220,
                    )
                    st.plotly_chart(fig_g, use_container_width=True)

                    fig_b = go.Figure()
                    fig_b.add_trace(go.Bar(
                        y=["Retention", "Churn Risk"],
                        x=[stay_p, churn_p],
                        orientation="h",
                        marker_color=[GREEN, RED],
                        marker_line_color="rgba(0,0,0,0)",
                        text=[f"{stay_p:.1f}%", f"{churn_p:.1f}%"],
                        textposition="outside",
                        textfont=dict(color=TEXT, size=11, family="Space Grotesk"),
                    ))
                    fig_b.add_vline(
                        x=threshold * 100, line_dash="dash",
                        line_color=TEXT, line_width=1.5,
                        annotation_text=f"Threshold {threshold*100:.1f}%",
                        annotation_font_color=SUB,
                        annotation_font_size=10,
                        annotation_position="top right",
                    )
                    fig_b.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=5, r=50, t=10, b=10),
                        height=120,
                        xaxis=dict(range=[0, 115], gridcolor="rgba(0,0,0,0)",
                                   tickfont=dict(color=MUTED), showticklabels=False),
                        yaxis=dict(tickfont=dict(color=TEXT, size=11),
                                   gridcolor="rgba(0,0,0,0)"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_b, use_container_width=True)
                #================================================

                # Actions
                with R3:
                    if pred:
                        actions = [
                            ("📞", "Call within 24 hours",       RED),
                            ("💰", "Offer personalised discount", AMBER),
                            ("📋", "Propose annual contract",      BLUE),
                            ("🛡️", "Add free Tech Support trial", PURPLE),
                            ("🎁", "Send loyalty reward offer",   GREEN),
                        ]
                        hdr_col, hdr_txt = RED,   "🚨 Retention Actions"
                    else:
                        actions = [
                            ("⭐", "Enrol in rewards programme",  GREEN),
                            ("📦", "Upsell premium services",     TEAL),
                            ("🔁", "Encourage contract upgrade",  BLUE),
                            ("📣", "Request referral or review",  PURPLE),
                            ("📊", "Monitor usage trends",        CYAN),
                        ]
                        hdr_col, hdr_txt = GREEN, "✅ Growth Actions"

                    rows_h = "".join(
                        f'<div style="display:flex;align-items:center;gap:9px;'
                        f'padding:8px 10px;border-radius:7px;margin-bottom:4px;'
                        f'background:{ac}0C;border:1px solid {ac}1A;">'
                        f'<span style="font-size:.88rem;">{ico}</span>'
                        f'<span style="font-size:.78rem;color:{SUB};">{txt}</span>'
                        f'</div>'
                        for ico, txt, ac in actions
                    )
                    md(f'<div style="background:{CARD};border:1px solid {BDR};'
                       f'border-radius:13px;padding:16px;">'
                       f'<div style="font-size:.74rem;font-weight:700;color:{hdr_col};'
                       f'font-family:Space Grotesk,sans-serif;margin-bottom:12px;">'
                       f'{hdr_txt}</div>{rows_h}</div>')

                # ── What Drives This Prediction ─────────────────────
                md("<br>")
                sec_hdr("🔍", "What Drives This Prediction?",
                        "Risk factors identified from EDA patterns for this customer profile")

                # Build risk factors — simple lists, no complex unpacking
                high_risk, med_risk, low_risk = [], [], []

                if "Month-to-month" in contract:
                    high_risk.append((
                        "📅 Month-to-month Contract",
                        "Historical churn rate: 42.7% — highest of all contract types",
                        "vs only 2.8% churn on 2-year plans",
                    ))
                if "Electronic" in payment:
                    high_risk.append((
                        "💳 Electronic Check Payment",
                        "Historical churn rate: 45.3% — highest of all payment methods",
                        "Customers on auto-pay churn significantly less",
                    ))
                if internet_svc == "Fiber optic":
                    high_risk.append((
                        "🌐 Fiber Optic Internet",
                        "Historical churn rate: 41.9% vs DSL at 18.9%",
                        "Price-to-value perception is the key driver",
                    ))
                if int(tenure) <= 12:
                    high_risk.append((
                        "🆕 New Customer (first 12 months)",
                        f"First-year churn rate: 47.7% — highest risk period",
                        f"This customer has {tenure} months tenure",
                    ))
                if online_sec == "No" and internet_svc != "No":
                    med_risk.append((
                        "🔒 No Online Security",
                        "Customers without it churn at 41.8%",
                        "Adding this service reduces dissatisfaction",
                    ))
                if tech_support == "No" and internet_svc != "No":
                    med_risk.append((
                        "🛠 No Tech Support",
                        "Customers without it churn at 41.6%",
                        "Proactive support prevents frustration",
                    ))
                if float(monthly) > 70:
                    med_risk.append((
                        "💸 High Monthly Charges",
                        f"${monthly:.0f}/month exceeds the $70 risk threshold",
                        "High charges correlate strongly with churn",
                    ))
                if senior == "Yes":
                    med_risk.append((
                        "👴 Senior Citizen",
                        "Churn rate 41.7% vs 24.3% for non-seniors",
                        "Dedicated support programmes improve retention",
                    ))
                if contract == "Two year":
                    low_risk.append((
                        "📋 Two-Year Contract",
                        "Churn rate only 2.8% — strongest retention signal",
                        "Long-term commitment is the best loyalty indicator",
                    ))
                if int(tenure) >= 48:
                    low_risk.append((
                        "🏆 Long-Tenure Customer",
                        f"{tenure} months loyalty — churn risk drops dramatically",
                        "Customers over 48 months churn at only 6.6%",
                    ))
                if contract == "One year":
                    low_risk.append((
                        "📋 One-Year Contract",
                        "Contracted customers churn far less than month-to-month",
                        "Encourage upgrade to 2-year for even better retention",
                    ))
                if online_sec == "Yes" and tech_support == "Yes":
                    low_risk.append((
                        "✅ Fully Supported Customer",
                        "Both online security and tech support active",
                        "Well-served customers have lower churn propensity",
                    ))

                # Fallback if nothing detected
                if not high_risk and not med_risk and not low_risk:
                    low_risk.append((
                        "✅ No Major Risk Factors",
                        "This customer does not match high-risk churn profiles",
                        "Continue monitoring engagement and usage metrics",
                    ))

                # Render in 3 columns — safe, no complex unpacking in f-strings
                all_factors = (
                    [(RED,   "High Risk Factor",   t) for t in high_risk] +
                    [(AMBER, "Medium Risk Factor", t) for t in med_risk]  +
                    [(GREEN, "Positive Signal",    t) for t in low_risk]
                )

                if all_factors:
                    ncols = min(3, len(all_factors))
                    if ncols < 1:
                        ncols = 1
                    fcols = st.columns(ncols, gap="medium")
                    for idx, factor in enumerate(all_factors):
                        col_c   = factor[0]
                        col_lbl = factor[1]
                        title_f, desc_f, note_f = factor[2]
                        card_html = risk_card_html(col_c, col_lbl, title_f, desc_f, note_f)
                        with fcols[idx % ncols]:
                            md(card_html)

                # Summary
                n_h = len(high_risk)
                n_m = len(med_risk)
                n_l = len(low_risk)
                s_col = RED if n_h >= 2 else AMBER if (n_h == 1 or n_m >= 2) else GREEN
                md(f'<div style="background:{s_col}0A;border:1px solid {s_col}28;'
                   f'border-radius:10px;padding:12px 16px;margin-top:10px;">'
                   f'<span style="font-size:.82rem;color:{TEXT};font-weight:600;">'
                   f'Risk Summary: </span>'
                   f'<span style="font-size:.82rem;color:{SUB};">'
                   f'{n_h} high-risk · {n_m} medium-risk · {n_l} positive signal(s). '
                   f'Churn probability: </span>'
                   f'<strong style="color:{tier_col};">{churn_p:.1f}% ({tier_lbl})</strong>'
                   f'</div>')

            except Exception as exc:
                st.error(f"**Prediction error:** {exc}")
                st.info(
                    "Check that all 5 PKL files in `models/` were generated by "
                    "running the notebook end-to-end with the same dataset."
                )

        else:
            divider()
            md(f'<div style="background:{CARD};border:1px dashed {BDR2};'
               f'border-radius:14px;padding:48px 24px;text-align:center;margin:8px 0;">'
               f'<div style="font-size:2.4rem;margin-bottom:12px;">🔮</div>'
               f'<div style="font-family:Space Grotesk,sans-serif;font-size:1rem;'
               f'font-weight:700;color:{TEXT};margin-bottom:8px;">Ready to Analyse</div>'
               f'<div style="font-size:.84rem;color:{SUB};line-height:1.7;'
               f'max-width:360px;margin:0 auto;">Complete the customer profile above '
               f'and click <strong style="color:{BLUE};">Run Churn Prediction</strong> '
               f'to see AI results with risk insights.</div></div>')

            md("<br>")
            sec_hdr("📌", "Key Churn Signals",
                    "Historical EDA patterns — top factors that drive customer churn")
            s1, s2 = st.columns(2, gap="large")
            with s1:
                for lb, rt, ac in [
                    ("Month-to-month contract",  "42.7% churn rate", RED),
                    ("Electronic check payment", "45.3% churn rate", RED),
                    ("Fiber optic internet",     "41.9% churn rate", RED),
                    ("Tenure ≤ 12 months",       "47.7% churn rate", RED),
                ]:
                    md(signal_row(lb, rt, ac))
            with s2:
                for lb, rt, ac in [
                    ("No online security",  "41.8% churn rate", AMBER),
                    ("No tech support",     "41.6% churn rate", AMBER),
                    ("Two-year contract",   " 2.8% churn rate", GREEN),
                    ("Tenure > 48 months",  " 6.6% churn rate", GREEN),
                ]:
                    md(signal_row(lb, rt, ac))


# ═══════════════════════════════════════════
# TAB 2 — EDA DASHBOARD
# ═══════════════════════════════════════════
with TAB2:
    if not ok_d:
        st.warning(
            "**dataset.csv not found.**  "
            "Place it in the same folder as app.py and restart Streamlit."
        )
        md(f'<div style="background:{CARD};border:1px dashed {BDR2};'
           f'border-radius:14px;padding:48px 24px;text-align:center;margin-top:12px;">'
           f'<div style="font-size:2.4rem;margin-bottom:12px;">📊</div>'
           f'<div style="font-family:Space Grotesk,sans-serif;font-size:1rem;'
           f'font-weight:700;color:{TEXT};margin-bottom:8px;">Dataset Required</div>'
           f'<div style="font-size:.84rem;color:{SUB};line-height:1.7;">'
           f'Add dataset.csv to the app folder to view interactive EDA charts.</div></div>')
    else:
        # KPIs
        sec_hdr("📊", "Dataset Overview",
                "7,043 Telco customers · 21 raw features · Kaggle Telco Churn dataset")
        kc2 = st.columns(6, gap="small")
        for col, (ic, lb, vl, ac, nt) in zip(kc2, [
            ("👥", "Total",       f"{len(df):,}",                              BLUE,   ""),
            ("🔴", "Churned",     f"{df['ChurnBin'].sum():,}",                 RED,    f"{df['ChurnBin'].mean()*100:.1f}%"),
            ("🟢", "Retained",    f"{(~df['ChurnBin'].astype(bool)).sum():,}", GREEN,  f"{(1-df['ChurnBin'].mean())*100:.1f}%"),
            ("📅", "Avg Tenure",  f"{df['tenure'].mean():.0f} mo",             PURPLE, "months"),
            ("💵", "Avg Monthly", f"${df['MonthlyCharges'].mean():.0f}",       AMBER,  "/month"),
            ("🧮", "Features",    "19",                                         TEAL,   "post-encoding"),
        ]):
            col.markdown(kpi(ic, lb, vl, ac, nt), unsafe_allow_html=True)

        divider()

        # ── 1. Churn Distribution ───────────────────────────────
        sec_hdr("📉", "1 · Churn Distribution",
                "73.5% No Churn vs 26.5% Churned — class imbalance handled with SMOTE on training data")

        no_ct  = int((df["Churn"] == "No").sum())
        yes_ct = int((df["Churn"] == "Yes").sum())

        ca, cb = st.columns(2, gap="large")
        with ca:
            fig = go.Figure(go.Bar(
                x=["No Churn", "Churned"],
                y=[no_ct, yes_ct],
                marker_color=[GREEN, RED],
                marker_line_color="rgba(0,0,0,0)",
                text=[f"{no_ct:,}", f"{yes_ct:,}"],
                textposition="outside",
                textfont=dict(color=TEXT, size=13),
            ))
            fig.update_layout(**plotly_layout("Churn Count", 300))
            fig.update_yaxes(title="Customers", gridcolor=BDR)
            fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with cb:
            fig = go.Figure(go.Pie(
                labels=["No Churn", "Churned"],
                values=[no_ct, yes_ct],
                hole=0.44,
                pull=[0, 0.05],
                marker=dict(colors=[GREEN, RED],
                            line=dict(color=BG, width=2)),
                textinfo="label+percent",
                textfont=dict(size=12),
            ))
            fig.update_layout(**plotly_layout("Churn Split (%)", 300))
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        md(f'<div style="background:{AMBER}0A;border:1px solid {AMBER}30;'
           f'border-radius:9px;padding:11px 16px;margin:6px 0;">'
           f'<strong style="color:{AMBER};">⚠ Class Imbalance:</strong>'
           f'<span style="color:{SUB};font-size:.84rem;"> 73.5% vs 26.5% — '
           f'SMOTE applied on training data only to prevent data leakage.</span></div>')

        # ── 2. Numerical Distributions ──────────────────────────
        divider()
        sec_hdr("📐", "2 · Numerical Feature Distributions",
                "Tenure, MonthlyCharges, TotalCharges — split by churn status")

        num_cols_list = ["tenure", "MonthlyCharges", "TotalCharges"]
        fig2a = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Tenure (months)", "Monthly Charges ($)", "Total Charges ($)"],
            horizontal_spacing=0.06,
        )
        for i, col_name in enumerate(num_cols_list, 1):
            for churn_val, nm, clr in [
                (False, "No Churn", BLUE),
                (True,  "Churned",  RED),
            ]:
                sub = df[df["ChurnBin"] == int(churn_val)][col_name]
                fig2a.add_trace(go.Histogram(
                    x=sub, name=nm, marker_color=clr,
                    opacity=0.7, nbinsx=30,
                    showlegend=(i == 1),
                    legendgroup=nm,
                ), row=1, col=i)
        lyt2a = plotly_layout("Distribution by Churn Status", 320)
        lyt2a["barmode"] = "overlay"
        fig2a.update_layout(**lyt2a)
        fig2a.update_xaxes(gridcolor=BDR, linecolor=BDR)
        fig2a.update_yaxes(gridcolor=BDR, linecolor=BDR)
        for ann in fig2a.layout.annotations:
            ann.font.color = SUB
            ann.font.size  = 11
        st.plotly_chart(fig2a, use_container_width=True)

        fig2b = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Tenure vs Churn", "Monthly Charges vs Churn",
                            "Total Charges vs Churn"],
            horizontal_spacing=0.06,
        )
        for i, col_name in enumerate(num_cols_list, 1):
            for churn_val, nm, clr in [
                ("No", "No Churn", BLUE),
                ("Yes", "Churned", RED),
            ]:
                fig2b.add_trace(go.Box(
                    y=df[df["Churn"] == churn_val][col_name],
                    name=nm, marker_color=clr, line_color=clr,
                    showlegend=(i == 1), legendgroup=nm,
                ), row=1, col=i)
        lyt2b = plotly_layout("Spread by Churn Status", 320)
        lyt2b["boxmode"] = "group"
        fig2b.update_layout(**lyt2b)
        fig2b.update_xaxes(gridcolor="rgba(0,0,0,0)")
        fig2b.update_yaxes(gridcolor=BDR)
        for ann in fig2b.layout.annotations:
            ann.font.color = SUB
            ann.font.size  = 11
        st.plotly_chart(fig2b, use_container_width=True)

        md(f'<div style="background:{CARD2};border:1px solid {BDR};'
           f'border-radius:9px;padding:11px 16px;margin:4px 0;">'
           f'<span style="font-size:.82rem;color:{SUB};">'
           f'<strong style="color:{RED};">Tenure:</strong> Bimodal — many new + many long-term customers &nbsp;·&nbsp; '
           f'<strong style="color:{AMBER};">MonthlyCharges:</strong> Churners tend to pay more &nbsp;·&nbsp; '
           f'<strong style="color:{BLUE};">TotalCharges:</strong> Right-skewed, correlated with tenure'
           f'</span></div>')

        # ── 3. Categorical Churn Rates ────────────────────────
        divider()
        sec_hdr("📊", "3 · Churn Rate by Category",
                "Which service and billing categories drive churn most?")

        cat_feats = [
            "Contract", "InternetService", "PaymentMethod",
            "TechSupport", "OnlineSecurity", "PaperlessBilling",
        ]
        fig3 = make_subplots(
            rows=2, cols=3,
            subplot_titles=cat_feats,
            vertical_spacing=0.2,
            horizontal_spacing=0.07,
        )
        for idx, feat in enumerate(cat_feats):
            r, c = divmod(idx, 3)
            cr = (df.groupby(feat)["Churn"]
                  .apply(lambda x: (x == "Yes").mean() * 100)
                  .sort_values(ascending=False)
                  .reset_index())
            cr.columns = [feat, "Pct"]
            bclrs = [RED if v > 30 else AMBER if v > 15 else GREEN
                     for v in cr["Pct"]]
            fig3.add_trace(go.Bar(
                x=cr[feat], y=cr["Pct"],
                marker_color=bclrs,
                marker_line_color="rgba(0,0,0,0)",
                text=[f"{v:.1f}%" for v in cr["Pct"]],
                textposition="outside",
                textfont=dict(color=TEXT, size=9),
                showlegend=False,
            ), row=r+1, col=c+1)

        lyt3 = plotly_layout("Churn Rate (%) by Feature Category", 540)
        fig3.update_layout(**lyt3)
        fig3.update_xaxes(tickfont=dict(size=8, color=SUB),
                          gridcolor="rgba(0,0,0,0)")
        fig3.update_yaxes(gridcolor=BDR, tickfont=dict(color=MUTED))
        for ann in fig3.layout.annotations:
            ann.font.color = SUB
            ann.font.size  = 11
        st.plotly_chart(fig3, use_container_width=True)

        md(f'<div style="background:{CARD2};border:1px solid {BDR};'
           f'border-radius:9px;padding:11px 16px;margin:4px 0;">'
           f'<span style="font-size:.82rem;color:{SUB};">'
           f'<strong style="color:{RED};">Contract:</strong> Month-to-month 42.7% vs Two-year 2.8% &nbsp;·&nbsp; '
           f'<strong style="color:{RED};">Payment:</strong> Electronic check 45.3% &nbsp;·&nbsp; '
           f'<strong style="color:{AMBER};">Internet:</strong> Fiber optic 41.9% vs DSL 18.9% &nbsp;·&nbsp; '
           f'<strong style="color:{AMBER};">Tech Support:</strong> Without 41.6% vs With 14.8%'
           f'</span></div>')

        # ── 4. Demographics ────────────────────────────────────
        divider()
        sec_hdr("👥", "4 · Demographics vs Churn",
                "Gender, Senior Citizen, Partner, Dependents breakdown")

        demo_feats = ["gender", "SeniorCitizen", "Partner", "Dependents"]
        d_cols = st.columns(4, gap="small")
        for i, feat in enumerate(demo_feats):
            with d_cols[i]:
                ct = (df.groupby(feat)["Churn"]
                      .apply(lambda x: (x == "Yes").mean() * 100)
                      .reset_index())
                ct.columns = [feat, "Pct"]
                # Convert x to string to handle int (SeniorCitizen = 0/1)
                x_labels = ct[feat].astype(str).tolist()
                clrs = [BLUE, RED] if len(ct) == 2 else [BLUE]*len(ct)
                fig_d = go.Figure(go.Bar(
                    x=x_labels,
                    y=ct["Pct"].tolist(),
                    marker_color=clrs,
                    marker_line_color="rgba(0,0,0,0)",
                    text=[f"{v:.1f}%" for v in ct["Pct"]],
                    textposition="outside",
                    textfont=dict(color=TEXT, size=10),
                ))
                fig_d.update_layout(**plotly_layout(feat, 240))
                fig_d.update_layout(margin=dict(l=5, r=5, t=40, b=5))
                fig_d.update_yaxes(title="Churn Rate (%)", range=[0, max(ct["Pct"])*1.32])
                fig_d.update_xaxes(gridcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_d, use_container_width=True)

        # ── 5. Tenure Groups ───────────────────────────────────
        divider()
        sec_hdr("⏱️", "5 · Churn Rate by Tenure Group",
                "0-12 months is the highest-risk window — 47.7% churn rate")

        tg = (df.groupby("TenureGroup", observed=True)["Churn"]
              .apply(lambda x: (x == "Yes").mean() * 100)
              .reset_index())
        tg.columns = ["Group", "Pct"]
        tg["Group"] = tg["Group"].astype(str)  # safe cast from Categorical

        tg_colors = [RED, AMBER, BLUE, TEAL]
        tc1, tc2 = st.columns([2, 1], gap="large")
        with tc1:
            fig_tg = go.Figure(go.Bar(
                x=tg["Group"].tolist(),
                y=tg["Pct"].tolist(),
                marker_color=tg_colors[:len(tg)],
                marker_line_color="rgba(0,0,0,0)",
                text=[f"{v:.1f}%" for v in tg["Pct"]],
                textposition="outside",
                textfont=dict(color=TEXT, size=12),
            ))
            fig_tg.update_layout(**plotly_layout("Churn Rate by Tenure Group", 300))
            fig_tg.update_yaxes(title="Churn Rate (%)", gridcolor=BDR)
            fig_tg.update_xaxes(title="Tenure Group", gridcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_tg, use_container_width=True)
        with tc2:
            for row_idx, row in tg.iterrows():
                ac = tg_colors[row_idx] if row_idx < len(tg_colors) else BLUE
                md(f'<div style="background:{CARD};border:1px solid {ac}30;'
                   f'border-left:3px solid {ac};border-radius:8px;'
                   f'padding:12px 14px;margin-bottom:7px;">'
                   f'<div style="font-size:.69rem;color:{MUTED};margin-bottom:2px;">'
                   f'{row["Group"]}</div>'
                   f'<div style="font-family:Space Grotesk,sans-serif;font-size:1.4rem;'
                   f'font-weight:700;color:{ac};">{row["Pct"]:.1f}%</div>'
                   f'</div>')

        # ── 6. Correlation Heatmap ─────────────────────────────
        divider()
        sec_hdr("🔥", "6 · Feature Correlation Heatmap",
                "Label-encoded · lower triangle only · key correlations with Churn")

        df_corr = df.drop(
            columns=["ChurnBin", "TenureGroup", "customerID"], errors="ignore"
        ).copy()
        le_tmp = LabelEncoder()
        # Fix: use include="object" with str handling for newer pandas
        obj_cols = [c for c in df_corr.columns
                    if df_corr[c].dtype == object or df_corr[c].dtype.name == "string"]
        for col_name in obj_cols:
            df_corr[col_name] = le_tmp.fit_transform(df_corr[col_name].astype(str))

        corr = df_corr.corr().round(2)
        # Mask upper triangle — replace with NaN
        mask = np.triu(np.ones_like(corr.values, dtype=bool), k=1)
        z_vals = corr.values.copy().astype(float)
        z_vals[mask] = np.nan

        # Build text array — show value or empty string (no "nan")
        text_vals = []
        for row_arr in z_vals:
            row_text = []
            for v in row_arr:
                row_text.append("" if np.isnan(v) else str(round(v, 2)))
            text_vals.append(row_text)

        # fig_hm = go.Figure(go.Heatmap(
        #     z=z_vals,
        #     x=corr.columns.tolist(),
        #     y=corr.index.tolist(),
        #     colorscale="RdBu_r",
        #     zmid=0,
        #     zmin=-1, zmax=1,
        #     text=text_vals,
        #     texttemplate="%{text}",
        #     textfont=dict(size=7, color="white"),
        #     hoverongaps=False,
        #  colorbar=dict(
        #         title=dict(
        #             text="r", 
        #             font=dict(color=SUB, size=10)
        #         ),
        #         tickfont=dict(color=SUB, size=9),
        #     ),
        # ))
        # hm_layout = plotly_layout("Feature Correlation Matrix", 520)
        # hm_layout["xaxis"] = dict(
        #     tickfont=dict(size=8, color=SUB), tickangle=45,
        #     gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
        # )
        # hm_layout["yaxis"] = dict(
        #     tickfont=dict(size=8, color=SUB),
        #     gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
        # )
        # hm_layout["margin"] = dict(l=120, r=20, t=50, b=120)
        # fig_hm.update_layout(**hm_layout)
        # st.plotly_chart(fig_hm, use_container_width=True)

        # md(f'<div style="background:{CARD2};border:1px solid {BDR};'
        #    f'border-radius:9px;padding:11px 16px;margin:4px 0;">'
        #    f'<span style="font-size:.82rem;color:{SUB};">'
        #    f'<strong style="color:{TEAL};">tenure</strong> negatively correlated with Churn — '
        #    f'long-term customers stay &nbsp;·&nbsp; '
        #    f'<strong style="color:{RED};">MonthlyCharges</strong> positively correlated — '
        #    f'expensive plans drive churn &nbsp;·&nbsp; '
        #    f'TotalCharges highly correlated with tenure (expected)'
        #    f'</span></div>')
        
        #===========================
        fig_hm = go.Figure(go.Heatmap(
            z=z_vals,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[
                [0.00, '#3b4cc0'], [0.15, '#6788ee'], [0.35, '#9abbff'], 
                [0.50, '#e2e2e2'], [0.65, '#f1a88d'], [0.85, '#d35c4e'], [1.00, '#b40426']
            ],
            zmid=0,
            zmin=-1, zmax=1,
            text=text_vals,
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=11),  # <-- FIXED: Plotly will now auto-contrast the text!
            hoverongaps=False,
            colorbar=dict(
                title=dict(
                    text="r", 
                    font=dict(color=SUB, size=12)
                ),
                tickfont=dict(color=SUB, size=11),
            ),
        ))
       
        
        # Increased height from 520 to 750 so it doesn't look shrunk
        hm_layout = plotly_layout("Feature Correlation Matrix", 750) 
        
        hm_layout["xaxis"] = dict(
            tickfont=dict(size=10, color=SUB), tickangle=45,  # Increased axis font size
            gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
        )
        hm_layout["yaxis"] = dict(
            tickfont=dict(size=10, color=SUB),                # Increased axis font size
            gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
        )
        hm_layout["margin"] = dict(l=120, r=20, t=50, b=120)
        
        fig_hm.update_layout(**hm_layout)
        st.plotly_chart(fig_hm, use_container_width=True)

        md(f'<div style="background:{CARD2};border:1px solid {BDR};'
           f'border-radius:9px;padding:11px 16px;margin:4px 0;">'
           f'<span style="font-size:.82rem;color:{SUB};">'
           f'<strong style="color:{TEAL};">tenure</strong> negatively correlated with Churn — '
           f'long-term customers stay &nbsp;·&nbsp; '
           f'<strong style="color:{RED};">MonthlyCharges</strong> positively correlated — '
           f'expensive plans drive churn &nbsp;·&nbsp; '
           f'TotalCharges highly correlated with tenure (expected)'
           f'</span></div>')
        #===========================

        # Raw data browser
        with st.expander("📋  Browse Raw Dataset"):
            flt = st.text_input("Filter by Churn (Yes / No / blank = all):", "")
            vdf = (df[df["Churn"].str.contains(flt, case=False, na=False)]
                   if flt else df)
            st.dataframe(
                vdf.drop(columns=["ChurnBin", "TenureGroup"], errors="ignore").head(300),
                use_container_width=True,
            )
            st.caption(f"Showing {min(300, len(vdf))} of {len(vdf):,} rows")


# ═══════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════
with TAB3:

    # Pipeline
    sec_hdr("🏆", "Model Training Pipeline",
            "End-to-end: raw data → SMOTE → tuning → best model → PKL artifacts")
    steps_html = "  ".join(
        f'<div style="background:{cl}12;border:1px solid {cl}30;border-radius:7px;'
        f'padding:7px 13px;font-size:.77rem;font-family:JetBrains Mono,monospace;'
        f'color:{cl};white-space:nowrap;">{s}</div>'
        for s, cl in [
            ("1. Data Cleaning",       BLUE),
            ("2. Feature Encoding",    CYAN),
            ("3. Train/Test Split",    TEAL),
            ("4. SMOTE Balancing",     GREEN),
            ("5. Baseline ×3 Models",  PURPLE),
            ("6. GridSearchCV Tuning", AMBER),
            ("7. Select by AUC-ROC",   RED),
            ("8. Save PKL Artifacts",  BLUE),
        ]
    )
    md(f'<div style="background:{CARD};border:1px solid {BDR};border-radius:11px;'
       f'padding:15px 17px;margin-bottom:20px;">'
       f'<div style="display:flex;gap:8px;flex-wrap:wrap;">{steps_html}</div></div>')

    # Performance table
    sec_hdr("📊", "Model Performance — Baseline vs Tuned",
            "Correct verified values from notebook execution")

    perf_df = pd.DataFrame({
        "Model":           ["XGBoost  ★ Best", "Random Forest", "Logistic Regression"],
        "Base Accuracy":   [0.7779, 0.7771, 0.7424],
        "Tuned Accuracy":  [0.7757, 0.7622, 0.7410],
        "Base F1":         [0.5865, 0.5890, 0.6191],
        "Tuned F1":        [0.6274, 0.6223, 0.6146],
        "Base AUC-ROC":    [0.8146, 0.8210, 0.8391],
        "Tuned AUC-ROC":   [0.8434, 0.8397, 0.8382],
        "Speed":           ["Medium", "Slow", "Fast"],
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # Metric bar charts
    md("<br>")
    mc_cols = st.columns(3, gap="medium")
    mnames  = ["XGBoost", "Random Forest", "Logistic Reg."]
    for i, (metric, vals, ac) in enumerate([
        ("Tuned Accuracy", [0.7757, 0.7622, 0.7410], BLUE),
        ("Tuned AUC-ROC",  [0.8434, 0.8397, 0.8382], AMBER),
        ("Tuned F1-Score", [0.6274, 0.6223, 0.6146], GREEN),
    ]):
        with mc_cols[i]:
            best   = max(vals)
            bclrs  = [ac if abs(v - best) < 1e-5 else BDR2 for v in vals]
            texts  = [f"{v:.4f} ★" if abs(v-best)<1e-5 else f"{v:.4f}" for v in vals]
            fig_mc = go.Figure(go.Bar(
                y=mnames, x=vals, orientation="h",
                marker_color=bclrs,
                marker_line_color="rgba(0,0,0,0)",
                text=texts, textposition="outside",
                textfont=dict(color=TEXT, size=10),
            ))
            fig_mc.update_layout(**plotly_layout(metric, 210))
            fig_mc.update_layout(margin=dict(l=8, r=65, t=42, b=8))
            fig_mc.update_xaxes(range=[min(vals)-0.05, max(vals)+0.07], gridcolor=BDR)
            fig_mc.update_yaxes(gridcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_mc, use_container_width=True)

    # Baseline vs Tuned comparison
    sec_hdr("📈", "Baseline vs Tuned Improvement",
            "F1-Score and AUC-ROC gain after GridSearchCV hyperparameter tuning")

    fig_bt = make_subplots(
        rows=1, cols=2,
        subplot_titles=["F1-Score: Baseline vs Tuned", "AUC-ROC: Baseline vs Tuned"],
        horizontal_spacing=0.1,
    )
    for cidx, (bv, tv, mn_col) in enumerate([
        ([0.5865, 0.5890, 0.6191], [0.6274, 0.6223, 0.6146], "F1"),
        ([0.8146, 0.8210, 0.8391], [0.8434, 0.8397, 0.8382], "AUC"),
    ], 1):
        all_v = bv + tv
        fig_bt.add_trace(go.Bar(
            name="Baseline", x=mnames, y=bv,
            marker_color=BDR2,
            marker_line_color="rgba(0,0,0,0)",
            text=[f"{v:.4f}" for v in bv], textposition="outside",
            textfont=dict(color=SUB, size=9),
            showlegend=(cidx == 1),
        ), row=1, col=cidx)
        fig_bt.add_trace(go.Bar(
            name="Tuned", x=mnames, y=tv,
            marker_color=BLUE,
            marker_line_color="rgba(0,0,0,0)",
            text=[f"{v:.4f}" for v in tv], textposition="outside",
            textfont=dict(color=TEXT, size=9),
            showlegend=(cidx == 1),
        ), row=1, col=cidx)
        fig_bt.update_yaxes(range=[min(all_v)-0.04, max(all_v)+0.06],
                            gridcolor=BDR, row=1, col=cidx)
        fig_bt.update_xaxes(gridcolor="rgba(0,0,0,0)", row=1, col=cidx)

    lyt_bt = plotly_layout("Baseline vs Tuned — All 3 Models", 340)
    lyt_bt["barmode"] = "group"
    fig_bt.update_layout(**lyt_bt)
    for ann in fig_bt.layout.annotations:
        ann.font.color = SUB
        ann.font.size  = 11
    st.plotly_chart(fig_bt, use_container_width=True)

    # Feature Importance
    if ok_m:
        sec_hdr("📌", "Feature Importance",
                f"Top 15 features driving {model_lbl} predictions")
        try:
            actual_m = arts["model"]
            if hasattr(actual_m, "named_steps"):
                actual_m = actual_m.named_steps.get("model", actual_m)
            if hasattr(actual_m, "feature_importances_"):
                imps = actual_m.feature_importances_
            elif hasattr(actual_m, "coef_"):
                imps = np.abs(actual_m.coef_[0])
            else:
                raise ValueError("No importances or coefficients on model object")

            fi = (pd.Series(imps, index=arts["cols"])
                  .sort_values(ascending=True)
                  .tail(15))
            top3 = sorted(fi.values)[-3]
            fi_clrs = [AMBER if v >= top3 else BLUE for v in fi.values]
            fi_txt  = [f"{v:.4f}" for v in fi.values]

            fig_fi = go.Figure(go.Bar(
                y=fi.index.tolist(),
                x=fi.values.tolist(),
                orientation="h",
                marker_color=fi_clrs,
                marker_line_color="rgba(0,0,0,0)",
                text=fi_txt, textposition="outside",
                textfont=dict(color=TEXT, size=9),
            ))
            fig_fi.update_layout(**plotly_layout(f"Feature Importances — {model_lbl}", 440))
            fig_fi.update_layout(margin=dict(l=8, r=70, t=50, b=8))
            fig_fi.update_xaxes(gridcolor=BDR, title="Importance Score")
            fig_fi.update_yaxes(gridcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_fi, use_container_width=True)
            md(f'<div style="font-size:.78rem;color:{MUTED};margin-top:4px;">'
               f'<span style="color:{AMBER};">■</span> Top 3 drivers &nbsp;'
               f'<span style="color:{BLUE};">■</span> Other features</div>')
        except Exception as e:
            st.info(f"Feature importance unavailable: {e}")
    else:
        st.info("Load model PKL files to see feature importance chart.")

    # Final metrics
    sec_hdr("📋", "Final Model Metrics — XGBoost",
            "Held-out test set · 1,409 samples · no data leakage")

    m5c = st.columns(5, gap="small")
    for col, (ic, lb, vl, ac) in zip(m5c, [
        ("🎯", "Accuracy",  f"{ACC*100:.2f}%", BLUE),
        ("⚡", "Precision", f"{PREC*100:.2f}%", PURPLE),
        ("🔍", "Recall",    f"{REC*100:.2f}%",  TEAL),
        ("📊", "F1 Score",  f"{F1:.4f}",        GREEN),
        ("📈", "AUC-ROC",   f"{AUC:.4f}",       AMBER),
    ]):
        col.markdown(kpi(ic, lb, vl, ac), unsafe_allow_html=True)

    # Classification report + hyperparams
    md("<br>")
    cr1, cr2 = st.columns(2, gap="large")

    with cr1:
        sec_hdr("📄", "Classification Report",
                "Per-class breakdown on test set — No Churn vs Churn")
        report_data = pd.DataFrame({
            "Class":     ["No Churn", "Churn", "Macro Avg", "Weighted Avg"],
            "Precision": [0.88, 0.56, 0.72, 0.80],
            "Recall":    [0.80, 0.71, 0.76, 0.78],
            "F1-Score":  [0.84, 0.63, 0.73, 0.78],
            "Support":   [1035, 374, 1409, 1409],
        })
        st.dataframe(report_data, use_container_width=True, hide_index=True)

    with cr2:
        sec_hdr("⚙️", "Best Hyperparameters",
                "XGBoost — GridSearchCV with 5-fold StratifiedKFold · scoring=F1")
        params = {}
        if ok_m:
            raw_params = m_info.get("best_params", {})
            if raw_params:
                params = {k.replace("model__", ""): v for k, v in raw_params.items()}
        if not params:
            params = {
                "colsample_bytree": 0.8,
                "gamma":            0,
                "learning_rate":    0.05,
                "max_depth":        3,
                "n_estimators":     200,
                "reg_lambda":       1,
                "subsample":        0.8,
            }
        rows_p = "".join(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:8px 0;border-bottom:1px solid {BDR}38;">'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:.79rem;'
            f'color:{CYAN};">{k}</span>'
            f'<span style="font-size:.79rem;color:{TEXT};font-weight:600;">{v}</span>'
            f'</div>'
            for k, v in params.items()
        )
        md(f'<div style="background:{CARD};border:1px solid {BDR};'
           f'border-radius:10px;padding:13px 15px;">{rows_p}</div>')

    # Business Insights
    divider()
    sec_hdr("💡", "Key Business Insights",
            "Actionable retention strategies from EDA and model feature importance")

    insights_data = [
        ("📅", "Contract Type",
         "Month-to-month customers churn at 42.7% vs only 2.8% on 2-year plans. "
         "Offer discounts to upgrade contract length.",
         RED),
        ("🆕", "New Customers",
         "First 12 months = 47.7% churn rate — highest-risk window. "
         "Invest heavily in onboarding and early engagement.",
         AMBER),
        ("💸", "Monthly Charges",
         "Bills over $70/month strongly correlate with churn. "
         "Introduce loyalty pricing and value bundles for high-payers.",
         AMBER),
        ("👴", "Senior Citizens",
         "Senior customers churn at 41.7% vs 24.3% for non-seniors. "
         "Dedicated support and simplified plans improve retention.",
         RED),
        ("💳", "Payment Method",
         "Electronic check = 45.3% churn — highest of all methods. "
         "Incentivise migration to automatic payment.",
         RED),
        ("🌐", "Fiber Optic",
         "Fiber optic churn: 41.9% vs DSL 18.9%. "
         "Price-to-value perception drives dissatisfaction — target with loyalty offers.",
         AMBER),
    ]
    for row_g in [insights_data[:3], insights_data[3:]]:
        rc = st.columns(3, gap="medium")
        for col, (ic, ti, de, ac) in zip(rc, row_g):
            col.markdown(insight_card(ic, ti, de, ac), unsafe_allow_html=True)
        md("<br>")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
md(f'<div style="margin-top:52px;padding:18px 0;border-top:1px solid {BDR};'
   f'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">'
   f'<div style="display:flex;align-items:center;gap:9px;">'
   f'<div style="width:28px;height:28px;border-radius:8px;background:{BLUE};'
   f'display:flex;align-items:center;justify-content:center;font-size:.88rem;">📡</div>'
   f'<span style="font-family:Space Grotesk,sans-serif;font-size:.88rem;'
   f'font-weight:700;color:{TEXT};">TeleChurn AI</span>'
   f'<span style="font-size:.76rem;color:{MUTED};">· Telco Customer Intelligence</span>'
   f'</div>'
   f'<div style="font-size:.7rem;color:{MUTED};">'
   f'XGBoost · SMOTE · GridSearchCV · Scikit-learn · Plotly · Streamlit · Python'
   f'</div></div>')