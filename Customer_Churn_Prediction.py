#!/usr/bin/env python
# coding: utf-8

# ============================================================
#  Customer Churn Prediction
#  All charts combined in ONE browser tab — dark modern theme
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score

# ── Dark theme ───────────────────────────────────────────────
DARK_BG   = '#0f1117'
CARD_BG   = '#161820'
GRID_CLR  = 'rgba(255,255,255,0.06)'
TEXT_CLR  = '#c9d1d9'
MUTED_CLR = '#6b7280'
PALETTE   = ['#7F77DD','#1D9E75','#D85A30','#EF9F27',
             '#378ADD','#D4537E','#888780','#5DCAA5']

pio.templates['dark_custom'] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_CLR, family='Arial, sans-serif', size=12),
        title=dict(font=dict(size=14, color=TEXT_CLR)),
        xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR,
                   tickfont=dict(color=MUTED_CLR, size=11)),
        yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR,
                   tickfont=dict(color=MUTED_CLR, size=11)),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=MUTED_CLR, size=11)),
        colorway=PALETTE,
    )
)
pio.templates.default = 'dark_custom'

# ── 1. Load dataset ──────────────────────────────────────────
print("Loading dataset...")
churn_dataset = pd.read_csv('Tel_Customer_Churn_Dataset.csv')
print(f"  Shape: {churn_dataset.shape}")

# ── 2. Preprocessing ─────────────────────────────────────────
# # Converting Churn column from Yes/No text to 1/0 numbers so ML models can process it
churn_dataset['Churn'] = churn_dataset['Churn'].map({'No': 0, 'Yes': 1}).astype(int)

cols_internet = ['OnlineBackup','StreamingMovies','DeviceProtection',
                 'TechSupport','OnlineSecurity','StreamingTV']
for col in cols_internet:
    churn_dataset[col] = churn_dataset[col].replace({'No internet service': 'No'})

churn_dataset['TotalCharges'] = churn_dataset['TotalCharges'].replace(' ', np.nan)
churn_dataset = churn_dataset[churn_dataset['TotalCharges'].notnull()]
churn_dataset = churn_dataset.reset_index()[churn_dataset.columns]
churn_dataset['TotalCharges'] = churn_dataset['TotalCharges'].astype(float)

print(f"  Churn counts: {churn_dataset['Churn'].value_counts().to_dict()}")

# ── 3. EDA aggregations ──────────────────────────────────────
churn_counts   = churn_dataset['Churn'].value_counts()
by_gender      = churn_dataset.groupby('gender').Churn.mean().reset_index()
by_techsupport = churn_dataset.groupby('TechSupport').Churn.mean().reset_index()
by_internet    = churn_dataset.groupby('InternetService').Churn.mean().reset_index()
by_payment     = churn_dataset.groupby('PaymentMethod').Churn.mean().reset_index()
by_contract    = churn_dataset.groupby('Contract').Churn.mean().reset_index()
by_tenure      = churn_dataset.groupby('tenure').Churn.mean().reset_index()

# ── 4. EDA dashboard figure ──────────────────────────────────
print("\nBuilding EDA dashboard...")

fig_eda = make_subplots(
    rows=3, cols=3,
    subplot_titles=[
        'Customer churn distribution',
        'Churn rate by gender',
        'Churn rate by tech support',
        'Churn rate by internet service',
        'Churn rate by payment method',
        'Churn rate by contract type',
        'Churn rate vs tenure (months)',
        '', ''
    ],
    specs=[
        [{'type': 'domain'}, {'type': 'xy'}, {'type': 'xy'}],
        [{'type': 'xy'},     {'type': 'xy'}, {'type': 'xy'}],
        [{'type': 'xy', 'colspan': 3}, None, None],
    ],
    vertical_spacing=0.13,
    horizontal_spacing=0.08,
)

fig_eda.add_trace(go.Pie(
    labels=['Retained', 'Churned'],
    values=churn_counts.values.tolist(),
    hole=0.62,
    marker=dict(colors=['#1D9E75','#7F77DD'],
                line=dict(color=DARK_BG, width=2)),
    textfont=dict(color=TEXT_CLR),
    hovertemplate='%{label}: %{value} (%{percent})<extra></extra>',
), row=1, col=1)

fig_eda.add_trace(go.Bar(
    x=by_gender['gender'],
    y=(by_gender['Churn']*100).round(1),
    marker_color=['#D4537E','#378ADD'], width=0.4,
    hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
), row=1, col=2)

fig_eda.add_trace(go.Bar(
    x=by_techsupport['TechSupport'],
    y=(by_techsupport['Churn']*100).round(1),
    marker_color=['#D85A30','#1D9E75'], width=0.4,
    hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
), row=1, col=3)

fig_eda.add_trace(go.Bar(
    x=by_internet['InternetService'],
    y=(by_internet['Churn']*100).round(1),
    marker_color=PALETTE[:3], width=0.4,
    hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
), row=2, col=1)

fig_eda.add_trace(go.Bar(
    x=(by_payment['Churn']*100).round(1),
    y=by_payment['PaymentMethod'],
    orientation='h',
    marker_color=['#D85A30','#EF9F27','#7F77DD','#1D9E75'],
    hovertemplate='%{y}: %{x:.1f}%<extra></extra>',
), row=2, col=2)

fig_eda.add_trace(go.Bar(
    x=by_contract['Contract'],
    y=(by_contract['Churn']*100).round(1),
    marker_color=['#D85A30','#EF9F27','#1D9E75'], width=0.4,
    hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
), row=2, col=3)

fig_eda.add_trace(go.Scatter(
    x=by_tenure['tenure'],
    y=(by_tenure['Churn']*100).round(1),
    mode='markers',
    marker=dict(size=6, color='#EF9F27',
                line=dict(width=0.5, color=DARK_BG)),
    hovertemplate='Tenure %{x}m: %{y:.1f}%<extra></extra>',
), row=3, col=1)

fig_eda.update_layout(
    height=900,
    title=dict(text='Customer churn — exploratory data analysis',
               font=dict(size=16, color=TEXT_CLR), x=0.01),
    paper_bgcolor=DARK_BG,
    plot_bgcolor=CARD_BG,
    showlegend=False,
    margin=dict(t=80, b=40, l=40, r=40),
)
for ann in fig_eda.layout.annotations:
    ann.font.color = MUTED_CLR
    ann.font.size  = 11

# ── 5. ML pipeline ───────────────────────────────────────────
print("\nRunning ML models (SVM may take ~1 min)...")

churn_ml = churn_dataset.copy()
churn_ml = pd.get_dummies(
    churn_ml,
    columns=['Contract','Dependents','DeviceProtection','gender',
             'InternetService','MultipleLines','OnlineBackup',
             'OnlineSecurity','PaperlessBilling','Partner',
             'PaymentMethod','PhoneService','SeniorCitizen',
             'StreamingMovies','StreamingTV','TechSupport'],
    drop_first=True
)

scaler = StandardScaler()
for col in ['tenure','MonthlyCharges','TotalCharges']:
    churn_ml[col] = scaler.fit_transform(churn_ml[[col]])

y = churn_ml['Churn']
X = churn_ml.drop(['Churn','customerID'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=50)

models_cfg = [
    ('Logistic Regression',    LogisticRegression(random_state=50, max_iter=500)),
    ('Support Vector Machine', SVC(kernel='linear', random_state=50, probability=True)),
    ('K-Nearest Neighbor',     KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)),
    ('Decision Tree',          DecisionTreeClassifier(criterion='gini', random_state=50)),
    ('Random Forest',          RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)),
]

results = {}
for name, model in models_cfg:
    print(f"  Training {name}...")
    model.fit(X_train, y_train)
    pred_y = model.predict(X_test)
    acc = round(accuracy_score(y_test, pred_y) * 100, 2)
    f1 = round(f1_score(y_test, pred_y) * 100, 2)
    precision = round(precision_score(y_test, pred_y) * 100, 2)
    recall = round(recall_score(y_test, pred_y) * 100, 2)
    results[name] = {
        'model': model, 'pred': pred_y,
        'accuracy': acc, 'f1': f1,
        'precision': precision, 'recall': recall
    }
    print(f"    Accuracy: {acc}% | F1: {f1}% | Precision: {precision}% | Recall: {recall}%")

best_name = max(results, key=lambda k: results[k]['accuracy'])
best_pred = results[best_name]['pred']
conf_mat  = confusion_matrix(y_test, best_pred)
tn, fp, fn, tp = conf_mat.ravel()
print(f"\nBest model: {best_name} ({results[best_name]['accuracy']}%)")

# Feature importance from Random Forest
rf_model = results['Random Forest']['model']
importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False).head(10)
print("\nTop 10 important features for churn prediction:")
print(feat_imp.to_string(index=False))

# ── 6. Model comparison figure ───────────────────────────────
print("\nBuilding ML dashboard...")

model_names = [n for n, _ in models_cfg]
accuracies  = [results[n]['accuracy'] for n in model_names]
sorted_pairs = sorted(zip(accuracies, model_names), reverse=True)
s_acc, s_names = zip(*sorted_pairs)
s_colors = ['#7F77DD' if n == best_name else '#2a2d3a' for n in s_names]

fig_ml = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Model accuracy comparison',
                    f'Confusion matrix — {best_name}'],
    column_widths=[0.55, 0.45],
    horizontal_spacing=0.10,
)

fig_ml.add_trace(go.Bar(
    x=list(s_acc),
    y=list(s_names),
    orientation='h',
    marker_color=s_colors,
    text=[f'{a:.1f}%' for a in s_acc],
    textposition='outside',
    textfont=dict(color=TEXT_CLR, size=11),
    hovertemplate='%{y}: %{x:.1f}%<extra></extra>',
    width=0.5,
), row=1, col=1)

conf_data = [
    ('True Negative',  tn, '#085041', '#9FE1CB'),
    ('False Positive', fp, '#501313', '#F7C1C1'),
    ('False Negative', fn, '#501313', '#F7C1C1'),
    ('True Positive',  tp, '#085041', '#9FE1CB'),
]
for i, (lbl, val, bg, tc) in enumerate(conf_data):
    fig_ml.add_annotation(
        xref='x2 domain', yref='y2 domain',
        x=(i % 2) * 0.55,
        y=1 - (i // 2) * 0.55,
        text=f'<b>{val}</b><br><span style="font-size:10px">{lbl}</span>',
        showarrow=False,
        font=dict(color=tc, size=14),
        bgcolor=bg,
        borderpad=18,
        width=130,
        align='center',
    )

fig_ml.update_layout(
    height=420,
    title=dict(text='ML model evaluation',
               font=dict(size=16, color=TEXT_CLR), x=0.01),
    paper_bgcolor=DARK_BG,
    plot_bgcolor=CARD_BG,
    showlegend=False,
    margin=dict(t=80, b=40, l=160, r=60),
)
for ann in fig_ml.layout.annotations:
    ann.font.color = MUTED_CLR
    ann.font.size  = 11

fig_ml.update_xaxes(range=[60, 90], ticksuffix='%',
                    gridcolor=GRID_CLR,
                    tickfont=dict(color=MUTED_CLR, size=11), row=1, col=1)
fig_ml.update_yaxes(gridcolor=GRID_CLR,
                    tickfont=dict(color=MUTED_CLR, size=11), row=1, col=1)
fig_ml.update_xaxes(visible=False, row=1, col=2)
fig_ml.update_yaxes(visible=False, row=1, col=2)

# ── 7. Write single combined HTML ────────────────────────────
html_eda = fig_eda.to_html(full_html=False, include_plotlyjs='cdn')
html_ml  = fig_ml.to_html(full_html=False, include_plotlyjs=False)

best_acc = results[best_name]['accuracy']

combined_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Customer Churn Prediction Dashboard</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0f1117;font-family:Arial,sans-serif;color:#c9d1d9;padding:28px 32px}}
h1{{font-size:20px;font-weight:500;color:#e2e8f0;margin-bottom:4px}}
.sub{{font-size:13px;color:#6b7280;margin-bottom:24px}}
.kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:28px}}
.kpi{{background:#161820;border:0.5px solid #2a2d3a;border-radius:8px;padding:14px 16px}}
.kpi.accent{{border-color:#534AB7}}
.kpi .lbl{{font-size:11px;color:#6b7280;margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em}}
.kpi .val{{font-size:22px;font-weight:500;color:#e2e8f0}}
.kpi.accent .val{{color:#AFA9EC}}
.kpi .hint{{font-size:11px;color:#4b5563;margin-top:3px}}
.sec{{font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;
      margin-bottom:10px;margin-top:28px;border-left:2px solid #534AB7;padding-left:10px}}
.chart-block{{background:#161820;border:0.5px solid #2a2d3a;
              border-radius:12px;padding:6px;margin-bottom:16px}}
@media(max-width:600px){{.kpi-row{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body>
<h1>Customer churn prediction dashboard</h1>
<div class="sub">Telecom dataset &mdash; 7,043 customers &mdash; 5 ML classifiers</div>

<div class="kpi-row">
  <div class="kpi">
    <div class="lbl">Total customers</div>
    <div class="val">7,043</div>
    <div class="hint">full dataset</div>
  </div>
  <div class="kpi">
    <div class="lbl">Churned</div>
    <div class="val">1,869</div>
    <div class="hint">26.5% of total</div>
  </div>
  <div class="kpi">
    <div class="lbl">Retained</div>
    <div class="val">5,174</div>
    <div class="hint">73.5% of total</div>
  </div>
  <div class="kpi accent">
    <div class="lbl">Best model accuracy</div>
    <div class="val">{best_acc}%</div>
    <div class="hint">{best_name}</div>
  </div>
</div>

<div class="sec">Exploratory data analysis</div>
<div class="chart-block">{html_eda}</div>

<div class="sec">Model evaluation</div>
<div class="chart-block">{html_ml}</div>

</body>
</html>"""

output_file = 'churn_dashboard.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(combined_html)

print(f"\nDone! Dashboard saved to '{output_file}'")
print("Opening in browser...")

import webbrowser, os
webbrowser.open('file://' + os.path.abspath(output_file))