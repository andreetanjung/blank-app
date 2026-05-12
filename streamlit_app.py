import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
import json
import io
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from optbinning import OptimalBinning

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PD Credit Scoring App",
    page_icon="💳",
    layout="wide",
)

st.title("💳 PD Credit Scoring — CIMB")
st.caption("Probability of Default model with Gen AI analyst powered by Claude")

# ─── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
anthropic_key = st.sidebar.text_input("Anthropic API Key", type="password",
                                       help="Required for the AI Analyst feature")
st.sidebar.markdown("---")
st.sidebar.info(
    "**Workflow**\n"
    "1. Upload your training CSV\n"
    "2. Train the model\n"
    "3. Upload evaluation CSV\n"
    "4. Score & download results\n"
    "5. Ask the AI Analyst anything"
)

# ─── Helpers ────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "purpose", "verification_status", "term", "mths_since_issue_d",
    "int_rate", "inq_last_6mths", "annual_inc", "estimated_monthly_debt",
]
WOE_COLS_LR = [f"{c}_woe" for c in FEATURE_COLS]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["credit_hist_tenure"] = df["mths_since_earliest_cr_line"] - df["mths_since_issue_d"]
    df["estimated_monthly_debt"] = (df["annual_inc"] / 12) * df["dti"]
    df["estimated_monthly_income"] = df["annual_inc"] / 12
    df["r_monthly_debt_to_income"] = df["estimated_monthly_debt"] / df["estimated_monthly_income"]
    df["monthly_int_rate"] = df["int_rate"] / 12
    df["r_monthly_intrate_to_income"] = df["monthly_int_rate"] / df["estimated_monthly_income"]
    return df


def calc_gini(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return abs(2 * auc - 1)


def assign_grade(df: pd.DataFrame, pred_col="pred_lr") -> pd.DataFrame:
    bins = [0, 0.1604, 0.4531, 0.7211, 0.8879, 0.9642, 0.9931, 1.0]
    labels = ["A", "B", "C", "D", "E", "F", "G"]
    df["temp_pct"] = df[pred_col].rank(pct=True, method="first")
    df["grade_new"] = pd.cut(df["temp_pct"], bins=bins, labels=labels, include_lowest=True)
    df.drop(columns=["temp_pct"], inplace=True)
    return df


def call_claude(messages: list, system: str, api_key: str) -> str:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": system,
            "messages": messages,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


# ─── Session state ───────────────────────────────────────────────────────────
for key in ["model", "woe_transformers", "trained_data", "eval_scored", "chat_history"]:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state["chat_history"] is None:
    st.session_state["chat_history"] = []

# ═════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["📊 Train Model", "🎯 Score & Evaluate", "📈 Analytics", "🤖 AI Analyst"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — TRAIN
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Step 1 — Train the PD Model")
    train_file = st.file_uploader("Upload training CSV (`pd_loan_data_train.csv`)", type="csv", key="train_upload")

    if train_file:
        with st.spinner("Loading & engineering features…"):
            data = pd.read_csv(train_file)
            data["good_bad"] = np.where(data["good_bad"] == 0, 1, 0)
            data = engineer_features(data)

        st.success(f"Loaded {len(data):,} rows · {data.shape[1]} columns")
        st.dataframe(data.head(5), use_container_width=True)

        # quick EDA
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Default Rate", f"{data['good_bad'].mean():.2%}")
        col_b.metric("Total Records", f"{len(data):,}")
        col_c.metric("Missing Values", f"{data.isnull().sum().sum():,}")

        if st.button("🚀 Train Logistic Regression Model", type="primary"):
            with st.spinner("Fitting WoE transformers & Logistic Regression…"):
                predictors = [c for c in data.columns if c not in ("loan_id", "grade", "good_bad", "data_type")]
                woe_transformers = {}
                for col in predictors:
                    dtype = "categorical" if data[col].dtype == "object" else "numerical"
                    ob = OptimalBinning(name=col, dtype=dtype, solver="cp", max_n_bins=4)
                    try:
                        ob.fit(data[col], data["good_bad"])
                        data[f"{col}_woe"] = ob.transform(data[col], metric="woe")
                        woe_transformers[col] = ob
                    except Exception:
                        pass

                # filter to the 8 selected WoE columns that are present
                available_woe = [c for c in WOE_COLS_LR if c in data.columns]

                # Replace inf/-inf with NaN, then fill NaN with 0 (neutral WoE)
                X_train = data[available_woe].replace([np.inf, -np.inf], np.nan).fillna(0)
                y_train = data["good_bad"]

                # Drop any remaining all-NaN columns
                X_train = X_train.astype(float)

                lr = LogisticRegression(max_iter=1000, random_state=42)
                lr.fit(X_train, y_train)

                data["pred_lr"] = lr.predict_proba(X_train)[:, 1]
                gini = calc_gini(y_train, data["pred_lr"])

                st.session_state["model"] = lr
                st.session_state["woe_transformers"] = woe_transformers
                st.session_state["trained_data"] = data

            st.success(f"✅ Model trained! Train Gini = **{gini:.4f}**")

            # Coefficients table
            coef_df = pd.DataFrame({
                "Feature": available_woe,
                "Coefficient": lr.coef_[0],
            }).sort_values("Coefficient", key=abs, ascending=False)
            st.subheader("Model Coefficients")
            st.dataframe(coef_df, use_container_width=True)

            # Calibration plot
            st.subheader("Calibration Plot (Train)")
            prob_true, prob_pred = calibration_curve(y_train, data["pred_lr"], n_bins=10, strategy="quantile")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
            ax.plot(prob_pred, prob_true, marker="o", color="steelblue", label="LR")
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title("Calibration — Train")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SCORE & EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Step 2 — Score Evaluation Data")

    if st.session_state["model"] is None:
        st.warning("⚠️ Please train the model first (Tab 1).")
    else:
        eval_file = st.file_uploader("Upload evaluation CSV (`pd_loan_data_test.csv`)", type="csv", key="eval_upload")

        if eval_file:
            with st.spinner("Scoring…"):
                eval_df = pd.read_csv(eval_file)
                eval_df = engineer_features(eval_df)

                woe_transformers = st.session_state["woe_transformers"]
                for col, ob in woe_transformers.items():
                    try:
                        eval_df[f"{col}_woe"] = ob.transform(eval_df[col], metric="woe")
                    except Exception:
                        pass

                available_woe = [c for c in WOE_COLS_LR if c in eval_df.columns]
                X_eval = eval_df[available_woe].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
                eval_df["pred_lr"] = st.session_state["model"].predict_proba(X_eval)[:, 1]
                eval_df = assign_grade(eval_df)
                st.session_state["eval_scored"] = eval_df

            st.success(f"✅ Scored {len(eval_df):,} records")

            # Grade distribution
            grade_summary = eval_df.groupby("grade_new", observed=True).agg(
                Count=("loan_id", "count"),
                Avg_PD=("pred_lr", "mean"),
                Min_PD=("pred_lr", "min"),
                Max_PD=("pred_lr", "max"),
            ).reset_index()
            st.subheader("Grade Distribution")
            st.dataframe(grade_summary, use_container_width=True)

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            colors = ["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#e74c3c", "#c0392b", "#8e44ad"]
            grade_counts = eval_df["grade_new"].value_counts().sort_index()
            ax2.bar(grade_counts.index.astype(str), grade_counts.values, color=colors[:len(grade_counts)])
            ax2.set_xlabel("Grade")
            ax2.set_ylabel("Count")
            ax2.set_title("Portfolio Grade Distribution")
            st.pyplot(fig2)

            # Download
            buf = io.BytesIO()
            eval_df.to_excel(buf, index=False, engine="openpyxl")
            st.download_button(
                "⬇️ Download Scored Results (.xlsx)",
                data=buf.getvalue(),
                file_name="scored_evaluation_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("📈 Model Analytics")

    data = st.session_state.get("trained_data")
    eval_df = st.session_state.get("eval_scored")

    if data is None:
        st.warning("⚠️ Train the model first.")
    else:
        st.subheader("Univariate Gini (Train Features)")
        woe_transformers = st.session_state["woe_transformers"]
        available_woe = [c for c in WOE_COLS_LR if c in data.columns]
        gini_rows = []
        for col in available_woe:
            try:
                g = calc_gini(data["good_bad"], data[col])
                gini_rows.append({"Feature": col, "Gini": round(g, 4)})
            except Exception:
                pass
        gini_table = pd.DataFrame(gini_rows).sort_values("Gini", ascending=False)
        st.dataframe(gini_table, use_container_width=True)

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.barh(gini_table["Feature"], gini_table["Gini"], color="steelblue")
        ax3.set_xlabel("Gini")
        ax3.set_title("Univariate Gini per Feature")
        ax3.invert_yaxis()
        st.pyplot(fig3)

        if eval_df is not None:
            st.subheader("Score Distribution — Evaluation Set")
            fig4, ax4 = plt.subplots(figsize=(7, 4))
            ax4.hist(eval_df["pred_lr"], bins=30, color="steelblue", edgecolor="white")
            ax4.set_xlabel("Predicted PD")
            ax4.set_ylabel("Count")
            ax4.set_title("PD Score Distribution")
            st.pyplot(fig4)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — AI ANALYST (Gen AI feature)
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("🤖 AI Credit Analyst — Powered by Claude")
    st.caption("Ask anything about the model, results, or credit risk concepts.")

    if not anthropic_key:
        st.warning("🔑 Enter your Anthropic API key in the sidebar to enable this feature.")
    else:
        # Build context for Claude
        context_parts = ["You are an expert credit risk analyst. Answer concisely and clearly."]

        model = st.session_state.get("model")
        trained_data = st.session_state.get("trained_data")
        eval_scored = st.session_state.get("eval_scored")

        if model is not None and trained_data is not None:
            available_woe = [c for c in WOE_COLS_LR if c in trained_data.columns]
            coef_info = dict(zip(available_woe, model.coef_[0].tolist()))
            train_gini = calc_gini(trained_data["good_bad"], trained_data["pred_lr"])
            context_parts.append(
                f"\n\nModel: Logistic Regression with WoE features.\n"
                f"Features & coefficients: {json.dumps(coef_info, indent=2)}\n"
                f"Train Gini: {train_gini:.4f}\n"
                f"Train default rate: {trained_data['good_bad'].mean():.4f}\n"
                f"Train size: {len(trained_data):,} rows"
            )

        if eval_scored is not None:
            grade_dist = eval_scored["grade_new"].value_counts().sort_index().to_dict()
            avg_pd = eval_scored["pred_lr"].mean()
            context_parts.append(
                f"\n\nEvaluation set: {len(eval_scored):,} rows\n"
                f"Average predicted PD: {avg_pd:.4f}\n"
                f"Grade distribution: {json.dumps({str(k): v for k, v in grade_dist.items()})}"
            )

        system_prompt = " ".join(context_parts)

        # Chat UI
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Ask about the model, risk grades, Gini, calibration…")

        if user_input:
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        reply = call_claude(
                            messages=st.session_state["chat_history"],
                            system=system_prompt,
                            api_key=anthropic_key,
                        )
                        st.write(reply)
                        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.error(f"API error: {e}")

        if st.button("🗑️ Clear chat"):
            st.session_state["chat_history"] = []
            st.rerun()
