"""
Streamlit Reddit Popularity Predictor (Multi-model)
- Upload train/val/test (optional). If not provided, uses a small synthetic sample.
- Train Linear Regression, Random Forest, Gradient Boosting (selectable).
- Shows validation MSE/MAE, Actual vs Predicted for each model.
- Single combined tree-based feature importance (top 10) shown once.
- Manual predictor: choose a model or predict with all trained models.
- Finish / Reset button clears session and returns to initial state.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# -------------------------
# Helper utilities
# -------------------------
def preprocess_basic(df):
    """Compute body_length and post_hour and ensure numeric columns exist."""
    df = df.copy()
    # body_length
    if 'body' in df.columns:
        df['body_length'] = df['body'].astype(str).apply(len)
    else:
        df['body_length'] = 0

    # created_utc -> post_hour
    if 'created_utc' in df.columns:
        try:
            # If created_utc is epoch seconds
            df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
        except Exception:
            # fallback - let pandas infer
            df['created_utc'] = pd.to_datetime(df['created_utc'], errors='coerce')
        df['post_hour'] = df['created_utc'].dt.hour.fillna(0).astype(int)
    else:
        if 'post_hour' not in df.columns:
            df['post_hour'] = 0

    # ensure ups/downs exist
    if 'ups' not in df.columns:
        df['ups'] = 0
    if 'downs' not in df.columns:
        df['downs'] = 0

    # Ensure a target column 'score' exists. If not, try common alternatives, otherwise synthesize.
    if 'score' not in df.columns:
        if 'popularity_score' in df.columns:
            df['score'] = df['popularity_score']
        elif 'popularity' in df.columns:
            df['score'] = df['popularity']
        else:
            # synthesize a reasonable proxy if no explicit target
            df['score'] = (df['ups'] - df['downs']) * 0.5

    # Force numeric and fill NaNs
    for col in ['ups','downs','body_length','post_hour','score']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df

def add_engineered_features(df):
    """Add a small set of engineered features used by models and for feature importance."""
    df = df.copy()
    # ensure columns exist to avoid KeyError
    for c in ['ups','downs','body_length','post_hour']:
        if c not in df.columns:
            df[c] = 0
    df['ups_downs_product'] = df['ups'] * df['downs']
    df['ups_squared'] = df['ups'] ** 2
    df['body_hour_interaction'] = df['body_length'] * df['post_hour']
    df['ups_log'] = np.log1p(df['ups'].clip(lower=0))
    df['downs_log'] = np.log1p(df['downs'].clip(lower=0))
    # safe ratio
    df['ups_to_downs'] = df['ups'] / (df['downs'] + 1)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df

def safe_split_df(df, train_frac=0.7, val_frac_of_remaining=0.5, seed=1):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    remaining = df.iloc[n_train:].reset_index(drop=True)
    n_val = int(len(remaining) * val_frac_of_remaining)
    val_df = remaining.iloc[:n_val].reset_index(drop=True)
    test_df = remaining.iloc[n_val:].reset_index(drop=True)
    return train_df, val_df, test_df

def create_sample_df(n=1000, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        'ups': rng.poisson(50, n),
        'downs': rng.poisson(5, n),
        'body': ['Example text'] * n,
        # store as timestamps; preprocess_basic will handle conversion
        'created_utc': pd.Timestamp('2024-01-01') + pd.to_timedelta(rng.randint(0, 86400, n), unit='s')
    })
    df['score'] = (df['ups'] - df['downs']) * 0.5 + rng.normal(0, 10, n)
    return safe_split_df(df)

# -------------------------
# Page header + sidebar
# -------------------------
st.set_page_config(layout="wide", page_title="Reddit Popularity - MultiModel")
st.title("Reddit Popularity Predictor (Multiple Models)")
st.write("Upload train/val/test CSVs (optional). Click *Train & Evaluate* to run selected models.")

with st.sidebar:
    st.header("Data Upload")
    train_file = st.file_uploader("train.csv (optional)", type=['csv'])
    val_file = st.file_uploader("val.csv (optional)", type=['csv'])
    test_file = st.file_uploader("test.csv (optional)", type=['csv'])
    st.markdown("---")
    st.header("Models to train")
    model_choices = st.multiselect("Models", options=['Linear Regression','Random Forest','Gradient Boosting'],
                                   default=['Linear Regression','Random Forest','Gradient Boosting'])
    st.markdown("---")
    train_btn = st.button("Train & Evaluate")
    finish_btn = st.button("Finish / Reset app")

# -------------------------
# Finish / reset behavior
# -------------------------
if finish_btn:
    # Clear selected session keys we use, then rerun app
    keys = list(st.session_state.keys())
    for k in keys:
        del st.session_state[k]
    # Re-run to show clean UI (support different Streamlit versions)
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            # if rerun not available, just pass (app state cleared)
            pass

# -------------------------
# Load data (uploaded or sample)
# -------------------------
train_df = pd.read_csv(train_file, engine='python', on_bad_lines='skip') if train_file else None
val_df = pd.read_csv(val_file, engine='python', on_bad_lines='skip') if val_file else None
test_df = pd.read_csv(test_file, engine='python', on_bad_lines='skip') if test_file else None

# If no uploaded train, use sample and split
if train_df is None:
    train_df, val_df, test_df = create_sample_df()
# If train uploaded but val/test missing, create splits from train
elif val_df is None and test_df is None:
    train_df, val_df, test_df = safe_split_df(train_df)
# If train+val uploaded but test missing, combine and split
elif val_df is not None and test_df is None:
    combined = pd.concat([train_df, val_df], ignore_index=True)
    train_df, val_df, test_df = safe_split_df(combined)

# Preprocess
train_df = preprocess_basic(train_df)
val_df = preprocess_basic(val_df)
test_df = preprocess_basic(test_df)

st.write(f"Rows — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

# -------------------------
# Prepare features & scaling
# -------------------------
base_features = ['ups','downs','body_length','post_hour']

X_train_base = train_df[base_features].copy()
y_train = train_df['score'].copy()
X_val_base = val_df[base_features].copy()
y_val = val_df['score'].copy()
X_test_base = test_df[base_features].copy()
y_test = test_df['score'].copy()

# Engineered features
X_train_enh = add_engineered_features(X_train_base)
X_val_enh = add_engineered_features(X_val_base)
X_test_enh = add_engineered_features(X_test_base)

# Scale
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_enh)
X_val_scaled = scaler.transform(X_val_enh)
X_test_scaled = scaler.transform(X_test_enh)

# -------------------------
# Ensure session_state storage exists
# -------------------------
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'last_predictions' not in st.session_state:
    st.session_state.last_predictions = {}

# -------------------------
# Training
# -------------------------
trained_models_local = {}
results = []
collected_importances = {}

if train_btn:
    if not model_choices:
        st.error("Please select at least one model in the sidebar before training.")
    else:
        with st.spinner("Training selected models..."):
            # Linear Regression
            if 'Linear Regression' in model_choices:
                lr = LinearRegression()
                lr.fit(X_train_scaled, y_train)
                preds_val = lr.predict(X_val_scaled)
                results.append({'Model':'Linear Regression', 'MSE': mean_squared_error(y_val, preds_val), 'MAE': mean_absolute_error(y_val, preds_val)})
                trained_models_local['Linear Regression'] = lr

            # Random Forest
            if 'Random Forest' in model_choices:
                rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
                rf.fit(X_train_scaled, y_train)
                preds_val = rf.predict(X_val_scaled)
                results.append({'Model':'Random Forest', 'MSE': mean_squared_error(y_val, preds_val), 'MAE': mean_absolute_error(y_val, preds_val)})
                trained_models_local['Random Forest'] = rf
                if hasattr(rf, "feature_importances_"):
                    collected_importances['Random Forest'] = rf.feature_importances_

            # Gradient Boosting
            if 'Gradient Boosting' in model_choices:
                gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
                gb.fit(X_train_scaled, y_train)
                preds_val = gb.predict(X_val_scaled)
                results.append({'Model':'Gradient Boosting', 'MSE': mean_squared_error(y_val, preds_val), 'MAE': mean_absolute_error(y_val, preds_val)})
                trained_models_local['Gradient Boosting'] = gb
                if hasattr(gb, "feature_importances_"):
                    collected_importances['Gradient Boosting'] = gb.feature_importances_

        # persist models and scaler in session_state so they survive reruns
        st.session_state.trained_models = trained_models_local
        st.session_state.scaler = scaler

        # results table (validation)
        results_df = pd.DataFrame(results).sort_values('MSE').reset_index(drop=True)
        st.subheader("Model comparison (on validation set)")
        st.dataframe(results_df)

        # -------------------------
        # Show Actual vs Predicted for each trained model (Test set)
        # -------------------------
        st.subheader("Actual vs Predicted (Test set) — one plot per model")
        for name, model_obj in trained_models_local.items():
            preds_test = model_obj.predict(X_test_scaled)
            mse_test = mean_squared_error(y_test, preds_test)
            mae_test = mean_absolute_error(y_test, preds_test)
            st.write(f"{name}** — Test MSE: {mse_test:.4f}, MAE: {mae_test:.4f}")

            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(y_test, preds_test, alpha=0.35)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=1)
            ax.set_xlabel("Actual Score")
            ax.set_ylabel("Predicted Score")
            ax.set_title(f"{name} - Actual vs Predicted")
            st.pyplot(fig)
            
        # WordCloud - only run if wordcloud installed and train_df has bodies
        try:
            from wordcloud import WordCloud
            if ('body' in train_df.columns) and train_df['body'].notna().any():
                high_score_posts = train_df.dropna(subset=['score', 'body'])
                high_score_posts = high_score_posts[high_score_posts['score'] > high_score_posts['score'].median()]
                high_score_text = " ".join(high_score_posts['body'].astype(str).tolist())
                if len(high_score_text.strip()) > 0:
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(high_score_text)
                    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                    ax_wc.imshow(wordcloud, interpolation='bilinear')
                    ax_wc.axis("off")
                    ax_wc.set_title("Word Cloud: Common Words in High-Scoring Reddit Posts", fontsize=14)
                    st.pyplot(fig_wc)
        except ModuleNotFoundError:
            st.warning("WordCloud not installed. Run pip install wordcloud to see this visualization.")
        except Exception:
            # don't break the app for any unexpected errors in visualization
            pass

        # -------------------------
        # Single combined feature importance (once)
        # -------------------------
        if collected_importances:
            feature_names = X_train_enh.columns.tolist()
            mats = []
            for nm, imp in collected_importances.items():
                arr = np.array(imp, dtype=float)
                if arr.shape[0] == len(feature_names):
                    mats.append(arr)
            if mats:
                avg_imp = np.mean(np.vstack(mats), axis=0)
                imp_series = pd.Series(avg_imp, index=feature_names).sort_values(ascending=False)
                st.subheader("Top 10 Features (combined importance from tree models)")
                st.table(imp_series.head(10))

        # Save best model (lowest val MSE) to session_state and disk
        if results:
            best_name = results_df.iloc[0]['Model']
            st.session_state.best_model_name = best_name
            st.session_state.trained_models = trained_models_local
            # Save best model to disk
            try:
                joblib.dump({'model': trained_models_local[best_name], 'scaler': scaler, 'features': X_train_enh.columns.tolist()}, "best_model_multi.joblib")
                st.write(f"Saved best model ({best_name}) to best_model_multi.joblib")
            except Exception:
                st.write("Warning: could not save model to disk (permission or environment issue).")

        # reset last_predictions (clear previous manual predictions)
        st.session_state.last_predictions = {}

# -------------------------
# Manual Predictor
# -------------------------
st.markdown("---")
st.header("Manual Predictor")

c1, c2, c3, c4 = st.columns(4)
ups_val = c1.number_input("ups", min_value=0, value=10, key="ui_ups")
downs_val = c2.number_input("downs", min_value=0, value=1, key="ui_downs")
body_len_val = c3.number_input("body_length", min_value=0, value=200, key="ui_body")
post_hour_val = c4.slider("post_hour", 0, 23, 12, key="ui_hour")

available_models = list(st.session_state.get('trained_models', {}).keys())
selected_model = st.selectbox("Choose model to predict with (or select none to use 'Predict all')",
                              options=['-- none --'] + available_models)

col_a, col_b = st.columns(2)
with col_a:
    btn_single = st.button("Predict with selected model")
with col_b:
    btn_all = st.button("Predict with all trained models")

def _build_input_df(u,d,b,h):
    return pd.DataFrame([{'ups':u,'downs':d,'body_length':b,'post_hour':h}])

if btn_single:
    if selected_model is None or selected_model == '-- none --':
        st.warning("Select a trained model from the dropdown (or use 'Predict with all').")
    else:
        models_map = st.session_state.get('trained_models', {})
        scaler_used = st.session_state.get('scaler', None)
        if models_map and scaler_used is not None and selected_model in models_map:
            model_obj = models_map[selected_model]
            inp = _build_input_df(ups_val, downs_val, body_len_val, post_hour_val)
            inp_enh = add_engineered_features(inp)
            inp_scaled = scaler_used.transform(inp_enh)
            try:
                p = float(model_obj.predict(inp_scaled)[0])
            except Exception:
                p = None
            st.session_state.last_predictions = {selected_model: p}
            if p is not None:
                st.success(f"Predicted Popularity (score) using {selected_model}: {p:.2f}")
            else:
                st.error("Prediction failed for selected model.")
        else:
            st.warning("No trained models/scaler available. Train models first.")

if btn_all:
    models_map = st.session_state.get('trained_models', {})
    scaler_used = st.session_state.get('scaler', None)
    if not models_map or scaler_used is None:
        st.warning("No trained models/scaler available. Train models first.")
    else:
        inp = _build_input_df(ups_val, downs_val, body_len_val, post_hour_val)
        inp_enh = add_engineered_features(inp)
        inp_scaled = scaler_used.transform(inp_enh)
        preds = {}
        for nm, mo in models_map.items():
            try:
                preds[nm] = float(mo.predict(inp_scaled)[0])
            except Exception:
                preds[nm] = None
        st.session_state.last_predictions = preds
        # show as table
        df_preds = pd.DataFrame.from_dict(preds, orient='index', columns=['predicted_score']).reset_index().rename(columns={'index':'model'})
        st.table(df_preds)

# show last predictions if any
if st.session_state.get('last_predictions'):
    last = st.session_state.last_predictions
    if isinstance(last, dict):
        if len(last) == 1:
            nm, val = next(iter(last.items()))
            if val is not None:
                st.info(f"Last prediction — {nm}: {val:.2f}")
        else:
            dfp = pd.DataFrame.from_dict(last, orient='index', columns=['predicted_score']).reset_index().rename(columns={'index':'model'})
            st.info("Last predictions:")
            st.table(dfp)

st.markdown("---")
st.write("Use *Finish / Reset app* in the sidebar to clear models, results and return to the initial state.")