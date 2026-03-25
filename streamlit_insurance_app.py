"""
streamlit_insurance_app.py

Single-file Streamlit app for your insurance charges model.
Edit only the lines/blocks that have a comment starting with "#change" immediately above them.

How to run:
1. pip install streamlit scikit-learn pandas joblib matplotlib seaborn shap
2. streamlit run streamlit_insurance_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings that might clutter the Streamlit interface (e.g., from SHAP or Matplotlib)
warnings.filterwarnings('ignore') 

st.set_page_config(page_title="Health Insurance Cost Predictor", layout="wide")

# -------------------------
# Top-level instructions
# -------------------------
st.title("Health Insurance Cost Predictor")
st.markdown("Estimate annual charges from user inputs or an uploaded dataset. Edit lines marked #change to match your model/data.")

# -------------------------
# Sidebar: Data & Model
# -------------------------
st.sidebar.header('Data & Model')

#change: If you want the app to look for a different CSV filename in the app folder, change it here.
#change
LOCAL_DATA_FILENAME = 'insurance.csv'

#change: If you want the app to look for different model filenames by default, change them here.
#change
DEFAULT_MODEL_FILENAMES = ('model.joblib', 'model.pkl')

use_local = st.sidebar.checkbox(f'Load local {LOCAL_DATA_FILENAME} if present', value=True)
uploaded_csv = st.sidebar.file_uploader('Or upload a CSV dataset (columns: age,sex,bmi,children,smoker,region,charges)', type=['csv'])

@st.cache_data
def load_dataset():
    if uploaded_csv is not None:
        return pd.read_csv(uploaded_csv)
    if use_local and os.path.exists(LOCAL_DATA_FILENAME):
        return pd.read_csv(LOCAL_DATA_FILENAME)
    return None

df = load_dataset()

if df is None:
    st.info('No dataset loaded. You can upload a CSV in the sidebar or place ' + LOCAL_DATA_FILENAME + ' in the app folder.')
else:
    st.success(f'Data loaded — {df.shape[0]} rows, {df.shape[1]} columns')
    if 'charges' in df.columns:
        if st.checkbox('Show raw data (first 50 rows)'):
            st.dataframe(df.head(50))

# -------------------------
# Model uploading / loading
# -------------------------
st.sidebar.markdown('---')
uploaded_model = st.sidebar.file_uploader('Upload model (joblib / pickle)', type=['joblib','pkl'])

@st.cache_resource
def load_model_from_file(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.sidebar.error(f'Failed to load model file {path}: {e}')
        return None

def load_model():
    # priority: uploaded model > local files > dummy
    if uploaded_model is not None:
        with open('uploaded_model.joblib','wb') as f:
            f.write(uploaded_model.getbuffer())
        m = load_model_from_file('uploaded_model.joblib')
        if m is not None:
            return m
    #change: If you saved your model under other names, add them to DEFAULT_MODEL_FILENAMES above.
    for fname in DEFAULT_MODEL_FILENAMES:
        if os.path.exists(fname):
            m = load_model_from_file(fname)
            if m is not None:
                return m
    # fallback dummy model (for demo/testing only)
    class Dummy:
        def predict(self, X):
            base = 2000
            cost = base + X['age']*30 + X['bmi']*40 + X['children']*200
            cost += np.where(X.get('smoker_yes', (X['smoker_no'] if 'smoker_no' in X else 0))==0, 0, 12000)
            return cost
        def __repr__(self):
            return 'DummyModel()'
    st.sidebar.warning('No model found — using a built-in dummy model for demo purposes. Place a trained model file in the app folder or upload one here.')
    return Dummy()

model = load_model()
st.sidebar.write('Loaded model:')
st.sidebar.write(getattr(model, '__class__', str(model))) # Corrected getattr access

# -------------------------
# Single prediction UI
# -------------------------
st.sidebar.markdown('---')
st.sidebar.header('Make a single prediction')

#change: Adjust the valid ranges if your model was trained on a different domain.
#change
age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=30)
sex = st.sidebar.selectbox('Sex', ['male','female'])
bmi = st.sidebar.slider('BMI', 10.0, 60.0, 24.0, step=0.1)
children = st.sidebar.number_input('Children', min_value=0, max_value=10, value=0)
smoker = st.sidebar.selectbox('Smoker', ['no','yes'])
region = st.sidebar.selectbox('Region', ['northeast','northwest','southeast','southwest'])

# -------------------------
# Feature preparation
# -------------------------
#change: This function MUST produce the exact features (names, order, transforms) your model expects.
#change: If you stored and loaded a pipeline (ColumnTransformer + estimator), you can keep this function minimal:
#change    return pd.DataFrame({'age':[age], 'sex':[sex], 'bmi':[bmi], 'children':[children], 'smoker':[smoker], 'region':[region]})
#change Otherwise, transform here to match the training-time features (one-hot, scaling names etc.).
#change
def prepare_features(age, sex, bmi, children, smoker, region):
    """
    Default simple encoder consistent with many insurance examples.
    Edit this block if your model expects different features, column names, or scaling.
    """
    # Ensure inputs are correctly handled as arrays/lists for consistent DataFrame creation
    if not isinstance(age, (np.ndarray, list)):
        age = [age]
        sex = [sex]
        bmi = [bmi]
        children = [children]
        smoker = [smoker]
        region = [region]
        
    df_x = pd.DataFrame({
        'age':age,
        'sex':sex,
        'bmi':bmi,
        'children':children,
        'smoker':smoker,
        'region':region
    })
    # Basic manual encodings (keep these if your model was not a pipeline).
    df_x['sex_male'] = (df_x['sex']=='male').astype(int)
    df_x['sex_female'] = (df_x['sex']=='female').astype(int)
    df_x['smoker_yes'] = (df_x['smoker']=='yes').astype(int)
    df_x['smoker_no'] = (df_x['smoker']=='no').astype(int)
    regions = ['northeast','northwest','southeast','southwest']
    for r in regions:
        df_x[f'region_{r}'] = (df_x['region']==r).astype(int)
    #change: If your model expects scaled numeric features, add the same scaler transformations here
    #change e.g. df_x['bmi'] = (df_x['bmi'] - BMI_MEAN)/BMI_STD
    keep_cols = ['age','bmi','children','sex_male','sex_female','smoker_yes','smoker_no'] + [f'region_{r}' for r in regions]
    return df_x[keep_cols]

# -------------------------
# Predict button
# -------------------------
if st.sidebar.button('Predict'):
    X = prepare_features(age, sex, bmi, children, smoker, region)
    try:
        #change: Some models require numpy arrays; if so, call model.predict(X.values) instead.
        pred = model.predict(X)[0]
        if np.isnan(pred) or np.isinf(pred):
            raise ValueError('Model returned invalid numeric value')
        st.sidebar.success(f'Estimated annual charges: ₹{pred:,.0f}')
    except Exception as e:
        st.sidebar.error(f'Prediction failed: {e}')

# -------------------------
# Main layout: visuals & explanations
# -------------------------
st.markdown('---')
col1, col2 = st.columns([1,1])

with col1:
    st.header('Data overview & visuals')
    if df is not None:
        # Basic distributions
        fig, axes = plt.subplots(2,2, figsize=(10,6))
        sns.histplot(df['age'], ax=axes[0,0], kde=True)
        axes[0,0].set_title('Age distribution')
        sns.histplot(df['bmi'], ax=axes[0,1], kde=True)
        axes[0,1].set_title('BMI distribution')
        sns.countplot(x='smoker', data=df, ax=axes[1,0])
        axes[1,0].set_title('Smoker count')
        sns.countplot(x='region', data=df, ax=axes[1,1])
        axes[1,1].set_title('Region counts')
        plt.tight_layout()
        st.pyplot(fig)

        # Scatter charges vs bmi colored by smoker
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df, ax=ax2, alpha=0.7)
        ax2.set_title('Charges vs BMI (colored by smoker)')
        st.pyplot(fig2)

        # Correlation heatmap on numeric columns
        numcols = ['age','bmi','children','charges']
        fig3, ax3 = plt.subplots(figsize=(5,4))
        sns.heatmap(df[numcols].corr(), annot=True, fmt='.2f', ax=ax3)
        ax3.set_title('Numeric correlation')
        st.pyplot(fig3)
    else:
        st.info('Load the dataset to see visuals here.')

with col2:
    st.header('Model explanation & feature importance')
    try:
        feat_names = prepare_features(30,'male',24.0,0,'no','northeast').columns.tolist()
        importance = None
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            st.write('Using feature_importances_ from model')
        elif hasattr(model, 'coef_'):
            coef = np.array(model.coef_).flatten()
            importance = np.abs(coef)
            st.write('Using absolute coefficients from linear model')
        if importance is not None:
            imp_df = pd.DataFrame({'feature':feat_names, 'importance':importance})
            imp_df = imp_df.sort_values('importance', ascending=False).head(10)
            fig4, ax4 = plt.subplots(figsize=(6,4))
            sns.barplot(x='importance', y='feature', data=imp_df, ax=ax4)
            ax4.set_title('Top feature importances')
            st.pyplot(fig4)
        else:
            st.info('Model does not expose feature_importances_ or coef_. You can use SHAP below for deeper explanations.')
    except Exception as e:
        st.error(f'Failed to compute simple feature importances: {e}')

    # SHAP
    #change: Toggle SHAP usage; set default to False to avoid slow starts. If you don't have shap installed, set to False.
    #change
    show_shap = st.checkbox('Run SHAP explanations (may be slow)', value=False)
    if show_shap:
        try:
            import shap
            #change: Background sample size for SHAP. Reduce for speed (e.g. 50).
            #change
            SHAP_BACKGROUND_SIZE = 50

            if df is not None:
                background = df.sample(min(SHAP_BACKGROUND_SIZE, len(df)), random_state=42)
                X_for_shap = prepare_features(background['age'].values, background['sex'].values,
                                                background['bmi'].values, background['children'].values,
                                                background['smoker'].values, background['region'].values)
            else:
                background = pd.DataFrame({'age':[30,40,50],'sex':['male']*3,'bmi':[22,28,33],'children':[0,1,2],'smoker':['no']*3,'region':['northeast']*3})
                X_for_shap = prepare_features(background['age'].values, background['sex'].values,
                                                background['bmi'].values, background['children'].values,
                                                background['smoker'].values, background['region'].values)

            # Build explainer
            try:
                # FIX APPLIED HERE: Replaced invalid model.class access with model.__class__.__name__
                if hasattr(shap, 'TreeExplainer') and (hasattr(model, 'feature_importances_') or model.__class__.__name__.lower().find('xg')!=-1 or model.__class__.__name__.lower().find('lgb')!=-1):
                    explainer = shap.TreeExplainer(model)
                else:
                    # KernelExplainer will be slow on large background sets
                    explainer = shap.KernelExplainer(model.predict, X_for_shap)
            except Exception as e:
                st.error(f'Could not initialize SHAP explainer: {e}')
                explainer = None

            if explainer is not None:
                if df is not None:
                    sample = df.sample(min(50, len(df)), random_state=1)
                    X_sample = prepare_features(sample['age'].values, sample['sex'].values, sample['bmi'].values, sample['children'].values, sample['smoker'].values, sample['region'].values)
                else:
                    X_sample = prepare_features(np.array([age]), np.array([sex]), np.array([bmi]), np.array([children]), np.array([smoker]), np.array([region]))

                shap_values = explainer.shap_values(X_sample)

                st.subheader('SHAP summary plot')
                try:
                    # Use st.pyplot context manager to handle Matplotlib warnings/issues better
                    with st.expander("SHAP Explanation Plot"):
                        fig_shap = plt.figure(figsize=(8,6))
                        # For single output models, shap_values is a 2D array, not a list of 2D arrays
                        if isinstance(shap_values, list) and len(shap_values) == 1:
                            shap_values = shap_values[0]
                        shap.summary_plot(shap_values, X_sample, feature_names=X_sample.columns, show=False)
                        st.pyplot(fig_shap)
                except Exception as e:
                    st.warning(f'Could not render full SHAP plot ({e}). Falling back to mean absolute SHAP bar plot.')
                    # Fallback bar plot
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    shap_df = pd.DataFrame({'feature': X_sample.columns, 'mean_abs_shap': mean_abs_shap}).sort_values('mean_abs_shap', ascending=False)
                    fig_bar, ax_bar = plt.subplots(figsize=(6,4))
                    sns.barplot(x='mean_abs_shap', y='feature', data=shap_df, ax=ax_bar)
                    ax_bar.set_title('Mean |SHAP value|')
                    st.pyplot(fig_bar)

                st.success('SHAP analysis completed')
        except Exception as e:
            st.error(f'Could not import shap or run explanation: {e}\nInstall shap via pip install shap')

# -------------------------
# Save sample predictions
# -------------------------
st.markdown('---')
if df is not None and st.button('Save sample predictions (first 200 rows)'):
    try:
        X_all = prepare_features(df['age'].values, df['sex'].values, df['bmi'].values, df['children'].values, df['smoker'].values, df['region'].values)
        #change: If your model requires numpy arrays for predict, call model.predict(X_all.values)
        preds = model.predict(X_all)
        out = df.copy().reset_index(drop=True)
        out['predicted_charges'] = preds
        out_csv = out.head(200).to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(out_csv).decode()
        href = f"data:file/csv;base64,{b64}"
        st.markdown(f"[Download sample predictions CSV]({href})")
    except Exception as e:
        st.error(f'Failed to compute predictions on full dataset: {e}')

st.markdown('\n---\n')
st.markdown('Note: This tool gives estimates & explanations for educational/demo purposes. For production deployments consider a proper API, authentication, and input validation.')

# -------------------------
# Optional helpful snippets shown to the user
# -------------------------
with st.expander('Developer notes & quick snippets'):
    st.markdown("*How to save a pipeline model (recommended)*")
    st.code("""from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import joblib

num_cols = ['age','bmi','children']
cat_cols = ['sex','smoker','region']
pre = ColumnTransformer([
  ('num', StandardScaler(), num_cols),
  ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
pipe = make_pipeline(pre, RandomForestRegressor(n_estimators=100, random_state=42))
# pipe.fit(X_train, y_train)
joblib.dump(pipe, 'model.joblib')""", language='python')

    st.markdown("*If your saved model expects a specific feature order*")
    st.code("""# Example: restore feature list together with model
joblib.dump({'model': pipe, 'feature_names': feature_names}, 'model_with_meta.joblib')
# Then in load_model(): loaded = joblib.load('model_with_meta.joblib'); model = loaded['model']; feature_names = loaded['feature_names']""", language='python')

    st.markdown("*Requirements*")
    st.code("streamlit\npandas\nscikit-learn\njoblib\nmatplotlib\nseaborn\nshap\n", language='text')
    