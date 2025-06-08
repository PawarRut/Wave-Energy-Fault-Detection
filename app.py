# -------------------------------
# app.py – Wave Energy Fault Detection
# -------------------------------

# Step 1 – Import libraries
import base64
from io import BytesIO
from PIL import Image
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# Helper – encode image to base64
# -------------------------------
def get_base64_img(img_path: str) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# -------------------------------
# Step 2 – Page config + background
# -------------------------------
st.set_page_config(page_title="Wave Energy Fault Detection",
                   layout="centered", page_icon="🌊")

# Inject background image (fills entire page)
bg_img = get_base64_img("background.png")        # ← make sure the file is beside app.py
page_bg_css = f"""
<style>
.stApp {{
    background: url('data:image/png;base64,{bg_img}') no-repeat center center fixed;
    background-size: cover;
}}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

# -------------------------------
# Step 3 – Heading & description
# -------------------------------
st.markdown("## 🌊 Wave Energy Fault Detection")
st.write(
    "Predict whether a fault will occur in a wave-energy converter based on real-time sensor data."
)

# -------------------------------
# Step 4 – Load dataset
# -------------------------------
df = pd.read_csv("wave_energy_converter_dataset.csv")

# Split features & target
X = df.drop("fault_detected", axis=1)
y = df["fault_detected"]

# -------------------------------
# Step 5 – Sidebar user input
# -------------------------------
st.sidebar.header("Enter Sensor Readings")

def collect_user_input() -> pd.DataFrame:
    """Return a one-row DataFrame with sidebar inputs."""
    inputs = {}
    for col in X.columns:
        col_min, col_max = X[col].min(), X[col].max()
        if col == "sea_state":
            inputs[col] = st.sidebar.selectbox(col.replace("_", " ").title(),
                                               sorted(X[col].unique()))
        else:
            inputs[col] = st.sidebar.slider(col.replace("_", " ").title(),
                                            float(col_min), float(col_max),
                                            float(col_min))
    return pd.DataFrame([inputs])

user_df = collect_user_input()

# -------------------------------
# Step 6 – Show user data
# -------------------------------
st.subheader("Your Input")
st.write(user_df)

# -------------------------------
# Step 7 – Train Random Forest
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
rf_clf = RandomForestClassifier(n_estimators=300, random_state=42)
rf_clf.fit(X_train, y_train)

# -------------------------------
# Step 8 – Prediction
# -------------------------------
pred = rf_clf.predict(user_df)[0]
pred_proba = rf_clf.predict_proba(user_df)[0][pred]

st.subheader("Prediction")
st.success("✅ No Fault Detected" if pred == 0 else "⚠️ Fault Detected")
st.caption(f"Confidence : {pred_proba:.1%}")

# -------------------------------
# Step 9 – Model accuracy
# -------------------------------
acc = accuracy_score(y_test, rf_clf.predict(X_test))
st.subheader("Model Accuracy on Test Set")
st.write(f"**{acc:.2%}**")

# -------------------------------
# Step 10 – Preview dataset
# -------------------------------
with st.expander("Preview Dataset"):
    st.dataframe(df.head())
