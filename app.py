import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# ---------------- UI ---------------- #
st.set_page_config(page_title="ML Model Dashboard", layout="wide")

st.title("📊 ML Model Performance Dashboard")

st.markdown("Compare multiple ML models on a preloaded dataset")

# ---------------- LOAD DATA ---------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")   # 🔥 your dataset file name
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- SELECT TARGET ---------------- #
target = st.selectbox("Select Target Column", df.columns)

# ---------------- PREPROCESSING ---------------- #
X = df.drop(columns=[target])
y = df[target]

# Encode categorical
X = pd.get_dummies(X)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------- TRAIN BUTTON ---------------- #
if st.button("🚀 Run Models"):

    results = []

    # -------- KNN -------- #
    k_values = range(1, 21)
    mse_values = []

    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        mse_values.append(mean_squared_error(y_test, y_pred))

    best_k = k_values[np.argmin(mse_values)]

    knn = KNeighborsRegressor(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    mse_knn = mean_squared_error(y_test, y_pred_knn)
    r2_knn = r2_score(y_test, y_pred_knn)

    results.append(["KNN", mse_knn, r2_knn])

    # -------- Linear Regression -------- #
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    results.append(["Linear Regression", mse_lr, r2_lr])

    # -------- Decision Tree -------- #
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    mse_dt = mean_squared_error(y_test, y_pred_dt)
    r2_dt = r2_score(y_test, y_pred_dt)

    results.append(["Decision Tree", mse_dt, r2_dt])

    # ---------------- DISPLAY ---------------- #
    st.subheader("📈 Model Comparison")

    results_df = pd.DataFrame(results, columns=["Model", "MSE", "R2 Score"])
    st.dataframe(results_df)

    # ---------------- BEST MODEL ---------------- #
    best_model = results_df.loc[results_df["R2 Score"].idxmax()]

    st.success(f"🏆 Best Model: {best_model['Model']}")

    # ---------------- K GRAPH ---------------- #
    st.subheader("📉 KNN Tuning (K vs MSE)")
    chart_data = pd.DataFrame({
        "K": list(k_values),
        "MSE": mse_values
    })
    st.line_chart(chart_data.set_index("K"))
