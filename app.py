import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Title
st.title("📊 ML Model Comparison App")

# Upload file
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Select target column
    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # Handle categorical (basic)
    X = pd.get_dummies(X)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Model selection
    model_name = st.selectbox(
        "Choose Model",
        ["KNN", "Linear Regression", "Decision Tree"]
    )

    # Train button
    if st.button("Train Model"):

        if model_name == "KNN":
            k = st.slider("Select K value", 1, 20, 5)
            model = KNeighborsRegressor(n_neighbors=k)

        elif model_name == "Linear Regression":
            model = LinearRegression()

        else:
            model = DecisionTreeRegressor()

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display
        st.success("Model Trained Successfully!")

        st.write("### 📈 Results")
        st.write(f"**MSE:** {mse:.2f}")
        st.write(f"**R² Score:** {r2:.4f}")

        # Interpretation
        if r2 < 0:
            st.error("Model is performing worse than baseline ⚠️")
        elif r2 < 0.5:
            st.warning("Model performance is moderate")
        else:
            st.success("Good model performance ✅")