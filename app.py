import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(page_title="Car ML Dashboard", layout="wide")

st.title("🚗 Advanced Car Price Dashboard")

# -------------------------
# LOAD DATA (with caching)
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("car_data.csv")

df = load_data()

# -------------------------
# SIDEBAR NAVIGATION
# -------------------------
menu = st.sidebar.selectbox("Menu", [
    "Dataset",
    "EDA Dashboard",
    "Model Training",
    "Prediction"
])

# -------------------------
# DATASET VIEW
# -------------------------
if menu == "Dataset":
    st.subheader("Dataset Preview")
    st.write(df.head())
    st.write(df.describe())

# -------------------------
# EDA DASHBOARD
# -------------------------
elif menu == "EDA Dashboard":

    st.subheader("📊 Exploratory Data Analysis")

    # Filter
    company = st.selectbox("Select Company", df["Car_Name"].unique())
    filtered_df = df[df["Car_Name"] == company]

    # KPI
    st.metric("Average Price", int(filtered_df["Selling_Price"].mean()))

    # Bar chart
    st.subheader("Car Count")
    st.bar_chart(df["Car_Name"].value_counts())

    # Price trend
    st.subheader("Price vs Year")
    st.line_chart(df.groupby("Year")["Selling_Price"].mean())

# -------------------------
# MODEL TRAINING
# -------------------------
elif menu == "Model Training":

    st.subheader("🤖 Train & Compare Models")

    # Encode
    df['Fuel'] = df['Fuel'].map({'Petrol':0, 'Diesel':1})
    df['Transmission'] = df['Transmission'].map({'Manual':0, 'Automatic':1})
    df['Car_Name'] = df['Car_Name'].astype('category').cat.codes

    X = df[['Year', 'Kms_Driven', 'Fuel', 'Transmission', 'Car_Name']]
    y = df['Selling_Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Models
    lr = LinearRegression()
    rf = RandomForestRegressor()

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Predictions
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)

    # Scores
    lr_score = r2_score(y_test, lr_pred)
    rf_score = r2_score(y_test, rf_pred)

    st.write("### Model Comparison")
    st.write(f"Linear Regression R²: {lr_score:.2f}")
    st.write(f"Random Forest R²: {rf_score:.2f}")

# -------------------------
# PREDICTION
# -------------------------
elif menu == "Prediction":

    st.subheader("🎯 Predict Car Price")

    model = pickle.load(open("model.pkl", "rb"))

    year = st.slider("Year", 2010, 2025, 2020)
    kms = st.number_input("Kms Driven", 1000, 100000, 20000)
    fuel = st.selectbox("Fuel", ["Petrol", "Diesel"])
    trans = st.selectbox("Transmission", ["Manual", "Automatic"])
    company = st.selectbox("Company", df["Car_Name"].unique())

    fuel_val = 0 if fuel == "Petrol" else 1
    trans_val = 0 if trans == "Manual" else 1
    company_val = df["Car_Name"].astype('category').cat.codes[df["Car_Name"] == company].iloc[0]

    if st.button("Predict"):
        input_data = pd.DataFrame([[year, kms, fuel_val, trans_val, company_val]],
                                 columns=['Year', 'Kms_Driven', 'Fuel', 'Transmission', 'Car_Name'])

        pred = model.predict(input_data)
        st.success(f"💰 Price: ₹ {int(pred[0])}")

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
st.sidebar.subheader("Feature Importance")

try:
    model = pickle.load(open("model.pkl", "rb"))
    importance = model.feature_importances_

    fig, ax = plt.subplots()
    ax.bar(['Year','Kms','Fuel','Trans','Company'], importance)
    st.sidebar.pyplot(fig)
except:
    pass