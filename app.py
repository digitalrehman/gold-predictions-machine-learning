from numpy import sqrt
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

st.set_page_config(page_title="Gold Price Prediction")
st.title("Gold Price Prediction - Machine Learning Project")

# Dataset URL
url = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"


# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv(url)
    return df


df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())

# Convert date
df["Date"] = pd.to_datetime(df["Date"])

df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month

df = df.drop("Date", axis=1)

# Features and target
X = df[["year", "month"]]
y = df["Price"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model selection
st.sidebar.header("Select Algorithm")

algo = st.sidebar.selectbox(
    "Algorithm", ["Linear Regression", "Decision Tree", "Random Forest"]
)

# Model training
if algo == "Linear Regression":
    model = LinearRegression()

elif algo == "Decision Tree":
    model = DecisionTreeRegressor()

else:
    model = RandomForestRegressor()

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Evaluation
col1, col2 = st.columns(2)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = sqrt(mse)

st.subheader("Model Performance")

col1.write("Mean Absolute Error:", mae)
col1.write("R2 Score:", r2)
col2.write("Mean Squared Error:", mse)
col2.write("Root Mean Squared Error:", rmse)

# Plot
st.subheader("Actual vs Predicted")

fig = plt.figure()

plt.plot(y_test.values, label="Actual")
plt.plot(pred, label="Predicted")

plt.legend()

st.pyplot(fig)


# Future prediction
st.sidebar.header("Predict Future Gold Price")

year = st.sidebar.number_input("Year", 2000, 2100, 2024)
month = st.sidebar.slider("Month", 1, 12, 1)

if st.sidebar.button("Predict Price"):

    result = model.predict([[year, month]])

    st.subheader("Predicted Gold Price")
    st.success(f"Estimated Price: {result[0]:.2f}")
