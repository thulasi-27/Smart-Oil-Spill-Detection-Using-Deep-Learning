import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Energy Prediction Dashboard", layout="wide")

# ---------- Load saved files ----------
try:
    model = joblib.load("model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()

# ---------- Helper functions ----------
def get_usage_category(value):
    if value < 1.5:
        return "Low Energy Consumption"
    elif value < 3.5:
        return "Medium Energy Consumption"
    else:
        return "High Energy Consumption"


def generate_future_predictions(base_input, hours=6):
    future_rows = []
    for i in range(hours):
        row = base_input.copy()
        row["hour"] = (row["hour"] + i) % 24
        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    future_df = future_df[feature_columns]
    preds = model.predict(future_df)
    future_df["Predicted_Global_Active_Power"] = preds
    return future_df


# ---------- Title ----------
st.title("Energy Consumption Prediction Dashboard")
st.write("Predict household energy consumption using a trained machine learning model.")

# ---------- Sidebar inputs ----------
st.sidebar.header("Enter Input Values")

global_reactive_power = st.sidebar.number_input(
    "Global Reactive Power", min_value=0.0, value=0.10, step=0.01
)
voltage = st.sidebar.number_input(
    "Voltage", min_value=0.0, value=240.0, step=0.1
)
global_intensity = st.sidebar.number_input(
    "Global Intensity", min_value=0.0, value=10.0, step=0.1
)
sub_metering_1 = st.sidebar.number_input(
    "Sub Metering 1", min_value=0.0, value=0.0, step=1.0
)
sub_metering_2 = st.sidebar.number_input(
    "Sub Metering 2", min_value=0.0, value=1.0, step=1.0
)
sub_metering_3 = st.sidebar.number_input(
    "Sub Metering 3", min_value=0.0, value=17.0, step=1.0
)

day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=1, step=1)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=1, step=1)
year = st.sidebar.number_input("Year", min_value=2000, max_value=2035, value=2010, step=1)
day_of_week = st.sidebar.number_input(
    "Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=0, step=1
)
hour = st.sidebar.number_input("Hour", min_value=0, max_value=23, value=12, step=1)
minute = st.sidebar.number_input("Minute", min_value=0, max_value=59, value=0, step=1)

# ---------- Main prediction ----------
input_dict = {
    "Global_reactive_power": global_reactive_power,
    "Voltage": voltage,
    "Global_intensity": global_intensity,
    "Sub_metering_1": sub_metering_1,
    "Sub_metering_2": sub_metering_2,
    "Sub_metering_3": sub_metering_3,
    "day": day,
    "month": month,
    "year": year,
    "day_of_week": day_of_week,
    "hour": hour,
    "minute": minute
}

input_df = pd.DataFrame([input_dict])[feature_columns]

if st.button("Predict Energy Consumption"):
    prediction = model.predict(input_df)[0]
    category = get_usage_category(prediction)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Power", f"{prediction:.4f} kW")
    col2.metric("Usage Category", category)
    col3.metric("Input Hour", f"{hour}:00")

    st.subheader("Prediction Result Chart")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Predicted Power"], [prediction])
    ax.set_ylabel("Kilowatts")
    ax.set_title("Predicted Global Active Power")
    st.pyplot(fig)

    st.subheader("Future 6-Hour Forecast")
    future_df = generate_future_predictions(input_dict, hours=6)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(
        future_df["hour"],
        future_df["Predicted_Global_Active_Power"],
        marker="o"
    )
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Predicted Power (kW)")
    ax2.set_title("Next 6 Hours Energy Forecast")
    st.pyplot(fig2)

    st.subheader("Forecast Table")
    st.dataframe(
        future_df[["hour", "Predicted_Global_Active_Power"]].rename(
            columns={
                "hour": "Hour",
                "Predicted_Global_Active_Power": "Predicted Power (kW)"
            }
        ),
        use_container_width=True
    )

# ---------- CSV Upload with preprocessing ----------
st.subheader("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload raw energy CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=";", low_memory=False)

        df.columns = df.columns.str.strip()
        df.replace("?", pd.NA, inplace=True)

        required_cols = [
            "Date",
            "Time",
            "Global_active_power",
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3"
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in uploaded CSV: {missing_cols}")
        else:
            numeric_cols = [
                "Global_active_power",
                "Global_reactive_power",
                "Voltage",
                "Global_intensity",
                "Sub_metering_1",
                "Sub_metering_2",
                "Sub_metering_3"
            ]

            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df.dropna(inplace=True)

            df["datetime"] = pd.to_datetime(
                df["Date"] + " " + df["Time"],
                format="%d/%m/%Y %H:%M:%S",
                errors="coerce"
            )

            df.dropna(subset=["datetime"], inplace=True)

            df["day"] = df["datetime"].dt.day
            df["month"] = df["datetime"].dt.month
            df["year"] = df["datetime"].dt.year
            df["day_of_week"] = df["datetime"].dt.dayofweek
            df["hour"] = df["datetime"].dt.hour
            df["minute"] = df["datetime"].dt.minute

            batch_input = df[feature_columns]
            df["Predicted_Global_Active_Power"] = model.predict(batch_input)

            st.success("Batch prediction completed successfully.")

            st.subheader("Batch Prediction Results")
            st.dataframe(
                df[[
                    "Date",
                    "Time",
                    "Global_active_power",
                    "Predicted_Global_Active_Power"
                ]].head(50),
                use_container_width=True
            )

            st.subheader("Actual vs Predicted (First 100 Rows)")
            chart_df = df[[
                "Global_active_power",
                "Predicted_Global_Active_Power"
            ]].head(100)

            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(chart_df["Global_active_power"].values, label="Actual")
            ax3.plot(chart_df["Predicted_Global_Active_Power"].values, label="Predicted")
            ax3.set_xlabel("Sample Index")
            ax3.set_ylabel("Global Active Power")
            ax3.set_title("Actual vs Predicted Energy Consumption")
            ax3.legend()
            st.pyplot(fig3)

    except Exception as e:
        st.error(f"Error processing file: {e}")

# ---------- Project description ----------
st.subheader("Project Description")
st.write(
    "This dashboard predicts household energy consumption using machine learning "
    "based on electrical and time-related input features. It also provides a "
    "short future forecast and supports batch prediction using raw CSV upload."
)

st.subheader("Technologies Used")
st.write(
    "- Python\n"
    "- Pandas\n"
    "- Scikit-learn\n"
    "- Matplotlib\n"
    "- Streamlit"
)
