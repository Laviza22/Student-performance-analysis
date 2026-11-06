
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "student_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "student_model.pkl")

# --------------------------------
# LOAD DATA
# --------------------------------

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Please place student_data.csv inside your project folder.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Load saved model
model_data = None

if os.path.exists(MODEL_PATH):
    model_data = joblib.load(MODEL_PATH)

# --------------------------------
# SIDEBAR NAVIGATION
# --------------------------------

st.sidebar.title("🎓 Student Analytics")

page = st.sidebar.selectbox(
    "Navigate Dashboard",
    [
        "Overview",
        "Exploratory Data Analysis",
        "Student Insights",
        "Machine Learning"
    ]
)

st.title("🎓 Student Performance Analysis")
st.caption("Data Science Dashboard | Student Academic Performance")

# --------------------------------
# PAGE 1: OVERVIEW
# --------------------------------

if page == "Overview":

    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", len(df))
    col2.metric("Average Final Grade", round(df["G3"].mean(), 2))
    col3.metric("Average Absences", round(df["absences"].mean(), 2))
    col4.metric("Average Age", round(df["age"].mean(), 1))

    st.subheader("Final Grade Distribution")

    fig = px.histogram(
        df,
        x="G3",
        nbins=20,
        title="Distribution of Final Grades",
        labels={"G3": "Final Grade"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Student Dataset")

    st.dataframe(df, use_container_width=True)

# --------------------------------
# PAGE 2: EXPLORATORY DATA ANALYSIS
# --------------------------------

elif page == "Exploratory Data Analysis":

    st.header("Exploratory Data Analysis")

    chart = st.selectbox(
        "Select Analysis",
        [
            "Final Grade Distribution",
            "Study Time vs Final Grade",
            "Absences vs Final Grade",
            "Gender Distribution",
            "Grade Comparison"
        ]
    )

    if chart == "Final Grade Distribution":

        fig = px.histogram(
            df,
            x="G3",
            nbins=20,
            title="Final Grade Distribution"
        )

    elif chart == "Study Time vs Final Grade":

        fig = px.box(
            df,
            x="studytime",
            y="G3",
            title="Study Time vs Final Grade",
            labels={
                "studytime": "Study Time Category",
                "G3": "Final Grade"
            }
        )

    elif chart == "Absences vs Final Grade":

        fig = px.scatter(
            df,
            x="absences",
            y="G3",
            title="Absences vs Final Grade",
            color="sex"
        )

    elif chart == "Gender Distribution":

        gender_count = df["sex"].value_counts().reset_index()
        gender_count.columns = ["Gender", "Students"]

        fig = px.pie(
            gender_count,
            names="Gender",
            values="Students",
            title="Student Gender Distribution"
        )

    else:

        grade_data = df[["G1", "G2", "G3"]].melt(
            var_name="Grade Period",
            value_name="Grade"
        )

        fig = px.box(
            grade_data,
            x="Grade Period",
            y="Grade",
            title="Comparison of G1, G2 and G3"
        )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# PAGE 3: STUDENT INSIGHTS
# --------------------------------

elif page == "Student Insights":

    st.header("Student Performance Insights")

    col1, col2 = st.columns(2)

    with col1:

        fig1 = px.histogram(
            df,
            x="studytime",
            color="sex",
            title="Study Time Distribution"
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        gender_performance = (
            df.groupby("sex")["G3"]
            .mean()
            .reset_index()
        )

        fig2 = px.bar(
            gender_performance,
            x="sex",
            y="G3",
            title="Average Final Grade by Gender"
        )

        st.plotly_chart(fig2, use_container_width=True)

    parental_education = (
        df.groupby("Medu")["G3"]
        .mean()
        .reset_index()
    )

    fig3 = px.bar(
        parental_education,
        x="Medu",
        y="G3",
        title="Mother's Education vs Student Performance",
        labels={"Medu": "Mother's Education Level"}
    )

    st.plotly_chart(fig3, use_container_width=True)

# --------------------------------
# PAGE 4: MACHINE LEARNING
# --------------------------------

elif page == "Machine Learning":

    st.header("Machine Learning Predictions")

    st.subheader("Random Forest Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", "1.165")
    col2.metric("MSE", "3.798")
    col3.metric("R² Score", "0.815")

    st.caption(
        "Model B uses G1 and G2 along with student information "
        "to predict the final grade G3."
    )

    if model_data is None:

        st.warning(
            "The saved model file is not available. "
            "Please restore the trained Random Forest model."
        )

    else:

        st.subheader("Predict a Student's Final Grade")

        numerical_columns = [
            col for col in df.select_dtypes(include="number").columns
            if col != "G3"
        ]

        categorical_columns = [
            col for col in df.select_dtypes(exclude="number").columns
        ]

        with st.form("prediction_form"):

            input_values = {}

            st.write("Enter the student's information:")

            for col in numerical_columns:

                minimum = float(df[col].min())
                maximum = float(df[col].max())
                default = float(df[col].median())

                input_values[col] = st.number_input(
                    col,
                    min_value=minimum,
                    max_value=maximum,
                    value=default
                )

            for col in categorical_columns:

                options = sorted(df[col].dropna().unique().tolist())

                input_values[col] = st.selectbox(
                    col,
                    options
                )

            submitted = st.form_submit_button("Predict Final Grade")

        if submitted:

            input_df = pd.DataFrame([input_values])

            encoded_input = pd.get_dummies(input_df)

            encoded_input = encoded_input.reindex(
                columns=model_data["columns"],
                fill_value=0
            )

            prediction = model_data["model"].predict(encoded_input)[0]

            st.success(
                f"Predicted Final Grade: {prediction:.2f} / 20"
            )

            if prediction >= 15:
                st.info("Strong predicted academic performance.")

            elif prediction >= 10:
                st.info("Satisfactory predicted academic performance.")

            else:
                st.warning(
                    "This student may need additional academic support."
                )
