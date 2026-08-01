import requests
import streamlit as st

st.set_page_config(page_title="ML Portal", page_icon="🤖")

st.title("🤖 Machine Learning Prediction Portal")
st.write("Enter values below to get predictions from your FastAPI backend.")

st.divider()

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal Length", value=5.1)
    sepal_width = st.number_input("Sepal Width", value=3.5)
with col2:
    petal_length = st.number_input("Petal Length", value=1.4)
    petal_width = st.number_input("Petal Width", value=0.2)

if st.button("Get Prediction", use_container_width=True):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Class: **{result['prediction_label']}**")
            st.info(f"Confidence: **{result['confidence'] * 100:.2f}%**")
        else:
            st.error("Backend server returned an error.")
    except Exception:
        st.error("Could not connect to FastAPI. Is your backend server running?")
