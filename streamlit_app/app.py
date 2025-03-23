import streamlit as st
import requests

st.title("Streamlit App")

# Connect to FastAPI
response = requests.get("http://fastapi:8000")
st.write("Response from FastAPI:", response.json())