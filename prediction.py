import joblib
import streamlit as st
import numpy as np

def predict(data):
    clf = joblib.load("rf_model.sav")
    return clf.predict(data)