#!/opt/homebrew/bin/python3
"""
Name: website.py
Purpose: To showcase the Gaussian Naïve Bayes classifier to recognize handwritten digits
"""

__author__ = "Ojas Chaturvedi"
__github__ = "github.com/ojas-chaturvedi"
__license__ = "MIT"

# Import necessary libraries
import streamlit as st
from streamlit.components.v1 import html
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import project code
from main import model, accuracy
from ImageProcessor import ImageProcessor


st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="random",
    # layout="wide",
)

# Add css to make the iframe fullscreen
st.markdown(
    """
    <style>
        iframe {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border: none;
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(
    """
    :white[Handwritten Digit Recognizer]
    """
)
st.markdown(
    ":white[An image classification web app to predict a numerical handwritten digit.]"
)
st.markdown(
    ":white[Accuracy of model with the MNIST dataset:] **:orange["
    + str(accuracy * 100)
    + "%]**"
)
