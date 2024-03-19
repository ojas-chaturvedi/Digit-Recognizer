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

