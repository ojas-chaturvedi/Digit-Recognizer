import streamlit as st

from streamlit.components.v1 import html

html(
    """
<html>
<head>
   <script src = "https://cdnjs.cloudflare.com/ajax/libs/tsparticles/1.18.11/tsparticles.min.js"> </script>
   <style>
      #particles {
         position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          width: 100%;
          height: 100%;
          z-index: -1;
          overflow: hidden;
         background-color: teal;
      }
   </style>
</head>
<body>
   <div id = "particles">
   </div>
   <script>
      tsParticles.load("particles", {
         particles: {
            number: {
               value: 1000
            },
            move: {
               enable: true
            },
            color: {
               value: "#272701"
            },
         }
      });
   </script>
</body>
</html>
""",
    height=20000,
    width=20000,
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

st.write("""
    # Handwritten Digit Recognizer
    """)
st.write("This is a simple image classification web app to predict a numerical handwritten digit.")
st.write("""
    Note: Digits within images must be clearly visible, in focus, and centered.
    There must be no other objects in the image (shadows, lines, etc.).
    """)
file = st.file_uploader("""
    Please upload an image file.
    """, type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

if file is not None:
    # Read the file as a byte stream
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)

    # Check if the byte stream is empty
    if file_bytes.size == 0:
        st.error("The uploaded file is empty. Please upload a valid image file.")
    else:
        # Use OpenCV to read the image data
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            st.error("Could not decode the image. Please upload a valid image file.")
        else:
            # Scale to 20x20, invert (like training)
            img = cv2.resize(255 - img, (20, 20), interpolation = cv2.INTER_AREA)

            # Make gray into black (uniform background like training)
            _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Make gray into black (uniform background like training)
            _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Display the processed image
            st.image(img, use_column_width=True)