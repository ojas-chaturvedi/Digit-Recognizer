import streamlit as st
from streamlit.components.v1 import html
import cv2
import numpy as np
import math
from scipy import ndimage
from keras.datasets import mnist

class NaiveBayes:
    def __init__(self):
        self.means = []
        self.variances = []
        self.priors = []

    def fit(self, x, y):
        self.classes = np.unique(y)

        for i in self.classes:
            self.priors.append(np.mean(y == i))
            x_n = x[y == i]
            self.means.append(np.mean(x_n, axis = 0))
            self.variances.append(np.var(x_n, axis = 0) + 0.01575)

    def predict(self, x):
        self.posteriors = []

        for i in self.classes:
            log_prior = np.log(self.priors[i])
            likelihood = np.sum(np.log(self.gaussian(x, self.means[i], self.variances[i])), axis = 1)
            posterior = likelihood + log_prior
            self.posteriors.append(posterior)
            
        self.posteriors = np.array(self.posteriors)
        if self.posteriors.ndim == 2:
            return np.argmax(self.posteriors, axis=0)
        else:
            return np.argmax(self.posteriors)

    def gaussian(self, x, mean, variance):
        numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator

# Get the training and testing data from the mnist library
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# Create the model, fit it, then test it
model = NaiveBayes()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
accuracy = np.mean(y_predicted == y_test)

class ProcessImage:
    def __init__(self, image):
        self.image = image

    def preprocess(self):
        # Read the image
        img = self.image

        # Scale to 20x20, invert (like training)
        img = cv2.resize(255 - img, (20, 20), interpolation = cv2.INTER_AREA)

        # img = cv2.GaussianBlur(img,(5,5),0)

        # Make gray into black (uniform background like training)
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove completely black (empty) rows/cols on all sides
        img = self.trim(img)

        # Center digit
        shiftx, shifty = self.getBestShift(img)
        shifted = self.shift(img, shiftx, shifty)
        img = shifted

        # DEBUG
        # cv2.imwrite("output.png", img)

        # Normalize the image
        img = img / 255.0

        # Reshape to 1D match the input of the model
        img = img.reshape(-1)

        return img

    def trim(self, img):
        while np.sum(img[0]) == 0:
            img = img[1:]

        while np.sum(img[:, 0]) == 0:
            img = np.delete(img, 0, 1)

        while np.sum(img[-1]) == 0:
            img = img[:-1]

        while np.sum(img[:, -1]) == 0:
            img = np.delete(img, -1, 1)

        rows, cols = img.shape

        if rows > cols:
            factor = 20.0 / rows
            rows = 20
            cols = int(round(cols * factor))
            img = cv2.resize(img, (cols, rows))
        else:
            factor = 20.0 / cols
            cols = 20
            rows = int(round(rows * factor))
            img = cv2.resize(img, (cols, rows))

        colsPadding = (
            int(math.ceil((28 - cols) / 2.0)),
            int(math.floor((28 - cols) / 2.0)),
        )
        rowsPadding = (
            int(math.ceil((28 - rows) / 2.0)),
            int(math.floor((28 - rows) / 2.0)),
        )
        img = np.pad(img, (rowsPadding, colsPadding), 'constant')

        return img

    def getBestShift(self, img):
        cy, cx = ndimage.center_of_mass(img)
        rows, cols = img.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)

        return shiftx, shifty

    def shift(self, img, sx, sy):
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows))

        return shifted

html(
    """
    <html>
        <head>
            <script src="http://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
            <style>
                body {
                    margin: 0;
                    background-color: teal;
                }
                #particles-js {
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    width: 100%;
                    height: 100%;
                    z-index: -1;
                    overflow: hidden;
                    background-color: teal;
                    background-image: url("");
                    background-repeat: no-repeat;
                    background-size: cover;
                    background-position: 50% 50%;
                }
            </style>
        </head>
        <body>
            <div id = "particles-js">
            </div>
            <script>
                particlesJS('particles-js',
                {
                    "particles": {
                        "number": {
                            "value": 80,
                            "density": {
                                "enable": true,
                                "value_area": 800
                            }
                        },
                        "color": {
                            "value": "#ffffff"
                        },
                        "shape": {
                            "type": "circle",
                            "stroke": {
                                "width": 0,
                                "color": "#000000"
                            },
                            "polygon": {
                                "nb_sides": 5
                            },
                            "image": {
                                "src": "img/github.svg",
                                "width": 100,
                                "height": 100
                            }
                        },
                        "opacity": {
                            "value": 0.5,
                            "random": false,
                            "anim": {
                                "enable": false,
                                "speed": 1,
                                "opacity_min": 0.1,
                                "sync": false
                            }
                        },
                        "size": {
                            "value": 5,
                            "random": true,
                            "anim": {
                                "enable": false,
                                "speed": 40,
                                "size_min": 0.1,
                                "sync": false
                            }
                        },
                        "line_linked": {
                            "enable": true,
                            "distance": 150,
                            "color": "#ffffff",
                            "opacity": 0.4,
                            "width": 1
                        },
                        "move": {
                            "enable": true,
                            "speed": 6,
                            "direction": "none",
                            "random": false,
                            "straight": false,
                            "out_mode": "out",
                            "attract": {
                                "enable": false,
                                "rotateX": 600,
                                "rotateY": 1200
                            }
                        }
                    },
                    "interactivity": {
                        "detect_on": "canvas",
                        "events": {
                            "onhover": {
                                "enable": true,
                                "mode": "repulse"
                            },
                            "onclick": {
                            "enable": true,
                            "mode": "push"
                            },
                            "resize": true
                        },
                        "modes": {
                            "grab": {
                                "distance": 400,
                                "line_linked": {
                                    "opacity": 1
                                }
                            },
                            "bubble": {
                                "distance": 400,
                                "size": 40,
                                "duration": 2,
                                "opacity": 8,
                                "speed": 3
                            },
                            "repulse": {
                                "distance": 200
                            },
                            "push": {
                                "particles_nb": 4
                            },
                            "remove": {
                                "particles_nb": 2
                            }
                        }
                    },
                    "retina_detect": true,
                });
            </script>
        </body>
    </html>
    """,
    height = 20000,
    width = 20000,
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
    unsafe_allow_html = True,
)

st.markdown(
    """
    # Handwritten Digit Recognizer
    """
)
st.markdown(
    "This is a simple image classification web app to predict a numerical handwritten digit."
)
st.markdown(
    "Accuracy of model with mnist dataset: **" + str(accuracy * 100) + "%**"
)
st.markdown(
    """
    Note: Digits within images must be clearly visible, in focus, and centered.
    There must be no other objects in the image (shadows, lines, etc.).
    """
)
file = st.file_uploader(
    """
    Please upload an image file.
    """,
    type = ["jpg", "png"],
)

if file is not None:
    # Read the uploaded file as a byte stream
    file_bytes = np.asarray(bytearray(file.read()), dtype = np.uint8)

    # Check if the byte stream is empty
    if file_bytes.size == 0:
        st.error("The uploaded file is empty. Please upload a valid image file.")
    else:
        # Use OpenCV to read the image data
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            st.error("Could not decode the image. Please upload a valid image file.")
        else:
            # Display the uploaded image
            st.image(file)

            # Process image
            image_processor = ProcessImage(img)
            final_image = image_processor.preprocess()

            # Predict procesed image
            predicted_digit = model.predict([final_image])
            st.markdown("Predicted Digit: **" + str(predicted_digit[0]) + "**")
