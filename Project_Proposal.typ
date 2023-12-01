#show heading: set text(font: "New Computer Modern")
#show heading: set block(above: 1.4em, below: 1em)
#set text(font: "New Computer Modern")
#show raw: set text(font: "New Computer Modern Mono")
#show par: set block(spacing: 0.55em)

#set text(size: 10pt)

#set page(margin: 1.75cm)

#set par(justify: true)

#let chiline() = { v(-3pt); line(length: 100%); v(-5pt) }

#set align(center)
= Handwriting Digit Recognizer --- Proposal
Capstone: The Art of Approximation\
Ojas Chaturvedi, Ritwik Jayaraman, Saianshul Vishnubhaktula, Zaheen Jamil

#set align(left)

== Language
#chiline()

Python, a simple and popular language for machine learning and data science due
to its extensive libraries and frameworks

== Objective
#chiline()

To develop a custom machine learning model which would be able to determine what
a digit is from an image of a handwritten single digit

== Implementation
#chiline()

==== Overview of Steps:
#chiline()

+ Data Exploration and Visualization
+ Data Preprocessing
+ Feature Engineering
+ Model Building
+ Model Training and Testing
+ Model Evaluation and Deployment
+ Hyperparameter Tuning and Optimization
+ Website/App Development

==== Potential Libraries:
#chiline()

- Pandas: For data manipulation and analysis
- NumPy: For numerical computing and working with arrays
- Matplotlib: For data visualization
- Scikit-learn: For data mining and analysis
- TensorFlow: For deep learning and complex neural network modeling
- Flask/Django: For backend web development
- SQLAlchemy: For SQL databases and Object-Relational Mapping

==== Manual Work:
#chiline()

- Making algorithms for data preprocessing and feature engineering
- Building custom model
- Training and testing model
- Creating website/app that can use the model and store results for future
  training of model
- Documentation of all steps

== Jobs
#chiline()

- Machine/Deep Learning Developers
  - Develops the machine learning model
  - Trains & tests the model
  - Makes the model usable in the website/app
- Data Analyst
  - Algorithm development for preprocessing and feature engineering
  - Will still contribute as a Machine Learning Developer
- GUI Developer
  - Makes the website/app and all of its functionality (UI)
  - Makes the model usable in the website/app
  - Will still contribute as a Machine Learning Developer