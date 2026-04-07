# Cyberbullying Detection in Bengali using Machine Learning

# Project Overview

This project focuses on detecting cyberbullying content in Bengali social media text using Natural Language Processing (NLP) and Machine Learning techniques.

The objective is to classify text into **bullying** and **non-bullying** categories based on linguistic patterns.

---

# Methodology

The project follows a standard machine learning pipeline:

* Text preprocessing (cleaning, normalization)
* Feature extraction using **TF-IDF (Term Frequency–Inverse Document Frequency)**
* Train/Test split for evaluation
* Training multiple supervised machine learning models



##  Models Implemented

The following models are used (aligned with research work):

* Logistic Regression
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest



##  Evaluation Metrics

Model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

## Technologies Used

* Python
* Pandas
* Scikit-learn


## Project Structure

```
main.py
preprocess.py
model.py
requirements.txt
README.md
```

---

##  Dataset Notice

The dataset used in this project is part of research work and is not publicly available.

To run this project, a dataset with the following format is required:

```
text,label
"example Bengali text",1
"another example",0
```

Where:

* **1 = Cyberbullying**
* **0 = Non-cyberbullying**

---

## Research Context

This project is based on research work in **cyberbullying detection in Bengali social media text**.

The full research explored multiple approaches including:

* TF-IDF based machine learning models
* Word2Vec embeddings
* Transformer-based models (e.g., BERT)

This repository presents a **clean and simplified implementation** of the machine learning pipeline for clarity and reproducibility.

---

## How to Run

1. Install required libraries:

```
pip install -r requirements.txt
```

2. Run the project:

```
python main.py
```

---

##  Author

Shihab Hossain Shuvo

MSc Data Science | Machine Learning & NLP Enthusiast
paper on google scolar:
https://scholar.google.co.uk/citations?hl=en&user=vCkaOnQAAAAJ
pdf available upon request
