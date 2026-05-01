# Iris Flower Classification — ML Project

A beginner-friendly machine learning project that classifies Iris flowers using two supervised learning algorithms: **Decision Tree** and **K-Nearest Neighbors (KNN)**.

---

## 📌 Overview

This project uses the classic [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) to train and evaluate two classification models and compares their performance.

| Property       | Value                              |
|----------------|------------------------------------|
| Dataset        | Iris (built-in via scikit-learn)   |
| Samples        | 150                                |
| Features       | 4 (sepal length/width, petal length/width) |
| Classes        | 3 (setosa, versicolor, virginica)  |
| Train/Test Split | 80% / 20% (random state = 42)   |

---

## 🧠 Models

### 1. Decision Tree Classifier
- `max_depth = 3`
- `random_state = 42`

### 2. K-Nearest Neighbors (KNN)
- `n_neighbors = 5`

Both models are evaluated using:
- **Accuracy Score**
- **Classification Report** (precision, recall, F1-score)
- **Confusion Matrix**

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.x installed along with the following packages:

```bash
pip install scikit-learn pandas numpy
```

### Run the project

```bash
python project.py
```

---

## 📊 Sample Output

```
==================================================
Iris Dataset Info
==================================================
Total samples   : 150
Features        : 4
Classes         : ['setosa', 'versicolor', 'virginica']

Training samples : 120
Testing samples  : 30

==================================================
Decision Tree Classifier
==================================================
Accuracy: 100.00%
...

==================================================
K-Nearest Neighbors (KNN) - K=5
==================================================
Accuracy: 100.00%
...

==================================================
Summary
==================================================
Decision Tree Accuracy : 100.00%
KNN Accuracy           : 100.00%
```

---

## 📁 Project Structure

```
ML-Project/
├── project.py   # Main script with data loading, training, and evaluation
└── README.md    # Project documentation
```

---

## 🛠 Technologies Used

- [Python 3](https://www.python.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

---

## 📄 License

This project is open-source and available for educational purposes.
