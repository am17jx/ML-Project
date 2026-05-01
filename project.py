
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

print("=" * 50)
print("Iris Dataset Info")
print("=" * 50)
print(f"Total samples   : {X.shape[0]}")
print(f"Features        : {X.shape[1]}")
print(f"Classes         : {list(target_names)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

print("\n" + "=" * 50)
print("Decision Tree Classifier")
print("=" * 50)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print(f"Accuracy: {dt_acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, dt_pred, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

print("\n" + "=" * 50)
print("K-Nearest Neighbors (KNN) - K=5")
print("=" * 50)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
print(f"Accuracy: {knn_acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, knn_pred, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, knn_pred))

print("\n" + "=" * 50)
print("Summary")
print("=" * 50)
print(f"Decision Tree Accuracy : {dt_acc * 100:.2f}%")
print(f"KNN Accuracy           : {knn_acc * 100:.2f}%")