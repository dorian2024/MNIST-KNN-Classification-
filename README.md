# Handwritten Digit Recognition using K-Nearest Neighbors (KNN)

**Week 3 Project – Machine Learning Internship**  


This project demonstrates handwritten digit classification using a custom-built **K-Nearest Neighbors (KNN)** model, developed entirely from scratch using **NumPy**, without relying on high-level ML libraries such as `scikit-learn`.

---

## Objective

The goal of the project was to construct a complete machine learning pipeline to classify digits (0–9) from the **MNIST dataset**. Key constraints included:
- No use of pre-built classifiers (e.g., `KNeighborsClassifier`)
- Entire model logic implemented using only core NumPy operations

---

## Data Preprocessing

- **Dataset**: MNIST Handwritten Digit Dataset
  - 70,000 grayscale images (28×28 pixels)
  - Each image is flattened into a 784-dimensional vector
- **Steps performed**:
  - Normalized pixel values to range [0, 1] by dividing by 255.0
  - Converted labels from strings to integers
  - Split dataset into:
    - 60% training
    - 20% validation
    - 20% testing

---

## KNN Model Implementation

The KNN model was implemented from scratch and includes the following core methods:

| Function            | Description |
|---------------------|-------------|
| `store_data(X, y)`  | Stores the training features and labels |
| `euclidean_distance(X)` | Efficient, vectorized implementation of the Euclidean distance |
| `predict(X, k)`     | Predicts labels for input data using majority vote (supports weighted voting) |
| `score(X, y)`       | Computes accuracy by comparing predictions with ground truth |

The Euclidean distance was computed using fully vectorized NumPy operations to optimize performance.

---

## Hyperparameter Tuning

The model was tested with different values of **k** to observe its impact on accuracy:
- The best performance was observed with **k = 5**, using **weighted voting** based on inverse distance.

---

## Performance Visualization

- Confusion matrix and accuracy scores were used to evaluate the classifier.
- Graphs were plotted to compare accuracy across different k values.

---

## Files in this Repository

- `knn_digit_classifier.ipynb` – Jupyter Notebook with the full implementation
- `mnist_train.csv`, `mnist_test.csv` – Dataset files (or links to download if too large)
- `README.md` – Project documentation

---

## 🛠 Technologies Used

- Python 3.x
- NumPy
- Matplotlib (for visualization)
- Jupyter Notebook

---

## 📌 Key Takeaways

- Successfully implemented a performant digit recognition system using only NumPy
- Gained deeper understanding of:
  - Distance metrics
  - Vectorized computation
  - Trade-offs in hyperparameter selection
- Achieved over **97% accuracy** with a simple, interpretable algorithm

---

## Acknowledgement

This project was completed as part of an academic internship and is intended for learning and demonstration purposes.
