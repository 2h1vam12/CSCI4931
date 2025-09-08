# Homework 1 – Deep Learning (Fall 2025)

### Title

**Getting Familiar with Artificial Neural Networks from Scratch**

### Course

CSCI 4931 – Deep Learning (Fall 2025, Undergraduate Version)

---

## 📌 Objective

* Gain hands-on experience building and training a simple neural network from scratch.
* Understand **forward propagation**, **backpropagation**, and **gradient descent**.
* Learn the importance of **debugging**, **hyperparameter tuning**, and **visualization** in deep learning.
* Practice explaining both **mathematical foundations** and **implementation reflections** clearly.

---

## 🧮 Part 1: Conceptual Warm-Up (No Coding)

Tasks included:

1. Deriving the gradient update rule for a single neuron with sigmoid activation.
2. Explaining the role of activation functions.
3. Discussing **overfitting** and techniques to mitigate it (e.g., regularization, dropout).
4. Discussing **underfitting** and how to address it (e.g., more complex models, feature engineering).
5. Manual forward-pass and gradient calculation on the small dataset:

| x1 | x2 | y |
| -- | -- | - |
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 1 |

---

## 💻 Part 2: Neural Network Implementation (Coding)

### Dataset

* **X**: Four binary samples with two features each.
* **Y**: Binary target labels.

### Architecture

* Input layer: 2 features
* Hidden layer: 2 neurons (with **sigmoid activation**)
* Output layer: 1 neuron (with **sigmoid activation**)
* Loss function: **Mean Squared Error (MSE)**

### Key Functions (implemented in `hw1-boilerplate.py`):

* `forward_propagation(X, W1, b1, W2, b2)`
* `backward(Z1, A1, Z2, A2, Y)`
* `update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2)`

### Training Loop

* Optimizer: Gradient Descent
* Epochs: 10,000
* Learning rate: Tuned between 0.01 and 0.1

### Visualizations

* **Loss Curve**: Shows loss decreasing across epochs.
* **Decision Boundary**: Displays how the network separates the dataset after training.

---

## 🔬 Part 3: Experimentation and Analysis

* **Learning Rates**: Compared 0.01, 0.05, and 0.1

  * Smaller learning rates converged slower but were more stable.
  * Larger learning rates sometimes caused oscillations.
* **Hidden Layer Size**: Tested sizes from 1 to 5

  * Too few neurons → underfitting.
  * Too many neurons → risk of overfitting given tiny dataset.

---

## ✍️ Part 4: Reflections

1. **Challenges in Backpropagation**

   * Biggest hurdle: getting matrix dimensions aligned.
   * Fixed by carefully tracking shapes and using NumPy broadcasting.

2. **Debugging Importance**

   * Debugging gradients helped identify vanishing/exploding values.
   * Strategy: printed partial outputs at each step and compared with manual calculations.

3. **Effect of Different Activations**

   * **Sigmoid**: Smooth, but prone to vanishing gradients.
   * **tanh**: Centered around 0, better gradient flow.
   * **ReLU**: Faster convergence but risk of “dying” neurons on such a small dataset.

---

## 📊 Deliverables

* **Code**: [`hw1-boilerplate.py`](./hw1-boilerplate.py) (with completed forward, backward, and update steps).
* **Report**: PDF containing

  * Part 1 short answers & math derivations.
  * Part 3 experiment results.
  * Part 4 reflections.
  * Plots: Loss curve & decision boundary.

---

## ✅ Grading Criteria (as per assignment sheet)

* **Correctness (40%)** – Implementation of ANN from scratch.
* **Clarity (20%)** – Well-documented code and clean explanations.
* **Analysis (20%)** – Insightful reflection and experimentation.
* **Effort (20%)** – Debugging, testing, and full report submission.

---

## 🧑‍💻 Author

Shivam Pathak
University of Colorado Denver – CSCI 4931
