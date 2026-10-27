import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="XOR Visualizer", layout="wide")

st.title("XOR & Perceptron Visualizer")
st.write(
    "Explore why a single perceptron struggles with XOR "
    "and how a neural network can learn it."
)

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# User selects hidden neurons
neurons = st.select_slider(
    "Number of hidden neurons",
    options=[1, 2, 4, 8],
    value=4
)

# Train models
perceptron = Perceptron(max_iter=1000, random_state=42)
perceptron.fit(X, y)

mlp = MLPClassifier(
    hidden_layer_sizes=(neurons,),
    activation="tanh",
    solver="lbfgs",
    max_iter=5000,
    random_state=42
)
mlp.fit(X, y)

# Display predictions
left, right = st.columns(2)

with left:
    st.subheader("Single Perceptron")
    p_pred = perceptron.predict(X)
    st.write("Predictions:", p_pred.tolist())
    st.metric("Correct predictions", f"{np.sum(p_pred == y)}/4")

with right:
    st.subheader(f"Neural Network ({neurons} neurons)")
    m_pred = mlp.predict(X)
    st.write("Predictions:", m_pred.tolist())
    st.metric("Correct predictions", f"{np.sum(m_pred == y)}/4")

# Create decision-boundary grid
a = np.linspace(-0.2, 1.2, 200)
b = np.linspace(-0.2, 1.2, 200)
A, B = np.meshgrid(a, b)
grid = np.c_[A.ravel(), B.ravel()]

pred_p = perceptron.predict(grid).reshape(A.shape)
pred_m = mlp.predict(grid).reshape(A.shape)

# Plot boundaries
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, regions, title in [
    (axes[0], pred_p, "Single Perceptron"),
    (axes[1], pred_m, f"Neural Network ({neurons} neurons)")
]:
    ax.contourf(
        A, B, regions,
        levels=[-0.5, 0.5, 1.5],
        colors=["lightblue", "mistyrose"],
        alpha=0.7
    )
    ax.scatter(X[y == 0, 0], X[y == 0, 1],
               color="blue", s=120, label="Output 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1],
               color="red", marker="x", s=120, label="Output 1")
    ax.set_xlabel("Input A")
    ax.set_ylabel("Input B")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
