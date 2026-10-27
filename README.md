XOR & Perceptron Visualizer :

An interactive machine learning project that demonstrates why a single perceptron cannot solve the XOR problem and how a neural network with a hidden layer can learn it.

About the Project :

The XOR (Exclusive OR) problem is a classic example in machine learning. A single-layer perceptron cannot correctly classify all four XOR input combinations because the data is not linearly separable.

This project visualizes the difference between a single perceptron and a Multi-Layer Perceptron (MLP) using Python, machine learning, and decision-boundary plots.

Features :

- Compare a single perceptron with a neural network.
- Experiment with 1, 2, 4, and 8 hidden neurons.
- View predictions and the number of correct classifications.
- Visualize decision boundaries for both models.
- Explore how hidden neurons affect the model's ability to learn XOR.

XOR Truth Table :

| Input A | Input B | Output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Technologies Used :

- Python
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Google Colab

Project Structure :

```text
xor-perceptron-visualizer/
├── app.py
├── requirements.txt
└── README.md
```

Run the Project Locally :

1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/xor-perceptron-visualizer.git
```

Replace `YOUR-USERNAME` with your GitHub username.

2. Open the project folder

```bash
cd xor-perceptron-visualizer
```

3. Install the dependencies

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

Learning Outcomes :

- Understand the limitations of a single perceptron.
- Learn the importance of hidden layers in neural networks.
- Understand the XOR problem and linear separability.
- Visualize machine learning predictions and decision boundaries.
- Gain practical experience with Python and Streamlit.

Future Improvements :

- Add more activation functions.
- Compare different neural network architectures.
- Improve the visualization and user interface.
- Add model accuracy comparisons and explanations.

Author:
Mohan Sai

GitHub: [mohansai2007](https://github.com/mohansai2007)

Portfolio: [mohan-sai-portfolio.netlify.app](https://mohan-sai-portfolio.netlify.app/)


This project was developed as a hands-on learning exercise to understand the fundamentals of perceptrons, neural networks, and the XOR classification problem.
