# **Movie Recommender System**
### **Matrix Factorization from Scratch using Gradient Descent**

This project implements a **movie recommendation system** using **Matrix Factorization (MF)** built completely from scratch in Python and NumPy. The model learns latent user and movie factors to predict ratings and evaluates performance using RMSE.

---

## **Project Overview**

The goal of this project is to predict user–movie ratings using **latent factor modeling**. Instead of similarity-based collaborative filtering, the system decomposes the user–item rating matrix into two lower-dimensional matrices and optimizes them using **stochastic gradient descent (SGD)**.

The implementation includes:
- Manual matrix factorization
- SGD-based optimization
- Regularization to prevent overfitting
- Train/test split
- RMSE-based evaluation
- Hyperparameter tuning

No external machine learning libraries are used for the core algorithm.

---

## **Technologies Used**

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib** (for evaluation and tuning plots)



---

## **How to Run**

### **1. Install dependencies**
```bash
pip install numpy pandas matplotlib

2. Download the dataset

This project uses the MovieLens 1M dataset.

Download it from:
https://grouplens.org/datasets/movielens/1m/

Place the file ratings.dat inside the data/ folder.

3. Run the project

Run the Python script:

python "MovieLens Recommender System.py"


Or open the notebook:

jupyter notebook "MovieLens Recommender System.ipynb"

