import numpy as np
import math
from sklearn.linear_model import LinearRegression
import pandas as pd


def gradiant_decent(x, y):
    m = 0
    b = 0
    iterations = 1000000
    n = len(x)
    prev_cost = 0
    learning_rate = 0.0002

    for i in range(iterations):
        y_predicted = m * x + b
        cost = (1 / n) * sum([val**2 for val in (y - y_predicted)])
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)
        m = m - learning_rate * md
        b = b - learning_rate * bd
        if math.isclose(prev_cost, cost, rel_tol=1e-20):
            break
        prev_cost = cost
        print(f"m={m}, b={b}, cost={cost}, iteration={i}")


df = pd.read_csv("test_scores.csv")

x = np.array(df["math"])
y = np.array(df["cs"])

gradiant_decent(x, y)

reg = LinearRegression()
reg.fit(df[["math"]], df.cs)

print(f"Linear regression: m = {reg.coef_} b = {reg.intercept_}")
