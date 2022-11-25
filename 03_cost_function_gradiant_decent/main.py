"""
    Gradiant decent algorithm
        It is used to minimize the number of iterations and predict the values without iterating whole dataset
    
    Cost Function 
        It is nothing but mean square error
"""

import numpy as np


def gradiant_decent(x, y):
    m = 0
    b = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.08

    for i in range(iterations):
        y_predicted = m * x + b
        cost = (1 / n) * sum([val**2 for val in (y - y_predicted)])
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)
        m = m - learning_rate * md
        b = b - learning_rate * bd
        print(f"m={m}, b={b}, cost={cost}, iteration={i}")


x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradiant_decent(x, y)
