import numpy as np
import cvxpy as cp
import time

def ni():

    r = 10
    a, b = 0, 0
    for i in range(10000):
        print(i)
        x = cp.Variable(1)
        y = cp.Variable(1)

        utility = x*x + 8*x*b/3-34*-(x[0]*x[0] - 2 * x[0] + x[1]*x[1] - x[1] + 5/4 + x[0] + x[1]) + r * (cp.power(y[0]-x[0],2) + cp.power(y[1]-x[1],2)) / 2

        constraints =[]

        constraints += [y[0] + y[1] - 1 <= 0]

        prob = cp.Problem(cp.Minimize(utility), constraints)

        result = prob.solve(solver='ECOS')

        print(x)
        print(utility.value)
        x = y.value
        if np.abs(utility.value) < 1e-15:
            break
    print(y.value)
def kkt():
    c = cp.Variable(1)

    utility = cp.power(c/2, 2) + cp.power(c/2, 2) + (c+2)/2 + (c+1)/2

    constraints = []

    constraints += [(c+2)/2+(c+1)/2 -1 <= 0]

    prob = cp.Problem(cp.Minimize(utility), constraints)

    result = prob.solve(solver='ECOS')

    print(c.value)
    print((c.value+2)/2, (c.value+1)/2)

ni()
kkt()