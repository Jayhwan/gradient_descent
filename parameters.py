import numpy as np
import cvxpy as cp
import time
import matplotlib.pyplot as plt

total_user = 100
time_step = 24
max_iter = 10000
alpha = 0.9956
beta_s = 0.99
beta_b = 1.01

p_soh = 1
p_l = 1

p_tax = 0.00001#0.0000001

q_max = 1000.
q_min = 0.

c_max = 100.
c_min = 100.