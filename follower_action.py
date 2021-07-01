import numpy as np
import cvxpy as cp
import time

total_user = 2
active_user = 2
time_step = 2

p_l = 1
p_soh = 1

alpha = 1#0.9956
beta_s = 1#0.99
beta_b = 1#1.01

p_tax = 0#0.0000001

q_max = 120.
q_min = 0.

c_max = 40.
c_min = 40.


def follower_action_kkt(a_o, load, indiv=False):

    if total_user == active_user:
        load_active = load
        load_passive = np.zeros((1, time_step))
    else:
        load_active = load[:active_user, :]
        load_passive = load[active_user:, :]

    if indiv:
        return 0

    c1 = cp.Variable((1, time_step))
    c2 = cp.Variable((1, time_step))

    c1_mat = c1
    c2_mat = c2
    for i in range(active_user - 1):
        c1_mat = cp.vstack([c1_mat, c1])
        c2_mat = cp.vstack([c2_mat, c2])

    x_s = ((p_l + p_soh) * a_o[0] - p_l * a_o[1] - p_l * c1_mat - (p_l + p_soh) * c2_mat - p_l * p_soh * load_active)/(p_soh*(p_soh+2*p_l))
    x_b = (p_l * a_o[0] - (p_l + p_soh) * a_o[1] - (p_l + p_soh) * c1_mat - p_l * c2_mat + p_l * p_soh * load_active)/(p_soh*(p_soh+2*p_l))
    l = x_s - x_b + load_active

    utility = cp.sum(cp.multiply(a_o[0], x_s) - cp.multiply(a_o[1], x_b))\
              - p_l * cp.sum(cp.power(cp.sum(l, axis=0), 2) + cp.multiply(cp.sum(l, axis=0), np.sum(load_passive, axis=0))) \
              - p_soh * cp.sum(cp.power(x_s, 2) + cp.power(x_b, 2))

    constraints = []
    constraints += [l >= 0, x_s >= 0, x_b >= 0]

    ess_matrix = np.fromfunction(np.vectorize(lambda i, j: 0 if i < j else np.power(alpha, i - j)),
                                 (time_step, time_step), dtype=float)
    q_ess = q_min * np.fromfunction(np.vectorize(lambda i, j: np.power(alpha, i - j + 1)), (time_step, 1), dtype=float)\
            + beta_s * ess_matrix @ cp.sum(x_s, axis=0).T - beta_b * ess_matrix @ cp.sum(x_b, axis=0).T

    constraints += [q_min <= q_ess, q_ess <= q_max]
    constraints += [cp.sum(x_s, axis=0) <= c_max]
    constraints += [cp.sum(x_b, axis=0) <= c_min]

    prob = cp.Problem(cp.Maximize(utility), constraints)

    start = time.time()

    result = prob.solve(solver = 'ECOS')

    end = time.time()

    return result, l.value, end - start

def follower_action(a_o, load, indiv=False):

    if total_user == active_user:
        load_active = load
        load_passive = np.zeros((1, time_step))
    else:
        load_active = load[:active_user, :]
        load_passive = load[active_user:, :]

    if indiv:
        return 0

    x_s = cp.Variable((active_user, time_step))
    x_b = cp.Variable((active_user, time_step))
    l = x_s - x_b + load_active

    utility = cp.sum(cp.multiply(a_o[0], x_s) - cp.multiply(a_o[1], x_b))\
              - p_l * cp.sum(cp.power(cp.sum(l, axis=0), 2) + cp.multiply(cp.sum(l, axis=0), np.sum(load_passive, axis=0))) \
              - p_soh * cp.sum(cp.power(x_s, 2) + cp.power(x_b, 2))

    constraints = []
    constraints += [l >= 0, x_s >= 0, x_b >= 0]

    ess_matrix = np.fromfunction(np.vectorize(lambda i, j: 0 if i < j else np.power(alpha, i - j)),
                                 (time_step, time_step), dtype=float)
    q_ess = q_min * np.fromfunction(np.vectorize(lambda i, j: np.power(alpha, i - j + 1)), (time_step, 1), dtype=float)\
            + beta_s * ess_matrix @ cp.sum(x_s, axis=0).T - beta_b * ess_matrix @ cp.sum(x_b, axis=0).T

    constraints += [q_min <= q_ess, q_ess <= q_max]
    constraints += [cp.sum(x_s, axis=0) <= c_max]
    constraints += [cp.sum(x_b, axis=0) <= c_min]

    prob = cp.Problem(cp.Maximize(utility), constraints)

    start = time.time()

    result = prob.solve(solver = 'ECOS')

    end = time.time()

    return result, l.value, end - start


a_o_1 = np.random.random((active_user, time_step))
a_o_2 = np.random.random((active_user, time_step))

a_o = np.zeros((2, active_user, time_step))
a_o_s = np.zeros((active_user, time_step))
a_o_b = np.zeros((active_user, time_step))
a_o[0] = np.minimum(a_o_1, a_o_2)
a_o[1] = np.maximum(a_o_1, a_o_2)
a_o_s = np.minimum(a_o_1, a_o_2)
a_o_b = np.maximum(a_o_1, a_o_2)

a_o = np.ones((2, active_user, time_step))
#load = 20*np.random.random((total_user, time_step))
load = 30*np.ones((total_user, time_step))
print(load)

print(follower_action_kkt(a_o, load))
print(follower_action(a_o, load))