import numpy as np
import cvxpy as cp
import time

total_user = 100
active_user = 10
time_step = 5
max_iter = 10000
alpha = 1#0.9956
beta_s = 1#0.99
beta_b = 1#1.01

p_soh = 0.000001
p_l = 1.

p_tax = 0#0.0000001

q_max = 120.
q_min = 0.

c_max = 40.
c_min = 40.


def operator_objective(act_user, t, load_matrix, operator_action=None, user_action=None):

    if total_user == act_user:
        load_active = load_matrix
        load_passive = np.zeros((1, t))
    elif act_user == 0:
        load_active = np.zeros((1, t))
        load_passive = load_matrix
    else:
        load_active = load_matrix[:act_user, :]
        load_passive = load_matrix[act_user:, :]

    if user_action is None:
        [x_s, x_b, l] = [np.zeros((1, t)), np.zeros((1, t)), np.zeros((1, t))]
    else:
        [x_s, x_b, l] = [user_action[0], user_action[1], user_action[2]]

    ec = - p_l * np.sum(np.power(np.sum(l, axis=0) + np.sum(load_passive, axis=0), 2))

    if a_o is None:
        tax = 0
    else:
        tax = - p_tax * np.sum(np.power(operator_action, 2))

    return ec + tax


def users_objective(act_user, t, load_matrix, operator_action=None, user_action=None):

    if total_user == act_user:
        load_active = load_matrix
        load_passive = np.zeros((1, time_step))
    elif act_user == 0:
        return 0
    else:
        load_active = load_matrix[:act_user, :]
        load_passive = load_matrix[act_user:, :]

    assert operator_action is not None and user_action is not None

    x = np.zeros(act_user)

    [x_s, x_b, l] = [user_action[0], user_action[1], user_action[2]]
    [p_s, p_b] = [operator_action[0], operator_action[1]]

    for i in range(act_user):
        trans = np.sum(np.multiply(p_s[i], x_s[i]) - np.multiply(p_b[i], x_b[i]))
        ec = - p_l * np.sum(np.multiply(l[i], np.sum(l, axis=0) + np.sum(load_passive, axis=0)))
        soh = p_soh * np.sum(np.multiply(x_s[i], np.sum(x_s, axis=0)))
        soh += p_soh * np.sum(np.multiply(x_b[i], np.sum(x_b, axis=0)))
        x[i] = trans + ec + soh

    return x


def constraint_value(act_user, t, load_matrix, operator_action=None, user_action=None):
    return 0


def initial_point(act_user, t):
    operator_action = np.ones((2, act_user, t))
    operator_action[1] *= 2
    return operator_action


def follower_action(a_o, load):
    return 0


def direction_finding(a_o, a_f, load):
    return 0


def step_size(a_o, a_f, load, d, r):
    update_coef = 0.9
    s = update_coef
    for i in range(10000):
        next_operator_action = a_o + s*r
        next_follower_action = follower_action(next_operator_action, load)
        update = True
        if operator_objective(next_operator_action, next_follower_action) >= operator_objective(a_o, a_f, load) - update_coef * s * d:
            if np.any(constraint_value(active_user, time_step, a_o, a_f, load) > 0):
                update = False
        else:
            update = False
        if update:
            return s
        else:
            s *= update_coef
    return 0


def iterations(active_user, time_step, load):
    init = initial_point(active_user, time_step)
    a_o = init
    a_f = follower_action(a_o, load)
    epsilon = 1e-12

    for i in range(max_iter):
        d, r = direction_finding(a_o, a_f, load)
        b = step_size(a_o, a_f, load, d, r)
        a_o = a_o + b * r
        a_f = follower_action(a_o, load)
        if b < epsilon:
            break

    return init, a_o, a_f

load = np.random.random((total_user, time_step))
a_o = initial_point(active_user, time_step)
a_f = follower_action(a_o, load)
d, r = direction_finding(a_o, a_f, load)
b = step_size(a_o, a_f, load, d, r)

init, a_o, a_f = iterations(active_user, time_step, load)


