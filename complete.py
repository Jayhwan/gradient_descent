import numpy as np
import cvxpy as cp
import time

total_user = 20
active_user = 20
time_step = 24
max_iter = 10000
alpha = 1#0.9956
beta_s = 1#0.99
beta_b = 1#1.01

p_soh = 1
p_l = 1.

p_tax = 0.000001#0.0000001

q_max = 100.
q_min = 0.

c_max = 40.
c_min = 40.


def decompose(act_user, t, load_matrix, operator_action, user_action):

    if total_user == act_user:
        load_active = load_matrix
        load_passive = np.zeros((1, t))
    elif act_user == 0:
        load_active = np.zeros((1, t))
        load_passive = load_matrix
    else:
        load_active = load_matrix[:act_user, :]
        load_passive = load_matrix[act_user:, :]

    if user_action is not None:
        [x_s, x_b, l] = [user_action[0], user_action[1], user_action[2]]
    else:
        [x_s, x_b, l] = [np.zeros((1, t)), np.zeros((1, t)), np.zeros((1, t))]

    if operator_action is not None:
        [p_s, p_b] = [operator_action[0], operator_action[1]]
    else:
        [p_s, p_b] = [np.zeros((1, t)), np.zeros((1, t))]

    return [x_s, x_b, l, p_s, p_b, load_active, load_passive]

def par(act_user, t, load_matrix, operator_action=None, user_action=None):

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, t, load_matrix, operator_action, user_action)
    load_total = np.sum(l, axis=0) + np.sum(load_passive, axis=0)
    return np.max(load_total)/np.average(load_total)

def operator_objective(act_user, t, load_matrix, operator_action=None, user_action=None):

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, t, load_matrix, operator_action, user_action)

    ec = - p_l * np.sum(np.power(np.sum(l, axis=0) + np.sum(load_passive, axis=0), 2))

    if act_user == 0:
        tax = 0
    else:
        tax = - p_tax * np.sum(np.power(p_s, 2) + np.power(p_b, 2))

    return ec + tax


def users_objective(act_user, t, load_matrix, operator_action=None, user_action=None):

    if act_user == 0:
        return np.zeros(0)

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, t, load_matrix, operator_action, user_action)

    x = np.zeros(act_user)

    for i in range(act_user):
        trans = np.sum(np.multiply(p_s[i], x_s[i]) - np.multiply(p_b[i], x_b[i]))
        ec = - p_l * np.sum(np.multiply(l[i], np.sum(l, axis=0) + np.sum(load_passive, axis=0)))
        soh = p_soh * np.sum(np.multiply(x_s[i], np.sum(x_s, axis=0)))
        soh += p_soh * np.sum(np.multiply(x_b[i], np.sum(x_b, axis=0)))
        x[i] = trans + ec + soh

    return x


def operator_constraint_value(act_user, t, load_matrix, operator_action=None, user_action=None):
    if act_user == 0:
        return np.zeros((1, t))

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, t, load_matrix, operator_action, user_action)

    x = np.zeros((2 * act_user, t))

    for i in range(t):
        for j in range(act_user):
            x[j * act_user] = - p_s[j, i]
            x[j * act_user +1] = p_s[j, i] - p_b[j, i]
    return x


def operator_constraint_value(act_user, t, load_matrix, operator_action=None, user_action=None):

    if act_user== 0:
        return np.zeros((1, t))

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, t, load_matrix, operator_action,
                                                                   user_action)

    x = np.zeros((3, act_user, t))

    for i in range(t):
        for j in range(act_user):
            x[0, j, i] = - p_s[j, i]
            x[1, j, i] = - p_b[j, i]
            x[2, j, i] = p_s[j, i] - p_b[j, i]

    return x


def follower_constraint_value(act_user, t, load_matrix, operator_action=None, user_action=None):

    if act_user == 0:
        return np.zeros((1, t))

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, t, load_matrix, operator_action, user_action)

    x = np.zeros((7 * act_user, t))

    for i in range(t):
        for j in range(act_user):
        # ESS SOH Constraints
            if i == 0:
                x[7 * j][i] = - q_min - beta_s * x_s[j, i] + beta_b * x_b[j, i]
                x[7 * j + 1][i] = q_min + beta_s * x_s[j, i] - beta_b * x_b[j, i] - q_max/act_user
            else:
                x[7 * j][i] = - alpha * (-x[7 * j, i-1]) - beta_s * x_s[j, i] + beta_b * x_b[j, i]
                x[7 * j + 1][i] = alpha * (-x[7 * j + 1, i-1]) + beta_s * x_s[j, i] - beta_b * x_b[j, i] - q_max/act_user
            # ESS Speed Constraints
            x[7 * j + 2][i] = x_s[j, i] - c_max/act_user
            x[7 * j + 3][i] = x_b[j, i] - c_min/act_user
            # Positive Constraints
            x[7 * j + 4][i] = - x_s[j][i]
            x[7 * j + 5][i] = - x_b[j][i]
            x[7 * j + 6][i] = - l[j][i]

    return x


def initial_point(act_user, t):
    operator_action = 1 * np.ones((2, act_user, t))
    #operator_action[1] *= 2
    return operator_action


def follower_action(a_o, load, indiv=False):

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

    utility = 0
    utility += cp.sum(cp.multiply(a_o[0], x_s) - cp.multiply(a_o[1], x_b))
    utility += - p_l * cp.sum(cp.power(cp.sum(l, axis=0), 2) + cp.multiply(cp.sum(l, axis=0), np.sum(load_passive, axis=0)))
    utility += - p_soh * cp.sum(cp.power(cp.sum(x_s, axis=0), 2) + cp.power(cp.sum(x_b, axis=0), 2))

    constraints = []
    constraints += [l >= 0]
    constraints += [x_s >= 0]
    constraints += [x_b >= 0]

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
    #print("# kkt #")
    #print("x_s :", x_s.value)#.reshape(active_user))
    #print("x_b :", x_b.value)#.reshape(active_user))
    #print("l   :", l.value)#.reshape(active_user))

    return result, x_s.value, x_b.value, l.value, end - start


def follower_action_indiv(a_o, load, indiv=False):

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

    utility = 0
    utility += cp.sum(cp.multiply(a_o[0], x_s) - cp.multiply(a_o[1], x_b))
    utility += - p_l * cp.sum(cp.power(cp.sum(l, axis=0), 2) + cp.multiply(cp.sum(l, axis=0), np.sum(load_passive, axis=0)))
    utility += - p_soh * cp.sum(cp.power(cp.sum(x_s, axis=0), 2) + cp.power(cp.sum(x_b, axis=0), 2))

    constraints = []
    constraints += [l >= 0]
    constraints += [x_s >= 0]
    constraints += [x_b >= 0]

    q_ess = np.zeros((active_user, time_step))
    for i in range(active_user):
        for j in range(time_step):
            if j == 0:
                q_ess[i, j] = q_min + beta_s * x_s[i, j] - beta_b * x_b[i, j]
            else:
                q_ess[i, j] = alpha * q_ess[i, j - 1] + beta_s * x_s[i, j] - beta_b * x_b[i, j]

    constraints += [q_min <= q_ess, q_ess <= q_max]
    constraints += [x_s <= c_max/active_user]
    constraints += [x_b <= c_min/active_user]

    prob = cp.Problem(cp.Maximize(utility), constraints)

    start = time.time()

    result = prob.solve(solver = 'ECOS')

    end = time.time()
    #print("# kkt #")
    #print("x_s :", x_s.value)#.reshape(active_user))
    #print("x_b :", x_b.value)#.reshape(active_user))
    #print("l   :", l.value)#.reshape(active_user))

    return result, x_s.value, x_b.value, l.value, end - start


def follower_constraints_derivative_matrix(act_user, t):
    x = np.zeros((4 + 3 * act_user, t, 2, t))

    for i in range(t):
        if i == 0:
            x[0][i][0][i] = act_user * (beta_s * p_l - beta_b * (p_l + p_soh))
            x[0][i][1][i] = act_user * (beta_s * (p_l + p_soh) - beta_b * p_l)
            x[1][i] = - x[0][i]
        else:
            x[0][i] = alpha * x[0][i-1]
            x[0][i][0][i] = act_user * (beta_s * p_l - beta_b * (p_l + p_soh))
            x[0][i][1][i] = act_user * (beta_s * (p_l + p_soh) - beta_b * p_l)
            x[1][i] = - x[0][i]

        x[2][i][0][i] = - act_user * p_l
        x[2][i][1][i] = - act_user * (p_l + p_soh)
        x[3][i][0][i] = - act_user * (p_l + p_soh)
        x[3][i][1][i] = - act_user * p_l

        for j in range(act_user):
            x[4 + 3 * j][i][0][i] = p_l
            x[4 + 3 * j][i][1][i] = p_l + p_soh
            x[4 + 3 * j + 1][i][0][i] = p_l + p_soh
            x[4 + 3 * j + 1][i][1][i] = p_l
            x[4 + 3 * j + 2][i][0][i] = - p_soh
            x[4 + 3 * j + 2][i][1][i] = p_soh

    x = x/(p_soh*p_soh*(p_soh+2*p_l))
    return x


def follower_constraints_derivative_matrix_indiv(act_user, t):
    x = np.zeros((7 * act_user, t, 2, t))

    for i in range(t):
        for j in range(act_user):
            if i == 0:
                x[7*j][i][0][i] = beta_s * p_l - beta_b * (p_l + p_soh)
                x[7*j][i][1][i] = beta_s * (p_l + p_soh) - beta_b * p_l
                x[7*j+1][i] = - x[7*j][i]
            else:
                x[7*j][i] = alpha * x[0][i-1]
                x[7*j][i][0][i] = beta_s * p_l - beta_b * (p_l + p_soh)
                x[7*j][i][1][i] = beta_s * (p_l + p_soh) - beta_b * p_l
                x[7*j+1][i] = - x[7*j][i]

            x[7*j+2][i][0][i] = - p_l
            x[7*j+2][i][1][i] = - (p_l + p_soh)
            x[7*j+3][i][0][i] = - (p_l + p_soh)
            x[7*j+3][i][1][i] = - p_l

            x[7*j+4][i][0][i] = p_l
            x[7*j+4][i][1][i] = p_l + p_soh
            x[7*j+5][i][0][i] = p_l + p_soh
            x[7*j+5][i][1][i] = p_l
            x[7*j+6][i][0][i] = - p_soh
            x[7*j+6][i][1][i] = p_soh

    x = x/(p_soh*p_soh*(p_soh+2*p_l))
    return x

def almost_same(a,b):
    if np.abs(a-b)<=5e-5:
        return True
    else:
        return False


def direction_finding(act_user, t, load_matrix, operator_action, user_action) :

    follower_gradient_matrix = follower_constraints_derivative_matrix(act_user, t)
    follower_constraints_value = follower_constraint_value(act_user, t, load_matrix, operator_action, user_action)
    d = cp.Variable(1)
    r_s = cp.Variable((act_user, t))
    r_b = cp.Variable((act_user, t))
    #r = cp.Variable((2, act_user, t))
    v = cp.Variable((2, t))
    g = cp.Variable((4 + 3 * act_user, t))

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, t, load_matrix, operator_action, user_action)

    c1 = x_s[0] - x_b[0] + load_active[0]

    objective = d

    constraints = []

    load_total = np.sum(l, axis=0) + np.sum(load_passive, axis=0)
    constraints += [cp.sum(cp.multiply(2*p_tax*p_s + 2*p_l/(p_soh+2*p_l) * np.ones((act_user, 1)) @ load_total.reshape(1, -1), r_s))
                    + cp.sum(cp.multiply(2*p_tax*p_b + 2*p_l/(p_soh+2*p_l) * np.ones((act_user, 1)) @ load_total.reshape(1, -1), r_b))
                    + cp.sum(cp.multiply(2*p_l/(p_soh+2*p_l) * load_total, v[0]))
                    - cp.sum(cp.multiply(2*p_l/(p_soh+2*p_l) * load_total, v[1])) <= d]

    for i in range(t):
        for j in range(act_user):
            constraints += [-r_s[j, i] <= p_s[j, i] + d]
            constraints += [-r_b[j, i] <= p_b[j, i] + d]
            constraints += [r_s[j, i] - r_b[j, i] <= - p_s[j, i] + p_b[j, i] + d]

    for i in range(t):
        constraints += [2*act_user*act_user/(p_soh*(p_soh+2*p_l)) * ((p_l+p_soh)*v[0][i]+p_l*v[1][i])
                        - (2*act_user-1)/(p_soh*(p_soh+2*p_l))*(p_l*cp.sum(r_s[:, i])-(p_soh+p_l)*cp.sum(r_b[:, i]))
                        + cp.sum(cp.multiply(follower_gradient_matrix[:, :, 0, i], g)) == 0]

        constraints += [2*act_user*act_user/(p_soh*(p_soh+2*p_l)) * (p_l*v[0][i]+(p_l+p_soh)*v[1][i]) +
                        - (2*act_user-1)/(p_soh*(p_soh+2*p_l))*(p_l*cp.sum(r_s[:, i])-(p_soh+p_l)*cp.sum(r_b[:, i]))
                        + cp.sum(cp.multiply(follower_gradient_matrix[:, :, 1, i], g)) == 0]

    for i in range(4+3*act_user):
        for j in range(t):
            if almost_same(follower_constraints_value[i, j], 0):
                constraints += [g[i, j] >= 0]
                constraints += [cp.sum(cp.multiply(follower_gradient_matrix[i, j], v)) == 0]
            else:
                constraints += [g[i, j] == 0]
                constraints += [cp.sum(cp.multiply(follower_gradient_matrix[i, j], v)) <= - follower_constraints_value[i, j] + d]

    constraints += [cp.sum(cp.power(cp.vstack([r_s, r_b]), 2)) <= 1]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    result = prob.solve(solver='ECOS')

    r = np.array([r_s.value, r_b.value])
    return result, d.value, r, v.value, g.value

def direction_finding_indiv(act_user, t, load_matrix, operator_action, user_action) :

    follower_gradient_matrix = follower_constraints_derivative_matrix(act_user, t)
    follower_constraints_value = follower_constraint_value(act_user, t, load_matrix, operator_action, user_action)
    d = cp.Variable(1)
    r_s = cp.Variable((act_user, t))
    r_b = cp.Variable((act_user, t))
    #r = cp.Variable((2, act_user, t))
    v = cp.Variable((2, t))
    g = cp.Variable((7 * act_user, t))

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, t, load_matrix, operator_action, user_action)

    c1 = x_s[0] - x_b[0] + load_active[0]

    objective = d

    constraints = []

    load_total = np.sum(l, axis=0) + np.sum(load_passive, axis=0)
    constraints += [cp.sum(cp.multiply(2*p_tax*p_s + 2*p_l/(p_soh+2*p_l) * np.ones((act_user, 1)) @ load_total.reshape(1, -1), r_s))
                    + cp.sum(cp.multiply(2*p_tax*p_b + 2*p_l/(p_soh+2*p_l) * np.ones((act_user, 1)) @ load_total.reshape(1, -1), r_b))
                    + cp.sum(cp.multiply(2*p_l/(p_soh+2*p_l) * load_total, v[0]))
                    - cp.sum(cp.multiply(2*p_l/(p_soh+2*p_l) * load_total, v[1])) <= d]

    for i in range(t):
        for j in range(act_user):
            constraints += [-r_s[j, i] <= p_s[j, i] + d]
            constraints += [-r_b[j, i] <= p_b[j, i] + d]
            constraints += [r_s[j, i] - r_b[j, i] <= - p_s[j, i] + p_b[j, i] + d]

    for i in range(t):
        constraints += [2*act_user*act_user/(p_soh*(p_soh+2*p_l)) * ((p_l+p_soh)*v[0][i]+p_l*v[1][i])
                        - (2*act_user-1)/(p_soh*(p_soh+2*p_l))*(p_l*cp.sum(r_s[:, i])-(p_soh+p_l)*cp.sum(r_b[:, i]))
                        + cp.sum(cp.multiply(follower_gradient_matrix[:, :, 0, i], g)) == 0]

        constraints += [2*act_user*act_user/(p_soh*(p_soh+2*p_l)) * (p_l*v[0][i]+(p_l+p_soh)*v[1][i]) +
                        - (2*act_user-1)/(p_soh*(p_soh+2*p_l))*(p_l*cp.sum(r_s[:, i])-(p_soh+p_l)*cp.sum(r_b[:, i]))
                        + cp.sum(cp.multiply(follower_gradient_matrix[:, :, 1, i], g)) == 0]

    for i in range(4+3*act_user):
        for j in range(t):
            if almost_same(follower_constraints_value[i, j], 0):
                constraints += [g[i, j] >= 0]
                constraints += [cp.sum(cp.multiply(follower_gradient_matrix[i, j], v)) == 0]
            else:
                constraints += [g[i, j] == 0]
                constraints += [cp.sum(cp.multiply(follower_gradient_matrix[i, j], v)) <= - follower_constraints_value[i, j] + d]

    constraints += [cp.sum(cp.power(cp.vstack([r_s, r_b]), 2)) <= 1]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    result = prob.solve(solver='ECOS')

    r = np.array([r_s.value, r_b.value])
    return result, d.value, r, v.value, g.value


def step_size(act_user, t, load_matrix, operator_action, user_action, d, r):
    update_coef = 0.9
    s = 100
    for i in range(10000):
        #print(i)
        next_operator_action = operator_action + s * r
        result, x_s, x_b, l, taken_time = follower_action(next_operator_action, load)
        next_follower_action = np.array([x_s, x_b, l])
        update = True
        if operator_objective(act_user, t, load_matrix, next_operator_action, next_follower_action) \
                >= operator_objective(act_user, t, load_matrix, operator_action, user_action) - update_coef * s * d:
            #print("good")
            if np.any(operator_constraint_value(active_user, time_step, load_matrix, next_operator_action, next_follower_action) > 0):
                update = False
        else:
            update = False
        if update:
            return s, next_operator_action, next_follower_action
        else:
            s *= update_coef
    return 0, operator_action, user_action


def iterations(act_user, t, load):
    init = initial_point(act_user, t)
    a_o = init
    result, x_s, x_b, l, taken_time = follower_action(a_o, load)
    a_f = np.array([x_s, x_b, l])

    print(operator_objective(act_user, t, load, a_o, a_f))
    print(par(act_user, t, load, a_o, a_f))
    epsilon = 1e-8

    for i in range(max_iter):
        print(i)
        print("dir")
        result, d, r, v, g = direction_finding(act_user, t, load, a_o, a_f)
        print("d :", result)
        print("step")
        b, _, _ = step_size(act_user, t, load, a_o, a_f, d, r)
        a_o = a_o + b * r
        print("step size : ", b)
        print("follower")
        result, x_s, x_b, l, taken_time = follower_action(a_o, load)
        a_f = np.array([x_s, x_b, l])
        print("operator objective :", operator_objective(act_user, t, load, a_o, a_f))
        print("par                :", par(act_user, t, load, a_o, a_f))
        #print("user buy           :", l)
        if b < epsilon:
            break

    return init, a_o, a_f

load = np.random.random((total_user, time_step))
load[0] *= 5
load[2] *= 3
#np.save("load.npy", load)
load = np.load("load.npy", allow_pickle=True)

print(load)
#a_o = initial_point(active_user, time_step)
#result, x_s, x_b, l, taken_time = follower_action(a_o, load)
#a_f = np.array([x_s, x_b, l])
#d, _, r, v, g = direction_finding(active_user, time_step, load, a_o, a_f)
#b = step_size(active_user, time_step, load, a_o, a_f, d, r)

init, a_o, a_f = iterations(active_user, time_step, load)


