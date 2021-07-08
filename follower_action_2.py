import numpy as np
import cvxpy as cp
import time

total_user = 100
active_user = 100
time_step = 24

p_l = 1
p_soh = 1

alpha = 1#0.9956
beta_s = 1#0.99
beta_b = 1#1.01

p_tax = 0.001#0.0000001

q_max = 120.
q_min = 0.

c_max = 40.
c_min = 40.

def follower_action_ni(a_o, load, indiv=False, init=None):

    if total_user == active_user:
        load_active = load
        load_passive = np.zeros((1, time_step))
    else:
        load_active = load[:active_user, :]
        load_passive = load[active_user:, :]
    gamma = 1

    #a = 1
    #x_s_tmp = np.array([[0], [a]])  # , [1/3, 0]]) # 2*load_active #np.zeros((active_user, time_step))
    #x_b_tmp = np.array([[a], [0]])  # , [0, 1/3]]) # load_active #np.zeros((active_user, time_step))
    #l_tmp = [[1 - a], [a]]  # , [1/3, 2/3]]#load_active
    x_s_tmp = np.zeros((active_user, time_step))
    x_b_tmp = np.zeros((active_user, time_step))
    l_tmp = load_active
    if init == None:
        x_s_tmp = np.zeros((active_user, time_step))
        x_b_tmp = np.zeros((active_user, time_step))
        l_tmp = load_active
    #else:
    #    [x_s_tmp, x_b_tmp, l_tmp] = init
    for i in range(20000):
        x_s = cp.Variable((active_user, time_step))
        x_b = cp.Variable((active_user, time_step))

        l = x_s - x_b + load_active

        #new_utility = cp.sum(cp.multiply(a_o[0], x_s) - cp.multiply(a_o[1], x_b))
        new_utility = 0
        prev_utility = 0
        for k in range(active_user):
            new_utility += cp.sum(cp.multiply(a_o[0][k], x_s[k]) - cp.multiply(a_o[1][k], x_b[k]))
            new_utility -= p_l * cp.sum(cp.power(l[k], 2) + cp.multiply(l[k], np.sum(l_tmp, axis=0) - l_tmp[k])) ###### 일반적으론 패시브유저도 고려해야함!!
            new_utility -= p_soh * cp.sum(cp.power(x_s[k], 2) + cp.multiply(x_s[k], np.sum(x_s_tmp, axis=0) - x_s_tmp[k]))
            new_utility -= p_soh * cp.sum(cp.power(x_b[k], 2) + cp.multiply(x_b[k], np.sum(x_b_tmp, axis=0) - x_b_tmp[k]))

            prev_utility += np.sum(np.multiply(a_o[0][k], x_s_tmp[k]) - np.multiply(a_o[1][k], x_b_tmp[k]))
            prev_utility -= p_l * np.sum(np.power(l_tmp[k], 2) + np.multiply(l_tmp[k], np.sum(l_tmp, axis=0) - l_tmp[k]))
            prev_utility -= p_soh * np.sum(np.power(x_s_tmp[k], 2) + np.multiply(x_s_tmp[k], np.sum(x_s_tmp, axis=0) - x_s_tmp[k]))
            prev_utility -= p_soh * np.sum(np.power(x_b_tmp[k], 2) + np.multiply(x_b_tmp[k], np.sum(x_b_tmp, axis=0) - x_b_tmp[k]))
        #prev_utility = np.sum(np.multiply(a_o[0], x_s_tmp) - np.multiply(a_o[1], x_b_tmp))\
        #          - p_l * np.sum(np.power(np.sum(l_tmp, axis=0), 2) + np.multiply(np.sum(l_tmp, axis=0), np.sum(load_passive, axis=0))) \
        #          - p_soh * np.sum(np.power(np.sum(x_s_tmp, axis=0), 2) + np.power(np.sum(x_b_tmp, axis=0), 2))

        difference = gamma*cp.sum(cp.power(cp.vstack([x_s, x_b]) - np.vstack([x_s_tmp, x_b_tmp]), 2))/2

        utility = new_utility - prev_utility - difference

        constraints = []
        constraints += [l >= 0]
        constraints += [x_s >= 0]
        constraints += [x_b >= 0]

        ess_matrix = np.fromfunction(np.vectorize(lambda i, j: 0 if i < j else np.power(alpha, i - j)),
                                     (time_step, time_step), dtype=float)
        q_ess = q_min * np.fromfunction(np.vectorize(lambda i, j: np.power(alpha, i - j + 1)), (time_step, 1), dtype=float)\
                + beta_s * ess_matrix @ cp.sum(x_s, axis=0).T - beta_b * ess_matrix @ cp.sum(x_b, axis=0).T

        if time_step == 1:
            for j in range(active_user):
                constraints += [- x_s[j][0] + x_b[j][0] - np.sum(x_s_tmp) + x_s_tmp[j][0] + np.sum(x_b_tmp) - x_b_tmp[j][0] <= 0]
        else:
            constraints += [q_min <= q_ess, q_ess <= q_max]

        constraints += [cp.sum(x_s, axis=0) <= c_max]
        constraints += [cp.sum(x_b, axis=0) <= c_min]

        prob = cp.Problem(cp.Maximize(utility), constraints)

        start = time.time()

        result = prob.solve(solver = 'ECOS')

        end = time.time()
        print(i)
        print("### ni ###")
        print(prev_utility)
        print(utility.value)
        print(difference.value)
        print("x_s :", x_s.value.reshape(active_user))
        print("x_b :", x_b.value.reshape(active_user))
        print("l   :", l.value.reshape(active_user))
        x_s_tmp = x_s.value
        x_b_tmp = x_b.value
        l_tmp = l.value

    return result, l.value, end - start, utility.value


def follower_action_ni_dist(a_o, load, indiv=False):

    if total_user == active_user:
        load_active = load
        load_passive = np.zeros((1, time_step))
    else:
        load_active = load[:active_user, :]
        load_passive = load[active_user:, :]
    gamma = 0
    a = 0.3000
    x_s_tmp = np.array([[0], [a]])  # , [1/3, 0]]) # 2*load_active #np.zeros((active_user, time_step))
    x_b_tmp = np.array([[a], [0]])  # , [0, 1/3]]) # load_active #np.zeros((active_user, time_step))
    l_tmp = [[1-a], [a]]  # , [1/3, 2/3]]#load_active

    #x_s_tmp = 2*load_active #np.zeros((active_user, time_step))
    #x_b_tmp = load_active #np.zeros((active_user, time_step))
    #l_tmp = load_active

    x_s_new = np.zeros((active_user, time_step))
    x_b_new = np.zeros((active_user, time_step))
    l_new = x_s_new - x_b_new + load_active


    for i in range(10000):
        total_utility = 0
        prev_utility = 0
        for k in range(active_user):
            x_s = cp.Variable((active_user, time_step))
            x_b = cp.Variable((active_user, time_step))
            l = x_s - x_b + load_active

            new_utility = 0
            new_utility += cp.sum(cp.multiply(a_o[0][k], x_s[k])) - cp.multiply(a_o[1][k], x_b[k])
            new_utility -= p_l * cp.sum(cp.power(l[k], 2) + cp.multiply(l[k], np.sum(l_tmp, axis=0) - l_tmp[k]))
            new_utility -= p_soh * cp.sum(cp.power(x_s[k], 2) + cp.multiply(x_s[k], np.sum(x_s_tmp, axis=0) - x_s_tmp[k]))
            new_utility -= p_soh * cp.sum(cp.power(x_b[k], 2) + cp.multiply(x_b[k], np.sum(x_b_tmp, axis=0) - x_b_tmp[k]))

            prev_utility += np.sum(np.multiply(a_o[0][k], x_s_tmp[k]) - np.multiply(a_o[1][k], x_b_tmp[k]))
            prev_utility -= p_l * np.sum(np.power(l_tmp[k], 2) + np.multiply(l_tmp[k], np.sum(l_tmp, axis=0) - l_tmp[k]))
            prev_utility -= p_soh * np.sum(np.power(x_s_tmp[k], 2) + np.multiply(x_s_tmp[k], np.sum(x_s_tmp, axis=0) - x_s_tmp[k]))
            prev_utility -= p_soh * np.sum(np.power(x_b_tmp[k], 2) + np.multiply(x_b_tmp[k], np.sum(x_b_tmp, axis=0) - x_b_tmp[k]))

            difference = gamma*cp.sum(cp.power(cp.vstack([x_s[k], x_b[k]]) - np.vstack([x_s_tmp[k], x_b_tmp[k]]), 2))/2

            utility = new_utility - difference

            constraints = []
            constraints += [l[k] >= 0, x_s[k] >= 0, x_b[k] >= 0]

            ess_matrix = np.fromfunction(np.vectorize(lambda i, j: 0 if i < j else np.power(alpha, i - j)),
                                         (time_step, time_step), dtype=float)
            q_ess = q_min * np.fromfunction(np.vectorize(lambda i, j: np.power(alpha, i - j + 1)), (time_step, 1), dtype=float)\
                    + beta_s * ess_matrix @ cp.sum(x_s, axis=0).T - beta_b * ess_matrix @ cp.sum(x_b, axis=0).T

            constraints += [q_min <= q_ess, q_ess <= q_max]
            constraints += [cp.sum(x_s, axis=0) <= c_max]
            constraints += [cp.sum(x_b, axis=0) <= c_min]

            if k!=0:
                constraints += [x_s[:k,:] == x_s_tmp[:k,:]]
                constraints += [x_b[:k,:] == x_b_tmp[:k,:]]
            if k!=active_user-1:
                constraints += [x_s[k+1:,:] == x_s_tmp[k+1:,:]]
                constraints += [x_b[k+1:,:] == x_b_tmp[k+1:,:]]

            prob = cp.Problem(cp.Maximize(utility), constraints)

            start = time.time()

            result = prob.solve(solver = 'ECOS')

            end = time.time()

            x_s_new[k,:] = x_s.value[k,:]
            x_b_new[k,:] = x_b.value[k,:]
            l_new[k,:] = l.value[k,:]
            total_utility += utility.value

        print("###",str(i),"###")
        print(prev_utility)
        print(total_utility - prev_utility)
        print(l.value)

        x_s_tmp = x_s_new
        x_b_tmp = x_b_new
        l_tmp = l_new

    return result, l.value, end - start

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
    print("# kkt #")
    print("x_s :", x_s.value)#.reshape(active_user))
    print("x_b :", x_b.value)#.reshape(active_user))
    print("l   :", l.value)#.reshape(active_user))

    return result, x_s.value, x_b.value, l.value, end - start


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
              - p_soh * cp.sum(cp.power(cp.sum(x_s, axis=0), 2) + cp.power(cp.sum(x_b, axis=0), 2))

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

    #print(a_o[0]-2*p_soh*x_s.value)
    #print(a_o[1]+2*p_soh*x_b.value)
    print("# No KKT #")
    print(x_s.value)
    print(x_b.value)
    print(l.value)
    return result, x_s.value, x_b.value, l.value, end - start


a_o_1 = np.random.random((active_user, time_step))
a_o_2 = np.random.random((active_user, time_step))

a_o = np.zeros((2, active_user, time_step))
#a_o_s = np.zeros((active_user, time_step))
#a_o_b = np.zeros((active_user, time_step))
a_o[0] = np.minimum(a_o_1, a_o_2)
a_o[1] = np.maximum(a_o_1, a_o_2)
#a_o_s = np.minimum(a_o_1, a_o_2)
#a_o_b = np.maximum(a_o_1, a_o_2)

#a_o = 1*a_o
#print(a_o)

#a_o = np.ones((2, active_user, time_step))

load = 5*np.random.random((total_user, time_step))
#load = np.array([[0,0],[0,1]])

#a_o = np.array([[[0.1], [0.2]], [[2], [3]]])
#load = np.array([[1],[0]])
#print(load)

#print(follower_action_kkt(a_o, load))
#print(follower_action(a_o, load))

#follower_action_ni_dist(a_o, load)
#print(follower_action(a_o, load))
_, x_s, x_b, l, _ = follower_action_kkt(a_o, load)
#follower_action_ni(a_o, load, init = [x_s, x_b, l])
print("load :", load)#.reshape(active_user))
print("a_o  :", a_o)