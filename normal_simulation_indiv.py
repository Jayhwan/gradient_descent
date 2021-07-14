import numpy as np
import matplotlib.pyplot as plt
from full import *
from complete import *


def find_good_initial_point(act_user, t, load_matrix):
    max_scatter = 300
    max_obj = -np.inf
    min_par = np.inf
    best_obj_a_o = None
    best_par_a_o = None
    for i in range(max_scatter):
        a_o = .5*np.random.random((2, act_user, t))+1*np.ones((2, act_user, t))
        a_o[0] = np.minimum(a_o[0], a_o[1])
        a_o[1] = np.maximum(a_o[0], a_o[1])
        result, x_s, x_b, l, time = follower_action(act_user, t, a_o, load_matrix)
        a_f = np.array([x_s, x_b, l])
        if result == - np.inf:
            print("fail")
            continue
        else:
            obj = operator_objective(act_user, t, load_matrix, a_o, a_f)
            pa = par(act_user, t, load_matrix, a_o, a_f)
            print(obj, pa)
            if obj > max_obj:
                best_obj_a_o = a_o
                best_obj_index = i
                max_obj = obj
            if pa < min_par:
                best_obj_a_o = a_o
                best_par_index = i
                min_par = pa
    print(max_obj, min_par, best_obj_index, best_par_index)
    return best_obj_a_o, best_par_a_o


#active_user = 100
#load = np.load("load_123.npy", allow_pickle=True)[:total_user, :time_step]
#a_o = np.ones((2, active_user, time_step))
#result, x_s, x_b, l, time = follower_action(active_user, time_step, a_o, load)
#a_f = np.array([x_s, x_b, l])
#print(operator_objective(active_user, time_step, load, a_o, a_f), par(active_user, time_step, load, a_o, a_f))

#best_obj_a_o, best_par_a_o = find_good_initial_point(active_user, time_step, load)
#np.save("best_a_o.npy", [best_obj_a_o, best_par_a_o])
#a_o = best_obj_a_o

load = np.load("load_123.npy", allow_pickle=True)[:total_user, :time_step]

best = []
exp_our_model = []
for i in range(101):
    print("active user number :", i)

    if i==0:
        a_o = None
        a_f = None
        best_a_o = None
    else:
        best_a_o, _ = find_good_initial_point(i, time_step, load)
        _, a_o, a_f = iterations(i, time_step, load, best_a_o)
    best += [best_a_o]
    exp_our_model += [[a_o, a_f, load]]
    np.save("best_action.npy", best)
    np.save("our_model_indiv.npy", exp_our_model)