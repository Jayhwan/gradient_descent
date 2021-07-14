import numpy as np
import matplotlib.pyplot as plt

our_model = np.load("our_model.npy", allow_pickle=True)
our_model_indiv = np.load("our_model_indiv.npy", allow_pickle=True)
print(np.shape(our_model))
print(np.shape(our_model_indiv))

x = our_model

p_l = 1
ec = np.zeros(len(x))
par = np.zeros(len(x))
for i in range(len(x)):
    [a_o, a_f, load] = x[i]
    if i == 0:
        ec[i] = p_l * np.sum(np.power(np.sum(load, axis=0), 2))
        par[i] = np.max(np.sum(load, axis=0))/np.average(np.sum(load, axis=0))
    else:
        [x_s, x_b, l] = a_f
        ec[i] = p_l * np.sum(np.power(np.sum(l, axis=0) + np.sum(load[i:], axis=0), 2))
        par[i] = np.max(np.sum(l, axis=0) + np.sum(load[i:], axis=0))/np.average(np.sum(l, axis=0) + np.sum(load[i:], axis=0))
plt.plot(range(len(x)), ec)
plt.show()