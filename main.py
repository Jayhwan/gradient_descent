import numpy as np
import cvxpy as cp
import time

total_user = 1
t = 2

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



def x_s_value(a_o, a_f, load):
    x_s = np.zeros(t)
    for i in range(t):
        x_s[i] = ((p_soh+p_l)*a_o[2*i]-p_l*a_o[2*i+1]-p_l*a_f[2*i]-(p_soh+p_l)*a_f[2*i+1]-p_soh*p_l*load[i])/(p_soh*(p_soh+2*p_l))
    return x_s

def x_b_value(a_o, a_f, load):
    x_b = np.zeros(t)
    for i in range(t):
        x_b[i] = (p_l*a_o[2*i]-(p_soh+p_l)*a_o[2*i+1]-(p_soh+p_l)*a_f[2*i]-p_l*a_f[2*i+1]+p_soh*p_l*load[i])/(p_soh*(p_soh+2*p_l))
    return x_b

def l_value(a_o, a_f, load):
    l = np.zeros(t)
    for i in range(t):
        l[i] = (a_o[2*i]+a_o[2*i+1]+a_f[2*i]-a_f[2*i+1]+p_soh*load[i])/(p_soh+2*p_l)
    return l

def operator_objective(a_o, a_f, load):
    return -p_l*(np.sum(np.power(np.array(l_value(a_o,a_f,load)),2)))-p_tax*np.sum(np.power(a_o,2))

def user_objective(a_o, a_f, load):
    x_s = x_s_value(a_o,a_f,load)
    x_b = x_b_value(a_o,a_f,load)
    l = l_value(a_o, a_f, load)

    return a_o[0]*x_s[0]-a_o[1]*x_b[0]+a_o[2]*x_s[1]-a_o[3]*x_b[1]-p_l*(np.sum(np.power(l,2)))-p_soh*(np.sum(np.power(x_s,2)+np.power(x_b,2)))

def follower_action(a_o, load):
    a_f = cp.Variable(4)
    x_s_0 = ((p_soh+p_l)*a_o[0]-p_l*a_o[1]-p_l*a_f[0]-(p_soh+p_l)*a_f[1]-p_soh*p_l*load[0])/(p_soh*(p_soh+2*p_l))
    x_s_1 = ((p_soh+p_l)*a_o[2]-p_l*a_o[3]-p_l*a_f[2]-(p_soh+p_l)*a_f[3]-p_soh*p_l*load[1])/(p_soh*(p_soh+2*p_l))
    x_b_0 = (p_l*a_o[0]-(p_soh+p_l)*a_o[1]-(p_soh+p_l)*a_f[0]-p_l*a_f[1]+p_soh*p_l*load[0])/(p_soh*(p_soh+2*p_l))
    x_b_1 = (p_l*a_o[2]-(p_soh+p_l)*a_o[3]-(p_soh+p_l)*a_f[2]-p_l*a_f[3]+p_soh*p_l*load[1])/(p_soh*(p_soh+2*p_l))
    l_0 = x_s_0-x_b_0+load[0]
    l_1 = x_s_1-x_b_1+load[1]
    objective = a_o[0]*x_s_0-a_o[1]*x_b_0+a_o[2]*x_s_1-a_o[3]*x_b_1-p_l*(cp.power(l_0,2)+cp.power(l_1,2))-p_soh*(cp.power(x_s_0,2)+cp.power(x_b_0,2)+cp.power(x_s_1,2)+cp.power(x_b_1,2))
    constraints = []
    constraints += [-beta_s*x_s_0+beta_b*x_b_0<=0]
    constraints += [beta_s*x_s_0-beta_b*x_b_0-q_max<=0]
    constraints += [alpha*(-beta_s*x_s_0+beta_b*x_b_0)-beta_s*x_s_1+beta_b*x_b_1<=0]
    constraints += [-alpha*(-beta_s*x_s_0+beta_b*x_b_0)+beta_s*x_s_1-beta_b*x_b_1-q_max<=0]
    constraints += [x_s_0 - c_max<=0]
    constraints += [x_s_1 - c_max <= 0]
    constraints += [x_b_0 - c_min <= 0]
    constraints += [x_b_1 - c_min <= 0]
    constraints += [-l_0<=0, -l_1<=0, -x_s_0<=0, -x_s_1<=0, -x_b_0<=0, -x_s_1<=0]

    prob = cp.Problem(cp.Maximize(objective), constraints)
    result = prob.solve(solver = 'ECOS')

    #print("x_s : ", x_s_0.value, x_s_1.value)
    #print("x_b : ", x_b_0.value, x_b_1.value)
    #print("l   : ", l_0.value, l_1.value)
    #print("C   : ", a_f.value)

    return a_f.value

def almost_same(a,b):
    if np.abs(a-b)<=1e-5:
        return True
    else:
        return False


def follower_constraints_func_values(a_o, a_f, load):
    x_s = x_s_value(a_o, a_f, load)
    x_b = x_b_value(a_o, a_f, load)
    l = l_value(a_o, a_f, load)
    constraints = np.zeros(7*t)
    constraints[0] = -beta_s*x_s[0]+beta_b*x_b[0]
    constraints[1] = beta_s*x_s[0]-beta_b*x_b[0]-q_max
    constraints[2] = alpha*(-beta_s*x_s[0]+beta_b*x_b[0])-beta_s*x_s[1]+beta_b*x_b[1]
    constraints[3] = alpha*(beta_s*x_s[0]-beta_b*x_b[0])+beta_s*x_s[1]-beta_b*x_b[1]-q_max
    constraints[4] = x_s[0]-c_max
    constraints[5] = x_b[0]-c_min
    constraints[6] = x_s[1]-c_max
    constraints[7] = x_b[1]-c_min
    constraints[8] = -l[0]
    constraints[9] = -x_s[0]
    constraints[10] = -x_b[0]
    constraints[11] = -l[1]
    constraints[12] = -x_s[1]
    constraints[13] = -x_b[1]
    return constraints

constraints_gradient_coefficient = np.ones((14,4))
constraints_gradient_coefficient[0] = [beta_s*p_l-beta_b*(p_soh+p_l), beta_s*(p_soh+p_l)-beta_b*p_l, 0, 0]
constraints_gradient_coefficient[1] = [-beta_s*p_l+beta_b*(p_soh+p_l), -beta_s*(p_soh+p_l)+beta_b*p_l, 0, 0]
constraints_gradient_coefficient[2] = [alpha*(beta_s*p_l-beta_b*(p_soh+p_l)), alpha*(beta_s*(p_soh+p_l)-beta_b*p_l), beta_s*p_l-beta_b*(p_soh+p_l), beta_s*(p_soh+p_l)-beta_b*p_l]
constraints_gradient_coefficient[3] = [alpha*(-beta_s*p_l+beta_b*(p_soh+p_l)), alpha*(-beta_s*(p_soh+p_l)+beta_b*p_l), -beta_s*p_l+beta_b*(p_soh+p_l), -beta_s*(p_soh+p_l)+beta_b*p_l]
constraints_gradient_coefficient[4] = [-p_l, -p_l-p_soh, 0, 0]
constraints_gradient_coefficient[5] = [-p_l-p_soh, -p_l, 0, 0]
constraints_gradient_coefficient[6] = [0, 0, -p_l, -p_l-p_soh]
constraints_gradient_coefficient[7] = [0, 0, -p_l-p_soh, -p_l]
constraints_gradient_coefficient[8] = [-p_soh, p_soh, 0, 0]
constraints_gradient_coefficient[9] = [p_l, p_l+p_soh, 0, 0]
constraints_gradient_coefficient[10] = [p_l+p_soh, p_l, 0, 0]
constraints_gradient_coefficient[11] = [0, 0, -p_soh, p_soh]
constraints_gradient_coefficient[12] = [0, 0, p_l, p_l+p_soh]
constraints_gradient_coefficient[13] = [0, 0, p_l+p_soh, p_l]
constraints_gradient_coefficient = constraints_gradient_coefficient*(1/(p_soh*(p_soh+2*p_l)))
#x = np.ones(4)
#print(np.sum(np.multiply(constraints_gradient_coefficient, x), axis=1))

#print(follower_constraints_func_values(a_o, a_f, load))

def direction_finding(a_o, a_f, load):
    d = cp.Variable(1)
    r = cp.Variable((4,1))
    v = cp.Variable((4,1))
    g = cp.Variable((14,1))

    x_s = x_s_value(a_o, a_f, load)
    x_b = x_b_value(a_o, a_f, load)
    l = l_value(a_o, a_f, load)

    objective = d

    constraints = []

    constraints += [2*p_l*(np.array([l[0],l[0],l[1],l[1]])@r)/(p_soh+2*p_l)+2*p_tax*(a_o@r)+2*p_l*(np.array([l[0],-l[0],l[1],-l[1]])@v)/(p_soh+2*p_l)<=d]

    A = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[1,-1,0,0],[0,0,1,-1]])
    b = np.array([a_o[0],a_o[1],a_o[2],a_o[3],-a_o[0]+a_o[1],-a_o[2]+a_o[3]])
    d_vector = cp.vstack([d,d,d,d,d,d])
    c = np.zeros((6,1))
    for i in range(6):
        c[i][0]=b[i]
    constraints += [A@r <= c+d_vector]

    B = 2*np.array([[p_soh+p_l,p_l,0,0],[p_l,p_soh+p_l,0,0],[0,0,p_soh+p_l,p_l],[0,0,p_l,p_soh+p_l]])
    C = np.array([[p_l,-p_soh-p_l,0,0],[p_soh+p_l,-p_l,0,0],[0,0,p_l,-p_soh-p_l],[0,0,p_soh+p_l,-p_l]])
    constraints += [B@v-C@r+constraints_gradient_coefficient.T@g==0]

    fcfv = follower_constraints_func_values(a_o, a_f, load)

    for i in range(14):
        if almost_same(fcfv[i], 0):
            constraints += [g[i]>=0]
            constraints += [constraints_gradient_coefficient[i]@v==0]
            #print("nonregular")
        else:
            constraints += [g[i]==0]
            constraints += [constraints_gradient_coefficient[i]@v <= -fcfv[i] + d]
            #print("regular")
    constraints += [cp.norm(r)<=1]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    result = prob.solve(solver='ECOS')
    #print("d : ", d.value)
    #print("r : ", np.array(r.value).T)
    #print("v : ", np.array(v.value).T)
    #print("g : ", np.array(g.value).T)
    return result, d.value, r.value, v.value, g.value

def step_size(a_o, a_f, load, d, r):

    updating_coeff = 0.1
    s = updating_coeff
    for i in range(10000):
        next_operator_action = a_o + s*r
        next_follower_action = follower_action(next_operator_action, load)
        update = True
        if operator_objective(next_operator_action, next_follower_action, load) >= operator_objective(a_o, a_f, load)-updating_coeff*s*d:
            for j in range(4):
                if -next_operator_action[j]>0:
                    update=False
            if next_operator_action[0]-next_operator_action[1]>0:
                update=False
            if next_operator_action[2]-next_operator_action[3]>0:
                update=False
        else:
            update = False
        if update:
            return s
        else:
            s *= updating_coeff

a_o = np.array([3,3,1,1]) #
load = [2, 4]
a_f = np.array(follower_action(a_o, load))

print("operator action    : ", a_o)
print("user grid buy      : ", l_value(a_o, a_f, load))
print("required load      : ", load)
print("operator objective : ", operator_objective(a_o, a_f, load))
print("user objective     : ", user_objective(a_o, a_f, load))

for i in range(1000):
    d, _, r, v, g = direction_finding(a_o, a_f, load)
    r = np.reshape(r, 4)
    b = step_size(a_o, a_f, load, d, r)
    a_o = np.array(a_o+b*r)
    a_f = np.array(follower_action(a_o, load))
    print("######### iter ", i, "##########")
    print("d                  : ", d)
    print("direction          : ", r)
    print("step size          : ", b)
    print("operator action    : ", a_o)
    print("user grid buy      : ", l_value(a_o, a_f, load))
    print("operator objective : ", operator_objective(a_o, a_f, load))
    print("user objective     : ", user_objective(a_o, a_f, load))