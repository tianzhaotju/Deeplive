import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


f_BBA = './result/QoE/QoE_al/BBA.csv'
f_RBA = './result/QoE/QoE_al/RBA.csv'
f_DYNAMIC = './result/QoE/QoE_al/DYNAMIC.csv'
f_Pensieve = './result/QoE/QoE_al/Pensieve.csv'
f_PDDQN = './result/QoE/QoE_al/PDDQN.csv'
f_Offline_optimal = './result/QoE/QoE_al/Offline optimal.csv'


def init(F):
    A = []
    with open(F,encoding = 'utf-8') as f:
        data = np.loadtxt(f,str,delimiter = ",")
        for i in data:
            A.append(float(i[2]))
    A = np.array(A)
    print(max(A))
    res = stats.relfreq(A, numbins=1000,defaultreallimits=(0,2.5))
    x = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size,res.frequency.size)
    y = np.cumsum(res.frequency)
    return x,y


RBA_x,RBA_y = init(f_RBA)
BBA_x,BBA_y = init(f_BBA)
DYNAMIC_x,DYNAMIC_y = init(f_DYNAMIC)
Pensieve_x,Pensieve_y = init(f_Pensieve)
PDDQN_x,PDDQN_y = init(f_PDDQN)
Offline_optimal_x,Offline_optimal_y = init(f_Offline_optimal)

plt.plot(RBA_x,RBA_y,color='magenta',linestyle='--',label='RBA')
plt.plot(BBA_x,BBA_y,color='darkviolet',linestyle='--',label='BBA')
plt.plot(DYNAMIC_x,DYNAMIC_y,color='orange',linestyle='--',label='DYNAMIC')
plt.plot(Pensieve_x,Pensieve_y,color='springgreen',linestyle='--',label='Pensieve')
plt.plot(PDDQN_x,PDDQN_y,color='red',linestyle='-',label='PDDQN-R')
plt.plot(Offline_optimal_x,Offline_optimal_y,color='grey',linestyle='--',label='Offline optimal')
plt.xlabel("Average QoE")
plt.ylabel("CDF")
plt.legend()
plt.savefig('')
plt.show()
