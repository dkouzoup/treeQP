
import numpy as np
import matplotlib.pyplot as plt

# define number of states
nx = 4

# read results
with open("xMPC.txt", "r") as f:
    xMPC = map(float, f)

with open("uMPC.txt", "r") as f:
    uMPC = map(float, f)

# define rest of dimensions
mpc_steps = len(xMPC)/nx - 1
n_masses = nx/2
nu = len(uMPC)/mpc_steps

# plot closed loop trajectories
for i in range(n_masses):
    plt.subplot(3,1,1)
    plt.plot(xMPC[i::nx])
    plt.ylabel("position")
    plt.title("closed loop trajectories")
    plt.subplot(3,1,2)
    plt.plot(xMPC[i+n_masses::nx])
    # plt.xlabel("MPC steps")
    plt.ylabel("velocity")

    for j in range(nu):
        plt.subplot(3,1,3)
        plt.plot(uMPC[j::nu])
        plt.xlabel("MPC steps")
        plt.ylabel("controls")

plt.show()

