
import numpy as np
import matplotlib.pyplot as plt

# read results
with open("n_masses.txt", "r") as f:
    n_masses = map(int, f)

with open("xMPC.txt", "r") as f:
    xMPC = map(float, f)

with open("uMPC.txt", "r") as f:
    uMPC = map(float, f)

with open("cpuTimes.txt", "r") as f:
    cpuTimes = map(float, f)

# define dimensions
mpc_steps = len(cpuTimes)
n_masses = n_masses[0]
nx = n_masses*2
nu = len(uMPC)/mpc_steps

# plot closed loop trajectories
for i in range(n_masses):
    plt.subplot(4,1,1)
    plt.plot(xMPC[i::nx])
    plt.ylabel("position")
    plt.title("closed loop trajectories")
    plt.subplot(4,1,2)
    plt.plot(xMPC[i+n_masses::nx])
    # plt.xlabel("MPC steps")
    plt.ylabel("velocity")

    for j in range(nu):
        plt.subplot(4,1,3)
        plt.step(range(mpc_steps),uMPC[j::nu])
        # plt.xlabel("MPC steps")
        plt.ylabel("controls")

#TODO(dimitris): axes from 0

cpuTimes_ms = [1e3*t for t in cpuTimes]

plt.subplot(4,1,4)
plt.plot(cpuTimes_ms)
plt.xlabel("MPC steps")
plt.ylabel("cpu time [ms]")

plt.show()

