import matplotlib.pyplot as plt
import numpy as np

# Numbered by number of GPUs.
dat1 = np.loadtxt("metrics_7_kk1_1gpus.dat")
dat2 = np.loadtxt("metrics_7_kk1_2gpus.dat")
dat4 = np.loadtxt("metrics_7_kk1_4gpus.dat")
dat8 = np.loadtxt("metrics_7_kk1_8gpus.dat")
dat40 = np.loadtxt("metrics_7_kk1_40gpus.dat")

# Single GPU data only has ~10M grid points.
plot1 = np.zeros((5, 2))
for i, ngpu in enumerate([1, 2, 4, 8, 40]):
    plot1[i,0] = ngpu
plot1[0,1] = dat1[1]
plot1[1,1] = dat2[0,1]
plot1[2,1] = dat4[0,1]
plot1[3,1] = dat8[0,1]
plot1[4,1] = dat40[0,1]

# Gather data for other plot (~100M grid points).
plot2 = np.zeros((4, 2))
for i, ngpu in enumerate([2, 4, 8, 40]):
    plot2[i,0] = ngpu
plot2[0,1] = dat2[1,1]
plot2[1,1] = dat4[1,1]
plot2[2,1] = dat8[1,1]
plot2[3,1] = dat40[1,1]

plt.plot(plot1[:,0], plot1[:,1], "o-", color="black")
plt.plot(plot2[:,0], plot2[:,1], "o-", color="#9f9f9f")
"""
plt.plot(dat3[:,0], dat3[:,1], "o-", color="#9a43da")
plt.plot(dat5[:,0], dat5[:,1], "o-", color="blue")
plt.plot(dat10[:,0], dat10[:,1], "o-", color="black")
plt.plot(dat20[:,0], dat20[:,1], "o-", color="#9f9f9f")
"""

plt.xlabel("Num. GPUs")
plt.ylabel("s/step")

plt.xscale("log")
plt.yscale("log")

plt.grid()

print(plot1)

natoms1 = 4*1**3
natoms3 = 4*3**3
natoms5 = 4*5**3
natoms10 = 4*10**3
natoms20 = 4*20**3

ngrid1 = "10M"
ngrid2 = "100M"
plt.legend([f"{ngrid1} grid points", f"{ngrid2} grid points"])

plt.savefig("metrics.png", dpi=500)
