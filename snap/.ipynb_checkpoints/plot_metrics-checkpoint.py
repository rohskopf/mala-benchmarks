import matplotlib.pyplot as plt
import numpy as np

dat1 = np.loadtxt("metrics_1_kk1.dat")
dat3 = np.loadtxt("metrics_3_kk1.dat")
dat5 = np.loadtxt("metrics_5_kk1.dat")
dat10 = np.loadtxt("metrics_10_kk1.dat")
dat20 = np.loadtxt("metrics_20_kk1.dat")

plt.plot(dat1[:,0], dat1[:,1], "o-", color="red")
plt.plot(dat3[:,0], dat3[:,1], "o-", color="#9a43da")
plt.plot(dat5[:,0], dat5[:,1], "o-", color="blue")
plt.plot(dat10[:,0], dat10[:,1], "o-", color="black")
plt.plot(dat20[:,0], dat20[:,1], "o-", color="#9f9f9f")

plt.xlabel("Num. grid points (million)")
plt.ylabel("Speed (M gridpoint-step/node-s)")

plt.xscale("log")
plt.yscale("log")

plt.grid()

natoms1 = 4*1**3
natoms3 = 4*3**3
natoms5 = 4*5**3
natoms10 = 4*10**3
natoms20 = 4*20**3
plt.legend([f"{natoms1} atoms", f"{natoms3} atoms", f"{natoms5} atoms", f"{natoms10} atoms", f"{natoms20} atoms"])

plt.savefig("metrics.png", dpi=500)
