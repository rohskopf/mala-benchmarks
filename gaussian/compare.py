import numpy as np

# Loop through 4 ranks and check:
for r in range(4):
    cpu = np.load(f"test_cpu_{r}.npy")
    kokkos = np.load(f"test_kokkos_{r}.npy")
    absdiff = abs(cpu - kokkos)
    maxdiff = np.max(absdiff)
    print(maxdiff)

print(np.shape(cpu))
print(np.shape(kokkos))