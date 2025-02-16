import emcee
import numpy as np


def get_argmax(arr):
    counts, bin = np.histogram(arr, bins=10000)
    return bin[np.argmax(counts)]


samples = emcee.backends.HDFBackend("state.h5", read_only=True).get_chain(flat=True)
print(
    "["
    + ",\n".join([f"{get_argmax(samples[:, i])}" for i in range(samples.shape[1])])
    + "],"
)
print("[" + ",\n".join([f"{el}" for el in samples.mean(axis=0)]) + "],")
