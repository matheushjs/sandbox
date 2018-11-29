import numpy as np
import pandas as pd
from scipy.stats import norm

normdown = norm.ppf(0.025)
normup = -normdown
samples = 100

for fname in ["raw.out", "war.out", "raw_war.out"]:
    data = pd.read_csv(fname, header=None, names=["colA", "colB"])

    print(fname)

    inst = np.array(data[data["colB"] == "instructions"]["colA"])
    cycl = np.array(data[data["colB"] == "cycles"]["colA"])
    ipc = inst / cycl
    print("IPC: {} +- {}".format(ipc.mean(), ipc.std()))
    print("IPC: [{}, {}]".format(normdown * ipc.std()/np.sqrt(samples) + ipc.mean(), normup * ipc.std()/np.sqrt(samples) + ipc.mean()))

    seconds = data[data["colB"] == "seconds"]["colA"]
    print("Seconds: {} +- {}".format(seconds.mean(), seconds.std()))
    print("Seconds: [{}, {}]".format(normdown * seconds.std()/np.sqrt(samples) + seconds.mean(), normup * seconds.std()/np.sqrt(samples) + seconds.mean()))

    miss = data[data["colB"] == "cache-misses"]["colA"]
    print("Miss: {} +- {}".format(miss.mean(), miss.std()))
    print("Miss: [{}, {}]".format(normdown * miss.std()/np.sqrt(samples) + miss.mean(), normup * miss.std()/np.sqrt(samples) + miss.mean()))

    print()
