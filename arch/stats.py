import numpy as np
import pandas as pd

for fname in ["raw.out", "war.out", "raw_war.out"]:
    data = pd.read_csv(fname, header=None, names=["colA", "colB"])

    print(fname)

    inst = np.array(data[data["colB"] == "instructions"]["colA"])
    cycl = np.array(data[data["colB"] == "cycles"]["colA"])
    ipc = inst / cycl
    print("IPC: {} +- {}".format(ipc.mean(), ipc.std()))

    seconds = data[data["colB"] == "seconds"]["colA"]
    print("Seconds: {} +- {}".format(seconds.mean(), seconds.std()))

    print()
