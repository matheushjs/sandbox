import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

labels   = ["1. Formulation", "2. Propose\nMethods", "3. Implement\nMethods", "4. Literature", "5. Articles", "6. Reports"]
times = {
    labels[0]: [(1, 7)], # (from, width), (from, width)
    labels[1]: [(2, 9)],
    labels[2]: [(2, 9)],
    labels[3]: [(1, 11)],
    labels[4]: [(8, 4)],
    labels[5]: [(4, 2), (10, 2)],
};
ywidth = 0.7
extraMargin = 0.5 # Top-most and bottom-most margins
initDate = dt.date(year=2020, month=2, day=1)

print(plt.rcParams.keys())
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.rcParams["xtick.minor.pad"] = 5
#plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.labelbottom"] = plt.rcParams["xlabel.bottom"] = False
#plt.rcParams["xtick.top"]    = plt.rcParams["xtick.labeltop"]    = plt.rcParams["xlabel.top"]    = True

fig, ax = plt.subplots(figsize=(12, 6))

for idx, label in enumerate(labels[::-1]):
    ax.broken_barh(times[label], (idx + 1 - ywidth/2, ywidth), facecolors="tab:blue")

allDates = [ initDate + dt.timedelta(days=31*i) for i in range(12) ]

def date_tick(idx, date):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    dmy = str(date).split("-")[::-1]
    if int(dmy[1]) == 1 or idx == 0:
        return months[int(dmy[1]) - 1] + " " + dmy[2]
    else:
        return months[int(dmy[1]) - 1]

# Y labels are positioned at [1, 2, 3, 4]...
# Horizontal bars have heights on [0.6, 1.4], [1.6, 2.4], ... (depends on ywidth)
ax.set_ylim(0.5 - extraMargin, len(labels) + 0.5 + extraMargin)
ax.set_xlim(1, 12)
ax.set_xlabel("Months", fontsize=14)
ax.set_xticks(np.arange(1, 12+0.1))
ax.set_xticklabels([ date_tick(idx, d) for idx, d in enumerate(allDates) ])
ax.set_yticks(np.arange(1, len(labels) + 0.1))
ax.set_yticklabels(labels) # It is inverted later
ax.grid(True, axis="x", c="k", ls="-", alpha=0.3)

ax.invert_yaxis()
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")

plt.savefig("project_schedule.png");
plt.show()
