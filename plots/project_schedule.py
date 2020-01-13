import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

labels = ["1. Formulation", "2. Propose\nMethods", "3. Implement\nMethods", "4. Literature", "5. Articles", "6. Reports"]
times  = {
    labels[0]: [(1, 7)], # Must be on format [(from, width), (from, width), ...]
    labels[1]: [(2, 8)],
    labels[2]: [(2, 9)],
    labels[3]: [(1, 12)],
    labels[4]: [(8, 4)],
    labels[5]: [(5, 2), (11, 2)],
};
ywidth      = 0.7 # Width of the horizontal bars
extraMargin = 0.5 # Top-most and bottom-most margins
initDate    = dt.date(year=2020, month=2, day=1)
nMonths     = 24  # Number of months in the gantt chart
minInterval = 2   # The minimal interval a task can use
titleSize   = 20  # font size
yTextPt     = 20  # font size
xTextPt     = 16  # font size
monthRepr   = "NUMBER" # "NUMBER" or "TEXT"

# Increase size of labels. Set any of them to a wrong value, and the error will tell you what are valid values.
# print(plt.rcParams.keys())
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"

# Add xticks to top and bottom
plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["xtick.top"]    = plt.rcParams["xtick.labeltop"] = True

# Remove yticks
plt.rcParams["ytick.minor.size"] = 0
plt.rcParams["ytick.major.size"] = 0

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_facecolor("#d7d7d7")

# Add the horizontal bars
for idx, label in enumerate(labels):
    ax.broken_barh(times[label], (idx + 1 - ywidth/2, ywidth), facecolors="#126262", zorder=2, edgecolor="black", linewidth=1.5)

# Generate all months of the project
allDates = [ initDate + dt.timedelta(days=31*i) for i in range(nMonths) if i % minInterval == 0 ]

# This function will generate the labels for the xticks
# "idx" is the index of the xtick, and date is the date in the format "YYYY-MM-DD"
SAVED_YEAR = ""
def date_tick(idx, date):
    global SAVED_YEAR, minInterval, monthRepr

    if monthRepr.upper() == "TEXT":
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    elif monthRepr.upper() == "NUMBER":
        months = [ str(i) for i in range(1, 13) ]
    else:
        raise Exception("monthRepr can only be 'TEXT' or 'NUMBER'.");

    dmy = str(date).split("-")[::-1]
    year = int(dmy[2])
    if year != SAVED_YEAR:
        SAVED_YEAR = year
        return dmy[2] + "\n" + months[int(dmy[1]) - 1]
    else:
        return months[int(dmy[1]) - 1]

# Horizontal bars have heights on [0.6, 1.4], [1.6, 2.4], ... (depends on ywidth)
ax.set_xlim(1, len(allDates) + 1)
ax.set_xlabel("Months", fontsize=titleSize, fontweight="bold", labelpad=14)
ax.set_xticks(np.arange(1, len(allDates)+1+0.1))
ax.set_xticks(np.arange(1, len(allDates) + 0.1)+0.5, minor=True)
ax.set_xticklabels([ ""                for idx, d in enumerate(allDates) ])
ax.set_xticklabels([ date_tick(idx, d) for idx, d in enumerate(allDates) ], { "weight": "bold", "size": xTextPt }, minor=True)

# Y labels are positioned at [1, 2, 3, 4]...
ax.set_ylim(0.5 - extraMargin, len(labels) + 0.5 + extraMargin)
ax.set_yticks(np.arange(1, len(labels) + 0.1))
ax.set_yticklabels(labels, { "weight": "bold", "size": yTextPt }) # It is inverted later
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)

# Set a faint grid along the x axis
ax.grid(True, axis="x", c="k", ls="-", alpha=0.15, zorder=1)

# Invert yaxis because bars were added from bottom to top
ax.invert_yaxis()

# The "Months" label is better on top
ax.xaxis.set_label_position("top")

plt.tight_layout()
plt.savefig("project_schedule.png");
plt.show()
