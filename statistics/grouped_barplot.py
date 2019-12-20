import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ["HMAC50", "HMAC100", "RSA50"]
access_token_means = [90.78, 90.16, 98.32]
access_token_error = [20.35, 28.39, 15.85]
get_data_means     = [7.01, 5.74, 7.57]
get_data_error     = [2.97, 2.31, 2.87]

x = np.arange(len(labels))  # the label locations
width = 0.35                # the width of the bars
value_offset = 0.9          # Offset of the value from the left-corner of the bar. 50% centers the value.

fig, ax = plt.subplots(figsize=(16, 10))
rects1 = ax.bar(x - width/2, access_token_means, width, label="Access Token", yerr=access_token_error, color="blue", capsize=5)
rects2 = ax.bar(x + width/2, get_data_means, width, label="Get Data", yerr=get_data_error, color="red", capsize=5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Time")
ax.set_title("Cost of each Cryptography Algorithm")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc="upper right")

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate("{}".format(height),
                    xy=(rect.get_x() + value_offset * rect.get_width(), height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom")

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
