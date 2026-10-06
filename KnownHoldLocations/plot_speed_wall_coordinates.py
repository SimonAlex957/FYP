import csv
import matplotlib.pyplot as plt
import numpy as np

with open("KnownHoldLocations/speed_wall_holds.csv", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

x = [float(row["x_m"]) for row in rows]
y = [float(row["y_m"]) for row in rows]
labels = [f"{row['panel']}-{row['hold']}" for row in rows]
colors = ["blue" if row["type"].strip().lower() == "foot" else "red" for row in rows]
hand_rows = [row for row in rows if row["type"].strip().lower() == "hand"]
hand_x = np.array([float(row["x_m"]) for row in hand_rows])
hand_y = np.array([float(row["y_m"]) for row in hand_rows])
line_slope, line_intercept = np.polyfit(hand_x, hand_y, 1)
line_x = np.linspace(hand_x.min(), hand_x.max(), 100)
line_y = line_slope * line_x + line_intercept

plt.figure(figsize=(6, 10))
plt.scatter(x, y, c=colors, label="Holds")

#line of best fit for hand holds
'''plt.plot(
    line_x,
    line_y,
    color="black",
    linestyle="--",
    label=f"Hand best fit: y = {line_slope:.2f}x + {line_intercept:.2f}",
)'''

for x_value, y_value, label in zip(x, y, labels):
    plt.annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)


plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Speed Wall Hold Coordinates")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig("KnownHoldLocations/speed_wall_coordinates.png", dpi=200)
plt.show()
