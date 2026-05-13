import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np

# Set font properties
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Define the data
data = [
    [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0],
    [1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0],
    [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1],
    [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]
]

# Extract rows with ones
rows_with_ones = [row for row in data if 1 in row]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the rows with ones as rectangles
for row in rows_with_ones:
    x = np.arange(4)
    y = np.zeros(4)
    width = 0.8
    height = 0.8
    ax.bar(x, height, width, bottom=y, color='blue')

# Set the limits and labels
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 2)
ax.set_xticks(np.arange(4))
ax.set_yticks([])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

# Save the figure
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\18.14.10\09_grapher\assets\0___Таблица_истинности_для_ДНФ_функции_f_x_y_z_.png'
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Close the figure
plt.close('all')