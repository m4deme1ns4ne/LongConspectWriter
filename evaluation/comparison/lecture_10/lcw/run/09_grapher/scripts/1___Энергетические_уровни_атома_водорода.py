import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import matplotlib.pyplot as plt

# Параметры
Z = 1
hbar = 1.054 * 10**-34  # Дж·с
E_n = lambda n: -13.6 * Z**2 / n**2

# Данные для графика
n_values = np.arange(1, 6)
E_values = E_n(n_values)

# Создание графика
plt.figure(figsize=(8, 6))
plt.plot(n_values, E_values, marker='o', linestyle='-', color='b')
plt.xlabel(r'$n$')
plt.ylabel(r'$E_n$ (эВ)')
plt.title(r'Энергетические уровни атома водорода')
plt.grid(True)

# Сохранение графика
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\22.24.34\09_grapher\assets\1___Энергетические_уровни_атома_водорода.png'
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех графиков
plt.close('all')