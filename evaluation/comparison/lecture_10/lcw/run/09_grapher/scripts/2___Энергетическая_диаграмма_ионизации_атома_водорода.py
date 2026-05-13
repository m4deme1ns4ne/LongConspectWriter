import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np

# Установка параметров для корректного отображения кириллицы и математики
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Создание данных
energies_eV = np.linspace(-13.6, 0, 100)
labels = ['Ground state', 'Ionization threshold']

# Создание графика
fig, ax = plt.subplots()
ax.plot(energies_eV, np.zeros_like(energies_eV), 'k-')
ax.scatter([-13.6], [0], color='red', label='Ionization threshold')

# Настройка осей и подписей
ax.set_xlabel(r'$E$ (эВ)')
ax.set_ylabel(r'$n$')
ax.set_title(r'Энергетическая диаграмма ионизации атома водорода')
ax.legend()

# Сохранение графика
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\22.24.34\09_grapher\assets\2___Энергетическая_диаграмма_ионизации_атома_водорода.png'
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех графиков
plt.close('all')