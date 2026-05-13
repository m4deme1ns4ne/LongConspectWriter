import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import matplotlib.pyplot as plt

# Установка параметров для корректного отображения кириллицы и минуса
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Данные для графика зависимости энергии активации от стадии реакции
energy_HH = [0, 72, 90, 48]
energy_DD = [0, 72, 90, 48]

# Создание графика
plt.figure(figsize=(10, 6))

# Построение линий для H–H и D–D
plt.plot(energy_HH, label='H–H')
plt.plot(energy_DD, label='D–D')

# Добавление подписей к точкам
plt.annotate('Начало', (energy_HH[0], energy_HH[0]), textcoords="offset points", xytext=(0,10), ha='center')
plt.annotate('Переходное состояние', (energy_HH[1], energy_HH[1]), textcoords="offset points", xytext=(0,10), ha='center')
plt.annotate('Максимум', (energy_HH[2], energy_HH[2]), textcoords="offset points", xytext=(0,10), ha='center')
plt.annotate('Конец', (energy_HH[3], energy_HH[3]), textcoords="offset points", xytext=(0,10), ha='center')

# Добавление заголовка и подписей осей
plt.title(r'Кинетика изотопного обмена (протий vs дейтерий)')
plt.xlabel(r'Энергия активации')
plt.ylabel(r'Стадия реакции')

# Добавление легенды
plt.legend()

# Сохранение графика
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\21.55.53\09_grapher\assets\4___Кинетика_изотопного_обмена__протий_vs_дейтерий_.png'
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех графиков
plt.close('all')