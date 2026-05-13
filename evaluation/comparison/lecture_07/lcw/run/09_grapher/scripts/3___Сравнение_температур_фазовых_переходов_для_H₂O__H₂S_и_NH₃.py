import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np

# Установка параметров для корректного отображения текста
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Данные для графика
data = [
    ('H2O', 0, 100),
    ('H2S', -80, -60),
    ('NH3', -78, -33)
]

# Создание фигуры и осей
fig, ax = plt.subplots()

# Построение столбчатой диаграммы
x = np.arange(len(data))
width = 0.25

ax.bar(x - width, [d[1] for d in data], width, label='Плавление')
ax.bar(x + width, [d[2] for d in data], width, label='Кипение')

# Подписи осей
ax.set_xlabel('Вещество')
ax.set_ylabel('Температура (°C)')

# Подписи точек
for i, d in enumerate(data):
    ax.text(i - width, d[1] + 2, f'{d[1]}°C', ha='center', va='bottom')
    ax.text(i + width, d[2] + 2, f'{d[2]}°C', ha='center', va='bottom')

# Подписи легенды
ax.legend()

# Названия веществ на оси X
ax.set_xticks(x)
ax.set_xticklabels([d[0] for d in data])

# Сохранение графика
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\20.37.18\09_grapher\assets\3___Сравнение_температур_фазовых_переходов_для_H₂O__H₂S_и_NH₃.png'
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех графиков
plt.close('all')