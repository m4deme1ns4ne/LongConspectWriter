import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np

# Установка параметров для корректного отображения текста
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Данные
elements = ["C", "H", "O", "N", "S", "P"]
min_percentages = [47, 6, 10, 1, 0.3, 1]
max_percentages = [47, 10, 25, 3, None, None]

# Создание столбчатой диаграммы
fig, ax = plt.subplots()

# Цвета для элементов
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']

# Добавление столбцов
for i, element in enumerate(elements):
    if max_percentages[i] is not None:
        ax.bar(element, max_percentages[i], color=colors[i], label=f'{element} ({max_percentages[i]}%)')
    else:
        ax.bar(element, min_percentages[i], color=colors[i], label=f'{element} ({min_percentages[i]}%)')

# Добавление легенды
ax.legend(title='Максимальное значение (%)')

# Настройка осей
ax.set_xlabel('Элементы')
ax.set_ylabel('Процент по массе тела (%)')
ax.set_title(r'Состав органогенных элементов в живых организмах по массе тела (%)')

# Сохранение графика
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\20.37.18\09_grapher\assets\1___Состав_органогенных_элементов_в_живых_организмах_по_массе_тела____.png'
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех графиков
plt.close('all')