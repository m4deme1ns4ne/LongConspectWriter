import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np

# Установка параметров для корректного отображения текста
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Данные для графика
x = ["Количество последовательностей", "Масса всех белков"]
y = [1e+130, 1e+100]
labels = ["~10¹³⁰", "> масса Вселенной (5×10⁵² кг)"]

# Создание фигуры и осей
fig, ax = plt.subplots()

# Построение графика
ax.bar(x, y, color=['blue', 'green'])

# Добавление меток
for i, v in enumerate(y):
    ax.text(i, v + 1e+90, labels[i], ha='center', va='bottom')

# Настройка логарифмической шкалы для оси Y
ax.set_yscale('log')

# Установка заголовка и меток осей
ax.set_title(r"Разнообразие белков: количество последовательностей и их суммарная масса")
ax.set_xlabel("Категория")
ax.set_ylabel("Значение")

# Сохранение графика
target_path = r"C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\21.17.32\09_grapher\assets\2___Разнообразие_белков__количество_последовательностей_и_их_суммарная_масса.png"
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех открытых графиков
plt.close('all')