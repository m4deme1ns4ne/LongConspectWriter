import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np

# Установка параметров для корректного отображения кириллицы и математики
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Данные для графика
isotope = ["H", "D"]
rate = [1.0, 0.17]

# Создание линейного графика
plt.figure(figsize=(8, 4))
plt.plot(isotope, rate, marker='o', linestyle='-', color='b')

# Настройка осей и подписей
plt.xlabel(r"Тип изотопа")
plt.ylabel(r"Скорость реакции")
plt.title(r"Сравнение скорости обмена протия и дейтерия")

# Сохранение графика
target_path = r"C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\21.55.53\09_grapher\assets\2___Сравнение_скорости_обмена_протия_и_дейтерия.png"
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех открытых графиков
plt.close('all')