import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np

# Данные для графика
isotope = ["H", "D"]
rate = [1.0, 0.17]

# Создание графика
plt.figure(figsize=(8, 4))
plt.bar(isotope, rate, color=['blue', 'red'])
plt.xlabel(r'$\text{Изотоп}$')
plt.ylabel(r'$\text{Скорость реакции}$')
plt.title(r'Зависимость скорости реакции от типа изотопа (H vs D)')
plt.ylim(0, 1.2)
plt.grid(True)

# Сохранение графика
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\21.55.53\09_grapher\assets\3___Энергетический_барьер_изотопного_обмена__протий_vs_дейтерий_.png'
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех графиков
plt.close('all')