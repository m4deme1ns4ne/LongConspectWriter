import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import matplotlib.pyplot as plt

# Параметры
n0_C7H8 = 0.1  # моль
n0_HNO3 = 0.5  # моль
n_product = 0.07  # моль

# Время
time = np.linspace(0, 3, 4)

# Количества веществ
C7H8 = n0_C7H8 - time * (n0_HNO3 / n_product)
HNO3 = n0_HNO3 - time * (n0_HNO3 / n_product)
product = time * (n0_HNO3 / n_product)

# Настройка параметров matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Создание графика
plt.figure(figsize=(10, 6))
plt.plot(time, C7H8, label=r'$\text{C}_7\text{H}_8$')
plt.plot(time, HNO3, label=r'$\text{HNO}_3$')
plt.plot(time, product, label=r'$\text{C}_7\text{H}_5\text{N}_3\text{O}_6$')

# Настройка осей и подписей
plt.xlabel(r'Время (с)')
plt.ylabel(r'Количество вещества (моль)')
plt.title(r'Кинетическая кривая реакции нитрования толуола')
plt.legend()

# Сохранение графика
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\21.55.53\09_grapher\assets\5___Кинетическая_кривая_реакции_нитрования_толуола.png'
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех графиков
plt.close('all')