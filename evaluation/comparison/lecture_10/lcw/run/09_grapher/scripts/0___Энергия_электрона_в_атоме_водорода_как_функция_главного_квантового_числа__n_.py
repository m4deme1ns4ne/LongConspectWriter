import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import matplotlib.pyplot as plt

# Установка параметров для корректного отображения кириллицы и минуса
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Определение значения боровского радиуса
bohr_radius = 5.29 * 10**-11

# Определение диапазона значений n
n_values = np.arange(1, 6)

# Вычисление энергии электрона для каждого значения n
E_n_values = -13.6 / n_values**2

# Создание графика
plt.figure(figsize=(8, 6))
plt.plot(n_values, E_n_values, marker='o', linestyle='-', color='b')

# Добавление подписей к осям
plt.xlabel(r'$n$ — главное квантовое число')
plt.ylabel(r'$E_n$ — энергия электрона (эВ)')

# Добавление заголовка
plt.title(r'Энергия электрона в атоме водорода как функция главного квантового числа $n$')

# Сохранение графика
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\22.24.34\09_grapher\assets\0___Энергия_электрона_в_атоме_водорода_как_функция_главного_квантового_числа__n_.png'
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех открытых графиков
plt.close('all')