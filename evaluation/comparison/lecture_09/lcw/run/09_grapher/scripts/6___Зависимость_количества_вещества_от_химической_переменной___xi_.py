import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import matplotlib.pyplot as plt

# Параметры реакции
n0_H2 = 0.5  # моль
n0_O2 = 0.7  # моль
nu_pr = 2    # коэффициент прореагирования водорода

# Диапазон химической переменной xi
xi_values = np.linspace(0, 0.25, 26)

# Количество вещества H2, O2 и H2O в зависимости от xi
n_H2 = n0_H2 - nu_pr * xi_values
n_O2 = n0_O2 - xi_values
n_H2O = 2 * xi_values

# Параметры для графика
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\21.55.53\09_grapher\assets\6___Зависимость_количества_вещества_от_химической_переменной___xi_.png'

# Создание графика
plt.figure(figsize=(8, 6))
plt.plot(xi_values, n_H2, label=r'$n(\text{H}_2)$')
plt.plot(xi_values, n_O2, label=r'$n(\text{O}_2)$')
plt.plot(xi_values, n_H2O, label=r'$n(\text{H}_2\text{O})$')

# Настройка графика
plt.xlabel(r'$\xi$')
plt.ylabel(r'Количество веществ (моль)')
plt.title(r'Зависимость количества веществ от химической переменной $\xi$')
plt.legend()
plt.grid(True)

# Сохранение графика
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех открытых графиков
plt.close('all')