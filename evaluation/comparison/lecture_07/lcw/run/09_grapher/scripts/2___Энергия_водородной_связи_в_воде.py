import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import matplotlib.pyplot as plt

# Установка параметров для корректного отображения текста
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Создание данных
x = np.linspace(0.3, 0.65, 10)
y_hydrogen_bond = (1/3) * x + (2/3) * (1 - x)

# Создание графика
plt.figure(figsize=(8, 6))
plt.plot(x, y_hydrogen_bond, label=r'$E_{\text{H-bond}} = \frac{1}{3}q_H + \frac{2}{3}(1 - q_H)$', color='blue')

# Настройка осей и подписей
plt.xlabel(r'Доля заряда $q_H$')
plt.ylabel(r'Сила связи $E_{\text{H-bond}}$')
plt.title(r'Энергия водородной связи в воде')
plt.legend()

# Сохранение графика
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\20.37.18\09_grapher\assets\2___Энергия_водородной_связи_в_воде.png'
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех графиков
plt.close('all')