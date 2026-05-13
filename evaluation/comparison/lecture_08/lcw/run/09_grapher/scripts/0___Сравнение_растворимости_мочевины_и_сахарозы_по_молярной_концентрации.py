import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np

# Установка параметров для корректного отображения текста
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Данные для графика
urea_concentrations = np.linspace(0, 12, 25)
urea_values = ['urea'] * len(urea_concentrations)

sucrose_concentrations = np.linspace(2.5, 2.8, 25)
sucrose_values = ['sucrose'] * len(sucrose_concentrations)

# Создание графика
plt.figure(figsize=(10, 6))

# Построение кривых
plt.plot(urea_concentrations, urea_values, label='Urea', marker='o')
plt.plot(sucrose_concentrations, sucrose_values, label='Sucrose', marker='s')

# Добавление подписей осей
plt.xlabel(r'Concentration (M)')
plt.ylabel('Substance')

# Добавление легенды
plt.legend()

# Сохранение графика
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\21.17.32\09_grapher\assets\0___Сравнение_растворимости_мочевины_и_сахарозы_по_молярной_концентрации.png'
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех открытых графиков
plt.close('all')