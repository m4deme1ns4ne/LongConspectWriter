import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np

# Настройки для кириллицы и отображения минуса
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Данные для графика
transitions = [
    {'transition': 'Lyman α', 'from_n': 2, 'to_n': 1, 'lambda_nm': 121.6},
    {'transition': 'Balmer H-alpha', 'from_n': 3, 'to_n': 2, 'lambda_nm': 656.3}
]

# Создание массивов для уровней энергии
n_values = np.arange(1, 4)
energy_levels = -13.6 * (n_values**-2)

# Создание графика
fig, ax = plt.subplots()

# Заполнение области между уровнями энергии
ax.fill_between(n_values, energy_levels, color='lightblue')

# Аннотация для каждого перехода
for transition in transitions:
    from_n = transition['from_n']
    to_n = transition['to_n']
    lambda_nm = transition['lambda_nm']
    delta_E = 13.6 * (from_n**-2 - to_n**-2)
    ax.annotate(f"{transition['transition']} ({lambda_nm} nm)", 
                 xy=(to_n, -13.6 * (to_n**-2)), 
                 xytext=(to_n + 0.1, -13.6 * (to_n**-2) - 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))

# Настройка осей и подписи
ax.set_xlabel(r'$n$')
ax.set_ylabel(r'$E$ (эВ)')
ax.set_title(r'Энергетические переходы в атоме водорода (спектроскопия)')

# Сохранение графика
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\22.24.34\09_grapher\assets\3___Энергетические_переходы_в_атоме_водорода__спектроскопия_.png'
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех графиков
plt.close('all')