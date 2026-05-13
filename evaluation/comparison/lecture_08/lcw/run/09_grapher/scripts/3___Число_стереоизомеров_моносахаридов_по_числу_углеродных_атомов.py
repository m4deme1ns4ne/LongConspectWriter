import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import matplotlib.pyplot as plt

# Устанавливаем параметры для корректного отображения текста
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Создаем данные
n_values = np.arange(3, 8)
aldose_isomers = 2 ** (n_values - 2)
keto_isomers = 2 ** (n_values - 3)

# Создаем график
fig, ax = plt.subplots()

# Рисуем кривые для альдоз и кетоз
ax.plot(n_values, aldose_isomers, label=r'$2^{(n-2)}$', marker='o')
ax.plot(n_values, keto_isomers, label=r'$2^{(n-3)}$', marker='s')

# Добавляем подписи и заголовок
ax.set_xlabel(r'Число углеродных атомов ($n$)')
ax.set_ylabel('Число стереоизомеров')
ax.set_title('Число стереоизомеров моносахаридов по числу углеродных атомов')

# Добавляем легенду
ax.legend()

# Сохраняем график
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\21.17.32\09_grapher\assets\3___Число_стереоизомеров_моносахаридов_по_числу_углеродных_атомов.png'
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрываем все открытые графики
plt.close('all')