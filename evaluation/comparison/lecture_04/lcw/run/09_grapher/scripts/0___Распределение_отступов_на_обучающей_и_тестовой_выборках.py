import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import matplotlib.pyplot as plt

# Параметры
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\19.12.37\09_grapher\assets\0___Распределение_отступов_на_обучающей_и_тестовой_выборках.png'

# Пример данных
train_margin = np.array([0.8, 1.2, -0.3, 0.5])
test_margin = np.array([0.7, 1.1, -0.4, 0.6])

# Создание гистограмм
plt.figure(figsize=(10, 6))

plt.hist(train_margin, bins=10, alpha=0.5, label='Train', color='blue')
plt.hist(test_margin, bins=10, alpha=0.5, label='Test', color='red')

# Настройка графика
plt.title('Распределение отступов на обучающей и тестовой выборках')
plt.xlabel('Отступ')
plt.ylabel('Частота')
plt.legend()

# Сохранение графика
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех открытых графиков
plt.close('all')