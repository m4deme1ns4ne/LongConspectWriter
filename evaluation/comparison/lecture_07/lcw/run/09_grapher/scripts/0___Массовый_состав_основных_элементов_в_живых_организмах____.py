import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt

# Данные для графика
elements = ['C', 'H', 'O', 'N', 'Others']
masses = [18, 10, 67.5, 3, 1.5]

# Создание круговой диаграммы
plt.figure(figsize=(8, 8))
plt.pie(masses, labels=elements, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Сохранение графика
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\20.37.18\09_grapher\assets\0___Массовый_состав_основных_элементов_в_живых_организмах____.png'
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех открытых графиков
plt.close('all')