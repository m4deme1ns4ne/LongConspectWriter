import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt

# Установка параметров для корректного отображения кириллицы и математики
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Данные для диаграммы
isotopes = ['H₂¹⁶O', 'H₂¹⁷O', 'H₂¹⁸O', 'D₂¹⁶O', 'D₂¹⁷O', 'D₂¹⁸O', 'T₂¹⁶O', 'T₂¹⁷O', 'T₂¹⁸O']
counts = [1, 1, 1, 1, 1, 1, 1, 1, 1]

# Создание круговой диаграммы
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=isotopes, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Сохранение графика
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\21.55.53\09_grapher\assets\0___Изотопное_разнообразие_молекул_H₂O_и_O₂.png'
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех открытых графиков
plt.close('all')