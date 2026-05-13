import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt

# Данные для графика
data = [('сахароза', 2.65), ('мочевина', 11)]

# Разделение данных на типы веществ и их концентрации
types = [item[0] for item in data]
concentrations = [item[1] for item in data]

# Создание графика
plt.figure(figsize=(8, 6))
plt.bar(types, concentrations, color=['blue', 'green'])

# Настройка осей и подписей
plt.xlabel(r'Тип соединения')
plt.ylabel(r'Молярная концентрация (моль/л)')

# Добавление легенды
plt.legend(['Сахароза', 'Мочевина'])

# Сохранение графика
target_path = r'C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\21.17.32\09_grapher\assets\1___Сравнение_растворимости_гидрофильных_соединений__сахароза_vs_мочевина.png'
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех графиков
plt.close('all')