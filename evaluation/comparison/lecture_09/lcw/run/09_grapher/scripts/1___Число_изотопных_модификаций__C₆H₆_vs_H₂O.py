import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt

# Установка параметров для корректного отображения кириллицы и минуса
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Данные для графика
substances = ["benzene", "water"]
isotopic_variants = [120000000, 3]

# Создание столбчатой диаграммы
plt.bar(substances, isotopic_variants, color=['blue', 'green'])

# Добавление подписей к столбцам
for i, v in enumerate(isotopic_variants):
    plt.text(i, v + 10000000, str(v), ha='center', va='bottom')

# Настройка заголовка и осей
plt.title(r"Число изотопных модификаций: C$_6$H$_6$ vs H$_2$O")
plt.xlabel("Состав")
plt.ylabel("Количество изотопных форм")

# Сохранение графика
target_path = r"C:\Users\alexa\conspectius-engine\data\runs\2026.05.12\21.55.53\09_grapher\assets\1___Число_изотопных_модификаций__C₆H₆_vs_H₂O.png"
plt.savefig(target_path, facecolor='white', bbox_inches='tight')

# Закрытие всех открытых графиков
plt.close('all')