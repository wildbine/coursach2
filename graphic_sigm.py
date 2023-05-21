import numpy as np
import matplotlib.pyplot as plt

# Создание массива значений x от -10 до 10 с шагом 0.1
x = np.arange(-10, 10, 0.1)

# Вычисление значений сигмоидальной функции для каждого значения x
y = 1 / (1 + np.exp(-x))

# Создание графика
plt.plot(x, y)

# Настройка осей
plt.axhline(0, color='gray', linewidth=0.5)  # Горизонтальная линия на уровне y = 0
plt.axhline(1, color='gray', linewidth=0.5)  # Горизонтальная линия на уровне y = 1
plt.axvline(0, color='gray', linewidth=0.5)  # Вертикальная линия на уровне x = 0
plt.xlabel('x')
plt.ylabel('sigmoid(x)')

# Настройка заголовка
plt.title('График сигмоидальной функции')

# Отображение графика
plt.show()
