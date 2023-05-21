import numpy as np
import matplotlib.pyplot as plt

# Создание массива значений x от -10 до 10 с шагом 0.1
x = np.arange(-10, 10, 0.1)

# Вычисление значений гиперболического тангенса для каждого значения x
y = np.tanh(x)

# Создание графика
plt.plot(x, y)

# Настройка осей
plt.axhline(0, color='gray', linewidth=0.5)  # Горизонтальная линия на уровне y = 0
plt.axvline(0, color='gray', linewidth=0.5)  # Вертикальная линия на уровне x = 0
plt.xlabel('x')
plt.ylabel('tanh(x)')

# Настройка заголовка
plt.title('График гиперболического тангенса')

# Отображение графика
plt.show()