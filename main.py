import numpy as np
import matplotlib.pyplot as plt

# Определяем функцию и её градиент
def func(x):
    x1, x2 = x
    return 3*x1**2 + 2*x2**2 + 4*x1*x2 - 5*x1 + 6*x2

def grad(x):
    x1, x2 = x
    grad_x1 = 6*x1 + 4*x2 - 5  # производная сначала от  x1
    grad_x2 = 4*x1 + 4*x2 + 6  # производная я по x2
    return np.array([grad_x1, grad_x2])

# Тут как на примере с урока просто оставляем данные но можем менять начальные значения
learning_rate = 0.01
num_iterations = 1000
x_initial = np.array([-14, 5])  # Начальные значения для x1 и x2

# Градиентный спуск
x_values = [x_initial]
for i in range(num_iterations):
    x_new = x_values[-1] - learning_rate * grad(x_values[-1])
    x_values.append(x_new)

x_final = x_values[-1]
print(f"Минимум достигается при x1 = {x_final[0]:.4f}, x2 = {x_final[1]:.4f}")
print(f"Значение функции в точке минимума = {func(x_final):.4f}")




'''
Классная работа которая предпоставлена была в 

import numpy as np
import matplotlib.pyplot as plt

# Определяем функцию и ее производную
def func(x):
    return (x - 5)**2

def grad(x):
    return 2 * (x - 5)

# Параметры градиентного спуска
learning_rate = 0.001
num_iterations = 50
x_initial = -14  # Начальное значение

# Градиентный спуск
x_values = [x_initial]
for i in range(num_iterations):
    x_new = x_values[-1] - learning_rate * grad(x_values[-1])
    x_values.append(x_new)


print(f"Минимум достигается при x = {x_values[-1]:.4f}")
print(f"Значение функции в точке минимума = {func(x_values[-1]):.4f}")


x = np.linspace(-15, 25)
y = func(x)

plt.figure(figsize=(5, 5))
plt.plot(x, y, label='f(x) = (x - 5)^2')
plt.scatter(x_values, [func(x) for x in x_values], color='red', label='Gradient Descent', zorder=5)
plt.plot(x_values, [func(x) for x in x_values], color='red', linestyle='dotted', zorder=5)
plt.title('Gradient Descent Optimization')

'''

'''
тз дз

Дана функция f(x) = 3x21+ 2x22+ 4x1x2−5x1+ 6x2- найти минимум функции с помощью метода градиентного спуска

'''
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()


'''
