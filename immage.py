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


x1 = np.linspace(x_final[0] - 5, x_final[0] + 5, 100)  # Диапазон вокруг найденного минимума
x2 = np.linspace(x_final[1] - 5, x_final[1] + 5, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = func([X1, X2])  # Вычисляем значения функции на сетке

# Построение графика
plt.figure(figsize=(10, 10))
cp = plt.contour(X1, X2, Z, levels=20, cmap='viridis')  # Линии уровня
plt.colorbar(cp, label='f(x1, x2)') #  Добавим шкалу цветов


# Преобразуем список x_values в массив numpy для удобства
x_values_np = np.array(x_values)
plt.plot(x_values_np[:, 0], x_values_np[:, 1], color='red', marker='o', markersize=5, label='Gradient Descent Path')

plt.title('Gradient Descent Optimization')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.show()
