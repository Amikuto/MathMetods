import numpy as np
import matplotlib.pyplot as plt

p1o1 = int(input("Потребление 1 отрасли 1: "))
p1o2 = int(input("Потребление 1 отрасли 2: "))
p2o1 = int(input("Потребление 2 отрасли 1: "))
p2o2 = int(input("Потребление 2 отрасли 2: "))
kpy1 = int(input("Конечное потребление y1: "))
kpy2 = int(input("Конечное потребление y2: "))
opx1 = int(input("Объем производства x1: "))
opx2 = int(input("Объем производства x2: "))
X = np.matrix([[p1o1, p1o2], [p2o1, p2o2]])
y = np.array([kpy1, kpy2])
x = np.array([opx1, opx2])
y2 = np.array(np.random.randint(50, high=300, size=2))

# X = np.matrix([[14, 42], [20, 28]])
# y = np.array([120, 150])
# x = np.array([176, 198])
# y2 = np.array([200, 130])


E = np.eye(len(X))
# print("Единичная матрица: \n", E)

A = np.zeros((2, 2))
A[0, 0] = X[0, 0] / x[0]
A[1, 0] = X[1, 0] / x[0]
A[0, 1] = X[0, 1] / x[1]
A[1, 1] = X[1, 1] / x[1]
print("Матрица прямых затрат: \n", A)

EA = E - A
# print("", EA)

S = np.linalg.inv(EA)
print("матрицы полных затрат: \n", S)

E1 = S @ EA
print("", E1)

if np.all(S > 0):
    print("Матрица A продуктивна")

np.linalg.eig(A)

d = np.linalg.eigvals(A)
print("Собственные значения матрицы: ", d)

lambda_A = np.max(d)
print("Определение числа Фробениуса: ", lambda_A)

P = np.linalg.eig(A)[1]
print("Собств. векторы A, стоящие в столбцах матр. P\n", P)

x1 = S @ y
print("Вектор объемов производства по отраслям (x1=x)", x1)

x2 = S @ y2
print("Вектор новых объемов производства по отраслям,→ (валового выпуска)", x2)

xc = np.array([0., 0.])
x11 = A[0, 0] * x2[0]
x21 = A[1, 0] * x2[0]
xc[0] = x2[0] - (x11 + x21)
x12 = A[0, 1] * x2[1]
x22 = A[1, 1] * x2[1]
xc[1] = x2[1] - (x12 + x22)
print("вектор чистой продукции отраслей xc: ", xc)


names = ["Отрасль 1", "Отрасль 2"]
plt.bar(names, xc, color=["orange", "blue"])
plt.title("чистая продукция отраслей xc")
plt.show()

plt.bar(names, x2, color=["orange", "blue"])
plt.title("Вектор новых объемов производства по отраслям,→ (валового выпуска)")
plt.show()

plt.bar(names, x1, color=["orange", "blue"])
plt.title("Вектор объемов производства по отраслям (x1=x)")
plt.show()
