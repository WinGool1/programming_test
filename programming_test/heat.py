# Добавление необходимых библиотек и модулей Python
import math
import numpy as np
import os
from matplotlib import pylab as plt
# Постоянные
exp = 2.7182
pi = 3.14
g = 9.81
l = 1.5  # Длина рассматриваемого участка
# Входные данные для пласта и скважины
L = [3000, 2400, 1500]  # Глубина залегания скважины
k = [1.3 * 10 ** (-12), 0.5 * 10 ** (-12), 1.1 * 10 ** (-12)]  # Проницаемость пласта
Pk = [32.06 * 10**6, 26.7 * 10**6, 17.3 * 10**6]  # Давление в пласте
Pc = [28.66 * 10**6, 23.3 * 10**6, 16.2 * 10**6]  # Давление на забое скважины
Rk = [550, 10000, 1000]  # Радиус контура питания
D = [0.273, 0.219, 0.159]  # Внешний диаметр скважины
d_s = [0.264, 0.210, 0.150]  # Внутренний диаметр скважины
d_T = 0.309  # Внутренний диаметр трубопровода
D_T = 0.325  # Внешний диаметр трубопровода
h = [10, 15, 8]  # Мощность пласта
pn_20 = 824  # Плотность нефти при 293 К
p_v = 999  # Плотность воды при 273 К
C_n = 1929.82  # Теплоёмкость нефти
lambda_n = 0.16  # Теплопроводность нефти
Ekv_shoroh = 0.014  # Эквивалентная шероховатость
lambda_soil = 2.0  # Теплопроводность грунта
lambda_steel = 46.0  # Теплопроводность стали
Tc = [[], [], []]
p_sm = [[], [], []]
v_sm = [[], [], []]
T = [[383], [365], [338]]  # Температура на глубине скважины
G = [[], [], []]
# Пустые массивые, которые будут заполняться по ходу расчёта скважины
Tg = [[], [], []]  # Температура грунта на поверхности
P = [[30.66 * 10**6], [24.3 * 10**6], [16.2 * 10**6]]
Re = []
x = []
for i in range(0, len(L)):
    x.append(np.arange(L[i], 1.5, -1.5))
    Tc[i].append(278 + 0.03 * L[i])
    lambda_v = 0.553 * (1 + 0.003 * (Tc[i][0] - 273))
    C_v = 4194 - 1.15 * (Tc[i][0] - 273) + 0.015 * ((Tc[i][0] - 273) ** 2)
    C = 0.9 * C_v + (1 - 0.9) * 1929.82
    b = 0.000886 + 1.6 * 0.000886 * 0.000886 * (Tc[i][0] - 293)
    lambda_sm = (1 - 0.9) * 0.16 + 0.9 * lambda_v
    p_sm[i].append(
        0.9 * (999 / (1 + 0.00012 * (Tc[i][0] - 273)))
        + 0.1 * (824 / (1 + 0.00088 * (Tc[i][0] - 293)))
    )
    nu_n = 0.00000514 * exp ** ((0.0791812 / 10) * (293 - Tc[i][0]))
    mu_n = nu_n * (824 / (1 + 0.00088 * (Tc[i][0] - 293)))
    mu_sm = mu_n * (1 + 2.5 * 0.9)  # расчёт кинематической вязкости смеси
    nu_sm = mu_sm / p_sm[i][0]  # расчёт динамической вязкости смеси
    Q_sm = -(2 * pi * h[i] * k[i] * (Pc[i] - Pk[i])) / (
        mu_sm * math.log10(Rk[i] / (d_s[i] * 0.5))
    )  # расчёт расхода по формуле Дюпюи
    G_sm = Q_sm * p_sm[i][0]
    G[i].append(G_sm)
    v_sm[i].append(4 * G_sm / (p_sm[i][0] * pi * d_s[i] ** 2))
    Re.append((v_sm[i][0] * d_s[i]) / nu_sm)  # Число Рейнольдса
    Pr = mu_sm * C / lambda_sm  # Число Прандтя
    Gr = g * d_s[i] ** 3 * b / nu_sm**2  # Число Грасгофа
    # РАСЧЁТ СКВАЖИНЫ
    if Re[i] < 2300:
        alpha = (
            ((0.17 * lambda_sm) / d_s[i])
            * (Re[i] ** 0.33)
            * (Pr**0.43)
            * (Gr**0.1)
            * 1
        )  # (Pr)
        lambda_t = 64 / Re[i]
        alpha_cor = 2.0
    elif Re[i] > 10000:
        alpha = (
            ((0.021 * lambda_sm) / d_s[i])
            * (Re[i] ** 0.8)
            * (Pr**0.43)
            * (Gr**0.1)
            * 1
        )  # (Pr)**0.25
        lambda_t = 0.067 * (158 / Re[i] + 2 * Ekv_shoroh) ** 0.2
        alpha_cor = 1.1
    elif 2300 < Re[i] < 10000:
        alpha_l = (
            ((0.17 * lambda_sm) / d_s[i])
            * (2300**0.33)
            * (Pr**0.43)
            * (Gr**0.1)
            * 1
        )  # (Pr)
        alpha_t = (
            ((0.021 * lambda_sm) / d_s[i])
            * (10000**0.8)
            * (Pr**0.43)
            * (Gr**0.1)
            * 1
        )  # (Pr)**0.25
        alpha = alpha_l + ((alpha_t - alpha_l) / 8000) * (Re[i] - 2000)
        lambda_t1 = 64 / 2300
        lambda_t2 = 0.067 * (158 / 10000 + 2 * Ekv_shoroh) ** 0.2
        lambda_t = lambda_t1 + ((lambda_t2 - lambda_t1) / (10000 - 2300)) * (
            Re[i] - 2300
        )
        alpha_cor = 2 - 0.297 * lambda_t
    lambda_tr = lambda_t * 1.05
    l_tr = lambda_tr * 1.5 * v_sm[i][0] ** 2 / (2 * d_s[i])
    # Коэффициент теплопередачи для скважины:
    k_therm = (
        1 / alpha
        + (d_s[i] / (2 * lambda_steel)) * math.log10(D[i] / d_s[i])
        + (d_s[i] / (2 * lambda_soil)) * math.log10(10)
    ) ** (-1)
    # Далее идёт расчёт параметров по всей глубине скважины
    for j in range(0, len(x[i])):
        t_gr = 279 + 0.03 * x[i][j]
        Tg[i].append(t_gr)
        t_sm = T[i][j] + (
            k_therm * pi * D[i] * l * (Tg[i][j] - T[i][j]) / (C * G[i][0])
        )
        T[i].append(t_sm)
        p_tmp = 0.9 * (999 / (1 + 0.00012 * (t_sm - 273))) + 0.1 * (
            824 / (1 + 0.00088 * (t_sm - 293))
        )
        p_sm[i].append(p_tmp)
        v_tmp = p_sm[i][j] * v_sm[i][j] / p_tmp
        v_sm[i].append(v_tmp)
        if j == 1:
            P_tmp1 = p_tmp * P[i][j] / p_sm[i][j] + p_tmp * (
                ((v_tmp**2 - v_sm[i][j] ** 2) / 2) - g * l - l_tr
            )
            P[i].append(P_tmp1)
        else:
            P_tmp = p_tmp * P[i][j] / p_sm[i][j] + p_tmp * (
                ((v_tmp**2 - v_sm[i][j] ** 2) / 2) - g * l - l_tr
            )
            P[i].append(P_tmp)
    print(
        "Параметры на выходе",
        "{0:.3f}".format(i + 1),
        "скважины",
        "\nТемпература:",
        "{0:.3f}".format(t_sm),
        "K",
        "\nПлотность:",
        "{0:.3f}".format(p_tmp),
        "кг/м^3",
        "\nСкорость:",
        "{0:.3f}".format(v_tmp),
        "м/с",
        "\nДавление:",
        "{0:.3f}".format(P_tmp),
        "Па",
    )
    input("Для продолжение нажмите Enter")
# Построение графиков на устье скважины
fig = plt.figure(1)
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)
plt.sca(ax1)
plt.plot(x[0], T[0][len(T[0]) : 0 : -1], "b-.")
plt.plot(x[1], T[1][len(T[1]) : 0 : -1], "r-")
plt.plot(x[2], T[2][len(T[2]) : 0 : -1], "g--")
plt.legend(["Первая скважина", "Вторая скважина", "Третья скважина"], loc=1)
plt.xlabel("Глубина H, м", color="gray")
plt.ylabel("Температура Т, К", color="gray")
plt.sca(ax2)
plt.plot(x[0], P[0][len(P[0]) : 0 : -1], "b-.")
plt.plot(x[1], P[1][len(P[1]) : 0 : -1], "r-")
plt.plot(x[2], P[2][len(P[2]) : 0 : -1], "g--")
plt.legend(["Первая скважина", "Вторая скважина", "Третья скважина"], loc=1)
plt.xlabel("Глубина H, м", color="gray")
plt.ylabel("Давление P, Па*10^7", color="gray")
plt.sca(ax3)
plt.plot(x[0], v_sm[0][len(v_sm[0]) : 0 : -1], "b-.")
plt.plot(x[1], v_sm[1][len(v_sm[1]) : 0 : -1], "r-")
plt.plot(x[2], v_sm[2][len(v_sm[2]) : 0 : -1], "g--")
plt.legend(["Первая скважина", "Вторая скважина", "Третья скважина"], loc=1)
plt.xlabel("Глубина H, м", color="gray")
plt.ylabel("Скорость v, м/с", color="gray")
plt.sca(ax4)
plt.plot(x[0], p_sm[0][len(p_sm[0]) : 0 : -1], "b-.")
plt.plot(x[1], p_sm[1][len(p_sm[1]) : 0 : -1], "r-")
plt.plot(x[2], p_sm[2][len(p_sm[2]) : 0 : -1], "g--")
plt.legend(["Первая скважина", "Вторая скважина", "Третья скважина"], loc=1)
plt.xlabel("Глубина H, м", color="gray")
plt.ylabel("Плотность р, кг/м^3", color="gray")
fig.tight_layout()
plt.show()
### Поворот трубы на 90 градусов
P_horiz = [[], [], []]
for i in range(0, len(P_horiz)):
    Em = 1.32 + (1 - (pi * (d_s[i] ** 2) / 4) / (pi * (0.309**2) / 4)) ** 2
    P_horiz[i].append(P[i][-1] - p_sm[i][-1] * Em * (v_sm[i][-1] ** 2) / (2))
H = 1.5
L2 = [500, 600, 700]
x2 = []
# РАСЧЁТ Трубопровода от каждой скважины
T_gr = 278.045
Re2 = []
l2 = 1
T2 = [[], [], []]
G2 = [[], [], []]
G2T = [[], [], []]
p_sm2 = [[], [], []]
v_sm2 = [[], [], []]
P2 = [[], [], []]
for k in range(0, len(P_horiz)):
    x2.append(np.arange(L2[k], 0, -1))
    G2[k].append(p_sm[k][-1] * v_sm[k][-1] * pi * d_T**2 / 4)
    T2[k].append(T[k][-1])
    p_sm2[k].append(p_sm[k][-1])
    v_sm2[k].append(v_sm[k][-1])
    P2[k].append(P_horiz[k][0])
    delta = D[k] - d_s[k]
    lambda_v2 = 0.553 * (1 + 0.003 * (278.045 - 273))
    lambda_sm2 = (1 - 0.9) * 0.16 + 0.9 * lambda_v2
    C_v2 = 4194 - 1.15 * (278.045 - 273) + 0.015 * ((278.045 - 273) ** 2)
    C2 = 0.9 * C_v2 + (1 - 0.9) * 1929.82
    nu_n2 = 0.00000514 * exp ** ((0.0791812 / 10) * (278 - T[k][-1]))
    mu_n2 = nu_n2 * (824 / (1 + 0.00088 * (T[k][-1] - 278)))
    mu_sm2 = mu_n2 * (1 + 2.5 * 0.9)
    nu_sm2 = mu_sm2 / p_sm[k][-1]
    Re2.append((v_sm[k][-1] * d_T) / nu_sm2)
    Pr2 = mu_sm2 * C2 / lambda_sm2
    Gr2 = g * d_T**3 * b * (T[i][-1] - T_gr) / nu_sm2**2
    if Re2[k] < 2300:
        alpha = (
            ((0.17 * lambda_sm2) / d_T)
            * (Re2[k] ** 0.33)
            * (Pr2**0.43)
            * (Gr2**0.1)
            * 1
        )  # (Pr)
        lambda_t = 64 / Re2[k]
        alpha_cor = 2.0
    elif Re2[k] > 10000:
        alpha = (
            ((0.021 * lambda_sm2) / d_T)
            * (Re2[k] ** 0.8)
            * (Pr2**0.43)
            * (Gr2**0.1)
            * 1
        )  # (Pr)**0.25
        lambda_t = 0.067 * (158 / Re2[k] + 2 * Ekv_shoroh) ** 0.2
        alpha_cor = 1.1
    elif 2300 < Re2[k] < 10000:
        alpha_l = (
            ((0.17 * lambda_sm2) / d_T)
            * (2300**0.33)
            * (Pr2**0.43)
            * (Gr2**0.1)
            * 1
        )  # (Pr)
        alpha_t = (
            ((0.021 * lambda_sm2) / d_T)
            * (10000**0.8)
            * (Pr2**0.43)
            * (Gr2**0.1)
            * 1
        )  # (Pr)**0.25
        alpha = alpha_l + ((alpha_t - alpha_l) / 8000) * (Re2[k] - 2000)
        lambda_t1 = 64 / 2300
        lambda_t2 = 0.067 * (158 / 10000 + 2 * Ekv_shoroh) ** 0.2
        lambda_t = lambda_t1 + ((lambda_t2 - lambda_t1) / (10000 - 2300)) * (
            Re2[k] - 2300
        )
        alpha_cor = 2 - 0.297 * lambda_t
    lambda_tr = lambda_t * 1.05
    l_tr = lambda_tr * 1.5 * v_sm[k][-1] ** 2 / (2 * d_T)
    alpha_soil = (
        2
        * lambda_soil
        / (D_T * math.log10(2 * H / D_T + ((2 * H / D_T) ** 2 - 1) ** 0.5))
    )
    k_therm_trub = ((1 / alpha) + (delta / (2 * lambda_steel)) + (1 / alpha_soil)) ** (
        -1
    )
    for j in range(0, len(x2[k])):
        t_sm2 = T2[k][j] + (
            k_therm_trub * pi * D_T * (T_gr - T[k][j]) / (C2 * G2[k][0])
        )
        T2[k].append(t_sm2)
        p_tmp = 0.9 * (999 / (1 + 0.00012 * (t_sm2 - 273))) + 0.1 * (
            824 / (1 + 0.00088 * (t_sm2 - 293))
        )
        p_sm2[k].append(p_tmp)
        v_tmp = p_sm2[k][j] * v_sm2[k][j] / p_tmp
        v_sm2[k].append(v_tmp)
        P_tmp = p_tmp * P2[k][j] / p_sm2[k][j] + p_tmp * (
            ((v_tmp**2 - v_sm2[k][j] ** 2) / 2) - l_tr
        )
        P2[k].append(P_tmp)
    G2T[k].append(p_sm2[k][-1] * v_sm2[k][-1] * pi * (d_T**2) / 4)
    print(
        "Параметры на выходе",
        "{0:.3f}".format(k + 1),
        "скважины",
        "\nТемпература:",
        "{0:.3f}".format(t_sm2),
        "K",
        "\nПлотность:",
        "{0:.3f}".format(p_tmp),
        "кг/м^3",
        "\nСкорость:",
        "{0:.3f}".format(v_tmp),
        "м/с",
        "\nДавление:",
        "{0:.3f}".format(P_tmp),
        "Па",
    )
    input("Для продолжение нажмите Enter")
fig = plt.figure(2)
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)
plt.plot(x2[0], T2[0][len(T2[0]) : 0 : -1], "b-.")
plt.plot(x2[1], T2[1][len(T2[1]) : 0 : -1], "r-")
plt.plot(x2[2], T2[2][len(T2[2]) : 0 : -1], "g--")
plt.legend(["Первая скважина", "Вторая скважина", "Третья скважина"], loc=1)
plt.xlabel("Глубина H, м", color="gray")
plt.ylabel("Температура Т, К", color="gray")
plt.sca(ax1)
plt.plot(x2[0], P2[0][len(P2[0]) : 0 : -1], "b-.")
plt.plot(x2[1], P2[1][len(P2[1]) : 0 : -1], "r-")
plt.plot(x2[2], P2[2][len(P2[2]) : 0 : -1], "g--")
plt.legend(["Первая скважина", "Вторая скважина", "Третья скважина"], loc=1)
plt.xlabel("Глубина H, м", color="gray")
plt.ylabel("Давление P, Па*10^6", color="gray")
plt.sca(ax2)
plt.plot(x2[0], v_sm2[0][len(v_sm2[0]) : 0 : -1], "b-.")
plt.plot(x2[1], v_sm2[1][len(v_sm2[1]) : 0 : -1], "r-")
plt.plot(x2[2], v_sm2[2][len(v_sm2[2]) : 0 : -1], "g--")
plt.legend(["Первая скважина", "Вторая скважина", "Третья скважина"], loc=1)
plt.xlabel("Глубина H, м", color="gray")
plt.ylabel("Скорость v, м/с", color="gray")
plt.sca(ax3)
plt.plot(x2[0], p_sm2[0][len(p_sm2[0]) : 0 : -1], "b-.")
plt.plot(x2[1], p_sm2[1][len(p_sm2[1]) : 0 : -1], "r-")
plt.plot(x2[2], p_sm2[2][len(p_sm2[2]) : 0 : -1], "g--")
plt.legend(["Первая скважина", "Вторая скважина", "Третья скважина"], loc=1)
plt.xlabel("Глубина H, м", color="gray")
plt.ylabel("Плотность р, кг/м^3", color="gray")
plt.sca(ax4)
fig.tight_layout()
plt.show()
# Расчёт тройничка
G_final = G2T[0][0] + G2T[1][0] + G2T[2][0]
T_final = (
    T2[0][-1] * G2T[0][0] + T2[1][-1] * G2T[1][0] + T2[2][-1] * G2T[2][0]
) / G_final
P_final = (
    P2[0][-1] * G2T[0][0] + P2[1][-1] * G2T[1][0] + P2[2][-1] * G2T[2][0]
) / G_final
# p_sm_3=
T_duct = []
p_duct = []
v_duct = []
P_duct = []
L3 = 1000  # длина трубопровода
x3 = np.arange(L3, 0, -1)
print(
    "Параметры на выходе из тройника",
    "\nТемпература:",
    "{0:.3f}".format(T_final),
    "К",
    "\nРасход:",
    "{0:.3f}".format(G_final),
    "кг/с",
    "\nДавление:",
    "{0:.3f}".format(P_final),
    "Па",
)
input("Для продолжение нажмите Enter")
for i in range(0, L3):
    if i == 0:
        t_tmp = T_final + (k_therm_trub * pi * D_T * (T_gr - T_final) / (C2 * G_final))
        T_duct.append(t_tmp)
        p_tmp = 0.9 * (999 / (1 + 0.00012 * (t_tmp - 273))) + 0.1 * (
            824 / (1 + 0.00088 * (t_tmp - 293))
        )
        p_duct.append(p_tmp)
        v_tmp = G_final / (p_tmp * pi * ((d_T**2) / 4))
        v_duct.append(v_tmp)
        P_tmp = p_tmp * P_final / 975 + p_tmp * (((v_tmp**2 - 0.385**2) / 2) - l_tr)
        P_duct.append(round(P_tmp, 3))
    else:
        t_tmp = T_duct[i - 1] + (
            k_therm_trub * pi * D_T * (T_gr - T_duct[i - 1]) / (C2 * G_final)
        )
        T_duct.append(t_tmp)
        p_tmp = 0.9 * (999 / (1 + 0.00012 * (t_tmp - 273))) + 0.1 * (
            824 / (1 + 0.00088 * (t_tmp - 293))
        )
        p_duct.append(p_tmp)
        v_tmp = p_duct[i - 1] * v_duct[i - 1] / p_tmp
        v_duct.append(v_tmp)
        P_tmp = p_tmp * P_duct[i - 1] / p_duct[i - 1] + p_tmp * (
            ((v_tmp**2 - v_duct[i - 1] ** 2) / 2) - l_tr
        )
        P_duct.append(round(P_tmp, 3))
G_ITOG = p_tmp * v_tmp * pi * ((d_T**2) / 4)
print(
    "Параметры на выходе",
    "{0:.3f}".format(i + 1),
    "скважины",
    "\nТемпература:",
    "{0:.3f}".format(t_tmp),
    "K",
    "\nПлотность:",
    "{0:.3f}".format(p_tmp),
    "кг/м^3",
    "\nСкорость:",
    "{0:.3f}".format(v_tmp),
    "м/с",
    "\nДавление:",
    "{0:.3f}".format(P_tmp),
    "Па",
    "\nРасход",
    "{0:.3f}".format(G_ITOG),
    "кг/с",
)
fig = plt.figure(3)
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)
plt.plot(x3, T_duct[len(T_duct) :: -1], "b-.")
plt.xlabel("Глубина H, м", color="gray")
plt.ylabel("Температура Т, К", color="gray")
plt.sca(ax1)
plt.plot(x3, P_duct[len(P_duct) :: -1], "b-.")
plt.xlabel("Глубина H, м", color="gray")
plt.ylabel("Давление P, Па*10^7", color="gray")
plt.sca(ax2)
plt.plot(x3, v_duct[len(v_duct) :: -1], "b-.")
plt.xlabel("Глубина H, м", color="gray")
plt.ylabel("Скорость v, м/с", color="gray")
plt.sca(ax3)
plt.plot(x3, p_duct[len(p_duct) :: -1], "b-.")
plt.xlabel("Глубина H, м", color="gray")
plt.ylabel("Плотность р, кг/м^3", color="gray")
plt.sca(ax4)
fig.tight_layout()
plt.show()
print('✨ 🍰 ✨')