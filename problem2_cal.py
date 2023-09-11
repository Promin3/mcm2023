import numpy as np
import pandas as pd
from shapely.geometry import Polygon

ST_s = [9, 10.5, 12, 13.5, 15]
phi = 39.4
H = 3
G0 = 1.366
w_mirror = 5.865
h_mirror = 3.4
h_tower = 80
h_hot = 8
d_hot = 7

# Day value
D_s = np.array([306, 337, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275])
# 𝛿 为太阳赤纬角
delta_s = np.arcsin(np.sin(2 * np.pi * D_s / 365) * np.sin(np.radians(23.45)))
# 𝜔 为太阳时角
omega_s = [np.pi / 12 * (ST - 12) for ST in ST_s]

alpha_jiaodu = {}
gamma_jiaodu = {}
alpha_s_values = {}
gamma_s_values = {}
for index, (D, delta) in enumerate(zip(D_s, delta_s)):
    month = index + 1
    alpha_s_values[month] = []
    gamma_s_values[month] = []
    alpha_jiaodu[month] = []
    gamma_jiaodu[month] = []
    for omega in omega_s:
        alpha_s = np.arcsin(
            np.cos(delta) * np.cos(np.radians(phi)) * np.cos(omega) + np.sin(delta) * np.sin(np.radians(phi)))
        tmp =  (np.sin(delta) - np.sin(alpha_s) * np.sin(np.radians(phi))) / (np.cos(alpha_s) * np.cos(np.radians(phi)))
        tmp = np.clip(tmp, -1, 1)
        gamma_s = np.arccos(tmp)
        alpha_s_values[month].append(alpha_s)  # 存入弧度，便于计算
        gamma_s_values[month].append(gamma_s)
        alpha_jiaodu[month].append(np.degrees(alpha_s))
        gamma_jiaodu[month].append(np.degrees(gamma_s))


DNI_values = {}
for month, alpha_s_day_values in alpha_s_values.items():
    DNI_values[month] = []
    for alpha_s in alpha_s_day_values:
        a = 0.4237 - 0.00821 * (6 - H) ** 2
        b = 0.5055 + 0.00595 * (6.5 - H) ** 2
        c = 0.2711 + 0.01858 * (2.5 - H) ** 2
        DNI = G0 * (a + b * np.exp(-c / np.sin(alpha_s)))
        DNI_values[month].append(DNI)

dataframe = pd.read_excel('A/positions_p2.xlsx')
mirror_pos = dataframe.to_numpy()

# eta_ref
eta_ref = 0.92


# eta_at
d_HR_s = np.array([np.sqrt((x+19.83) ** 2 + (y-24) ** 2 + (h_tower - h_mirror) ** 2) for x, y in mirror_pos])

# 调试得到d_HR数值均小于1000
eta_at_values = np.array([0.99321 - 0.0001176 * d_HR + 1.97e-8 * d_HR ** 2 for d_HR in d_HR_s])

eta_at = sum(eta_at_values)/len(eta_at_values)

# eta_sb_values
# 输入当时候时间点对应的alpha_s, gamma_s,以及前后mirror对应的坐标，获得遮挡效率。前面的镜面为A，后面的镜面为B
def cal_eta_sb(alpha_s, gamma_s, x_A, y_A, x_B, y_B):
    d1 = np.array([-3, 3])
    d2 = np.array([3, 3])
    d3 = np.array([-3, -3])
    d4 = np.array([-3, -3])
    s11, s21, s31 = np.cos(alpha_s) * np.cos(gamma_s), np.sin(gamma_s) * np.cos(alpha_s), np.sin(alpha_s)

    T_matrix = np.array([[-np.sin(alpha_s), -np.sin(gamma_s) * np.cos(alpha_s), np.cos(gamma_s) * np.cos(alpha_s)],
                         [np.cos(alpha_s), -np.sin(gamma_s) * np.sin(alpha_s), np.cos(gamma_s) * np.sin(alpha_s)],
                         [0, np.cos(gamma_s), np.sin(gamma_s)]])

    d_matrix_1 = np.array([[d1[0]],
                           [d1[1]],
                           [0]])
    d_matrix_2 = np.array([[d2[0]],
                           [d2[1]],
                           [0]])
    d_matrix_3 = np.array([[d3[0]],
                           [d3[1]],
                           [0]])
    d_matrix_4 = np.array([[d4[0]],
                           [d4[1]],
                           [0]])

    O_A = np.array([[x_A],
                    [y_A],
                    [h_mirror]])

    O_B = np.array([[x_B],
                    [y_B],
                    [h_mirror]])

    d_prime_1 = np.dot(T_matrix, d_matrix_1) + O_A
    d_prime_2 = np.dot(T_matrix, d_matrix_2) + O_A
    d_prime_3 = np.dot(T_matrix, d_matrix_3) + O_A
    d_prime_4 = np.dot(T_matrix, d_matrix_4) + O_A

    d_double_prime1 = np.dot(T_matrix.T, d_prime_1 - O_B)
    d_double_prime2 = np.dot(T_matrix.T, d_prime_2 - O_B)
    d_double_prime3 = np.dot(T_matrix.T, d_prime_3 - O_B)
    d_double_prime4 = np.dot(T_matrix.T, d_prime_4 - O_B)

    S1 = np.array([s11, s21, s31])

    S2 = np.dot(T_matrix, S1)
    # 太阳光线在B中坐标
    s12, s22, s32 = S2[0], S2[1], S2[2]

    x_B_1 = (s32 * d_double_prime1[0][0] - s12 * d_double_prime1[2][0]) / s32
    y_B_1 = (s32 * d_double_prime1[1][0] - s22 * d_double_prime1[2][0]) / s32
    x_B_2 = (s32 * d_double_prime2[0][0] - s12 * d_double_prime2[2][0]) / s32
    y_B_2 = (s32 * d_double_prime2[1][0] - s22 * d_double_prime2[2][0]) / s32
    x_B_3 = (s32 * d_double_prime3[0][0] - s12 * d_double_prime3[2][0]) / s32
    y_B_3 = (s32 * d_double_prime3[1][0] - s22 * d_double_prime3[2][0]) / s32
    x_B_4 = (s32 * d_double_prime4[0][0] - s12 * d_double_prime4[2][0]) / s32
    y_B_4 = (s32 * d_double_prime4[1][0] - s22 * d_double_prime4[2][0]) / s32

    polygon1 = Polygon([(3, 3), (3, -3), (-3, -3), (-3, 3)])
    polygon2 = Polygon([(x_B_1, y_B_1), (x_B_2, y_B_2), (x_B_3, y_B_3), (x_B_4, y_B_4)])

    # 计算两个四边形的交集
    intersection = polygon1.intersection(polygon2)

    # 检查是否有交集
    if intersection.is_empty:
        intersection_area = 0.0
    else:
        intersection_area = intersection.area

    shadow_loss = intersection_area / 36
    eta_sb = 1 - shadow_loss
    # eta_sb = 0.98
    return eta_sb


selected_position = pd.read_excel('A/筛选后的数据.xlsx')
selected_position = selected_position.iloc[:, :2].to_numpy()  # dataframe要转化为numpy array


# 输入当时候时间点对应的alpha_s, gamma_s,获得对应的平均阴影遮挡效率
def cal_eta_sb_average(alpha_s, gamma_s):
    eta_collections = []
    for i in range(0, selected_position.shape[0] - 4, 3):
        eta1 = cal_eta_sb(alpha_s, gamma_s, selected_position[i][0], selected_position[i][1],
                          selected_position[i + 1][0], selected_position[i + 1][1])
        eta_collections.append(eta1)
        eta2 = cal_eta_sb(alpha_s, gamma_s, selected_position[i][0], selected_position[i][1],
                          selected_position[i + 2][0], selected_position[i + 2][1])
        eta_collections.append(eta2)
        eta3 = cal_eta_sb(alpha_s, gamma_s, selected_position[i + 1][0], selected_position[i + 1][1],
                          selected_position[i + 3][0], selected_position[i + 3][1])
        eta_collections.append(eta3)
        eta4 = cal_eta_sb(alpha_s, gamma_s, selected_position[i + 2][0], selected_position[i + 2][1],
                          selected_position[i + 3][0], selected_position[i + 3][1])
        eta_collections.append(eta4)
    # 最后的3个mirror不能凑成一组
    eta5 = cal_eta_sb(alpha_s, gamma_s, selected_position[24][0], selected_position[24][1],
                      selected_position[25][0], selected_position[25][1])
    eta_collections.append(eta5)
    eta6 = cal_eta_sb(alpha_s, gamma_s, selected_position[24][0], selected_position[24][1],
                      selected_position[26][0], selected_position[26][1])
    eta_collections.append(eta6)
    eta_sb_average = sum(eta_collections) / len(eta_collections)
    return eta_sb_average


merged_alpha_gamma = {}
for key in alpha_s_values.keys():
    merged_list = [(x, y) for x, y in zip(alpha_s_values[key], gamma_s_values[key])]
    merged_alpha_gamma[key] = merged_list

eta_sb_month_list = {}
for month, alpha_gamma_tupleList in merged_alpha_gamma.items():
    eta_sb_month_list[month] = []
    for alpha_gamma_tuple in alpha_gamma_tupleList:
        tmp = cal_eta_sb_average(alpha_gamma_tuple[0], alpha_gamma_tuple[1])
        eta_sb_month_list[month].append(tmp)

eta_sb_averageList = []
for month, list in eta_sb_month_list.items():
    avg = sum(list) / len(list)
    eta_sb_averageList.append(avg)

eta_sb_year_average = sum(eta_sb_averageList)/len(eta_sb_averageList)



# eta_cos
d_n = np.array(
    [np.sqrt(h_tower ** 2 + (mirror_pos[i][0]+19.83) ** 2 + (mirror_pos[i][1]-24) ** 2) for i in range(mirror_pos.shape[0])])
x_s = mirror_pos[:, 0]
y_s = mirror_pos[:, 1]


# 输入当前时间点，获得所有mirror的余弦损失序列
def cal_eta_cos(alpha_s, gamma_s):
    # 针对每个mirror计算余弦损失序列
    cos_list = []
    for d, x, y in zip(d_n, x_s, y_s):
        tmp = d + np.sin(alpha_s) * h_tower + x * np.cos(alpha_s) * np.sin(gamma_s) + y * np.cos(alpha_s) * np.sin(gamma_s)
        if np.isnan(tmp) or tmp < 0:
            continue
        cos_item = np.sqrt(d + np.sin(alpha_s) * h_tower + x * np.cos(alpha_s) * np.sin(gamma_s) +
                           y * np.cos(alpha_s) * np.sin(gamma_s)) / np.sqrt(2 * d)
        cos_list.append(cos_item)
    return sum(cos_list)/len(cos_list)


eta_cos_month_list = {}
for month, alpha_gamma_tupleList in merged_alpha_gamma.items():
    eta_cos_month_list[month] = []
    for alpha_gamma_tuple in alpha_gamma_tupleList:
        eta_cos_month_list[month].append(cal_eta_cos(alpha_gamma_tuple[0], alpha_gamma_tuple[1]))

eta_cos_averageList = []
for month, list in eta_cos_month_list.items():
    eta_cos_averageList.append(sum(list) / len(list))

eta_cos_year_average = sum(eta_cos_averageList)/len(eta_cos_averageList)


shadow_loss = 0.02


# eta_trunc_values
eta_trunc_month_list = {}
for month, DNI_list in DNI_values.items():
    eta_trunc_month_list[month] = []
    for index, DNI in enumerate(DNI_list):
        energy_absorbedByCollector = 0.92 * DNI * 6 * 6
        total_reflection_energy = DNI * 6 * 6
        loss_energy = shadow_loss * DNI * 6 * 6
        eta_trunc_month_list[month].append(energy_absorbedByCollector / (total_reflection_energy - loss_energy))

eta_trunc_averageList = []
for month, list in eta_trunc_month_list.items():
    eta_trunc_averageList.append(sum(list)/len(list))

eta_trunc_year_average = sum(eta_trunc_averageList)/len(eta_trunc_averageList)

eta = []
for i in range(12):
    eta.append(eta_sb_averageList[i] * eta_cos_averageList[i] * eta_trunc_averageList[i] * eta_ref * eta_at)

eta_average = sum(eta)/len(eta)


A_i = w_mirror * w_mirror
E_field = {}
N = mirror_pos.shape[0]
for index, (month, DNI_list) in enumerate(DNI_values.items()):
    E_field[month] = []
    for DNI in DNI_list:
        E_field[month].append(DNI * N * A_i * eta[index])

E_field_month_average = []
for month, list in E_field.items():
    E_field_month_average.append(sum(list)/len(list))

E_field_year_average = sum(E_field_month_average)/len(E_field_month_average)

E_field_permirror = []
for e in E_field_month_average:
    E_field_permirror.append(e/(N * w_mirror * w_mirror))

E_field_permirror_year = sum(E_field_permirror)/len(E_field_permirror)

