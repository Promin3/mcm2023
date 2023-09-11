import random
import math
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# （h_build, h_mirror, w_mirror, x_t, y_t, x_1, y_1, ..... x_i, y_i）

n = 3000

NUM_VARIABLES = 5 + n

VARIABLE_BOUNDS = [(2, 6), (2, 8), (2, 8), (-250, 250), (-250, 250)]
VARIABLE_BOUNDS.extend([(-350, 350)] * n)


def evaluate(individual):
    eta_sum = 0
    cnt = 0
    for i in range(5, len(individual), 2):
        eta_at = 0.99321 - 0.0001176 * math.sqrt((individual[i] - individual[3]) ** 2 +
                                                 (individual[i + 1] - individual[4]) ** 2 + (80 - individual[0]) ** 2) \
                 + 1.97e-8 * ((individual[i] - individual[3]) ** 2 + (individual[i + 1] - individual[4]) ** 2 + (
                    80 - individual[0]) ** 2)
        eta = eta_at * 0.98 * 0.763 * 0.92 * 0.939
        eta_sum += eta
        cnt += 1
    return eta_sum / cnt,


def h_build_h_mirror_is_vaild(individual):
    if individual[0] > individual[1] / 2:
        return True
    else:
        return False


def w_m_h_m_is_vaild(individual):
    return individual[2] >= individual[1]


def check_distance_to_origin(individual):
    for i in range(5, len(individual), 2):
        x = individual[i]
        y = individual[i + 1]
        dist = math.sqrt(x ** 2 + y ** 2)
        if dist >= 350:
            return False
    return True


def check_tower_distance_to_origin(individual):
    x = individual[3]
    y = individual[4]
    dist = math.sqrt(x ** 2 + y ** 2)
    if dist >= 350:
        return False
    return True


def check_distance(individual):
    def euclidean_distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    points = [(individual[i], individual[i + 1]) for i in range(5, len(individual), 2)]
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = euclidean_distance(points[i], points[j])
            if distance < individual[2] + 5:
                return False
    return True


def check_distance_to_target(individual):
    def euclidean_distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    for i in range(5, len(individual), 2):
        x = individual[i]
        y = individual[i + 1]
        dist = euclidean_distance((x, y), (individual[3], individual[4]))
        if dist < 100:
            return False
    return True


toolbox = base.Toolbox()
for i in range(NUM_VARIABLES):
    toolbox.register(f"attr_float_{i}", random.uniform, VARIABLE_BOUNDS[i][0], VARIABLE_BOUNDS[i][1])
toolbox.register("individual", tools.initCycle, creator.Individual,
                 tuple(getattr(toolbox, f"attr_float_{i}") for i in range(NUM_VARIABLES)), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.decorate("evaluate", tools.DeltaPenalty(h_build_h_mirror_is_vaild, 10))
toolbox.decorate("evaluate", tools.DeltaPenalty(w_m_h_m_is_vaild, 10))
toolbox.decorate("evaluate", tools.DeltaPenalty(check_distance, 100000))
toolbox.decorate("evaluate", tools.DeltaPenalty(check_distance_to_origin, 1000000000000000000000))
toolbox.decorate("evaluate", tools.DeltaPenalty(check_distance_to_target, 100000))
toolbox.decorate("evaluate", tools.DeltaPenalty(check_tower_distance_to_origin, 10000))


toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # Gaussian mutation
toolbox.register("select", tools.selTournament, tournsize=3)

# Create an initial population
population = toolbox.population(n=50)

# Run the genetic algorithm
NGEN = 200  # Number of generations
CXPB = 0.02  # Crossover probability
MUTPB = 0.02  # Mutation probability

for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# Get the best individual and its fitness value
best_ind = tools.selBest(population, k=1)[0]
best_fitness = best_ind.fitness.values[0]
print("Best fitness:", best_fitness)
print("Best individual: ", best_ind)

# 去掉前3个数
coordinates = best_ind[3:]

# 每两个数为一组坐标
coordinate_pairs = [(coordinates[i], coordinates[i + 1]) for i in range(0, len(coordinates), 2)]

# 创建一个DataFrame以便写入Excel
coordinate_df = pd.DataFrame(coordinate_pairs, columns=["X", "Y"])

data = coordinate_df

# 提取第一行点的坐标
first_point = data.iloc[0][['X', 'Y']]

# 计算每个数据点与第一行点的距离
data['Distance'] = np.sqrt((data['X'] - first_point['X'])**2 + (data['Y'] - first_point['Y'])**2)

# 根据距离筛选数据
filtered_data = data.iloc[0:1].append(data[data['Distance'] >= 100])

filtered_data['ra'] = np.sqrt(filtered_data['X']**2 + filtered_data['Y']**2)

filtered_data = filtered_data[filtered_data['ra'] <= 350]

# 可以选择将筛选后的数据保存为新的文件
filtered_data.to_excel('A/coordinates_p2.xlsx', index=False)

plt.figure(figsize=(20, 20))
plt.scatter(filtered_data.iloc[1:]['X'], filtered_data.iloc[1:]['Y'], color='blue', s=200)
plt.scatter(filtered_data.iloc[0]['X'], filtered_data.iloc[0]['Y'], color='red', s=400)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Distribution of Points')
plt.show()


print(evaluate(best_ind)[0] * best_ind[1] * best_ind[2] * n * 0.001)