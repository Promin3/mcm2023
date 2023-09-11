import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# data = pd.read_excel('A/附件.xlsx')
# data = pd.read_excel('coordinates.xlsx')
data = pd.read_excel('A/E3.xlsx')
mirror_pos = data.to_numpy()
plt.figure(figsize=(10, 10))
# plt.scatter(data['x坐标 (m)'], data['y坐标 (m)'])
plt.scatter(data['x坐标 (m)'], data['y坐标 (m)'])
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Distribution of Points')
plt.show()
