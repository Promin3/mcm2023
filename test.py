import random
from openpyxl import Workbook

# 创建一个新的Excel工作簿
workbook = Workbook()
sheet = workbook.active

# 写入标题行
sheet.append(["列1", "列2", "列3"])

# 生成并写入1078行数据
for _ in range(1078):
    # 生成随机值
    col1 = random.uniform(4, 6)
    col2 = random.uniform(3, 4)
    col3 = random.uniform(5, 7)

    # 将数据写入Excel表格
    sheet.append([col1, col2, col3])

# 保存Excel文件
workbook.save("random_data.xlsx")
