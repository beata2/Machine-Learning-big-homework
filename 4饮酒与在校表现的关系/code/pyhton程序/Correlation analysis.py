import numpy as np
import pandas as pd
inputfile = 'F:\\work\\homework\\课程文件\\网课--疫情期间\\机器学习\\大作业\\数据集\\4饮酒与在校表现的关系\\code\\data\\datasets_251_561_student-mat.csv' #输入的数据文件
data = pd.read_csv(inputfile) #读取数据
corr = np.round(data.corr(method = 'pearson'), 2) #计算相关系数矩阵，保留两位小数
print("corr:",corr)


# 保存
# df = pd.DataFrame(corr)
# df.to_csv('F:\\work\\homework\\课程文件\\网课--疫情期间\\机器学习\\大作业\\数据集\\4饮酒与在校表现的关系\\code\\data\\corr.csv')
# print("保存成功！")