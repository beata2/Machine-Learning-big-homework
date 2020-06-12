# 数据处理----替换元素
import numpy as np
import pandas as pd
# inputfile = 'F:\\work\\homework\\课程文件\\网课--疫情期间\\机器学习\\大作业\\数据集\\4饮酒与在校表现的关系\\code\\data\\datasets_251_561_student-por.csv' #输入的数据文件
inputfile = 'F:\\work\\homework\\课程文件\\网课--疫情期间\\机器学习\\大作业\\数据集\\4饮酒与在校表现的关系\\code\\data\\datasets_251_561_student-mat.csv' #输入的数据文件
data = np.array(pd.read_csv(inputfile)) #读取数据

# 换值
data[data=='GP'] = 1 ;data[data=='MS'] = 2 ;         data[data=='F'] = 1; data[data=='M'] = 2
data[data=='U'] = 1 ;data[data=='R'] = 2 ;         data[data=='LE3'] = 1; data[data=='GT3'] = 2
data[data=='T'] = 1 ;data[data=='A'] = 2 ;         data[data=='teacher'] = 1; data[data=='health'] = 2;data[data=='services'] = 3; data[data=='at_home'] = 5;data[data=='other'] = 4
data[data=='home'] = 1 ;data[data=='reputation'] = 2 ;data[data=='course'] = 3 ;  data[data=='other'] = 4 ;        data[data=='mother'] = 1; data[data=='father'] = 2;data[data=='other'] = 4
data[data=='yes'] = 1 ;data[data=='no'] = 2


df = pd.DataFrame(data)
# df.to_csv('F:\\work\\homework\\课程文件\\网课--疫情期间\\机器学习\\大作业\\数据集\\4饮酒与在校表现的关系\\code\\data\\por_to_num.csv')
df.to_csv('F:\\work\\homework\\课程文件\\网课--疫情期间\\机器学习\\大作业\\数据集\\4饮酒与在校表现的关系\\code\\data\\mat_to_num.csv')
print("保存成功！")


