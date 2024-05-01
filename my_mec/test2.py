import numpy as np
import copy
# print(pow(10, -174 / 10) * 0.001)
# print(np.round(3.0000000000000004, 4))
# trans_time = [456.4, float("inf")]
# for i in range(len(trans_time)):
#     if trans_time[i] < float("inf"):
#         print(str(trans_time[i]) + " is not a inf num")
#     else:
#         print(str(trans_time[i]) + " is a inf")
# print(np.round(3.444,1))
# a = []
# a.append(2)
# print(a[a.index(2)])

# b = [4, 4]
# print(np.argmax(b))
# a = [(2,3),(4,5)]
# for i in a:
#     print(i[0])
# a = (1,3)
# b = (4,)
# print(a+b)
# a= [1,3,4]
# if 2 not in a:
#     a.append(2)
# print(a)
# remain_servers = [1,2,3,4,5]
# for i in remain_servers:  # 对于过载服务器上即将被迁移的用户来说
#     accepted_servers =copy.copy(remain_servers)
#     if i in accepted_servers:
#         accepted_servers.remove(i)
#     print(accepted_servers)
# print(-5+8)
# print(pow(10, -174 / 10))
# a = []
# a.append(2)
# a.append(3)
# a.append(5)
# a.append(1)
# print(np.argmax(a))
import xlrd
import matplotlib.pyplot as plt

# 调节字体
# plt.rcParams['font.sans-serif'] = ['simsun']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 导入excel文件，以及第几张表
data = xlrd.open_workbook('D:\Program Files\Anaconda3\Lib\site-packages\gym\envs\my_mec\logM\DataFigs.xlsx') # Prices Ratio
table = data.sheets()[0]
# print(table.nrows)  # 50001

row_num = 1
row_num2 = 2
x = ["P-DQN", "Random", "Greedy", "CLF", "HTR"]
# yAxis = [table.col_values(0)[row_num], table.col_values(1)[row_num], table.col_values(2)[row_num],
#          table.col_values(3)[row_num], table.col_values(4)[row_num]]
yAxis = [table.col_values(0)[row_num2], table.col_values(1)[row_num2], table.col_values(2)[row_num2],
         table.col_values(3)[row_num2], table.col_values(4)[row_num2]]
# print(yAxis[0])

plt.bar(x[0], yAxis[0], 0.6, label='P-DQN')
plt.bar(x[1], yAxis[1], 0.6)
plt.bar(x[2], yAxis[2], 0.6)
plt.bar(x[3], yAxis[3], 0.6)
plt.bar(x[4], yAxis[4], 0.6)

plt.xlim(xmin=-1, xmax=5)
plt.ylim(ymin=0, ymax=100)
plt.xlabel('总的任务数量N=100', fontproperties="SimSun", fontsize=12)
# plt.ylabel('所有任务的总能耗', fontproperties="SimSun", fontsize=12)
# plt.title("不同算法与总的能耗关系图",fontproperties="SimSun", fontsize=12)

plt.ylabel('成功完成的任务数量', fontproperties="SimSun", fontsize=12)
plt.title("不同算法与任务完成情况的关系图",fontproperties="SimSun", fontsize=12)
# plt.legend()
# plt.show()
plt.savefig(r'D:\Program Files\Anaconda3\Lib\site-packages\gym\envs\my_mec\logM\Fig_user_N100_.jpg', dpi=600, format='jpg')
