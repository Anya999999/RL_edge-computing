import xlrd
import matplotlib
import matplotlib.pyplot as plt

# 调节字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 导入excel文件，以及第几张表
data = xlrd.open_workbook('D:\Program Files\Anaconda3\Lib\site-packages\gym\envs\my_mec\log\Datas59\PDQN_N1.xlsx') # AllRecords3SB1
table = data.sheets()[0]
# print(table.nrows)  # 50001

row_num = 1 # 10


xAxis = table.col_values(0)[row_num:row_num + 4]
print(xAxis)
    # 第一个图的数据
column_num = 1 #改这里 7
tA = table.col_values(column_num)[row_num:row_num + 4]   # 1 7   13
# print(tA)
#     # print(t1)
#     # print(tA)
#     # 第二个图的数据
tB = table.col_values(column_num+1)[row_num:row_num + 4] #4
    # 第三个图的数据
tC = table.col_values(column_num+2)[row_num:row_num + 4]
tD = table.col_values(column_num+3)[row_num:row_num + 4]
tE = table.col_values(column_num+4)[row_num:row_num + 4]
# tF = table.col_values(column_num+5)[row_num:row_num + 4]
print(tA,tB,tC,tD,tE)

# 误差
err_row_num = 7
dA = table.col_values(column_num)[err_row_num:err_row_num + 4]
dB = table.col_values(column_num+1)[err_row_num:err_row_num + 4]
dC = table.col_values(column_num+2)[err_row_num:err_row_num + 4]
dD = table.col_values(column_num+3)[err_row_num:err_row_num + 4]
dE = table.col_values(column_num+4)[err_row_num:err_row_num + 4]
# dF = table.col_values(column_num+5)[err_row_num:err_row_num + 4]
print(dA, dB, dC, dE)
#     # 作图
# plt.plot(xAxis, tA, label='I-PDQN', linestyle='dashed', marker='o')
#     # plt.axes([1, 199, 0, 50])
# plt.plot(xAxis, tB, label='Random-C', linestyle='dashed', marker='<')
# plt.plot(xAxis, tC, label='Greedy-C', linestyle='dashed', marker='s')
# plt.plot(xAxis, tD, label='NO-C', linestyle='dashed', marker='>')
# plt.plot(xAxis, tE, label='HTR-C', linestyle='dashed', marker='x')

# plt.plot(xAxis, tA, label='P_DQN', linestyle='dashed', marker='o')
#     # plt.axes([1, 199, 0, 50])
# plt.plot(xAxis, tB, label='Random', linestyle='dashed', marker='<')
# plt.plot(xAxis, tC, label='Greedy', linestyle='dashed', marker='s')
# plt.plot(xAxis, tD, label='NO', linestyle='dashed', marker='>')
# plt.plot(xAxis, tE, label='HTR', linestyle='dashed', marker='x')
# plt.plot(xAxis, tF, label='DQN', linestyle='dashed', marker='1')

#-------------NNNNNNNNNNNN-----------------------
# plt.xlim(xmin=45, xmax=205)  #能耗
# plt.ylim(ymin=0, ymax=10)
# plt.xlabel('用户数量', fontproperties="SimSun", fontsize=14)
# plt.ylabel('平均能耗', fontproperties="SimSun", fontsize=14)
# plt.title("用户数量与平均能耗关系", fontproperties="SimSun", fontsize=14)
# #
# plt.xlim(xmin=45, xmax=205)   #用户数量的坐标
# plt.ylim(ymin=5, ymax=150)
# plt.xlabel('用户数量', fontproperties="SimSun", fontsize=14)
# plt.ylabel('总服务的用户数量',fontproperties="SimSun", fontsize=14)
# plt.title("用户数量与总服务的用户数量关系", fontproperties="SimSun", fontsize=14)

# plt.xlim(xmin=45, xmax=205)  # 平均服务时间
# plt.ylim(ymin=1.5, ymax=4.5)
# plt.xlabel('用户数量', fontproperties="SimSun", fontsize=14)
# plt.ylabel('平均服务时间', fontproperties="SimSun", fontsize=14)
# plt.title("用户数量与平均服务时间关系", fontproperties="SimSun", fontsize=14)
#--------------------------FFFFFFFFFFFF-----------
# plt.xlim(xmin=3.8, xmax=7.2)  # 成本的坐标
# plt.ylim(ymin=0, ymax=11)
# plt.xlabel('边缘服务器计算性能', fontproperties="SimSun", fontsize=14)
# plt.ylabel('平均能耗', fontproperties="SimSun", fontsize=14)
# plt.title("边缘服务器计算性能与平均能耗关系", fontproperties="SimSun", fontsize=14)

# plt.xlim(xmin=3.8, xmax=7.2)  #用户数量的坐标
# plt.ylim(ymin=10, ymax=105)
# plt.xlabel('边缘服务器计算性能', fontproperties="SimSun", fontsize=14)
# plt.ylabel('总服务的用户数量',fontproperties="SimSun", fontsize=14)
# plt.title("边缘服务器计算性能与总服务的用户数量关系", fontproperties="SimSun", fontsize=14)
#
# plt.xlim(xmin=3.8, xmax=7.2)  # 时间的坐标
# plt.ylim(ymin=1, ymax=4.5)
# plt.xlabel('边缘服务器计算性能', fontproperties="SimSun",fontsize=14)
# plt.ylabel('平均服务时间', fontproperties="SimSun", fontsize=14)
# plt.title("边缘服务器计算性能与平均服务时间关系", fontproperties="SimSun", fontsize=14)

#-----------------------------B----------
# plt.xlim(xmin=9.5, xmax=25.5)  # 能耗的坐标
# plt.ylim(ymin=0, ymax=10)
# plt.xlabel('边缘服务器带宽', fontproperties="SimSun", fontsize=14)
# plt.ylabel('平均能耗', fontproperties="SimSun", fontsize=14)
# plt.title("边缘服务器带宽与平均能耗关系", fontproperties="SimSun", fontsize=14)
#
# plt.xlim(xmin=9.5, xmax=25.5)  #用户数量的坐标
# plt.ylim(ymin=10, ymax=105)
# plt.xlabel('边缘服务器带宽', fontproperties="SimSun", fontsize=14)
# plt.ylabel('总服务的用户数量',fontproperties="SimSun", fontsize=14)
# plt.title("边缘服务器带宽与总服务的用户数量关系", fontproperties="SimSun", fontsize=14)

# plt.xlim(xmin=9.5, xmax=25.5)  # 时间的坐标
# plt.ylim(ymin=1, ymax=4.5)
# plt.xlabel('边缘服务器带宽', fontproperties="SimSun", fontsize=14)
# plt.ylabel('平均服务时间', fontproperties="SimSun", fontsize=14)
# plt.title("边缘服务器带宽与平均服务时间关系", fontproperties="SimSun", fontsize=14)

# plt.figure(figsize=(9,6), dpi=1200)  # 图片长宽和清晰度
ax = plt.gca()

# ----N-----
ax.set_aspect(0.6)  # N
plt.xlim(xmin=45, xmax=205)   # N
plt.ylim(ymin=5, ymax=140)
plt.xlabel('用户数量', fontproperties="SimSun", fontsize=12)
plt.ylabel('总服务的用户数量',fontproperties="SimSun", fontsize=12)

# -- F---
# ax.set_aspect(0.020)
# plt.xlim(xmin=3.8, xmax=7.2)     # F
# plt.ylim(ymin=10, ymax=105)
# plt.xlabel('边缘服务器计算性能', fontproperties="SimSun", fontsize=12)
# plt.ylabel('总服务的用户数量',fontproperties="SimSun", fontsize=12)

#---B------------
# ax.set_aspect(0.10)
# plt.xlim(xmin=9, xmax=26)     # F
# # plt.xlim(xmin=14, xmax=31)
# plt.ylim(ymin=20, ymax=105)
# plt.xlabel('边缘服务器带宽',fontproperties="SimSun", fontsize=12)
# plt.ylabel('总服务的用户数量',fontproperties="SimSun", fontsize=12)

#---P-DQN
# plt.errorbar(xAxis, tA, yerr=dA, label='P-DQN', linestyle='-')
#     # plt.axes([1, 199, 0, 50])
# plt.errorbar(xAxis, tB, yerr=dB, label='Random', linestyle='--')
# plt.errorbar(xAxis, tC, yerr=dC, label='Greedy', linestyle='-.')
# plt.errorbar(xAxis, tD, yerr=dD, label='NO', linestyle=':')
# plt.errorbar(xAxis, tE, yerr=dE, label='HTR', linestyle='dashed')   # , color='#ff81c0'
# plt.errorbar(xAxis, tF, yerr=dF, label='DQN', linestyle='dashdot', color='#ff81c0')  # , color='y'

# ---I-PDQN---
# plt.errorbar(xAxis, tA, yerr=dA, label='I-PDQN', linestyle='-')
#     # plt.axes([1, 199, 0, 50])
# plt.errorbar(xAxis, tB, yerr=dB, label='Random-C', linestyle='--')
# plt.errorbar(xAxis, tC, yerr=dC, label='Greedy-C', linestyle='-.')
# plt.errorbar(xAxis, tD, yerr=dD, label='NO-C', linestyle=':')
# plt.errorbar(xAxis, tE, yerr=dE, label='HTR-C', linestyle='dashed')

plt.legend()

# plt.gca().spines["top"].set_alpha(0.3)
plt.savefig(r'./log/Datas59/Figs/PDQN_N.svg',dpi=1200, format='svg',bbox_inches ='tight')      # 保存图片
# plt.show()

# fig = plt.gcf()
# fig.savefig(r'1.svg',dpi=1200, format='svg')

# plt.show()
# plt.savefig(r'D:\Program Files\Anaconda3\Lib\site-packages\gym\envs\my_mec\log\Figs58\PDQN_NUser.svg',dpi=1200, format='svg')

