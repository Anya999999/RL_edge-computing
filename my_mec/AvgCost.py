import xlrd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 调节字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
plt.rcParams['mathtext.fontset'] = 'stix'
fonten = FontProperties(fname='Times New Roman')
plt.rcParams['font.size'] = 11  # 更改label字体大小
plt.tick_params(labelsize=12) # 更改坐标轴的刻度

# plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 导入excel文件，以及第几张表
data = xlrd.open_workbook('D:\Program Files\Anaconda3\Lib\site-packages\gym\envs\my_mec\log\Datas59\PDQN_N1.xlsx') # Prices Ratio
table = data.sheets()[0]

row_num = 1 # 10

xAxis = table.col_values(0)[row_num:row_num + 4]
print(xAxis)
column_num = 7 # 改这里 7
tA = table.col_values(column_num)[row_num:row_num + 4]   # 1 7   13
tB = table.col_values(column_num+1)[row_num:row_num + 4]    # 4

tC = table.col_values(column_num+2)[row_num:row_num + 4]
tD = table.col_values(column_num+3)[row_num:row_num + 4]
tE = table.col_values(column_num+4)[row_num:row_num + 4]
#tF = table.col_values(column_num+5)[row_num:row_num + 4]
print(tA,tB,tC,tD,tE)

# 误差
err_row_num = 7
dA = table.col_values(column_num)[err_row_num:err_row_num + 4]
dB = table.col_values(column_num+1)[err_row_num:err_row_num + 4]
dC = table.col_values(column_num+2)[err_row_num:err_row_num + 4]
dD = table.col_values(column_num+3)[err_row_num:err_row_num + 4]
dE = table.col_values(column_num+4)[err_row_num:err_row_num + 4]
#dF = table.col_values(column_num+5)[err_row_num:err_row_num + 4]
print(dA, dB, dC, dE)
fig = plt.gcf()
ax = plt.gca()
# ----N----
ax.set_aspect(8)
plt.xlim(xmin=45, xmax=205)  # N
plt.ylim(ymin=0, ymax=10.5)
plt.xlabel('用户数量', fontproperties="SimSun", fontsize=12)
plt.ylabel('平均能耗$\mathrm{(J)}$', fontproperties="SimSun", fontsize=12)

# --F---
# ax.set_aspect(0.15)
# plt.ylim(ymin=0, ymax=11)
# plt.xlim(xmin=3.8, xmax=7.2)     # F
# plt.xlabel('边缘服务器计算性能$\mathrm{(GHz)}$', fontproperties="SimSun", fontsize=12)
# plt.ylabel('平均能耗$\mathrm{(J)}$', fontproperties="SimSun", fontsize=12)

#-----B----
# ax.set_aspect(0.8)
# # plt.xlim(xmin=14, xmax=31)     #  P-DQN
# plt.xlim(xmin=9, xmax=26)
# plt.ylim(ymin=0, ymax=10.5) # I-PDQN
# # plt.ylim(ymin=0, ymax=10)
# #
# plt.xlabel(r'边缘服务器带宽$\mathrm{(MHz)}$',fontproperties="SimSun", fontsize=12)
# plt.ylabel(r'平均能耗$\mathrm{(J)}$', fontproperties="SimSun", fontsize=12)

#---P_DQN
plt.errorbar(xAxis, tA, yerr=dA, label='P-DQN', linestyle='-')
plt.errorbar(xAxis, tB, yerr=dB, label='Random', linestyle='--')
plt.errorbar(xAxis, tC, yerr=dC, label='Greedy', linestyle='-.')
plt.errorbar(xAxis, tD, yerr=dD, label='NO', linestyle=':')
plt.errorbar(xAxis, tE, yerr=dE, label='HTR', linestyle='dashed')   # , color='#ff81c0'

# ---I-PDQN---
# plt.errorbar(xAxis, tA, yerr=dA, label='I-PDQN', linestyle='-')
#     # plt.axes([1, 199, 0, 50])
# plt.errorbar(xAxis, tB, yerr=dB, label='Random-C', linestyle='--')
# plt.errorbar(xAxis, tC, yerr=dC, label='Greedy-C', linestyle='-.')
# plt.errorbar(xAxis, tD, yerr=dD, label='NO-C', linestyle=':')
# plt.errorbar(xAxis, tE, yerr=dE, label='HTR-C', linestyle='dashed')

#plt.errorbar(xAxis, tF, yerr=dF, label='DQN', linestyle='dashdot', color='#ff81c0')  # , color='y'
plt.legend()

plt.savefig(r'./log/Datas59/Figs/PDQN_N_Cost.svg',dpi=1200, format='svg', bbox_inches ='tight')      # 保存图片

