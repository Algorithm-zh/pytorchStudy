import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# 大多数matplotlib实用程序位于pyplot子模块下
#在图中从位置(0,0)到位置(6，250)画一条线
xpoints = np.array([0,6])
ypoints = np.array([0,20])
# plot函数用于在图表中绘点
plt.plot(xpoints,ypoints)

#绘制多点
xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 9])
plt.plot(xpoints,ypoints)
plt.show()

#默认x点
# 不指定x轴上的点，则默认0，1，2，3.。。。

#标记
# 关键字参数marker，指定标记强调每个点
plt.plot(ypoints, marker = "o")
plt.show()

#格式化字符串fmt
#语法为：marker|line|color
plt.plot(ypoints, 'o:b')#表示用o标记每个点，用虚线画，线为蓝色
plt.show()

#设置尺寸大小markersize，简写为ms
plt.plot(ypoints, 'o:b', ms = 20)
plt.show()

#标记颜色
# 使用mec标记边缘的颜色
# 使用mfc标记内部的颜色
plt.plot(ypoints, 'o:b', ms = 20, mec = 'g', mfc = 'r')
plt.show()

#线条
# 虚实下线用ls表示
# 颜色用color或c
# 宽度用linewidth或lw
# 可以成对画
# plt.plot(x1, y1, x2, y2)


#标签
#xlabel和ylabel函数为x轴和y轴设置标签,使用title设置标题
#设置字体为楷体
plt.rcParams["font.sans-serif"] = ["KaiTi"]
#使用fontdict参数来设置标题和标签的字体属性
font = {'family':'sans-serif','color':'blue','size':20}
plt.xlabel("卡路里",fontdict=font)
plt.ylabel("超级卡路里",fontdict=font)
plt.title("牛逼",fontdict=font)
plt.plot(xpoints,ypoints)
plt.show()


#网格线
plt.xlabel("卡路里")
plt.ylabel("超级卡路里")
plt.title("牛逼")
plt.plot(xpoints,ypoints)
plt.grid()
plt.show()
#axis指定要显示哪个轴的网格线
# plt.grid(axis='x')


#多图
#plot1
plt.subplot(1,2,1)#一行两列第一张子图
plt.plot(xpoints,ypoints);
#plot2
plt.subplot(1,2,2)#一行两列第二张子图
plt.plot(xpoints,ypoints)
plt.show()
#title可以为每个子图加标题
#suptitle可以为所有图加总标题


#散点图
plt.scatter(xpoints,ypoints)
plt.show()
#给每个点上色，只能用c而不能用color
#用数组传入
colors = np.array(['red','blue','green','black'])
plt.scatter(xpoints,ypoints,c = colors)
plt.show()
#颜色图，具体看详细文档


#柱状图
#使用bar函数画图
x = np.array(['A','B','C','D'])
y = np.array([3,8,1,10])
plt.bar(x,y)
plt.show()
#使用barh可以画水平柱状图
#width和height可以设置宽度和水平图的高度



#直方图
# 用hist()函数来创建直方图
#用numpy随机生成一个包含250个值的数组，集中在170左右，标准差为10
x = np.random.normal(170,10,250)
plt.hist(x)
plt.show()


#饼图
#pie()绘制饼图
#labels设置标签
x = np.array([21,34,56,72])
label = ["西瓜","苹果","香蕉","桃子"]
#startangle可以设置开始画的角度
#Explode可以让某一块突出
myexplode = [0.1,0,0,0]
plt.pie(x,labels=label,explode=myexplode,shadow=True)
#可以用shadow设置阴影
#用colors设置颜色，传入对应数组
#legend可以设置图例，也可以设置标题
plt.legend(title = '标题')
plt.show()
