#numpy是处理数组的python库
#提供一个数组对象，比传统的python列表快
#numpy中的数组对象称为ndarray
#数组被存储在内存中的一个连续位置，所以进程可以非常快的操作


import numpy as np

#数组
# 可以用array创建一个ndarray对象
arr = np.array([1,2,3,4,5])
print(type(arr))
#创建ndarray，可以将列表、元组或任何类似数组对象传给array，转换为ndarray
arr = np.array((1,2,3,4))
print(arr)
#维度
arr = np.array([[1,2,3],[2,3,4]])
print(arr)
#ndim查看维度
print(arr.ndim)
#可以用ndmin参数定义维数
arr = np.array([1,2,3,4],ndmin=5)
print(arr)#[[[[[1 2 3 4]]]]]
#访问多维数组
arr = np.array([[1,2,3],[2,3,4]])
print(arr[0,1])#访问第一维中的第二个元素
#负索引可以从后往前找
#数组裁剪[start:end:step]
# 不传递start，视为0，不传递end，视为该维度内数组的长度，不传递step，视为1
#start-end，左闭右开
print(arr[1,1:])


#数据类型
# i代表整数，u代表无符号整数，f浮点，c复合浮点数，m时间区间，M时间，O对象，S字符串，Uunicode字符串，V固定的其他类型的内存块
#dtype可以返回数组的数据类型
# arr.dtype
# 可以创建时使用dtype指定数据类型
#4字节整数的数组
arr = np.array([1,2,3,4], dtype='i4')
# astype()可以复制数组并且修改数据类型
newarr = arr.astype('f')
print(newarr)


#副本和视图
#副本是一个新数组，视图只是原始数组的视图
# 副本拥有数据，对副本修改不会影响原数组，视图正好相反
arr = np.array([1,2,3,4], dtype='i4')
x = arr.copy()
arr[0] = 3
print(arr)
print(x)
#视图,只有视图有base属性
y = arr.view()
arr[0] = 9
print(arr)
print(y)


#数组的形状
#每个维度中元素的数量
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr.shape)
#重塑数组，重塑的数组数量必须相同
newarr = arr.reshape(9)
print(newarr)
newarr2 = newarr.reshape(3,3)
print(newarr2)
#重塑之后返回的是视图
#可以使用未知的维度
# 传递-1作为值，numpy将自动计算数字
arr = np.array([1,2,3,4,5,6,7,8])
newarr = arr.reshape(2,2,-1)
print(newarr)
print(newarr.shape)
#展平数组指将多维数组转化为1维数组
# 可以用reshape(-1)
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
newarr = arr.reshape(-1)
print(newarr)


#numpy的数组操作

#数组迭代
#使用for循环
# for x in arr
#使用nditer迭代数组可以直接迭代高维数组，不需要多层for循环
for x in np.nditer(arr):
    print(x)
#可以使用op_dtypes参数传递期望的数据类型，以在迭代时更改元素的数据类型
#numpy不会就地更改元素的数据类型，需要一些其他空间来执行操作，此额外空间称为buffer，为了在nditer()中启用它，我们传递参数flags=["buffered"]
arr =np.array([1,2,3])
for x in np.nditer(arr,flags=['buffered'],op_dtypes=['S']):
    print(x)
#ndenumerate()方法可以迭代数组,idx表示索引
for idx, x in np.ndenumerate(arr):
    print(idx,x)

#数组连接
arr1 = np.array([[1,2],[3,4]])
arr2 = np.array([[5,6],[7,8]])
# arr = np.concatenate((arr1,arr2), axis=1)#axis = 1按行连
arr = np.stack((arr1,arr2))#多加一个括号
print(arr,arr.shape)
#hstack，vstack，dstack按不同方式堆叠
#数组分割
# np.array_split(arr,4)#不能均分系统会自动调配
# np.split(arr,3)#均分，如果不能均分会报错

#数组搜索
#where方法
print(arr)
x = np.where(arr == 4)
print(x)#返回的是位置
#排序搜索
arr = np.array([1,3,5,7])
x = np.searchsorted(arr,3)#从左往右找第一个大于3的(]
print(x)
x = np.searchsorted(arr,3,side='right')#从右往左找[)
print(x)
x = np.searchsorted(arr,[3,5,7])#可以找多个值


#数组过滤
# 可以使用布尔数组来过滤数组
arr = np.array([1,3,5,7])
x = [True,False,True,False]
newarr = arr[x]
print(newarr)
#简单过滤器写法
filter_arr = arr > 3
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

