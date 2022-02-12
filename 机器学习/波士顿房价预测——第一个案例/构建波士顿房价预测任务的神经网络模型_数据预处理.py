# 导入需要用到的package
import numpy as np
import json
# 步骤细节
# '''
# # 读入训练数据
# '''
# datafile = './work/housing.data'
# data = np.fromfile(datafile, sep=' ')
# #print(data)
#
# '''
# # 数据形状变换
# '''
# # 读入之后的数据被转化成1维array，其中array的第0-13项是第一条数据，第14-27项是第二条数据，以此类推....
# # 这里对原始数据做reshape，变成N x 14的形式
# feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS',
#                  'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
# feature_num = len(feature_names)
# data = data.reshape([data.shape[0] // feature_num, feature_num])
# # 查看数据
#
# print(data.shape) #读取 行列
# #print('data:',data[0:2,0])  #二维的切片方式  [0:2,0] =[0:2][0]
#
# '''
# # 数据集划分
# '''
# # 我们将80%的数据用作训练集，20%用作测试集
# # 通过打印训练集的形状，可以发现共有404个样本，每个样本含有13个特征和1个预测值。
# ratio = 0.8
# offset = int(data.shape[0] * ratio) #506*0.8
# training_data = data[:offset] #切片 data[0:offset]
# print('training_data:',training_data.shape)
#
# '''
# # 数据归一化处理
# # '''
# # 对每个特征进行归一化处理，使得每个特征的取值缩放到0~1之间。
# # 这样做有两个好处：
# # 一是模型训练更高效
# # 二是特征前的权重大小可以代表该变量对预测结果的贡献度（因为每个特征值本身的范围相同）。
# # 计算train数据集的最大值，最小值，平均值
# maximums, minimums, avgs = \
#                      training_data.max(axis=0), \
#                      training_data.min(axis=0), \
#      training_data.sum(axis=0) / training_data.shape[0]
# # 对数据进行归一化处理
# for i in range(feature_num):
#     #print(maximums[i], minimums[i], avgs[i])
#     data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])


'''将上述三步封装成函数;return training_data, test_data'''
def load_data():
    '''
    # 读入训练数据
    '''
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ')
    # print(data)

    '''
    # 数据形状变换
    '''
    # 读入之后的数据被转化成1维array，其中array的第0-13项是第一条数据，第14-27项是第二条数据，以此类推....
    # 这里对原始数据做reshape，变成N x 14的形式
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                     'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    # 查看数据

    #print(data.shape)  # 读取 行列
    # print('data:',data[0:2,0])  #二维的切片方式  [0:2,0] =[0:2][0]

    '''
    # 数据集划分
    '''
    # 我们将80%的数据用作训练集，20%用作测试集
    # 通过打印训练集的形状，可以发现共有404个样本，每个样本含有13个特征和1个预测值。
    ratio = 0.8
    offset = int(data.shape[0] * ratio)  # 506*0.8
    training_data = data[:offset]  # 切片 data[0:offset]
    #print('training_data:', training_data.shape)

    '''
    # 数据归一化处理
    # '''
    # 对每个特征进行归一化处理，使得每个特征的取值缩放到0~1之间。
    # 这样做有两个好处：
    # 一是模型训练更高效
    # 二是特征前的权重大小可以代表该变量对预测结果的贡献度（因为每个特征值本身的范围相同）。
    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = \
        training_data.max(axis=0), \
        training_data.min(axis=0), \
        training_data.sum(axis=0) / training_data.shape[0]
    # 对数据进行归一化处理
    for i in range(feature_num):
        # print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset] #506*0.8=404
    test_data = data[offset:]
    return training_data, test_data

# 获取数据
training_data, test_data = load_data()
x = training_data[:, :-1] #不包括最后一个
y = training_data[:, -1:] #只取最后一个

if __name__ == '__main__':
    # 查看数据
    print(training_data[0])
    print('x[0]:',x[0])
    print('y[0]:',y[0])

