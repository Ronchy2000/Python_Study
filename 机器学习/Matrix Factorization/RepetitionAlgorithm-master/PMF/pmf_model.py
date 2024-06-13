from __future__ import print_function
import numpy as np
from numpy.random import RandomState
import pickle
import os
import copy
from evaluations import *
class PMF():
    '''
    a class for this Double Co-occurence Factorization model
    '''
    # initialize some paprameters                                      #原来是50
    def __init__(self, R, lambda_alpha=1e-2, lambda_beta=1e-2, latent_size=10, momuntum=0.8,
                 lr=0.001, iters=200, seed=None):
        #  超参数
        self.lambda_alpha = lambda_alpha
        self.lambda_beta = lambda_beta
        #  按一定比例保留之前的梯度
        self.momuntum = momuntum
        #  用户对电影的评分矩阵NxM
        self.R = R
        # ？？
        self.random_state = RandomState(seed)
        #  迭代次数
        self.iterations = iters
        #  学习率
        self.lr = lr
        #  指示函数，此处用矩阵表示，1表示用户对电影打分，0表示未打分（1表示有数据，0表示无数据))
        self.I = copy.deepcopy(self.R)
        self.I[self.I != 0] = 1
        #  生成用户和电影的特征矩阵，U的维度是NxD,V的维度是DxM
        self.U = 10*self.random_state.rand(np.size(R, 0), latent_size)
        self.V = 10*self.random_state.rand(np.size(R, 1), latent_size)
        print("U.shape",self.U.shape) # U.shape (943, 20)
        print("V.shape",self.V.shape) # V.shape (1682, 20)

    def loss(self):
        # the loss function of the model
        # 也就是论文中的目标函数E，目的是最小化loss
        loss = np.sum(self.I*(self.R-np.dot(self.U, self.V.T))**2) + self.lambda_alpha*np.sum(np.square(self.U)) + self.lambda_beta*np.sum(np.square(self.V))
        return loss
    def predict(self, data, index):
        # data是验证集，取用户和电影这两个维度
        #index_data = np.array([[int(ele[0]), int(ele[1])] for ele in data], dtype=int)  # 维度： len(ele)x2
        # print("index_data:",index_data)
        # print("index_data.shape:",index_data.shape)  #index_data.shape: (20000, 2)
        '''
        self.U.take(index_data.take(0, axis=1), axis=0):根据用户id获得对应的U矩阵，
        '''
        # print("index_data.take(0, axis=1):",index_data.take(0, axis=1))

        #u_features = self.U.take(index_data.take(0, axis=1), axis=0)  # U是NxD维度，index_data.take(0, axis=1)取的是用户信息（每一行的第0个值）
        #v_features = self.V.take(index_data.take(1, axis=1), axis=0)  #
        # u_features = np.r_[self.U,self.U]  #纵向拼接
        # v_features = np.r_[  np.tile(self.V[3-1],(self.U.shape[0],1) )
        #                     ,np.tile(self.V[11-1],(self.U.shape[0],1) )
        #                     ]
        u_features = np.tile(self.U,(len(index),1))
        v_features =  np.tile(self.V[index[0], :], (self.U.shape[0], 1))
        if len(index[1:])  > 0:
            for i in index[1:]:
                print("i:",i)
                tmp = np.tile(self.V[i, :], (self.U.shape[0], 1))
                v_features = np.concatenate((v_features,tmp),axis= 0) #垂直组合
        print("u_features.shape:",u_features.shape)
        # print("v_features.shape:",len(v_features[0]),len(v_features))
        print("v_features.shape:",v_features.shape )
        # v_features = self.V
        # print("self.U:",self.U[410],self.U[209],self.U[198])
        # print("u_features",u_features)
        #print("u_features.shape", u_features.shape) #  (20000, 20)
        #print("v_features.shape", v_features.shape)  # (20000, 20)
        '''
        axis= 0 对a的横轴进行操作，在运算的过程中其运算的方向表现为纵向运算,axis= 1 对a的纵轴进行操作，在运算的过程中其运算的方向表现为横向运算
        '''
        #按行求和 ——————问题可能所在
        preds_value_array = np.sum(u_features*v_features, axis = 1) # 计算预测的评分,u_features*v_features是NxM维，横向求和之后是Nx1维度
        # print("u_features",u_features)
        # print("v_features",v_features)
        # print("u_features*v_features",u_features*v_features)
        # print("u_features*v_features.shape", (u_features * v_features).shape) # 对应元素相乘
        # print("preds_value_array:",preds_value_array)
        return preds_value_array

    def train(self, train_data=None, vali_data=None, vali_index = None):
        '''
        # training process
        :param train_data: train data with [[i,j],...] and this indicates that K[i,j]=rating
        :param lr: learning rate
        :param iterations: number of iterations
        :return: learned V, T and loss_list during iterations
        '''
        train_loss_list = []
        vali_rmse_list = []
        last_vali_rmse = None

        # monemtum
        momuntum_u = np.zeros(self.U.shape)  # NxD维度
        momuntum_v = np.zeros(self.V.shape)  # DxM维度

        for it in range(self.iterations):
            # 梯度下降
            # derivate of Vi，U的梯度，整个矩阵
            grads_u = np.dot(self.I*(self.R-np.dot(self.U, self.V.T)), -self.V) + self.lambda_alpha*self.U

            # derivate of Tj，V的梯度，整个矩阵
            grads_v = np.dot((self.I*(self.R-np.dot(self.U, self.V.T))).T, -self.U) + self.lambda_beta*self.V

            # update the parameters
            '''
            momuntum_u、momuntum_v保存的是前一次迭代所得到的梯度，乘以self.momuntum表示按比例保留之前所得到的梯度，如果是同方向则是加速作用，如果是反方向则是缓冲作用
            '''
            momuntum_u = (self.momuntum * momuntum_u) + self.lr * grads_u
            momuntum_v = (self.momuntum * momuntum_v) + self.lr * grads_v
            # 更新U、V矩阵
            self.U = self.U - momuntum_u
            self.V = self.V - momuntum_v

            # training evaluation
            # 计算训练时的损失
            train_loss = self.loss()
            # 将训练时的损失保存在数组中
            train_loss_list.append(train_loss)
            # 输入验证集对模型进行预测，获得预测的R
            vali_preds = self.predict(vali_data,vali_index)
            # 与真实的评分计算均方根误差
            real = vali_data.flatten('F') #按数值方向展成一维

            vali_rmse = RMSE(real, vali_preds)
            # 将每次的rmse保存到列表中
            vali_rmse_list.append(vali_rmse)

            print('traning iteration:{: d} ,loss:{: f}, vali_rmse:{: f}'.format(it, train_loss, vali_rmse))
            # 训练截止条件：last_vali_rmse不为空且rmse比前一次迭代的大或相等
            if last_vali_rmse and (last_vali_rmse - vali_rmse) <= 0:
                print('convergence at iterations:{: d}'.format(it))
                break
            else:
                last_vali_rmse = vali_rmse
        # 返回训练得到的U、V、loss_list、rmse_list
        return self.U, self.V, train_loss_list, vali_rmse_list
