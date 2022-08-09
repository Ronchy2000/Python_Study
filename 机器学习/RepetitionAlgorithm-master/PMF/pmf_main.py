from __future__ import print_function
from evaluations import *
from pmf_model import *
import pandas as pd

print('PMF Recommendation Model Example')

# choose dataset name and load dataset, 'ml-1m', 'ml-10m'
dataset = 'ml-100k'
processed_data_path = os.path.join(os.getcwd(), 'processed_data', dataset)
user_id_index = pickle.load(open(os.path.join(processed_data_path, 'user_id_index.pkl'), 'rb'),encoding='bytes')
item_id_index = pickle.load(open(os.path.join(processed_data_path, 'item_id_index.pkl'), 'rb'),encoding='bytes')
data = np.loadtxt(os.path.join(processed_data_path, 'data.txt'), dtype=float)

# set split ratio
ratio = 0.6
# train_data = data[:int(ratio*data.shape[0])]  # 前60%为训练集
# vali_data = data[int(ratio*data.shape[0]):int((ratio+(1-ratio)/2)*data.shape[0])]  # 从训练集中取20%作为验证集
# test_data = data[int((ratio+(1-ratio)/2)*data.shape[0]):]  # 后20%作为测试集

# print("train_data.shape",train_data.shape) #(60000, 3)
# print("vali_data.shape",vali_data.shape)   #(20000, 3)
# print("test_data.shape",test_data.shape)   #(20000, 3)

# NUM_USERS = max(user_id_index.values()) + 1
# NUM_ITEMS = max(item_id_index.values()) + 1
NUM_PATH = 1404
NUM_Corner = 14
# print("R的大小：{}x{}".format(NUM_USERS,NUM_ITEMS))
# print('dataset density:{:f}'.format(len(data)*1.0/(NUM_USERS*NUM_ITEMS)))

#----------------------------------------------------------
#将用于train的数据，填充成  NUM_USERS  x  NUM_ITEMS 矩阵中
# R = np.zeros([NUM_USERS, NUM_ITEMS])
# for ele in train_data:
#     R[int(ele[0]), int(ele[1])] = float(ele[2])
    #print(R[int(ele[0]), int(ele[1])])
# print("R:",R)
#===========================
R = np.zeros([NUM_PATH,NUM_Corner])

df = pd.read_csv("mydata2_corner1-corner14.csv")
feature_name,target_name = [],[]
header = list(df.columns.values)
target_name = header[:] #拷贝给target_name
R = np.array(df.values)
R[:,[3,11,5,13]] = 0

vali_data = df.values[:,[3,11]]  # path *2
test_data = df.values[:,[5,13]]  # path *2
'''
训练集：Corner1-Corner14（除去验证测试）
验证集：Corner3，Corner11
测试集：Corner5，Corner13
'''


#----------------------------------------------------------
# construct model
print('training model.......')
lambda_alpha = 0.01
lambda_beta = 0.01
latent_size = 20
lr = 3e-5
iters = 1000
model = PMF(R=R, lambda_alpha=lambda_alpha, lambda_beta=lambda_beta, latent_size=latent_size, momuntum=0.9, lr=lr, iters=iters, seed=1)
print('parameters are:ratio={:f}, reg_u={:f}, reg_v={:f}, latent_size={:d}, lr={:f}, iters={:d}'.format(ratio, lambda_alpha, lambda_beta, latent_size,lr, iters))
# U, V, train_loss_list, vali_rmse_list = model.train(train_data=train_data, vali_data=vali_data)
#
# print('testing model.......')
# preds = model.predict(data=test_data)
# test_rmse = RMSE(preds, test_data[:, 2])

# print('test rmse:{:f}'.format(test_rmse))
