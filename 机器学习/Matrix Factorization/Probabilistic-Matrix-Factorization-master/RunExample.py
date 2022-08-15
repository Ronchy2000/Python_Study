import matplotlib.pyplot as plt
import numpy as np
from LoadData import load_rating_data, spilt_rating_dat
from sklearn.model_selection import train_test_split
from ProbabilisticMatrixFactorization import PMF
import pandas as pd

if __name__ == "__main__":
    file_path = "data/ml-100k/u.data"
    pmf = PMF()
    pmf.set_params({"num_feat": 15, "epsilon": 0.0001, "_lambda": 0.01, "momentum": 0.008, "maxepoch": 100, "num_batches": 10,
                    "batch_size": 1000})
    # ratings = load_rating_data(file_path) #返回 nx3的矩阵
    df = pd.read_csv("timing_flattern_3列.csv")
    print(df.values)
    ratings = np.array(df.values[:,1:])
    # ratings = ratings.astype(np.int32)
    print("ratings:",ratings)
    print("ratings.shape:",ratings.shape)
    print(len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat)
    train, test = train_test_split(ratings, test_size=0.2) #sklearn的函数  # spilt_rating_dat(ratings)
    # print("train:",train,"test:",test)
    # print("train:", train.shape, "test:", test.shape)
    pmf.fit(train, test)

    # Check performance by plotting train and test errors
    plt.plot(range(pmf.maxepoch), pmf.rmse_train, marker='o', label='Training Data')
    plt.plot(range(pmf.maxepoch), pmf.rmse_test, marker='v', label='Test Data')
    plt.title('The MovieLens Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
    print("precision_acc,recall_acc:" + str(pmf.topK(test)))
