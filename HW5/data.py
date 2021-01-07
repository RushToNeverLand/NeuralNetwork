import numpy as np
import matplotlib.pyplot as plt
    
def generate_data(n, r=10, w=6, d=1):
    theta1 = np.random.uniform(0, np.pi, size=n)
    theta2 = np.random.uniform(-np.pi, 0, size=n)
    w1 = np.random.uniform(-w/2, w/2, size=n)
    w2 = np.random.uniform(-w/2, w/2, size=n)
    one = np.ones_like(theta1)
    
    # data_A_i = [1, coord_x, coord_y, label], label = 1 or -1
    data_A = np.array([one, (r+w1)*np.cos(theta1), (r+w1)*np.sin(theta1), one]).T
    data_B = np.array([one, r + (r+w2)*np.cos(theta2), -d + (r+w2)*np.sin(theta2), -one]).T
    return data_A, data_B
    
class Data:
    def __init__(self, n, r=10, w=6, d=1):
        self.n = n          # 数据对数，A、B区域各有n个点
        self.r = r          # 半径
        self.w = w          # 圆环的宽度
        self.d = d          # 区域B的向下偏移量
        self.data_A = []    # 区域A的数据集
        self.data_B = []    # 区域B的数据集
        self.data_AB = []   # 混合区域A和区域B的数据集
    
    def get_data(self):
        self.data_A, self.data_B = generate_data(self.n, self.r, self.w, self.d)
        all_data = np.vstack([self.data_A,self.data_B])
        np.random.shuffle(all_data)
        self.data_AB = all_data
    
    def plot(self):
        fig = plt.figure()
        plt.scatter(self.data_A[:, 1], self.data_A[:, 2], marker='x')
        plt.scatter(self.data_B[:, 1], self.data_B[:, 2], marker='+')
        plt.show()
        
    
if __name__ == "__main__":
    # data_A, data_B = generate_data(1000)
    # print(data_A)
    # print(data_B)
    # fig = plt.figure()
    # plt.scatter(data_A[:,1], data_A[:, 2], marker='x')
    # plt.scatter(data_B[:,1], data_B[:, 2], marker='+')
    # plt.show()
    
    data_set = Data(1000)
    data_set.get_data()
    print(data_set.data_AB)
    data_set.plot()

