%%
clc
clear all
close all
% 创建一个0-99的一维矩阵
z = [ in range(100)];
z_watch = np.mat(z);
 
% 创建一个方差为1的高斯噪声，精确到小数点后两位
noise = np.round(np.random.normal(0, 1, 100), 2);
noise_mat = np.mat(noise);
 
% 将z的观测值和噪声相加
z_mat = z_watch + noise_mat;
 
% 定义x的初始状态，即位置和速度
x_mat = np.mat([[0, ], [0, ]]);
% 定义初始状态协方差矩阵,这个矩阵在迭代中会被更新
p_mat = np.mat([[0, 0], [0, 0]]);
% 定义状态转移矩阵
% 第一行算位置，1：老的位置 1：delta_t=1 每秒钟采一次样
% 第二行算速度，0：速度跟老的位置没关系 1：跟上一次最优估计的速度一样
f_mat = np.mat([[1, 1], [0, 1]]);
% 定义状态转移协方差矩阵，这里我们把协方差设置的很小，因为觉得状态转移矩阵准确度高
q_mat = np.mat([[0.0001, 0], [0, 0.0001]]);
% 定义观测矩阵，如果状态有两项，观测只有一项，那么观测矩阵H是一个[1 0]，如果观测的有两项这两项那么观测的矩阵是[1 1]；
h_mat = np.mat([1, 0]);
% 定义观测噪声协方差
r_mat = np.mat([1]);
 
for i = 1:1:100
    x_predict = f_mat * x_mat;
    p_predict = f_mat * p_mat * f_mat.T + q_mat;
    kalman = p_predict * h_mat.T / (h_mat * p_predict * h_mat.T + r_mat);
    x_mat = x_predict + kalman * (z_mat[0, i] - h_mat * x_predict);
    p_mat = (np.eye(2) - kalman * h_mat) * p_predict;
 
    plt.plot(x_mat[0, 0], x_mat[1, 0], 'ro', markersize=1);
 
plt.show();