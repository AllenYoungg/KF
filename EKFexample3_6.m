%% example 3.6

clc
clear all
close all


%% 系统描述（具有加性噪声的非线性系统）
% 状态函数：x(k)=0.5*x(k-1)+25*x(k-1)/(1+x(k-1)^2)+8*cos(1.2*(k-1))+w(k-1)
% 观测方程：y(k)=x(k)^2/20+v(k)

T=50;                 % 总时间
Q=1;                  % 过程噪声矩阵
R=1;                  % 测量噪声矩阵
w=sqrt(Q)*randn(1,T); % 产生过程噪声
v=sqrt(R)*randn(1,T); % 产生观测噪声

x=zeros(1,T);         
x(1)=0.1;             % x_0=0.1
y=zeros(1,T);
y(1)=x(1)^2/20+v(1);  % 计算初值

for k=2:T
    x(k)=0.5*x(k-1)+25*x(k-1)/(1+x(k-1)^2)+8*cos(1.2*(k-1))+w(k-1);
    y(k)=x(k)^2/20+v(k);
end


%% EKF滤波算法
Xekf=zeros(1,T);
Xekf(1)=x(1);
Yekf=zeros(1,T);
Yekf(1)=y(1);
P=2*eye(1);                                                         % 估计协方差阵P,P(0)=2

for k=2:T
    Xn=0.5*Xekf(k-1)+25*Xekf(k-1)/(1+Xekf(k-1)^2)+8*cos(1.2*(k-1)); % 状态预测
    Zn=Xn^2/20;                                                     % 观测预测
    F=0.5+25 *(1-Xn^2)/((1+Xn^2)^2);                                % 求状态矩阵F
    H=Xn/10;                                                        % 求观测矩阵
    P=F*P*F'+Q;                                                     % 协方差预测    
    K=P*H'/(H*P*H'+R);                                              % 求卡尔曼增益
    Xekf(k)=Xn+K*(y(k)-Zn);                                         % 状态更新
    P=(eye(1)-K*H)*P;                                               % 协方差阵更新
end


%% 画图与误差分析
Xvar=zeros(1,T);
for k=1:T
    Xvar(k)=(Xekf(k)-x(k))^2;
end

% 滤波效果
figure
hold on;box on;
plot(x,'-ko','MarkerFace','g');
plot(Xekf,'-ks','MarkerFace','b');
legend('真实值','预测值');
xlabel('时间/s');
ylabel('状态');
title('滤波效果');

% 误差分析
figure
hold on;box on;
plot(Xvar,'-ko','MarkerFace','g');
xlabel('时间/s');
ylabel('均方误差');
title('误差分析');