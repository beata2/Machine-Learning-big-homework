% 用主成分分析法
clc;
clear all;
%% 
% ------------读数据
load sn11.txt  %把原始的数据保存在纯文本文件sn22.txt中 
[m,n]=size(sn11); 
x0=sn11(:,1:n-3);  %扰动变量因子
y0 = mean([sn11(:,n),sn11(:,n),sn11(:,n)],2);   % 参考对比值
r=corrcoef(x0)  %计算相关系数矩阵
% ----------输入数据标准化处理 
xb=zscore(x0);  
yb=zscore(y0);  
% -----------主成分分析,s对应变换主因子后矩阵，t特征值，c主成分的系数 
[c,s,t]=princomp(xb) 
% ----------计算累积贡献率
contr=cumsum(t)/sum(t)  
num=input('主成分的个数:')   
fprintf('\n主成分变量回归方程：') 
hg=s(:,1:num)\yb;  
hg=c(:,1:num)*hg;  
hg2=[mean(y0)-std(y0)*mean(x0)./std(x0)*hg, std(y0)*hg'./std(x0)]  
fprintf('y=%f',hg2(1)); 
for i=1:n-3     
    fprintf('+%f*x%d',hg2(i+1),i); 
end
