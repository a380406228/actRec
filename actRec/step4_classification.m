% 分类，得到最后结果
% step1:kmean聚类
% step2:svm分类，直方图交叉核

% param setting
iterMax = 50;
numFeatures = 225;    % number of features to learn
batchNumPatches = 2000; 

% for kmeans
nClusters = 4000;
eps = 0.5;

% for toupiao
track_num = 200;

% for batch
sample_class_num = 6;
dnum = 4;
sample_num = 25;
sample_total = sample_num*dnum*sample_class_num;
act_name = {'boxing','handclapping','handwaving','jogging','running','walking'};

% for test
sample_class_num = 6;
dnum = 1;
sample_num = 5;
sample_total = sample_num*dnum*sample_class_num;

% step1:先加载pool_max,矫正格式（转置）后输出到txt文件
loadpath = sprintf('data\\pool_max_NF%d_BN%d_iter%d.mat',numFeatures,batchNumPatches,iterMax);
pool_max_batch = load(loadpath);
pool_max_batch = pool_max_batch.pool_max_batch;

% 将数据输出到txt文件保存
savepath = sprintf('data\\pool_max_NF%d_BN%d_iter%d.txt',numFeatures,batchNumPatches,iterMax);
x= pool_max_batch';
fid=fopen(savepath,'wt');
[m,n]=size(x);
for i=1:1:m
    for j=1:1:n
        if j==n
            fprintf(fid,'%f\n',x(i,j));
            if mod(i,1000)==0
                disp(i);
            end;
        else
            fprintf(fid,'%f ',x(i,j));
        end;
    end;
end;
disp(savepath);
fclose(fid);


% kmeans聚类
% mykmeans input.txt output.txt rows cols nClusters eps
loadpath = sprintf('data\\pool_max_NF%d_BN%d_iter%d.txt',numFeatures,batchNumPatches,iterMax);
bestLabelPath = sprintf('data\\bestLabels_NF%d_BN%d_iter%d.txt',numFeatures,batchNumPatches,iterMax);

pool_max_load = load(loadpath);
rows = size(pool_max_load,1); % 6000
cols = size(pool_max_load,2); % 225
commandstr = sprintf('mykmeans.exe %s %s %d %d %d %f',loadpath,bestLabelPath,rows,cols,nClusters,eps);
disp(commandstr)
system(commandstr);

% 对得到的bestLabels进行投票
% 一个样本200条轨迹，一共30个样本，6类，每类5种
bestLabels = load(bestLabelPath);
feature = zeros(sample_total,nClusters);
tmp = zeros(1,track_num);
for sp=1:sample_total
    if mod(sp,50)==0
        disp(sp)
    end;
    tmp = bestLabels([1+(sp-1)*track_num:sp*track_num])';
    for i=1:track_num %1--200
        for j=1:nClusters %1--4000
            if(j==tmp(i))
                feature(sp,j) = feature(sp,j)+1;
            end;
        end;
    end;    
end;

% 分类
tr_set = [1 4 11 12 13 14 15 16 17 18 19 20 21 23 24 25];
tt_set = [2 3 5 6 7 8 9 10 22];
train_data=[];
test_data=[];
train_label = [];
test_label = [];

pcnt = 0;
for acti = 1:sample_class_num %1:6
    for peri = 1:sample_num % 1:25
        for di = 1:dnum % 1:4
            pcnt = pcnt+1;
            % train_data
            for tr_id=1:size(tr_set,2)
                if peri==tr_set(tr_id)
                    train_data=[train_data; feature(pcnt,:)];
                    train_label=[train_label; acti];
                end;
            end;
            % test_data
            for tt_id=1:size(tt_set,2)
                if peri==tt_set(tt_id)
                    test_data=[test_data; feature(pcnt,:)];
                    test_label=[test_label; acti];
                end;
            end;            
        end;
    end;
end;

tr_num = size(train_data,1)
tt_num = size(test_data,1)

% Linear Kernel
model_linear = libsvmtrain(train_label, train_data, '-t 0');
[predict_label_L, accuracy_L, dec_values_L] = libsvmpredict(test_label, test_data, model_linear);

% 直方图交叉核
% 使用的核函数 K(x,x') = sum(min(lamda(x),lamda(x') ) )
ktrain4 = ones(tr_num,tr_num);
m = nClusters
for i = 1:tr_num
    %disp(i)
    for j = 1:tr_num            
        mintmp=zeros(1,m);
        for k=1:m
            if train_data(i,k)<train_data(j,k)
                mintmp(k)=train_data(i,k);
            else
                mintmp(k)=train_data(j,k);
            end;
        end;
        ktrain4(i,j) = sum(mintmp);    

    end
end
Ktrain4 = [(1:tr_num)',ktrain4];
model_precomputed4 = libsvmtrain(train_label, Ktrain4, '-t 4');

disp('*****************************************************')

ktest4 = ones(tt_num,tr_num);
for i = 1:tt_num
    %disp(i)
    for j = 1:tr_num
        mintmp=zeros(1,m);
        for k=1:m
            if test_data(i,k)<train_data(j,k)
                mintmp(k)=test_data(i,k);
            else
                mintmp(k)=train_data(j,k);
            end;
        end;
        ktest4(i,j) = sum(mintmp);
    end
end
Ktest4 = [(1:tt_num)', ktest4];
[predict_label_P4, accuracy_P4, dec_values_P4] = libsvmpredict(test_label, Ktest4, model_precomputed4);






