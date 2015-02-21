% 前提是要得到每个样本的原始patches特征
% 用C++写的提取patches的特征，得到600个样本的原始特征，每个200个特征
% 每行一个特征，分别是15帧的patches（32*32=1024）大小15帧，一共1024*15=15360列，200行。
% 路径：D:\myproject\data\dataoriginrand200
% person01_boxing_d1_origin.txt--...

% param setting
track_length = 15;
track_num = 200;
size_origin = 32;
size_new = 16;

ext_ratio = 0.1;
patches_origin_num = track_length*track_num; % 15*200=3000
patches_ext_num = int32(patches_origin_num*ext_ratio); %每个样本需要抽取的patches数,3000*0.1=300


addpath('D:\\myproject\\data\\dataoriginrand200');
feaorigin = load('person01_boxing_d1_origin.txt');

% step1:随机提取每个样本中10%的patches（以样本为单位平均采样）
% 每个样本15*200=3000个patches，平均抽取300个，一共300*600个patches
% 提取的同时进行resize降维
% 得到：16*16=256行（特征维数），180000列（300*600）的抽样和resize后的特征。

% step2:进行预处理，白化
% 预处理：1.resize 2.normalize 3.ZCA

rand_idx = randperm(patches_origin_num);
rand_idx = rand_idx([1:patches_ext_num]);
rand_idx = sort(rand_idx);

patches_ext = zeros(size_new*size_new,patches_ext_num);
pcnt = 0;
for patches_ext_idx = 1:patches_ext_num
    pcnt = pcnt+1;
    randone_idx = rand_idx(patches_ext_idx)-1;
    rand_r = floor(randone_idx/track_length)+1;
    rand_c = mod(randone_idx,track_length)+1;
    
    patch_row = feaorigin(rand_r,:);
    patch_one = patch_row([size_origin*size_origin*(rand_c-1)+1:size_origin*size_origin*rand_c]);
    patch_one = reshape(patch_one,size_origin,size_origin); %调整为32*32的矩阵
    patch_one = patch_one'; % 转置，后面也要对应保持一致
    patch_one = imresize(patch_one,[size_new size_new]); % 重新调整大小
    patch_one = reshape(patch_one,1,size_new*size_new); %重新调整为一行
    
    patches_ext(:,pcnt) = patch_one'; %调整为列并赋值给大的数组
end;

figure(1);  display_network(patches_ext);
patches_normal = normalizeData(patches_ext);
figure(2);  display_network(patches_normal);

% step2: ZCA
% load data
x = patches_normal;

x = x-repmat(mean(x,1),size(x,1),1); %求每一列的均值
% Implement PCA to obtain xRot
xRot = zeros(size(x)); % You need to compute this
[n m] = size(x);
sigma = (1.0/m)*x*x';
[u s v] = svd(sigma);
xRot = u'*x;

% Check your implementation of PCA
covar = zeros(size(x, 1)); % You need to compute this
covar = (1./m)*xRot*xRot';

% Visualise the covariance matrix. You should see a line across the
% diagonal against a blue background.
figure('name','Visualisation of covariance matrix');
imagesc(covar);

% Step 2: Find k, the number of components to retain
%  Write code to determine k, the number of components to retain in order
%  to retain at least 99% of the variance.

k = 0; % Set k accordingly
ss = diag(s);
% for k=1:m
%    if sum(s(1:k))./sum(ss) < 0.99
%        continue;
% end
% 其中cumsum(ss)求出的是一个累积向量，也就是说ss向量值的累加值
% 并且(cumsum(ss)/sum(ss))<=0.99是一个向量，值为0或者1的向量，为1表示满足那个条件
k = length(ss((cumsum(ss)/sum(ss))<=0.99));

% Step 3: Implement PCA with dimension reduction
xHat = zeros(size(x));  % You need to compute this
xHat = u*[u(:,1:k)'*x;zeros(n-k,m)];

% Visualise the data, and compare it to the raw data
% You should observe that the raw and processed data are of comparable quality.
% For comparison, you may wish to generate a PCA reduced image which
% retains only 90% of the variance.

figure('name',['PCA processed images ',sprintf('(%d / %d dimensions)', k, size(x, 1)),'']);
display_network(xHat);
figure('name','Raw images');
display_network(x);

% Step 4a: Implement PCA with whitening and regularisation
%  Implement PCA with whitening and regularisation to produce the matrix
%  xPCAWhite. 

epsilon = 0.1;
xPCAWhite = zeros(size(x));
xPCAWhite = diag(1./sqrt(diag(s)+epsilon))*u'*x;
figure('name','PCA whitened images');
display_network(xPCAWhite);

% Step 4b: Check your implementation of PCA whitening 
covar = (1./m)*xPCAWhite*xPCAWhite';

% Visualise the covariance matrix. You should see a red line across the
% diagonal against a blue background.
figure('name','Visualisation of covariance matrix');
imagesc(covar);

% Step 5: Implement ZCA whitening
%  Now implement ZCA whitening to produce the matrix xZCAWhite. 
%  Visualise the data and compare it to the raw data. You should observe
%  that whitening results in, among other things, enhanced edges.
xZCAWhite = zeros(size(x));
xZCAWhite = u*xPCAWhite;

% Visualise the data, and compare it to the raw data.
% You should observe that the whitened images have enhanced edges.
figure('name','ZCA whitened images');
display_network(xZCAWhite);
figure('name','Raw images');
display_network(x);










