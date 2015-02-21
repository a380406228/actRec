% 批量提取样本特征
% param setting
track_length = 15;
track_num = 200;
size_origin = 32;
size_new = 16;

ext_ratio = 0.1;
patches_origin_num = track_length*track_num; % 15*200=3000
patches_ext_num = int32(patches_origin_num*ext_ratio); %每个样本需要抽取的patches数,3000*0.1=300

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


addpath('D:\\myproject\\data\\dataoriginrand200');
patches_ext_batch = zeros(size_new*size_new,patches_ext_num*sample_total);
pcnt = 0;
for acti = 1:sample_class_num %1:6
    for peri = 1:sample_num % 1:25
        shi = floor(peri/10);
        ge = mod(peri,10);
        for di = 1:dnum % 1:4
            path = sprintf('person%d%d_%s_d%d_origin.txt',shi,ge,act_name{acti},di);
            disp(path);
            feaorigin = load(path);
            
            rand_idx = randperm(patches_origin_num);
            rand_idx = rand_idx([1:patches_ext_num]);
            rand_idx = sort(rand_idx);
            
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

                patches_ext_batch(:,pcnt) = patch_one'; %调整为列并赋值给大的数组
            end;
                        
        end;
    end;
end;
disp(pcnt);

patches_normal_batch = normalizeData(patches_ext_batch);
% 随机显示
subnum = 400;
randshow_idx = randperm(patches_ext_num*sample_total);
randshow_idx = randshow_idx([1:subnum]);
randshow_idx = sort(randshow_idx);
figure(1);  display_network(patches_normal_batch(:,randshow_idx));

% ZCA
x = patches_normal_batch;
x = x-repmat(mean(x,1),size(x,1),1); %求每一列的均值

% Implement PCA to obtain xRot
xRot = zeros(size(x)); % You need to compute this
[n m] = size(x);
sigma = (1.0/m)*x*x';
[u s v] = svd(sigma);
xRot = u'*x;

% Step 2: Find k, the number of components to retain
k = 0; % Set k accordingly
ss = diag(s);
k = length(ss((cumsum(ss)/sum(ss))<=0.99));

% Step 3: Implement PCA with dimension reduction
xHat = zeros(size(x));  % You need to compute this
xHat = u*[u(:,1:k)'*x;zeros(n-k,m)];

% Step 4: Implement PCA with whitening and regularisation
epsilon = 0.1;
xPCAWhite = zeros(size(x));
xPCAWhite = diag(1./sqrt(diag(s)+epsilon))*u'*x;

xZCAWhite = zeros(size(x));
xZCAWhite = u*xPCAWhite;

% 随机显示
figure('name','ZCA whitened images');
display_network(xZCAWhite(:,randshow_idx));

savepath = sprintf('data\\xZCAWhite.mat');
%save xZCAWhite xZCAWhite
save(savepath,'xZCAWhite')



            
            
            