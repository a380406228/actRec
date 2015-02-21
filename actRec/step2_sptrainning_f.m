function []=step2_sptrainning_f()
% 稀疏编码训练
% 根据预处理得到的patches进行稀疏编码训练
% 准备数据：./data/xZCAWhite.mat

addpath data;
addpath minFunc;

% STEP 0: Initialization
isTopo = 0;
numFeatures = 225;    % number of features to learn
batchNumPatches = 2000; % number of patches per batch
iterMax = 200;


track_length = 15;
track_num = 200;

ext_ratio = 0.1;
track_length = 15;
patches_origin_num = track_length*track_num; % 15*200=3000
patches_ext_num = int32(patches_origin_num*ext_ratio); %每个样本需要抽取的patches数,3000*0.1=300

sample_class_num = 6;
dnum = 4;
sample_num = 25;
sample_total = sample_num*dnum*sample_class_num;
act_name = {'boxing','handclapping','handwaving','jogging','running','walking'};

% % for test
% sample_class_num = 3;
% dnum = 1;
% sample_num = 3;
% sample_total = sample_num*dnum*sample_class_num;


numPatches = patches_ext_num*sample_total;   % number of patches
patchDim = 16;         % patch dimension
visibleSize = patchDim * patchDim; %单通道灰度图，256维，学习225个特征

% dimension of the grouping region (poolDim x poolDim) for topographic sparse coding
poolDim = 3;

lambda = 5e-5;  % L1-regularisation parameter (on features)
epsilon = 1e-5; % L1-regularisation epsilon |x| ~ sqrt(x^2 + epsilon)
gamma = 1e-2;   % L2-regularisation parameter (on basis)

% STEP 1: Sample patches
images = load('xZCAWhite.mat');
patches = images.xZCAWhite;
display_network(patches(:, 1:400));

% STEP 3: Iterative optimization
% Initialize options for minFunc
%options.Method = 'lbfgs';
options.Method = 'cg';
options.display = 'off';
options.verbose = 0;

% Initialize matrices
weightMatrix = rand(visibleSize, numFeatures);%256*225
featureMatrix = rand(numFeatures, batchNumPatches);%225*2000

% Initialize grouping matrix
assert(floor(sqrt(numFeatures)) ^2 == numFeatures, 'numFeatures should be a perfect square');
donutDim = floor(sqrt(numFeatures));
assert(donutDim * donutDim == numFeatures,'donutDim^2 must be equal to numFeatures');

groupMatrix = zeros(numFeatures, donutDim, donutDim); % 225*15*15
groupNum = 1;
for row = 1:donutDim
    for col = 1:donutDim 
        groupMatrix(groupNum, 1:poolDim, 1:poolDim) = 1; % poolDim=3
        groupNum = groupNum + 1;
        groupMatrix = circshift(groupMatrix, [0 0 -1]);
    end
    groupMatrix = circshift(groupMatrix, [0 -1, 0]);
end
groupMatrix = reshape(groupMatrix, numFeatures, numFeatures);%121*121

% if isequal(questdlg('Initialize grouping matrix for topographic or non-topographic sparse coding?', 'Topographic/non-topographic?', 'Non-topographic', 'Topographic', 'Non-topographic'), 'Non-topographic')
%     groupMatrix = eye(numFeatures); %非拓扑结构时的groupMatrix矩阵
% end
if(0 == isTopo)
    groupMatrix = eye(numFeatures); % 非拓扑结构时的groupMatrix矩阵
end;

% Initial batch
indices = randperm(numPatches);%1*9000
indices = indices(1:batchNumPatches);%1*2000
batchPatches = patches(:, indices);                           

fprintf('%6s%12s%12s%12s%12s\n','Iter', 'fObj','fResidue','fSparsity','fWeight');
%warning off;
for iteration = 1:iterMax   
    % iteration = 1;
    error = weightMatrix * featureMatrix - batchPatches;
    error = sum(error(:) .^ 2) / batchNumPatches;  %说明重构误差需要考虑样本数
    fResidue = error;
    num_batches = size(batchPatches,2);
    R = groupMatrix * (featureMatrix .^ 2);
    R = sqrt(R + epsilon);    
    fSparsity = lambda * sum(R(:));    %稀疏项和权值惩罚项不需要考虑样本数
    
    fWeight = gamma * sum(weightMatrix(:) .^ 2);
    
    %上面的那些权值都是随机初始化的
    fprintf('  %4d  %10.4f  %10.4f  %10.4f  %10.4f\n', iteration, fResidue+fSparsity+fWeight, fResidue, fSparsity, fWeight)
               
    % Select a new batch
    indices = randperm(numPatches);
    indices = indices(1:batchNumPatches);
    batchPatches = patches(:, indices);                    
    
    % Reinitialize featureMatrix with respect to the new
    % 对featureMatrix重新初始化，按照网页教程上介绍的方法进行
    featureMatrix = weightMatrix' * batchPatches;
    normWM = sum(weightMatrix .^ 2)';
    featureMatrix = bsxfun(@rdivide, featureMatrix, normWM); 
    
    % Optimize for feature matrix    
    options.maxIter = 20;
    %给定权值初始值，优化特征值矩阵
    [featureMatrix, cost] = minFunc( @(x) sparseCodingFeatureCost(weightMatrix, x, visibleSize, numFeatures, batchPatches, gamma, lambda, epsilon, groupMatrix), ...
                                           featureMatrix(:), options);
    featureMatrix = reshape(featureMatrix, numFeatures, batchNumPatches);                                      
    weightMatrix = (batchPatches*featureMatrix')/(gamma*num_batches*eye(size(featureMatrix,1))+featureMatrix*featureMatrix');
    [cost, grad] = sparseCodingWeightCost(weightMatrix, featureMatrix, visibleSize, numFeatures, batchPatches, gamma, lambda, epsilon, groupMatrix);
      
    
    figure(2);
    display_network(weightMatrix);
end

savepath = sprintf('data\\weightMatrix_NF%d_BN%d_iter%d.mat',numFeatures,batchNumPatches,iterMax);
save(savepath,'weightMatrix');




