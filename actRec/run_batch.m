clear all
clc
close all

% step1:prepros,resize and ZCA
disp('**************step1_preprospatches_batch_f******************\n');
step1_preprospatches_batch_f();

% step2:sparse coding for weightMatrix
clear all
disp('**************step2_sptrainning_f***************************\n');
step2_sptrainning_f();

% step3:get featureMatrix
clear all
disp('**************step3_getfeatureMatrix_batch_f****************\n');
step3_getfeatureMatrix_batch_f();

% step4:kmeans and svm
clear all
disp('**************step4_classification_f************************\n');
step4_classification_f();
