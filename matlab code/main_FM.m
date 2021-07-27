function main_FM(FM_Index, pic_size)
% clc;
% close all;
% clear all;

% load data;
data_ori = load(['../python code/res110_pruning_fmValue/FM' num2str(FM_Index) '_value/1.txt']);

% pic_size = 32;
pic_pixel = pic_size*pic_size;

data_tmp = reshape(data_ori,pic_pixel,length(data_ori)/pic_pixel);

tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Parameters initialization.
options.verbose = 0;
options.initial_rank = 'auto'; 
options.inf_flag = 1; 
options.MAXITER = 500;
options.UPDATE_BETA = 1; 
options.mode = 'VB';
%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Calculate pruning indices...\n')
BB = 1;
kk = 0.5;  
E_hat = zeros(size(data_tmp,1),size(data_tmp,2));
for i = 1:50 
    thre = sum(sum(E_hat))/(200);
    if (thre > 0.5)
        kk = kk + 0.5;
    elseif (thre < 0.1)
        kk = kk + 0.1;
    else
        kk = kk + thre;
    end
    [X_hat, A_hat, B_hat, E_hat] = VBAS(data_tmp,options,kk);
%     [a,b,c1] = kmeans(E_hat(:,1),2);
%     BB = c1(1) / c1(2); % Calculate the thershold.
end
time = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pruning
rho1 = 0.4;

score1 = sum(abs(E_hat));
[~, Index1] = sort(score1, 'ascend');
pruning_index1 = Index1(1:floor(rho1*length(data_ori)/pic_pixel));

filename1 = fopen(['/Res110_PruningIndex/FM' num2str(FM_Index) '_1.txt'],'wt');
fprintf(filename1,'%g\n', eval(['pruning_index1']));

% pruning
rho2 = 0.6;

score2 = sum(abs(E_hat));
[~, Index2] = sort(score2, 'ascend');
pruning_index2 = Index2(1:floor(rho2*length(data_ori)/pic_pixel));

filename2 = fopen(['/Res110_PruningIndex/FM' num2str(FM_Index) '_2.txt'],'wt');
fprintf(filename2,'%g\n', eval(['pruning_index2']));

% pruning
rho3 = 0.7;

score3 = sum(abs(E_hat));
[~, Index3] = sort(score3, 'ascend');
pruning_index3 = Index3(1:floor(rho3*length(data_ori)/pic_pixel));

filename3 = fopen(['/Res110_PruningIndex/FM' num2str(FM_Index) '_3.txt'],'wt');
fprintf(filename3,'%g\n', eval(['pruning_index3']));

% pruning
rho4 = 0.8;

score4 = sum(abs(E_hat));
[~, Index4] = sort(score4, 'ascend');
pruning_index4 = Index4(1:floor(rho4*length(data_ori)/pic_pixel));

filename4 = fopen(['/Res110_PruningIndex/FM' num2str(FM_Index) '_4.txt'],'wt');
fprintf(filename4,'%g\n', eval(['pruning_index4']));


save(['/Res110_PruningIndex/FM_S' num2str(FM_Index) '.mat'],'E_hat')
% fprintf('end\n')

end