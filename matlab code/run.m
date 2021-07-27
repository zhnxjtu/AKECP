clc
clear
close all

mkdir('Res110_PruningIndex')

for i = 1:108

    if i <= 36
        pic_size = 32;
    elseif i > 36 && i <= 72
        pic_size = 16;
    elseif i > 72 
        pic_size = 8;
    end
    
    fprintf('The %d-th layer is processing, FM size is %d... \n', i, pic_size);
    main_FM(i, pic_size);
end