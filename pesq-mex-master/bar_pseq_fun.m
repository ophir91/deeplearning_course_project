function bar_pseq_fun(SNR,type_noise)
    x=[];
    if type_noise==1
        type_noise='speech';
    elseif type_noise==2
        type_noise = 'babble';  
    elseif type_noise==3
        type_noise = 'factory';
    end
    fprintf('start compiling ...\n');
    mex *.c -output ./bin/PESQ_MEX

    addpath('./bin');
    fprintf('\n ======================================= \n source codes are compiled successfully.\n');
    ref_path = strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_de.WAV');

% 
%     deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762.WAV');
%     x = [x; pesq(ref_path,deg_path)];
%     deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_de_no_net.WAV');
%     x = [x; pesq(ref_path,deg_path)];


    deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_n.WAV');
    x = [x; pesq(ref_path,deg_path)];

    deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_de.WAV');
    x = [x; pesq(ref_path,deg_path)];

    deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_n_de.WAV');
    x = [x; pesq(ref_path,deg_path)];

    deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_n_de_phn_hard.WAV');
    x = [x; pesq(ref_path,deg_path)];

    deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_n_de_phn_soft4.WAV');
    x = [x; pesq(ref_path,deg_path)];
        
    deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_n_de_phn_soft39.WAV');
    x = [x; pesq(ref_path,deg_path)];
    
    deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_de_moe2.WAV');
    x = [x; pesq(ref_path,deg_path)];

    deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_de_moe6.WAV');
    x = [x; pesq(ref_path,deg_path)];
    
    deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_de_moe20.WAV');
    x = [x; pesq(ref_path,deg_path)];
    
    deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_de_moe39.WAV');
    x = [x; pesq(ref_path,deg_path)];
%     deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_n_de_no_net.WAV');
%     x = [x; pesq(ref_path,deg_path)];
    
    somenames = {strcat('snr   ',int2str(SNR)),strcat('snr2   ',int2str(SNR))};

    bar([x(:,1)';x(:,2)']);
    set(gca,'xticklabel',somenames);
    grid on
%     legend('orginal','original de no net','original de','noise de','noise de phn hard','noise de phn soft','noise', 'noise n de no net');
    legend('noise','noise de','noise de one net' ,'noise de phn hard','noise de phn soft4','noise de phn soft39', 'noise n de moe2', 'noise n de moe6', 'noise n de moe20', 'noise n de moe39');

    
    dir_name =  strcat('/media/ophir/DATA1/Asaf/deep_project/python/Barchart_',type_noise)
    if exist(dir_name, 'dir')
       warning('dir exist')
    else
      mkdir(dir_name)
    end
    saveas(gcf,strcat(dir_name,'/SNR_',int2str(SNR),'_Barchart.png'))

    title('PSEQ');
    
    disp(x(:,1)');






end

