
type_noise='speech'


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

y=[];
name = [];
num_tests = 12;
SNR = [10,5,0,-5];
z= zeros(8,2,num_tests,length(SNR));

for ii =1:length(SNR)
    for index=1:num_tests
        x=[];
        ref_path = strcat('/media/ophir/DATA1/Asaf/deep_project/python/results/',type_noise,'_',int2str(SNR(ii)),'/',int2str(index),'_de.WAV');
        deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/results/',type_noise,'_',int2str(SNR(ii)),'/',int2str(index),'_n.WAV');
        x = [x; pesq(ref_path,deg_path)];

        deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/results/',type_noise,'_',int2str(SNR(ii)),'/',int2str(index),'_n_de.WAV');
        x = [x; pesq(ref_path,deg_path)];

        deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/results/',type_noise,'_',int2str(SNR(ii)),'/',int2str(index),'_n_de_phn_hard.WAV');
        x = [x; pesq(ref_path,deg_path)];

        deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/results/',type_noise,'_',int2str(SNR(ii)),'/',int2str(index),'_n_de_phn_soft4.WAV');
        x = [x; pesq(ref_path,deg_path)];

        deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/results/',type_noise,'_',int2str(SNR(ii)),'/',int2str(index),'_de_moe6.WAV');
        x = [x; pesq(ref_path,deg_path)];


        deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/results/',type_noise,'_',int2str(SNR(ii)),'/',int2str(index),'_de_moe39.WAV');
        x = [x; pesq(ref_path,deg_path)];

        deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/results/',type_noise,'_',int2str(SNR(ii)),'/',int2str(index),'_de_moe2_lstm.WAV');
        x = [x; pesq(ref_path,deg_path)];
        deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/results/',type_noise,'_',int2str(SNR(ii)),'/',int2str(index),'_de_moe39_lstm.WAV');
        x = [x; pesq(ref_path,deg_path)];

        z(:,:,index,ii) = x(:,:);
    end
    name = [name; cell(strcat('snr',{'  '},num2str(SNR(ii))))];
end

somenames = name;

mean_z = squeeze(mean(z,3))


figure(1)
bar(mean_z(:,1:2:end)');
title(strcat('PESQ for ',{' '},type_noise,' noise'))
ylabel('pesq');
set(gca,'xticklabel',somenames);
legend('noise','single FC-DNN' ,'PHN-base hard','PHN-base soft', 'DMoE6',  'DMoE39', 'DRMoE2',  'DRMoE39');

dir_name =  strcat('/media/ophir/DATA1/Asaf/deep_project/python/Barchart_',type_noise)
if exist(dir_name, 'dir')
    warning('dir exist')
else
    mkdir(dir_name)
end
saveas(gcf,strcat(dir_name,'/Barchart.png'))








