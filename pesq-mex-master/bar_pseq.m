fprintf('start compiling ...\n');
mex *.c -output ./bin/PESQ_MEX

if ~exist('x')
    x = 0.1
end
x

addpath('./bin');
fprintf('\n ======================================= \n source codes are compiled successfully.\n');
SNR = -10
ref_path = strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_de_no_net.WAV');


deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762.WAV');
x(1) = pesq(ref_path,deg_path);
deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_de_no_net.WAV');
x(2) = pesq(ref_path,deg_path);

deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_de.WAV');
x(3) = pesq(ref_path,deg_path);

deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_n_de.WAV');
x(4) = pesq(ref_path,deg_path);

deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_n_de_phn_hard.WAV');
x(5) = pesq(ref_path,deg_path);

deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_n_de_phn_soft.WAV');
x(6) = pesq(ref_path,deg_path);

deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_n.WAV');
x(7) = pesq(ref_path,deg_path);

deg_path =strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/train_SI762_n_de_no_net.WAV');
x(8) = pesq(ref_path,deg_path)

somenames = {strcat('snr   ',int2str(SNR)),'snr 100'};

bar([x;x]);
set(gca,'xticklabel',somenames);
grid on
legend('orginal','original de no net','original de','noise de','noise de phn hard','noise de phn soft','noise', 'noise n de no net');

saveas(gcf,strcat('/media/ophir/DATA1/Asaf/deep_project/python/',int2str(SNR),'/Barchart.png'))

title('PSEQ');



% 
% [ref, ~] = audioread(ref_path);
% [deg, fs] = audioread(deg_path);
% snr(ref,deg)
