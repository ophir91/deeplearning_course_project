%% compile and test file

fprintf('start compiling ...\n');

mex *.c -output ./bin/PESQ_MEX

addpath('./bin');

fprintf('\n ======================================= \n source codes are compiled successfully.\n');

fprintf('start testing\n');

[ref, ~] = audioread('/media/ophir/DATA1/Asaf/deep_project/pesq/pesq-mex-master/audio/speech.wav');
[deg, fs] = audioread('/media/ophir/DATA1/Asaf/deep_project/pesq/pesq-mex-master/audio/speech_bab_0dB.wav');
fprintf('testing narrowband.\n');
disp(pesq_mex(ref, deg, fs, 'narrowband'));


fprintf('testing wideband.\n');
disp(pesq_mex(ref, deg, fs, 'wideband'));

fprintf('testing both.\n');
disp(pesq_mex(ref, deg, fs, 'both'));

fprintf('done.\n');



