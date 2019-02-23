function [outputArg] = pesq(ref_path,deg_path)
%% compile and test file

    fprintf('start testing\n');
% ref_path='/media/ophir/DATA1/Asaf/deep_project/pesq/pesq-mex-master/audio/speech.wav';
% deg_path='/media/ophir/DATA1/Asaf/deep_project/pesq/pesq-    [deg, fs] = audioread('/media/ophir/DATA1/Asaf/deep_project/pesq/pesq-mex-master/audio/speech_bab_0dB.wav');

    [ref, ~] = audioread(ref_path);
    [deg, fs] = audioread(deg_path);
%     fprintf('testing narrowband.\n');
%     disp(pesq_mex(ref, deg, fs, 'narrowband'));
% 
% 
%     fprintf('testing wideband.\n');
%     disp(pesq_mex(ref, deg, fs, 'wideband'));
% 
%     fprintf('testing both.\n');
%     disp(pesq_mex(ref, deg, fs, 'both'));
% 
%     fprintf('done.\n');

    outputArg(1) = pesq_mex(ref, deg, fs, 'wideband');
    outputArg(2) = pesq_mex(ref, deg, fs, 'narrowband');
end

