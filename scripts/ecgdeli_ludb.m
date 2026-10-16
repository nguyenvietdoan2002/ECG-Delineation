% ECGdeli 1.1 on every LUDB record (decision 78). Called from MATLAB with two variables already set:
%
%   matlab -batch "ecgdeli_root='<ECGdeli checkout>'; work='<this repo>/work/ecgdeli'; run('<this repo>/scripts/ecgdeli_ludb.m')"
%
% Both paths must be absolute: run() executes the script from its own folder.
%
% Reads work/ludb_input.mat (signals: records x samples x 12 in mV, ids, fs) written by
% scripts/ecgdeli_ludb.py export, and writes work/ludb_fpt.mat: fpt, a records x 12 cell array of
% fiducial point tables (1-based samples, one row per beat), plus failed, msg, n_samples and the
% MATLAB version. Filtering and delineation follow ECGdeli's Example/Annotate_ExampleECG.m at v1.1:
% 1-40 Hz band-pass, 50 Hz notch, isoline correction, Annotate_ECG_Multi.

addpath(genpath(ecgdeli_root));
S = load(fullfile(work, 'ludb_input.mat'));
fs = double(S.fs);
n_records = size(S.signals, 1);
n_samples = size(S.signals, 2);
fpt = cell(n_records, 12);
failed = false(n_records, 1);
msg = repmat({''}, n_records, 1);
t0 = tic;
for k = 1:n_records
    ecg = reshape(S.signals(k, :, :), n_samples, 12);
    try
        filtered = ECG_High_Low_Filter(ecg, fs, 1, 40);
        filtered = Notch_Filter(filtered, fs, 50, 1);
        [filtered, ~, ~, ~] = Isoline_Correction(filtered);
        [~, cells] = Annotate_ECG_Multi(filtered, fs);
        for j = 1:12
            fpt{k, j} = double(cells{j, 1});
        end
    catch err
        failed(k) = true;
        msg{k} = err.message;
    end
    if mod(k, 20) == 0
        fprintf('%d of %d records, %.0f s\n', k, n_records, toc(t0));
    end
end
matlab_version = version;
save(fullfile(work, 'ludb_fpt.mat'), 'fpt', 'failed', 'msg', 'n_samples', 'matlab_version', '-v7');
fprintf('done: %d records, %d failed, %.0f s\n', n_records, sum(failed), toc(t0));
