% Multi-subject preprocessing loop (example input for eeglab2mne)
datapath = '/data/study';
outpath  = fullfile(datapath, 'derivatives');
subjects = {'sub01', 'sub02', 'sub03'};

for s = 1:length(subjects)
    EEG = pop_loadset('filename', [subjects{s} '.set'], 'filepath', datapath);
    EEG = pop_select(EEG, 'nochannel', {'HEOG' 'VEOG'});
    if EEG.srate > 250
        EEG = pop_resample(EEG, 250);
    end
    EEG = pop_eegfiltnew(EEG, 'locutoff', 0.1, 'hicutoff', 30);
    EEG = pop_reref(EEG, []);
    EEG = pop_epoch(EEG, {'S  1' 'S  2'}, [-0.2 0.8]);
    EEG = pop_rmbase(EEG, [-200 0]);
    EEG = pop_saveset(EEG, 'filename', sprintf('%s_epo.set', subjects{s}), 'filepath', outpath);
end
