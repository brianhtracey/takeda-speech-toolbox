
import glob
import numpy as np
import os

from compute_features import WaveformAnalyzer
import general_utils as util
import segment_and_timing as seg


# set pre-processing and other params
runpar = {
        'notch_filter_freq': -1, 'normalize_waveform': False,
        'voiced': True , # analyzed voiced vs. unvoiced?
        'do_dfa': False, 'segmentation_method': 'praat_feinberg',
        'min_pause_sec': 0.3, 'transcript_dir': None}

# set up directory for outputs and QC plots
if True:
    # CLAC
    data_dir = '/Users/ieu8424/Library/CloudStorage/OneDrive-Takeda/Documents/CLAC-Dataset/max_phonation/32k'
    s = data_dir + '/' + 's*.wav'  # for Fjona data
    cut_at_start_sec = 0.2;
    duration_kept = 0.8;
else:
    cut_at_start_sec = 0.75;
    duration_kept = 2;
    if True:
        data_dir = '/Users/ieu8424/Library/CloudStorage/OneDrive-SharedLibraries-Takeda/MIT-Takeda Program (Extranet) - cleaned_data/denoised_all_renamed_audio_files_with_gender/all_renamed_audio_files_with_gender'
        s = data_dir + '/' + '*AAAH*.wav'  # for Fjona data
    else:
        data_dir = '/Users/ieu8424/Library/CloudStorage/OneDrive-Takeda/Documents/WATCH-PD/baseline_data_for_paper/verbalphonation'
        s = data_dir + '/' + '*.wav'  # for WPD data
#data_dir = '/Users/ieu8424/Documents/MIT_voice/Fjona_rename/ftd_renamed'
#data_dir = '/Users/ieu8424/Documents/MIT_voice/Fjona_rename/healthy_renamed'
#s = data_dir + '/' + '*AAAH*.wav'  # for Fjona data

from datetime import date
today = date.today()
thisdate = today.strftime("%B_%d_%y")
res_dir = data_dir + '/aa_phonation_segmented_' + thisdate
if not os.path.exists(res_dir):
    os.makedirs(res_dir)



# get list of all files in directory

full_fnames = glob.glob(s) # os.listdir(data_dir)

# instantiate a waveform analyzer object
analyzer = WaveformAnalyzer()

#full_fnames = ['/Users/ieu8424/Documents/MIT_voice/Sri_ML_Codes_Dec2021/Data/verbalarticulation_998346_0.wav']
#full_fnames = ['/Users/ieu8424/Documents/MIT_voice/Sri_ML_Codes_Dec2021/Data/verbalarticulation_990118_0.wav']

unsegmented_file_list = []
cnt = 0

for wvfile in full_fnames:
    # collect file info together for later use
    ddur, basename, ext = util.get_fileparts(wvfile)

    cnt = cnt + 1
    print('%d of %d files processed: %s' % (cnt, len(full_fnames), basename))

    # read in wave file as parselmouth
    y, sr, nchan = util.read_wave(wvfile)

    import scipy.io.wavfile as sc_wavread
    sr, y = sc_wavread.read(wvfile)

    print(sr)
    if len(y.shape) > 1:
        print('*** ' + basename + ' has multiple channels ********')
        y = y[:, 0]

    analyzer.load_waveform(
            y, sr, num_channels = nchan, notch_filter_freq = runpar['notch_filter_freq'],
            normalize_waveform = runpar['normalize_waveform'])
    # was valid data read in?
    if analyzer.has_data():
        # March 19, 2020 - call code for segmenting based on parselmouth voiced frames detection
        seg_starts, seg_ends = seg.voicing_based_segmentation_sustainedvowel(
                analyzer.getSound())  # optional 2nd argument will cause plotting
        seg_starts = [seg_starts]  # put in a list for compatability with vu code
        seg_ends = [seg_ends]

        sel_start, sel_end, sel_long_end = seg.filter_seg_timesV(
                seg_starts, seg_ends, pad_at_start = cut_at_start_sec,
                len_to_keep = duration_kept)

    if len(sel_start) == 0:
        print('no segment found')
        unseg_dict = {}
        unseg_dict.update({'file': basename})
        unsegmented_file_list.append(unseg_dict)

    else:
        # save out the sound clip and save a pictture
        util.plot_waveform_and_segmentation(
                analyzer.getSound(), sel_start, sel_end, sel_long_end, basename, res_dir)

        littleSnd = analyzer.getSound().extract_part(
                from_time=sel_start[0], to_time=sel_long_end[0], preserve_times = True)

        # rescale to avoid clip...
        mxabs = np.max(np.abs(littleSnd.values))
        if mxabs > 1.0:
            littleSnd.values = littleSnd.values / mxabs
        littleSnd.save(res_dir + '/' +  basename + "_seg.wav", 'WAV')
        print(' ')

