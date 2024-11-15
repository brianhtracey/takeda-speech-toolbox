
import glob
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd

from compute_features import WaveformAnalyzer
import general_utils as util
import segment_and_timing as seg


def plot_dist(snd, distmatrix, res_dir, name):
    fig, ax = plt.subplots(figsize = (6, 6))
    im = ax.pcolor( -10 * np.log10(distmatrix + 0.001), vmin = -15, vmax = -3)
    ax.set_title(name)
    plt.colorbar(im)
    fig.savefig(os.path.join(res_dir, name) + '.pdf', format = "pdf")
    plt.close(fig)

def plot_strip(snd,diststrip, res_dir, name):
    fig, ax = plt.subplots(figsize = (6, 6))
    im = plt.pcolor(np.transpose(1 - diststrip), vmin = 0.5, vmax = 1)
    ax.set_title(name)
    plt.colorbar(im)
    fig.savefig(os.path.join(res_dir, name) + '.pdf', format = "pdf")
    plt.close(fig)

def plotme(snd, nucleii, speechseg_on,speechseg_off, res_dir, name):
    print(name)
    fig, ax = plt.subplots(figsize = (10, 2))
    ax.plot(snd.xs(), snd.values.T)
    for xc in nucleii:
        ax.axvline(x = xc, color = 'k', linestyle = ':', alpha = 0.2)

    #ax.axvspan(0.0, speechseg_on[0], facecolor='0.2', alpha=0.2)
    for igap, t_on in enumerate(speechseg_off):
        if igap < len(speechseg_off) - 1:
            t_off = speechseg_off[igap]
            t_on = speechseg_on[igap + 1]
            ax.axvspan(t_off, t_on, facecolor = '0.2', alpha = 0.2)
    ax.set_title(name)

    fig.savefig(os.path.join(res_dir, name) + '.pdf', format = "pdf")
    plt.close(fig)

def filterme(y, sr, verbose = False):
    from scipy import signal
    numtaps = np.int32(4 / 50 * sr) + 1
    f = [100 / (sr / 2), 3000 / (sr / 2)]
    b = signal.firwin(numtaps, f, window = 'hanning', pass_zero = False)
    yfilt = signal.filtfilt(b, 1, y)
    if verbose:
        fv, H = signal.freqz(b, 1, fs = sr, worN = 2000)
        plt.figure()
        plt.plot(fv, abs(H), 'b')

        plt.figure()
        f, t, Sxx = signal.spectrogram(yfilt, sr)
        plt.pcolormesh(t, f, np.log10(Sxx), shading = 'gouraud')

    return yfilt

## DEPRECATED
def parse_homerolled(voicing_starts_mine, voicing_stops_mine, min_pause_len=0.3):
    assert (len(voicing_starts_mine) == len(voicing_stops_mine)), "needs same number of starts, stops!"

    import numpy as np
    syll_nuclei = (np.asarray(voicing_starts_mine) + np.asarray(voicing_stops_mine)) / 2.0

    pause_dur = []
    speech_starts_mine = [voicing_starts_mine[0]]
    speech_stops_mine = []
    for i in range(len(voicing_starts_mine) - 1):
        this_pause = voicing_starts_mine[i + 1] - voicing_stops_mine[i]
        if this_pause > min_pause_len:  # apr
            pause_dur.append(this_pause)
            speech_stops_mine.append(voicing_stops_mine[i])
            speech_starts_mine.append(voicing_starts_mine[i + 1])

    # finish off with end of last voiced segment
    speech_stops_mine.append(voicing_stops_mine[-1])

    return speech_starts_mine, speech_stops_mine, syll_nuclei, pause_dur


def distance_mat(s, t, dist):  # Rmani
    N = s.shape[0]
    M = t.shape[0]

    dist_mat = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            dist_mat[i, j] = dist(s[i], t[j])
    return dist_mat


def distance_strip(s, t, nsamps, dist):  # Rmani
    N = s.shape[0]
    M = t.shape[0]

    N_est = N - nsamps
    dist_strip = np.zeros((N_est, nsamps))

    for i in range(N_est):
        for offset in range(nsamps):
            dist_strip[i, offset] = dist(s[i], t[i + offset])
    return dist_strip

def compute_dist_mat_remove_silence(sndObject, t1, t2, res_dir, name):  # Rmani
    # remove silence from the beginning and end of the file
    if False:
        myaudio = AudioSegment.from_wav(filepath)
        myaudio = myaudio.split_to_mono()[0]
        dBFS = myaudio.dBFS
        detected_nonsilence = silence.detect_nonsilent(
                myaudio, min_silence_len = 250, silence_thresh = dBFS - 16)

        sr, y = wavfile.read(filepath)
        if len(detected_nonsilence) > 0:
            start = detected_nonsilence[0][0] / 1000
            if len(detected_nonsilence) > 1:
                end = detected_nonsilence[-1][1] / 1000
                y = y[int(start * sr):int(end * sr)]
            else:
                end = detected_nonsilence[0][1] / 1000
                y = y[int(start * sr):int(end * sr)]
    else:
        snd_silence_removed = sndObject.extract_part(
                from_time = t1, to_time = t2, preserve_times = True)
        y = np.squeeze(snd_silence_removed.values.T)
        sr = snd_silence_removed.get_sampling_frequency()


    # Compute the MFCCs
    import python_speech_features
    mfcc_time_step_sec = 0.025#  0.01 # 0.01 = 10 ms is the default value for the package
    mfcc2 = python_speech_features.mfcc(y, sr, winlen = 0.025, winstep = mfcc_time_step_sec)

    # Compute the distance matrix
    from scipy.spatial import distance
    if False:
        dist_mat_silence_removed = distance_mat(mfcc2, mfcc2, distance.euclidean)
        plot_dist(snd_silence_removed, dist_mat_silence_removed, "distance_" + name)
    else:
        sec_to_analyze = 2.0
        nframes_to_analyze = np.int32(sec_to_analyze / mfcc_time_step_sec)
        #dist_mat_silence_removed_e = distance_strip(mfcc2, mfcc2,nframes_to_analyze, distance.euclidean)
        dist_mat_silence_removed = distance_strip(mfcc2, mfcc2, nframes_to_analyze, distance.cosine)
        plot_strip(snd_silence_removed, dist_mat_silence_removed, res_dir, "distance_" + name)

    return dist_mat_silence_removed

######## done helper codes


#### MAIN START

runpar = {
        'notch_filter_freq': 50, 'normalize_waveform': True,
        'voiced': True,  # analyzed voiced vs. unvoiced?
        'do_dfa': False, 'segmentation_method': 'praat_feinberg',
        'min_pause_sec': 0.1, 'max_pause_sec': 2.0,
        'transcription_dir':'/Users/ieu8424/Documents/MIT_voice/Adam_FTD_and_Healthy_data/AWS_transcripts/all_monologue_AWS'}

# set up directory for outputs and QC plots
data_dir  = '/Users/ieu8424/Library/CloudStorage/OneDrive-SharedLibraries-Takeda/MIT-Takeda Program (Extranet) - cleaned_data/denoised_all_renamed_audio_files_with_gender/16k';

#data_dir = '/Users/ieu8424/Documents/MIT_voice/WavByTask/monologue'
#data_dir = '/Users/ieu8424/Documents/MIT_voice/WavByTask/pata'
#data_dir = '/Users/ieu8424/Documents/MIT_voice/WavByTask/pataka'

from datetime import date
today = date.today()
thisdate = today.strftime("%B_%d_%y")
res_dir = data_dir + '/qc_ptak_' + thisdate
if not os.path.exists(res_dir):
    os.makedirs(res_dir)


# get list of all files in directory
s = data_dir + '/*PTAK*.wav'
full_fnames = glob.glob(s) # os.listdir(data_dir)

timing_praat_list = []
timing_aws_list = []

acst_feat_prev_list = []
acst_feat_new_list = []
# instantiate a waveform analyzer object
analyzer = WaveformAnalyzer()

# can do a subset for training.
#full_fnames = full_fnames[0:4:100]

cnt = 0
for wvfile in full_fnames:

    # collect file info together for later use
    ddur, basename, ext = util.get_fileparts(wvfile)
    thisfile = basename  #[19:25]

    cnt = cnt + 1
    print('%d of %d files processed: %s' % (cnt, len(full_fnames), basename))

    # read in wave file as parselmouth
    y, sr, nchan = util.read_wave(wvfile)

    if nchan != 1:
        print(wvfile + 'has multiple channels... but load_waveform can handle!')

    analyzer.load_waveform(
            y, sr, notch_filter_freq = runpar['notch_filter_freq'],
            normalize_waveform = runpar['normalize_waveform'])

    # get an SNR estimate on overall waveform
    est_snr_dB, noise_dB = analyzer.get_SNR_info()

    # collect file info together for later use
    fileinfo_dict = {
            'file': thisfile, 'numClipped': analyzer.get_num_clipped(),
            'est_snr_dB': est_snr_dB, 'noise_dB': noise_dB}

    # get a filtered waveform?  tried filtering as described in DeJong and Wempe, i.e.100-5 kHz, but doesn't help mjuch
    if False:
        yin = np.squeeze(analyzer.getSound().values.T)
        yfilt = filterme(yin, sr, verbose = False)
        import parselmouth as pm
        sndFilt = pm.Sound(yfilt, sr)

    # now, run the different segmentation methods.
    # set pre-processing and other params

    speech_starts_praat, speech_stops_praat, syll_nuclei_praat, pauses_praat = seg.feinberg_speech_rate(
            analyzer.getSound(), minpause = runpar['min_pause_sec'])
    # seg.feinberg_speech_rate(sndFilt, minpause=runpar['min_pause_sec'])
    pauses_praat = seg.discard_overlong_pauses(
            pauses_praat, max_believable_pause_sec = runpar['max_pause_sec'])
    pauses_praat = seg.discard_tooshort_pauses(
            pauses_praat, min_believable_pause_sec = runpar['min_pause_sec'])

    timing_feat_praat = seg.get_timing_features_syl(
            speech_starts_praat, speech_stops_praat, syll_nuclei_praat,
            pauses_praat, check_seg_change = True)
    timing_feat_praat.update(fileinfo_dict)
    timing_praat_list.append(timing_feat_praat)

    plotme(analyzer.getSound(), syll_nuclei_praat, speech_starts_praat, speech_stops_praat, res_dir, thisfile + "_praat" )

    if False:
        # ==================  AWS  =====================

        speech_starts_aws, speech_stops_aws, pauses_aws, seg_success = seg.aws_based_segmentation(
                basename, runpar['transcription_dir'], runpar['max_pause_sec'])
        pauses_aws = seg.discard_overlong_pauses(
                pauses_aws, max_believable_pause_sec = runpar['max_pause_sec'])
        pauses_aws = seg.discard_tooshort_pauses(
                pauses_aws, min_believable_pause_sec = runpar['min_pause_sec'])

        syll_nuclei_aws = speech_starts_aws  # hack for compatibility
        check_seg_change = False  # speech starts, stops are same as syllabi

        if seg_success:
            timing_feat_aws = seg.get_timing_features_syl(
                    speech_starts_aws, speech_stops_aws, syll_nuclei_aws,
                    pauses_aws, check_seg_change = check_seg_change)
            timing_feat_aws.update(fileinfo_dict)
            timing_aws_list.append(timing_feat_aws)

            plotme(analyzer.getSound(),syll_nuclei_aws, speech_starts_aws, speech_stops_aws, res_dir, thisfile + "_aws")

    if False:
        # R'mani distance matrix
        t1 = speech_starts_aws[0]
        t2 = speech_stops_aws[-1]
        dist_mtrx = compute_dist_mat_remove_silence(
                analyzer.getSound(), t1, t2, res_dir, thisfile)

    do_acoust = False
    if do_acoust:
        # now, try passing these into acoustic feature land
        print('running prev')
        acst_feat = analyzer.get_features_by_segment(
                speech_starts_praat, speech_stops_praat,
                do_voicedVsunvoiced = runpar['voiced'], do_dfa = runpar['do_dfa'])
        acst_feat.update(fileinfo_dict)
        acst_feat_prev_list.append(acst_feat)

        # try alternate
        print('running new')
        acst_feat_new = analyzer.get_features_across_segments(
                speech_starts_praat, speech_stops_praat,
                do_voicedVsunvoiced = runpar['voiced'], do_dfa = runpar['do_dfa'])

        acst_feat_new.update(fileinfo_dict)
        acst_feat_new_list.append(acst_feat_new)

# convert to pandas and then save - timing
df_feat_praat = pd.DataFrame(timing_praat_list)
df_feat_aws = pd.DataFrame(timing_aws_list)
df_feat_praat.to_csv(os.path.join(res_dir, "praat_timing.csv"))
df_feat_aws.to_csv(os.path.join(res_dir, "aws_timing.csv"))

if do_acoust:
    # convert to pandas and then save - acoustics
    df_feat_ac_prev = pd.DataFrame(acst_feat_prev_list)
    df_feat_ac_prev.to_csv(os.path.join(res_dir, "ac_prev.csv"))

    df_feat_ac_new = pd.DataFrame(acst_feat_new_list)
    df_feat_ac_new.to_csv(os.path.join(res_dir, "ac_new.csv"))

