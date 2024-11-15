
import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import re

from compute_features import WaveformAnalyzer
import general_utils as util
import segment_and_timing as seg


##########################################

# Dec 1 2021: added run parameters as input dictionary
# example:
# runpar = {'notch_filter_freq': -1,
#                'normalize_waveform'= True,
#                 'voiced':True , # analyzed voiced vs. unvoiced?
#                 'do_dfa': False,
#                 'segmentation_method':'AWS',
#                 'min_pause_sec': 0.3,
#                 'transcript_dir':'(path to directory))',
# NOTE 100 ms is shortest pause examined in https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-016-0096-7/tables/3
# but 0.3 sec seems more common

##########################################

def process_phonation_files(file_list, qc_dir, runpar, doplot=True):
    print('phonation files: ignoring segmentation_type parameter')

    feat_long_dict_list = []  # set up empty lists for feature for longer time segment
    feat_dict_list = []  # set up empty dictionary for main (shorter) time segment

    unsegmented_file_list = []  # catch names of files we can't segment

    # instantiate a waveform analyzer object
    analyzer = WaveformAnalyzer()

    if runpar['debug']
        file_list = file_list[0:9]  # use this line to debug smaller number!

    cnt = 0
    for ff in file_list:
        # get basename
        base = os.path.basename(ff)
        fname = os.path.splitext(base)[0]
        cnt = cnt + 1
        print('file %d of %d : %s' % (cnt, len(file_list), base))

        y, sr, nchan = util.read_wave(ff)
        analyzer.load_waveform(
                y, sr, nchan, notch_filter_freq = runpar['notch_filter_freq'],
                normalize_waveform = runpar['normalize_waveform'])

        # was valid data read in?
        if analyzer.has_data():
            # March 19, 2020 - call code for segmenting based on parselmouth voiced frames detection
            seg_starts, seg_ends = seg.voicing_based_segmentation_sustainedvowel(analyzer.getSound())  # optional 2nd argument will cause plotting
            seg_starts = [seg_starts]  # put in a list for compatability with vu code
            seg_ends = [seg_ends]

            sel_start, sel_end, sel_long_end = seg.filter_seg_timesV(
                    seg_starts, seg_ends, pad_at_start = 0.75, len_to_keep = 3)
            if doplot:
                util.plot_waveform_and_segmentation(
                        analyzer.getSound(), sel_start, sel_end, sel_long_end, fname, qc_dir)

            # get an SNR estimate on overall waveform
            est_snr_dB, noise_dB =  analyzer.get_SNR_info()

            # collect file info together for later use
            fileinfo_dict = {
                    'file': fname, 'numClipped': analyzer.get_num_clipped(),
                    'est_snr_dB': est_snr_dB, 'noise_dB': noise_dB}

            if len(sel_start) == 0:
                print('no segment found')
                unseg_dict = {}
                unseg_dict.update({'file': fname})
                unsegmented_file_list.append(unseg_dict)
            else:
                # process each segment - but for sustained vowel, there is jst 1 segment!
                feat_dict_long = analyzer.get_features_across_segments(
                        sel_start, sel_long_end, do_voicedVsunvoiced = runpar['voiced'],
                        do_dfa = runpar['do_dfa'])

                # core chunk of phonation
                feat_dict = analyzer.get_features_across_segments(
                        sel_start, sel_end, do_voicedVsunvoiced = runpar['voiced'],
                        do_dfa = runpar['do_dfa'])
                # for each, add in file info, then store
                if not feat_dict_long == []:
                    feat_dict_long.update(fileinfo_dict)
                    feat_long_dict_list.append(feat_dict_long)

                if not feat_dict == []:
                    feat_dict.update(fileinfo_dict)
                    feat_dict_list.append(feat_dict)

                # save out the sound clip
                if False:  # maybe use runpar['debug']
                    littleSnd = analyzer.getSound().extract_part(
                            from_time = sel_start[0], to_time = sel_end[0],
                            preserve_times = True)
                    littleSnd.save(qc_dir + '/' + fname + "_seg.wav", 'WAV')

    # convert lists of dictionaries into pandas datafromes
    # TODO consider sorting columns into a more readable order
    df = pd.DataFrame(feat_dict_list)
    df_long = pd.DataFrame(feat_long_dict_list)

    unsegDf = pd.DataFrame(unsegmented_file_list)

    return(df, df_long, unsegDf)

################################################################

def process_ddk_or_reading(file_list_with_path, runpar, transcription_dir=None):
    feat_dict_list = []

    # instantiate a waveform analyzer object
    analyzer = WaveformAnalyzer()

    if runpar['debug']:
        file_list_with_path = file_list_with_path[0:9]  # use this line to debug smaller number!

    exception_counter = 0
    cnt = 0
    for wvfile in file_list_with_path:
        # collect file info together for later use
        _, basename, _ = util.get_fileparts(wvfile)
        fileinfo_dict = {'file': basename}

        cnt = cnt + 1
        print('%d of %d files processed: %s' % (cnt, len(file_list_with_path), basename))
        try:
            # read in wave file as parselmouth
            y, sr, nchan = util.read_wave(wvfile)
            if nchan != 1:
                print(wvfile + 'has multiple channels')
            analyzer.load_waveform(
                    y, sr, nchan, notch_filter_freq = runpar['notch_filter_freq'],
                    normalize_waveform = runpar['normalize_waveform'])

            # trim interviewer speech
            if runpar['remove_interviewer_speech']:
                # read in manually marked interviewer speech and clean up
                interviewerDF = pd.read_csv(
                        '/Users/ieu8424/Takeda - MIT-Takeda Program (Extranet) - derived_features/all_interviewer.txt',
                        header = None)
                interviewerDF.columns = ['files', 't_start', 't_stop']
                basetmp = re.sub('\_enhanced$', '', basename)
                basetmp = re.sub('\_raw$', '', basetmp)
                littleDF = interviewerDF[interviewerDF["files"] == basetmp]

                if len(littleDF.index) > 0:  # did we get a match?
                    int_start = littleDF['t_start'].tolist()
                    int_end = littleDF['t_stop'].tolist()

                    cut_sound = util.remove_contaminated_regions(analyzer.getSound(), int_start, int_end)
                    # reload it...
                    analyzer.getSound()  # check works
                    y2 = np.squeeze(cut_sound.values)
                    analyzer.load_waveform(y2, sr, 1, normalize_waveform = False)
                    analyzer.getSound()  # verified this works

            # get an SNR estimate on overall waveform
            est_snr_dB, noise_dB = analyzer.get_SNR_info()
            # collect file info together for later use
            fileinfo_dict = {
                    'file': basename, 'numClipped': analyzer.get_num_clipped(),
                    'est_snr_dB': est_snr_dB, 'noise_dB': noise_dB}

            if runpar['segmentation_method'] == 'praat_feinberg':
                speech_starts, speech_stops, syll_nuclei, pauses = seg.feinberg_speech_rate(
                        analyzer.getSound(), minpause=runpar['min_pause_sec'])
                seg_success = True  # always will run
                check_seg_change = True  # speech starts, stops NOT linked to syll_nuclei

            elif runpar['segmentation_method'] == 'aws':
                # here, 'starts' and 'stops' are per word
                speech_starts, speech_stops, pauses, seg_success = seg.aws_based_segmentation(
                        basename, transcription_dir, runpar['max_pause_sec'])
                syll_nuclei = speech_starts  # hack for compatibility
                check_seg_change = False  # speech starts, stops are same as syllabi

            elif runpar['segmentation_method'] == 'forced_aligner':
                aligner_dir = runpar['transcript_dir']  # assume that aligner output is in data directory..
                # here, 'starts' and 'stops' are per word
                speech_starts, speech_stops, pauses, seg_success = seg.aligner_based_segmentation(
                        basename, aligner_dir, runpar['max_pause_sec'])
                syll_nuclei = speech_starts  # hack for compatibility
                check_seg_change = False  # speech starts, stops are same as syllabi

            elif runpar['segmentation_method'] == 'homerolled':  # my method
                ## TODO: seg.homerolled_voicing_based_segmentation_ddk() ??
                speech_starts, speech_stops = seg.voicing_based_segmentation_ddk(analyzer.getSound())
                # fname, qc_dir)  # optional 2nd argument will cause plotting
                pauses = seg.get_pause_list(speech_starts, speech_stops)
                syll_nuclei = speech_starts  # hack for compatibility
                seg_success = True  # always will run
                check_seg_change = False  # speech starts, stops are same as syllabi

            # did selected method fail? if so, run praat_feinberg
            seg_source = runpar['segmentation_method']
            if not seg_success:
                speech_starts, speech_stops, syll_nuclei, pauses = seg.feinberg_speech_rate(
                        analyzer.getSound(), minpause=runpar['min_pause_sec'])
                check_seg_change = True  # speech starts, stops NOT linked to syll_nuclei
                seg_source = 'praat_feinberg_backup'

            # get timing features after filtering on pause length
            pauses = seg.discard_overlong_pauses(pauses, max_believable_pause_sec = runpar['max_pause_sec'])
            pauses = seg.discard_tooshort_pauses(pauses, min_believable_pause_sec = runpar['min_pause_sec'])
            timing_feat = seg.get_timing_features_syl(
                    speech_starts, speech_stops, syll_nuclei, pauses,
                    check_seg_change = check_seg_change)
            timing_feat["seg_method"] = seg_source

            feat_dict = analyzer.get_features_across_segments(
                    speech_starts, speech_stops, do_voicedVsunvoiced = runpar['voiced'],
                    do_dfa = runpar['do_dfa'])

            if not feat_dict == []:
                feat_dict.update(timing_feat)  # add timing features to acoustic features
                feat_dict.update(fileinfo_dict)  # add in file info
                feat_dict_list.append(feat_dict)

        except:
            print('***** SOMETHING BAD HAPPENED ***')
            exception_counter += 1
    # TODO there are some longer pauses that are really gaps between sentences.  Add a parser for known phrases....
    # for now, 1) discard very long pauses and b) look at median and IQR

    # convert to pandas and then save
    df_feat = pd.DataFrame(feat_dict_list)

    return df_feat

