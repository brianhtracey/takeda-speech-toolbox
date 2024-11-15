#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 16, 2018

@author: btracey
"""

import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
import parselmouth as pm
from scipy import ndimage as spim
import textgrids


################################################################

def process_one_textgrid_file(tgrid_file):
    # parse the filename
    # block, readingnum = parse_reading_file_name(os.path.basename(tgrid_file))

    # read the textgrid file
    grid1 = textgrids.TextGrid(tgrid_file)

    # ** process words
    # pull out word starts and stops, skipping silence ('sil') and short pause ('sp')
    word_start = []
    word_end = []
    transcript = ''
    pause_dur = []
    for word in grid1['words']:
        w = word.text.transcode()
        if w.isupper():  # capture the words (not 'sil' or 'sp')
            transcript = transcript + w + ' '
            word_start.append(word.xmin)
            word_end.append(word.xmax)
        elif w == 'sp':  # capture the silent pauses
            pdur = word.xmax - word.xmin
            pause_dur.append(pdur)
            if pdur > 1.0:
                print('longer pause: ', pdur)
        else:
            print('skipping word: ' + w)
            # TODO assert this is 'sil'

    # TODO extract phonemes and pass them out
    # skipping phonemenes for now
    # for phoneme in grid1['phones']:
    #     print(phoneme)
    #     p = phoneme.text.transcode()
    #     v = phoneme.containsvowel()  # this is nice
    #     d = phoneme.dur

    return (word_start, word_end, pause_dur)


#############################

def process_one_AWS_Transcribe_file(json_dir, jfile):
    f = open(os.path.join(json_dir, jfile), "r")
    data = json.loads(f.read())
    f.close()

    job = data['jobName']
    wavfile = job[:-22]
    print(job)

    res = data['results']
    transcripts = res['transcripts']
    transcript = transcripts[0]['transcript']
    transcript = transcript.replace(',', '')  # strip commas so can save to CSV!
    all_items = res['items']

    word_start = []
    word_end = []
    pause_dur = []
    for item in all_items:
        for k, v in item.items():
            if k == 'type' and v == 'pronunciation':
                word_start.append(float(item['start_time']))
                word_end.append(float(item['end_time']))
                if len(word_start) > 1:
                    this_pause = word_start[-1] - word_end[-2]
                    if this_pause > 0.0:
                        pause_dur.append(this_pause)

    return (word_start, word_end, pause_dur)


##########################################

def discard_overlong_pauses(pauses_list, max_believable_pause_sec=1.5):
    filtered = filter(lambda x: x <= max_believable_pause_sec, pauses_list)
    # above is filter object.  convert to list when passing out
    return list(filtered)

def discard_tooshort_pauses(pauses_list, min_believable_pause_sec):
    filtered = filter(lambda x: x > min_believable_pause_sec, pauses_list)
    # above is filter object.  convert to list when passing out
    return list(filtered)

########## START HELPER FUNCTIONS FOR HOMEGROWN SEGMENTATON

## PRIVATE
def remove_small_objects_via_opening(frames_in, width_sec, deltat):
    # set up structure elements for opening: operation removes small objects
    open_width = np.int(width_sec / deltat)
    if open_width > 2:  # becasue we will discard first, last
        strel_open = np.zeros(open_width)
        strel_open[1:-1] = 1
        frames_out = spim.binary_opening(frames_in, strel_open)
    else:
        frames_out = frames_in

    return (frames_out)


## PRIVATE
def fill_small_holes_via_closing(frames_in, width_sec, deltat):
    close_width = np.int(width_sec / deltat)
    if close_width > 2:  # becasue we will discard first, last
        strel_close = np.zeros(close_width)
        strel_close[1:-1] = 1
        frames_out = spim.binary_closing(frames_in, strel_close)
    else:
        frames_out = frames_in

    return (frames_out)


def filter_seg_timesV(seg_starts, seg_ends, pad_at_start=0.5, len_to_keep=2.5):
    """
    do some filtering on the segments found to select part for analysis
    rule: find the first segment that is at least (pad_at_start+len_to_keep sec long.
    Discard the firstpad_at_start sec, keep the next len_to_keep sec
    if no such segments, then return empty list

    returns sel_start, sel_end, sel_end_longer
    """
    sel_start = []
    sel_end = []
    sel_end_longer = []

    not_found = True
    for iseg in range(len(seg_starts)):
        seg_dur = seg_ends[iseg] - seg_starts[iseg]
        if (not_found and (seg_dur > (pad_at_start + len_to_keep))):
            t_start = seg_starts[iseg] + pad_at_start
            sel_start.append(t_start)
            sel_end.append(t_start + len_to_keep)
            sel_end_longer.append(max(t_start + len_to_keep, seg_ends[iseg] - pad_at_start))
            not_found = False

    return sel_start, sel_end, sel_end_longer


def get_pause_list(seg_start_sec, seg_end_sec, min_pause_len=0.0):
    """
    gets a list of pauses long than min_pause_len sec
    returns pause_dur list
    """
    assert (len(seg_start_sec) == len(seg_end_sec)), "needs same number of starts, stops!"

    pause_dur = []
    for i in range(len(seg_start_sec) - 1):
        this_pause = seg_start_sec[i + 1] - seg_end_sec[i]
        if this_pause > min_pause_len:
            pause_dur.append(this_pause)

    return pause_dur


def merge_segments_by_gap(seg_starts_in, seg_ends_in, merge_thresh_sec=0.2):
    """
    takes list of segments (same # of starts, ends) and merges events where gap is < merge_thresh_sec
    returns seg_start, seg_end
    """
    assert (len(seg_starts_in) == len(seg_ends_in)), "needs same number of starts, stops!"

    seg_start = []
    seg_end = []
    in_a_seg = False
    gap_list_debug = []
    last_seg = len(seg_starts_in) - 1
    for iseg in range(last_seg):
        gap_dur = seg_starts_in[iseg + 1] - seg_ends_in[iseg]
        gap_list_debug.append(gap_dur)  # for later checking
        if not in_a_seg:  # start a segment
            seg_start.append(seg_starts_in[iseg])
            in_a_seg = True
        elif gap_dur > merge_thresh_sec:
            # we were in a segment, but gap to next seg is too big
            seg_end.append(seg_ends_in[iseg])
            in_a_seg = False
        elif (iseg == last_seg - 1):
            # we were at the very last segment, so need to finish it
            seg_end.append(seg_ends_in[iseg + 1])

    if len(seg_end) == (len(seg_start) - 1):
        seg_end.append(seg_ends_in[iseg + 1])

    return seg_start, seg_end


def require_consistent_amplitude(pmSnd, voiced_in, stdmult=4, doplot=False, figname='none'):
    BIGGEST_BELIEVABLE_SD = 3
    intns = pmSnd.to_intensity(time_step = 0.010000)
    intensity_values_dB = np.squeeze(intns.values.T)  # dB for SPL, i.e. RMS amplitude
    tvec = intns.dt * np.arange(0, intns.n_frames)

    nel = min(np.size(voiced_in), intns.n_frames)
    voiced = np.copy(voiced_in[0:nel])
    indx = np.arange(0, nel)
    intensity_values_dB = intensity_values_dB[0:nel]
    p = np.polyfit(indx[voiced == True], intensity_values_dB[voiced == True], 2)
    int_fit = np.polyval(p, indx)
    iv = intensity_values_dB - int_fit
    sdval = min(BIGGEST_BELIEVABLE_SD, np.std(iv[voiced == True]))
    upper = np.zeros_like(int_fit) + stdmult * sdval
    lower = np.zeros_like(int_fit) - stdmult * sdval

    out_of_range = np.squeeze((iv > upper) | (iv < lower))
    voiced[out_of_range] = False

    if doplot:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(intensity_values_dB)
        plt.plot(int_fit)
        plt.subplot(3, 1, 2)
        plt.plot(iv)
        plt.plot(upper)
        plt.plot(lower)
        plt.subplot(3, 1, 3)
        plt.plot(voiced_in[0:nel])
        plt.plot(voiced)
        # plt.show()  # need to comment this to run in loop; else, waits until this figure is closed
        plt.savefig("ampl_consist{y}.png".format(y = figname))
        plt.close()

    return (voiced)


## PRIVATE
def find_voiced_regions(pmSound, closing_width_sec, opening_width_sec, fmin=75.0):
    # get pitch and use this to find time with voiced / unvoiced
    fmax = 500.0
    pitch_ac = pmSound.to_pitch_ac(pitch_floor = fmin, pitch_ceiling = fmax)
    pitch_ac_values = pitch_ac.selected_array['frequency']
    frame_voiced_raw = pitch_ac_values > 0
    frame_dt = pitch_ac.dt

    # fill in small holes
    frame_voiced1 = fill_small_holes_via_closing(frame_voiced_raw, closing_width_sec, frame_dt)
    # set up structure elements for closing: operation fills in small holes

    # remove small segments via opening
    frame_voiced = remove_small_objects_via_opening(frame_voiced1, opening_width_sec, frame_dt)

    return (frame_voiced, frame_voiced_raw, frame_dt)


## PRIVATE
def find_higherintensity_regions(pmSound, deltat, intens_threhsold, closing_width_sec, opening_width_sec):
    # extract intensity
    intns = pmSound.to_intensity(time_step = deltat)
    intensity_values_dB = intns.values.T  # dB for SPL, i.e. RMS amplitude
    intens_thresh_raw = np.squeeze(intensity_values_dB > intens_threhsold)

    # fill in small holes
    intens_thresh1 = fill_small_holes_via_closing(intens_thresh_raw, closing_width_sec, deltat)
    # set up structure elements for closing: operation fills in small holes

    # remove small segments via opening
    intens_thresh = remove_small_objects_via_opening(intens_thresh1, opening_width_sec, deltat)

    return (intens_thresh)


## PRIVATE
def find_biggest_voiced_seg(frame_voiced):
    # find start and end of largest voiced segment
    lab, nlab = spim.label(frame_voiced)
    lenmax = 0
    segmax = 0
    for iseg in range(1, nlab + 1):
        if np.sum(lab == iseg) > lenmax:  # or store array and use np.argmax
            lenmax = np.sum(lab == iseg)
            segmax = iseg

    ibiggest_samps = np.nonzero(lab == segmax)
    ibiggest_samps = np.squeeze(ibiggest_samps)

    return (ibiggest_samps[0], ibiggest_samps[-1], ibiggest_samps)


####### END HELPER FUNCTIONS FOR SEGMENTATION


####### START  HOMEGROWN SEGMENTATION FUNCTIONS
# NEW  CODE, March 19
# idea: detecting voiced sounds (mostly vowels) might be more selective than the Google VAD, which
# detects generic speech (both voiced and unvoiced speech).  So, use parselmouth to find voiced frames
# and then use morphological processing to clean it up

def voicing_based_segmentation_sustainedvowel(y_snd, title_string=None):
    voiced, voiced_raw, dt_voiced = find_voiced_regions(y_snd, 0.1, 0.3)
    if sum(voiced == True) > 0:
        voiced = require_consistent_amplitude(
                y_snd, voiced, stdmult = 4, doplot = False, figname = title_string)
    voice_seg = find_biggest_voiced_seg(voiced)

    start_sec = voice_seg[0] * dt_voiced
    stop_sec = voice_seg[1] * dt_voiced

    # do a plot?
    if not (title_string == None):
        intns = y_snd.to_intensity(time_step = 0.010000)
        intensity_values_dB = intns.values.T  # dB for SPL, i.e. RMS amplitude

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(y_snd.xs(), y_snd.values.T)
        plt.title(title_string)
        plt.subplot(2, 1, 2)
        plt.plot(voiced_raw)
        # plt.plot(voiced)
        plt.plot(intensity_values_dB / np.max(intensity_values_dB))
        plt.show()

    # return start, end of segment separately
    return (start_sec, stop_sec)


def homerolled_voicing_based_segmentation_ddk(y_snd, title_string=None, qcdir=None):
    # initiialize output
    start_sec = []
    stop_sec = []

    # first, find biggest region of voicing, allowing for gaps between syllables
    voiced_coarse, voiced_raw, dt_voiced = find_voiced_regions(
            y_snd, closing_width_sec = 1, opening_width_sec = 0.1)

    if not all(voiced_coarse == False):  # did we find anything?
        # voiced = require_consistent_amplitude(y_snd, voiced, stdmult = 4, doplot = True, figname = title_string)
        tmp = find_biggest_voiced_seg(voiced_coarse)
        idx_coarse = tmp[2]
        voiced_coarse_samps = np.zeros_like(voiced_coarse, dtype = bool)
        voiced_coarse_samps[idx_coarse] = True
        # array above segments the general time period when there is speech - smooting through little dropouts in voicing

        # next, try to estimate an intensity threshold
        intns = y_snd.to_intensity(time_step = dt_voiced)
        intensity_values_dB = intns.values.T  # dB for SPL, i.e. RMS amplitude

        minlen = np.min([len(intensity_values_dB), len(voiced_coarse_samps)])  # make sure all vectors are same length
        intensity_values_dB = intensity_values_dB[0:minlen]
        voiced_coarse_samps = voiced_coarse_samps[0:minlen]
        voiced_raw = voiced_raw[0:minlen]

        # use the 'coarse' segmentation to blank out any voiced samples detection outside the main speech area
        voiced_finer_samps = voiced_raw[0:minlen] * voiced_coarse_samps

        # find threshold based on finer-scale voicing detections
        # mid_voiced_intens = np.percentile(intensity_values_dB[voiced_finer_samps],25)
        # dec 7 - seems senstive to this! started with 25
        low_voiced_intens = np.percentile(intensity_values_dB[voiced_finer_samps], 25)
        # high_unvoiced_intens = np.percentile(intensity_values_dB[voiced_finer_samps==False],75)

        # use this threshodl to get intensity-based segments
        higher_intens_samps = find_higherintensity_regions(
                y_snd, dt_voiced, low_voiced_intens, closing_width_sec = 5 * dt_voiced,
                opening_width_sec = 3 * dt_voiced)

        # apply coarse and fine regions - then clean up voiced regions using some morphological filtering
        voiced_final = np.logical_or(voiced_coarse_samps * higher_intens_samps[0:minlen], voiced_finer_samps)
        voiced_final = remove_small_objects_via_opening(voiced_final, 3 * dt_voiced, dt_voiced)
        voiced_final = fill_small_holes_via_closing(voiced_final, 5 * dt_voiced, dt_voiced)
        # get segment starts and ends

        # create lists with the labeled segments
        lab, nlab = spim.label(higher_intens_samps)
        for iseg in range(1, nlab + 1):  # iseg=0 is unvoiced, so skip it
            this_seg = np.squeeze(np.nonzero(lab == iseg))
            if this_seg.size > 1:  # this_seg[-1] crashes if this_seg is only 1 element long
                start_sec.append(this_seg[0] * dt_voiced)
                stop_sec.append(this_seg[-1] * dt_voiced)

        # do a plot?
        if not (title_string == None):
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(y_snd.xs(), y_snd.values.T)
            plt.title(title_string)
            plt.subplot(2, 1, 2)
            plt.plot(voiced_raw * voiced_coarse_samps, linestyle = ':')
            plt.plot(voiced_final)
            plt.plot(intensity_values_dB / np.max(intensity_values_dB))
            plt.axhline(y = low_voiced_intens / np.max(intensity_values_dB), color = 'r', linestyle = '-.')
            # plt.show()

            fname_save = title_string + '_voicing_seg_v2'
            # plt.show()  # need to comment this to run in loop; else, waits until this figure is closed
            plt.savefig("{x}/{y}.png".format(x = qcdir, y = fname_save))
            plt.close()

    # return start, end of segment separately
    return (start_sec, stop_sec)


##  code to use AWS output

def aws_based_segmentation(file_basename, transcription_dir, max_pause=2):
    # find matching json AWS Transcribe file, if any
    # first, remove any parentheses, which were stripped before AWS processing
    split_string = file_basename.split("(", 1)
    file_basename = split_string[0]
    json_file = glob.glob(os.path.join(transcription_dir, file_basename + '*.json'))  # os.listdir(data_dir)

    if len(json_file) == 1:
        aws_success = True
        # found a transcription.  Get timing cues from it, for comparison with Textgrid
        word_starts_aws, word_stops_aws, pauses = process_one_AWS_Transcribe_file(
                transcription_dir, json_file[0])
        pauses = discard_overlong_pauses(pauses, max_believable_pause_sec = max_pause)
    else:
        aws_success = False
        word_starts_aws = []
        word_stops_aws = []
        pauses = []

    # return start, end of segment separately
    return word_starts_aws, word_stops_aws, pauses, aws_success

##  code to use forced aligner output

def aligner_based_segmentation(file_basename, aligner_dir, max_pause=2):
    # read output of Montreal forced aligner, which is in textgrid file

    # find matching Textgrid file, if any
    tg_file = glob.glob(os.path.join(aligner_dir, file_basename + '*.TextGrid'))  # os.listdir(data_dir)
    if len(tg_file) == 1:  # found a an aligner output file
        align_success = True
        word_starts_align, word_stops_align, pauses  = process_one_textgrid_file(tg_file[0])

        pauses = discard_overlong_pauses(pauses, max_believable_pause_sec = max_pause)
    else:
        align_success = False
        word_starts_align = []
        word_stops_align = []
        pauses = []

    # return start, end of segment separately
    return word_starts_align, word_stops_align, pauses, align_success


##### syllable nuclei code - see  https://github.com/drfeinberg/PraatScripts

# Translated to Python in 2019 by David Feinberg
# I changed all the variable names so they are human readable
#
# slight modifications, Dec 2021, Brian Tracey:

# def speech_rate(filename):  original from Feinberg
#     silencedb = -25
#     mindip = 2
#     minpause = 0.3
#     sound = pm.Sound(filename)


def feinberg_speech_rate(sound, silencedb=-25, mindip=2, minpause=0.3):
    intensity = sound.to_intensity(50)
    threshold, silence_threshold = find_silence_thresholds(intensity, silencedb)
    textgrid = pm.praat.call(intensity, "To TextGrid (silences)", silence_threshold, minpause, 0.1, "silent", "sounding")

    sound_from_intensity = filter_sound_from_intensity(intensity)
    timecorrection = time_correction_for_converted_sound(sound, sound_from_intensity)
    timepeaks = estimate_peak_times(sound_from_intensity, threshold)
    validtime = find_valid_peaks(intensity, timepeaks, mindip)
    voicedpeak = find_voiced_times(sound, validtime, textgrid)
    syll_nuclei = np.asarray(voicedpeak) * timecorrection

    speakStarts, speakStops = find_speak_starts_and_stops(textgrid)
    pauses = get_pause_list(speakStarts, speakStops, min_pause_len = minpause)
    return speakStarts, speakStops, syll_nuclei, pauses

def find_silence_thresholds(intensity, silencedb):
    min_intensity = pm.praat.call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_intensity = pm.praat.call(intensity, "Get maximum", 0, 0, "Parabolic")
    # get .99 quantile to get maximum (without influence of non-speech sound bursts)
    max_99_intensity = pm.praat.call(intensity, "Get quantile", 0, 0, 0.99)
    # estimate Intensity threshold
    threshold = max_99_intensity + silencedb
    silence_threshold = threshold - max_intensity
    if threshold < min_intensity:
        threshold = min_intensity
    return threshold, silence_threshold

def filter_sound_from_intensity(intensity):
    intensity_matrix = pm.praat.call(intensity, "Down to Matrix")
    # sndintid = sound_from_intensity_matrix
    return pm.praat.call(intensity_matrix, "To Sound (slice)", 1)

def time_correction_for_converted_sound(sound, sound_from_intensity):
    originaldur = sound.get_total_duration()
    # use total duration, not end time, to find out duration of intensity_duration
    # in order to allow nonzero starting times.
    # TODO: check... is this the same as sound_from_intensity.get_total_duration()
    intensity_duration = pm.praat.call(sound_from_intensity, "Get total duration")
    # calculate time correction due to shift in time for Sound object versus intensity object
    return originaldur / intensity_duration

def estimate_peak_times(sound_from_intensity, threshold):
    point_process = pm.praat.call(sound_from_intensity, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
    # estimate peak positions (all peaks)
    numpeaks = pm.praat.call(point_process, "Get number of points")
    t = [pm.praat.call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]

    timepeaks = []
    for time in t:
        # TODO: check... is this the same as Sound.get_value(time)
        value = pm.praat.call(sound_from_intensity, "Get value at time", time, "Cubic")
        if value > threshold:
            timepeaks.append(time)
    return timepeaks

def find_valid_peaks(intensity, timepeaks):
    # Fill array with valid peaks: only intensity values if preceding dip in intensity is greater than mindip
    validtime = []
    for p, time in enumerate(timepeaks):
        if p == len(timepeaks) - 1:
            break  # Looking forward one, so don't process the last value
        this_intensity = pm.praat.call(intensity, "Get value at time", time, "Cubic")
        dip = pm.praat.call(intensity, "Get minimum", time, timepeaks[p + 1], "None")
        diffint = abs(this_intensity - dip)
        if diffint > mindip:
            validtime.append(time)
    return validtime

def find_voiced_times(sound, validtime, textgrid):
    # Look for only voiced parts
    pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
    voicedpeak = []
    for time in validtime:
        whichinterval = pm.praat.call(textgrid, "Get interval at time", 1, time)
        whichlabel = pm.praat.call(textgrid, "Get label of interval", 1, whichinterval)
        value = pitch.get_value_at_time(time)
        if not math.isnan(value):
            if whichlabel == "sounding":
                voicedpeak.append(time)
    return voicedpeak

def find_speak_starts_and_stops(textgrid):
    silencetier = pm.praat.call(textgrid, "Extract tier", 1)
    silencetable = pm.praat.call(silencetier, "Down to TableOfReal", "sounding")
    # pull out pause time info
    speakStarts = []
    speakStops = []
    npauses = pm.praat.call(silencetable, "Get number of rows")
    speakingtot = 0
    for ipause in range(npauses):
        pause = ipause + 1
        beginsound = pm.praat.call(silencetable, "Get value", pause, 1)
        endsound = pm.praat.call(silencetable, "Get value", pause, 2)
        speakStarts.append(beginsound)
        speakStops.append(endsound)
    return speakStarts, speakStops


## timing feature extraction

def get_timing_features_words(word_starts_sec, word_end_sec, pause_dur_sec, min_meaningful_pause=0.1):
        # get timing cues from segmented speech
        # folow Mundt et al, 2007, depression voice

        # 100 ms is shortest pause examined in https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-016-0096-7/tables/3
        # so it is taken here as defautl min_meaningful_pause
        # inputs are lists.  Convert them to np arrays
        pause_dur_sec = np.array(pause_dur_sec)
        word_starts_sec = np.array(word_starts_sec)
        word_end_sec = np.array(word_end_sec)

        assert (len(word_starts_sec) == len(word_end_sec)), "needs same number of starts, stops!"

        # first, initialize all outputs (to 0 or nan as appropriate)
        pause_total_time = 0.0
        total_time = 0.0
        pause_len_mean = np.nan
        pause_len_std = np.nan
        pause_len_median = np.nan
        pause_iqr = np.nan
        pause_frac = np.nan
        words_per_sec = np.nan
        median_word_length = np.nan
        print('TODO: how to handle nan pauses? causes problems later')

        # pause stats
        # threshold to find # pauses longer than min_meaningful_pause
        pause_dur_sec = pause_dur_sec[pause_dur_sec > min_meaningful_pause]

        num_pauses = len(pause_dur_sec)
        if num_pauses > 0:
            pause_total_time = np.sum(pause_dur_sec)
            pause_len_mean = np.mean(pause_dur_sec)
            pause_len_std = np.std(pause_dur_sec)

            # do quartile statistics versions of above
            pause_len_median = np.median(pause_dur_sec)
            q75, q25 = np.percentile(pause_dur_sec, [75, 25])
            pause_iqr = q75 - q25

        # word stats
        num_words = len(word_starts_sec)
        if num_words > 0:
            # set total speaking time as sum(word_dur_sec) + sum(pause_dur_sec)
            # as this automatically skips long silences between readings
            total_time = np.sum(word_end_sec - word_starts_sec) + pause_total_time
            words_per_sec = float(num_words) / total_time
            median_word_length = np.median(word_end_sec - word_starts_sec)

        if total_time > 0:
            pause_frac = pause_total_time / total_time

        timing_dict = {
                'total_time': total_time, 'num_words': num_words,
                'words_per_sec': words_per_sec,
                'median_word_length': median_word_length,
                'num_pauses': num_pauses, 'pause_len_mean': pause_len_mean,
                'pause_len_std': pause_len_std, 'pause_time': pause_total_time,
                'pause_frac': pause_frac, 'pause_median': pause_len_median,
                'pause_iqr': pause_iqr}

        return timing_dict


def get_timing_features_syl(speech_starts_sec, speech_stops_sec, syl_nuclei_sec, pause_dur_sec, check_seg_change=False):
        # get timing cues from segmented speech
        # folow Mundt et al, 2007, depression voice, also Feinburg code

        # assumptions about 'speech' inputs.
        # If nothing else, set speech_starts_sec = [0] and speech_starts_sec = [file length]
        assert (len(speech_starts_sec) > 0)
        assert (len(speech_starts_sec) == len(speech_stops_sec))

        # inputs are lists.  Convert them to np arrays
        pause_dur_sec = np.array(pause_dur_sec)
        speech_starts_sec = np.array(speech_starts_sec)
        speech_stops_sec = np.array(speech_stops_sec)
        syl_nuclei_sec = np.array(syl_nuclei_sec)

        # basic timing stats speaking time
        pause_total_time = np.sum(pause_dur_sec)
        phonationtime = np.sum(speech_stops_sec - speech_starts_sec)
        # set total speaking time as sum(word_dur_sec) + sum(pause_dur_sec)
        # as this automatically skips long silences between readings
        total_time = phonationtime + pause_total_time
        pause_frac = pause_total_time / total_time

        # pause stats
        num_pauses = len(pause_dur_sec)
        if num_pauses > 0:
            pause_len_mean = np.mean(pause_dur_sec)
            pause_len_std = np.std(pause_dur_sec)

            # do quartile statistics versions of above
            pause_len_median = np.median(pause_dur_sec)
            q75, q25 = np.percentile(pause_dur_sec, [75, 25])
            pause_iqr = q75 - q25
        else:
            pause_len_mean = np.nan
            pause_len_std = np.nan
            pause_len_median = np.nan
            pause_iqr = np.nan

        # syllable stats
        num_syll = len(syl_nuclei_sec)

        syl_speakingrate = np.nan
        syl_articulationrate = np.nan
        syl_avg_duration = np.nan
        syl_median_spacing = np.nan
        syl_iqr_spacing = np.nan

        if num_syll > 0:
            syl_speakingrate = num_syll / total_time
            syl_articulationrate = num_syll / phonationtime
            syl_avg_duration = 1 / syl_articulationrate

            # look at stats on differences
            syl_spacing_sec = np.diff(syl_nuclei_sec)
            if check_seg_change:
                # label each nucleus by its speech chunk
                chunk_label = -1 * np.ones_like(syl_nuclei_sec)
                for ch in range(len(speech_starts_sec)):
                    ix = np.where(np.logical_and(
                            syl_nuclei_sec >= speech_starts_sec[ch],
                            syl_nuclei_sec <= speech_stops_sec[ch]))
                    ix = np.squeeze(ix)
                    chunk_label[ix] = ch

                # now only keep syllable jumps that don't cross chunk boundaries
                chunk_change = np.diff(chunk_label)
                syl_spacing_sec = syl_spacing_sec[chunk_change == 0]

            if len(syl_spacing_sec) > 5:
                syl_median_spacing = np.median(syl_spacing_sec)
                # was a check for debugging only syl_mean_spacing = np.mean(syl_spacing_sec)
                q75, q25 = np.percentile(syl_spacing_sec, [75, 25])
                syl_iqr_spacing = q75 - q25
            else:
                syl_median_spacing = np.nan
                syl_iqr_spacing = np.nan

            # redo mean after outliner detection.  doesn't work that well...
            #outlier_thresh = q75 + 1.5*syl_iqr_spacing
            #syl_spacing_sec_OLrem = syl_spacing_sec[syl_spacing_sec < outlier_thresh]
            #syl_mean_OLrem = np.mean(syl_spacing_sec_OLrem)

        timing_dict = {
                'total_time': total_time, 'phonationtime': phonationtime,
                'pause_total_time': pause_total_time, 'pause_frac': pause_frac,
                'num_pauses': num_pauses, 'pause_len_mean': pause_len_mean,
                'pause_len_std': pause_len_std, 'pause_median': pause_len_median,
                'pause_iqr': pause_iqr, 'num_syll': num_syll,
                'syl_speakingrate': syl_speakingrate,
                'syl_articulationrate': syl_articulationrate,
                'syl_avg_duration': syl_avg_duration,
                'syl_median_spacing': syl_median_spacing,
                'syl_iqr_spacing': syl_iqr_spacing}

        return timing_dict

