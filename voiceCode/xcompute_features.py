#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:27:10 2018
@author: btracey

# this file contains code for extracting acoustic features using parselmouth, 
# for one single parselmouth object

"""

import librosa
import matplotlib.pyplot as plt
import numpy as np
import parselmouth as pm
import scipy.signal

import dfa


class WaveformAnalyzer:
    def __init__(self):
        self.pm_sound = []
        self.nclip = 0
        self.data_is_loaded = False

    def has_data(self):
        return self.data_is_loaded

    def getSound(self):
        return self.pm_sound

    def get_num_clipped(self):
        return self.nclip

    ## DEPRECATED
    def dict_mean(self, dict_list):
        mean_dict = {}
        for key in dict_list[0].keys():
            mean_dict[key] = np.mean([d[key] for d in dict_list], axis=0)
        return mean_dict
    # TODO weight this by length of each utterance

    ## DEPRECATED
    def dict_weighted_mean(self, dict_list):
        # do a duration-weighted average of values in all segments in the list

        # pull out durations and find fraction of total = weight
        durs = [d.get('dur', None) for d in dict_list]
        durs = np.array(durs)
        dur_wt = durs / np.sum(durs)

        # do the wieghted average
        mean_dict = {}
        for key in dict_list[0].keys():
            data = [d[key] for d in dict_list]
            # are the values all nan?
            if np.isnan(data).all():
                mean_dict[key] = np.nan
            else:
                mean_dict[key] = np.nansum(data * dur_wt) # np.average(data, weights=dur_wt) average doesn't handle weights

        # do a little check...
        wt_dur = np.sum(durs * dur_wt)
        fracdiff = np.abs(wt_dur - mean_dict['dur']) / mean_dict['dur']
        assert (fracdiff < 0.001), "check weighting calculation!"

        return mean_dict

    ## PRIVATE
    def filter_waveform(self, y, sr, notch_freq=60., notch_bw=2.0):
        """Reads a .wav file.
        Takes the path, and returns (PCM audio data, sample rate).
        """

        # power line notch filter
        quality = notch_freq / notch_bw  # quality of 30 gives about 25 dB notch at 60 Hz
        b, a = scipy.signal.iirnotch(notch_freq, quality, sr)
        yfilt = scipy.signal.lfilter(b, a, y)

        # remove any DC offset
        yfilt = yfilt - np.mean(yfilt)

        debugit = False
        if debugit:
            freq, h = scipy.signal.freqz(b, a, fs=sr,worN=200000)
            fig, ax = plt.subplots(2, 1, figsize=(8, 6))
            ax[0].plot(freq, 20 * np.log10(abs(h)), color='blue')
            ax[0].set_title("Frequency Response")
            ax[0].set_ylabel("Amplitude (dB)", color='blue')
            ax[0].set_xlim([0, 100])
            ax[0].set_ylim([-25, 10])
            ax[0].grid()
            ax[1].plot(freq, np.unwrap(np.angle(h)) * 180 / np.pi, color='green')
            ax[1].set_ylabel("Angle (degrees)", color='green')
            ax[1].set_xlabel("Frequency (Hz)")
            ax[1].set_xlim([0, 100])
            ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
            ax[1].set_ylim([-90, 90])
            ax[1].grid()
            plt.show()

        return yfilt

    def load_waveform(self, y, sr, num_channels, full_range_counts=32768,
                      notch_filter_freq = -1, normalize_waveform=True):
        self.data_is_loaded = len(y) > 0
        if self.data_is_loaded:

            if type(y)==bytes:
                # convert the waveform to parselmouth sound object
                ynp = np.fromstring(y, dtype=np.int16)
                # if stereo, just grab channel 1
                ynp = ynp[::num_channels]
                #TODO consider allowing an input that would chose which channel
                self.nclip = np.sum(np.abs(ynp) > (full_range_counts - 2))
                # de-mean the waveform to eliminate DC offset
                ynp = ynp - np.mean(ynp)
                ynp = ynp / full_range_counts  # scale so stays in range -1 to 1

            elif type(y)==np.ndarray:   #'y.dtype == 'float32': # to handle denoised inputs
                ynp = y
            else:
                print('going splat')


            if notch_filter_freq > 0:
                ynp = self.filter_waveform(ynp, sr, notch_freq = notch_filter_freq)

            if normalize_waveform:
                ynp0 = ynp
                ynp = (ynp - np.mean(ynp,axis=0)) #/  np.std(ynp, axis=0)

            self.pm_sound = pm.Sound(ynp, sr)

    def get_SNR_info(self, noise_ptile = 3, signal_ptile = 95):
        # get stats to compute SNR.  Should be computed for entire recording
        intns = self.pm_sound.to_intensity()
        intensity_values_dB = intns.values.T  # dB for SPL, i.e. RMS amplitude
        # multiply by 2 below to go to power
        noise_pow_dB = 2 * np.percentile(intensity_values_dB, noise_ptile)
        sig_pow_dB = 2 * np.percentile(intensity_values_dB, signal_ptile)
        est_snr_power_dB = sig_pow_dB - noise_pow_dB
        #print(est_snr_power_dB)

        return est_snr_power_dB, noise_pow_dB

    def get_features_across_segments(self,word_starts_sec, word_end_sec, do_voicedVsunvoiced=True, do_dfa = False):

        # loop over all segments
        assert (len(word_starts_sec) == len(word_end_sec)), "needs same number of starts, stops!"

        # get timing features
        #timing_feat = self.get_timing_features(word_starts_sec, word_end_sec, pause_dur_sec)

        # get acoustic features

        # pull out start of first to end of last segment
        # but re-start time from zero (don't preseve times) to avoid confusing feature code
        t1 = word_starts_sec[0]
        t2 = word_end_sec[-1]
        sound_seg = self.pm_sound.extract_part(from_time=t1, to_time=t2, preserve_times=False)
        seg_start_shifted = [x - t1 for x in word_starts_sec]
        seg_end_shifted = [x - t1 for x in word_end_sec]

        # instantiate object and get features

        if sound_seg.get_total_duration()>0.1:  # parselmouth pitch needs at least 0.064 sec for 100 Hz min pitch
            try:
                featureCalculator = ComputeAcousticFeaturesCore(sound_seg)
                featureCalculator.set_parameters(
                        calc_voicedVsUn = do_voicedVsunvoiced,
                        calc_voicedVsUn_mfcc = False, do_dfa = do_dfa)
                # change default parameters on line below
                acoust_feat = featureCalculator.get_features(
                        seg_start_shifted, seg_end_shifted)
            except:
                acoust_feat = []
                print('extraction failed')

        return acoust_feat

########## DONE WITH CLASS ############


##########  START NEW CLASS  #######


## PRIVATE
class ComputeAcousticFeaturesCore:
    def __init__(self, pm_sound):
        self.pm_sound = pm_sound
        self.set_parameters() # set parameters with defaults


    def set_parameters(self, do_dfa=False, pitchmin=70.0, pitchmax = 300.0,
                       sgram_frame_sec=0.02, calc_voicedVsUn=True, calc_voicedVsUn_mfcc = False):
            self.params = {"do_dfa": do_dfa, "pitchmin":pitchmin, "pitchmax":pitchmax,
                           "sgram_frame_sec":sgram_frame_sec, "calc_voicedVsUn":calc_voicedVsUn,
                           "calc_voicedVsUn_mfcc":calc_voicedVsUn_mfcc}

    def get_features(self, seg_starts_sec=None, seg_ends_sec=None, do_all_only=True):
        # public
        # Given a parselmouth sound object, computes all the standard features
        duration_sec = self.pm_sound.get_total_duration()

        # TODO: BT - run these as time-aligned or just whole-record?
        pulsed_dict  = self.get_pulsed_stats()
        harm_dict = self.get_harmonicity()
        cep_dict = self.get_cepstral_peak_prominence()

        t_vec, f_vec, pitch_values, SGram, rmsdB = self.timealigned_pitch_and_spect()
        pitch_dict = self.get_pitch_stats(pitch_values)

        if seg_starts_sec is None: # didn't pass in seg start, stop
            sil_voic_unvoic = self.find_silent_voiced_unvoiced(rmsdB, pitch_values)
        else: # we did!
            sil_voic_unvoic = self.find_silent_voiced_unvoiced_SEG(
                    t_vec, pitch_values, seg_starts_sec, seg_ends_sec)

        frac_voiced_of_nonsilent = np.sum(sil_voic_unvoic == 2) / (np.sum(sil_voic_unvoic > 0))
        frac_voiced = np.sum(sil_voic_unvoic == 2) / len(sil_voic_unvoic)
        frac_silent = np.sum(sil_voic_unvoic == 0) / len(sil_voic_unvoic)

        # compute librosa features
        mfccs, spec_contrast, spec_flatness = self.calc_librosa_features_from_spectrogram(SGram)

        # these function will compute feats for all non-silent frames, and optionally for voiced vs. unvoiced
        spec_dict = self.get_spec_contrast_and_flatness(spec_contrast, spec_flatness, sil_voic_unvoic)
        amp_dict = self.get_amplitude_stats(rmsdB,sil_voic_unvoic)
        mfcc_dict = self.mfcc_wrapper(mfccs, sil_voic_unvoic)

        # TODO do alpha only on non-silent frames
        if self.params["do_dfa"]:  # DFA is slow, so optinally skip it
            dfa_alpha = self.get_dfa_alpha()
        else:
            dfa_alpha = np.nan

        # merge all features into one dictionary..
        feat_dict = {}
        feat_dict.update(pitch_dict)
        feat_dict.update(pulsed_dict)
        feat_dict.update(amp_dict)
        feat_dict.update(spec_dict)
        feat_dict.update(harm_dict)
        feat_dict.update(cep_dict)
        feat_dict.update(mfcc_dict)
        feat_dict.update({'dur': duration_sec, 'dfa_alpha': dfa_alpha,
                          'frac_voiced':frac_voiced, 'frac_silent':frac_silent,
                          'frac_voiced_of_nonsilent':frac_voiced_of_nonsilent})

        return(feat_dict)

    # All PRIVATE
    def add_dpostfix(self, d_in,postfix):
        d_out = dict(("{}_{}".format(k, postfix), v) for k, v in d_in.items())
        return d_out

    def find_silent_voiced_unvoiced(self, rmsdB_vec, pitch_vec, silence_thresh_dB = 25):
        # estimates silent frames based on RMS intensity; uses pitch to find voiced vs unvoiced frames
        assert rmsdB_vec.size==pitch_vec.size, "rms and pitch vectors different size"

        suv_vec = np.ones_like(rmsdB_vec)  # default to unvoiced
        # set threshold based on max VOICED region to avoid bangs, etc
        if np.sum(np.isfinite(pitch_vec)) > 0:
            mxRms = np.max(rmsdB_vec[np.isfinite(pitch_vec)])
        else:
            mxRms = np.max(rmsdB_vec)

        is_silent = rmsdB_vec < (mxRms-silence_thresh_dB)
        suv_vec[is_silent] = 0
        suv_vec[np.isfinite(pitch_vec)] = 2
        return suv_vec

    def find_silent_voiced_unvoiced_SEG(self, t_vec, pitch_vec, seg_starts_sec, seg_ends_sec):
        # estimates silent frames based on RMS intensity; uses pitch to find voiced vs unvoiced frames
        assert t_vec.size==pitch_vec.size, "t_vec and pitch vectors cannot be different size"

        # 0) default to silence
        suv_vec = np.zeros_like(t_vec)

        # 1) set regions in voiced regions to Unvoiced, by default
        dt = t_vec[1]-t_vec[0]  # this is half-width of librosa window
        for ch in range(len(seg_starts_sec)):
            ix = np.where(np.logical_and(t_vec >= seg_starts_sec[ch],
                                         t_vec <= seg_ends_sec[ch]+dt))
            ix = np.squeeze(ix)
            suv_vec[ix] = 1

        # 2) set voiced regions where pitch is finite
        suv_vec[np.isfinite(pitch_vec)] = 2


        return suv_vec

    def timealigned_pitch_and_spect(self):
        # librosa - intensity and spectrogram
        y = self.pm_sound.values
        y = np.ndarray.flatten(y)  # flatten it to a true 1-d array that librosa can deal with
        sr = self.pm_sound.sampling_frequency
        nsamp = int(sr * self.params["sgram_frame_sec"])
        hop_samp = int(nsamp / 2) # set pitch frames and spectrogram hoplength to be 2x as fast as spectrogram frames

        # compute STFT and get time and freq axis for it
        SGram = librosa.stft(y=y, n_fft= nsamp, hop_length = hop_samp)
        nf = SGram.shape[0]
        # TODO: confirm f_vec no longer needed
        f_vec = librosa.fft_frequencies(sr=sr, n_fft=1 + (2 * (nf - 1)))
        nt = SGram.shape[1]
        t_vec = librosa.frames_to_time(
                np.arange(nt), sr = sr, n_fft = nsamp, hop_length = hop_samp)

        # get rms power.  Get from SGram to ensure there are no size mismatches
        #rms = librosa.feature.rms(y=y,frame_length=hop_samp, hop_length=hop_samp)
        rms = librosa.feature.rms(S=SGram, frame_length=nsamp, hop_length=hop_samp)
        # see Praat manual (Intensity: get Standard deviation): Praat does std dev, etc on db-transfomred intensity
        rmsdB = np.squeeze(librosa.amplitude_to_db(rms))

      # get pitch values, find voiced
        dt_pitch = self.params["sgram_frame_sec"] / 5
        pitch_ac = self.pm_sound.to_pitch_ac(
                time_step = dt_pitch, pitch_floor = self.params["pitchmin"],
                pitch_ceiling = self.params["pitchmax"])
        pitch_val_fast = pitch_ac.selected_array['frequency']
        if np.sum(pitch_val_fast)==0:
            print('** no voiced frames found before interp **')
        pitch_val_fast[pitch_val_fast == 0] = np.nan  # replace unvoiced samples by NaN

        # set up time vector, and inteprolate
        #t_pitch = pitch_ac.t1 + np.arange(pitch_val_fast.size) * dt_pitch
        t_pitch =  np.arange(pitch_val_fast.size) * dt_pitch # use relative time

        pitch_values = np.interp(t_vec, t_pitch, pitch_val_fast,
                                           left=np.nan, right = np.nan)
        if np.nansum(pitch_values) == 0:
            print(' ** no voiced frames found after interp ** ')

        # ix_unvoiced = np.argwhere(np.isnan(pitch_values))
        # ix_voiced = np.where(np.isnan(pitch_values)==False)

        assert t_vec.size == rmsdB.size, "rmsdB is wrong length"
        assert t_vec.size == pitch_values.size, "pitch is wrong length"
        assert t_vec.size ==  SGram.shape[1], "spectrogram is wrong length"

        return t_vec, f_vec, pitch_values, SGram, rmsdB

    def calc_librosa_features_from_spectrogram(self, SGram):
        # move me
        sr = self.pm_sound.sampling_frequency
        Sm = librosa.feature.melspectrogram(  # BT may 26 bumped fmax from 8 to 12k
                S = np.abs(SGram) ** 2, sr = sr, n_mels = 40, fmax = 12000)
        mfcc = librosa.feature.mfcc(S = librosa.power_to_db(Sm), n_mfcc = 13)
        sz = np.shape(mfcc)
        if sz[1] > 9: # by default, need
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order = 2)
        else:
            mfcc_delta = np.full([sz[0], sz[1]], np.nan)  # set up matrix of nan's
            mfcc_delta2 = np.full([sz[0], sz[1]], np.nan)  # set up matrix of nan's

        mfcc_all = np.concatenate((mfcc, mfcc_delta), axis = 0)
        #mfcc_all = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis = 0)

        spectral_contrast = librosa.feature.spectral_contrast(S = np.abs(SGram))
        spectral_flatness = librosa.feature.spectral_flatness(S = np.abs(SGram))

        return mfcc_all, spectral_contrast, spectral_flatness

    def get_spec_contrast_and_flatness(self, spectral_contrast, spectral_flatness, suv_vec):
        # flatness is a 1-d array; only compute stats for non-silent periods
        spec_dict = self.get_mean_sd_spread(spectral_flatness[:,suv_vec>0], 'spec_flat')
        if self.params["calc_voicedVsUn"]:
            sfU = spectral_flatness[:, suv_vec == 1]
            tmp = self.get_mean_sd_spread(sfU, 'spec_flat')
            tmp = self.add_dpostfix(tmp, 'U')
            spec_dict.update(tmp)
            sfV = spectral_flatness[:, suv_vec == 2]
            tmp = self.get_mean_sd_spread(sfV, 'spec_flat')
            tmp = self.add_dpostfix(tmp, 'V')
            spec_dict.update(tmp)

        # contrast is computed by octave band
        nb = spectral_contrast.shape[0] # number of bands
        assert suv_vec.size == spectral_contrast.shape[1], "suv size should match spectral_contrast size"

        fmin = 100 # DANGER HARD-CODED!  this is the librosa default value
        octa = np.zeros(nb + 2)  # define bands - follows librosa code
        octa[1:] = fmin * (2.0 ** np.arange(0, nb + 1))

        for k in range(nb):
            labl = "contrst_%d_%d" % (octa[k], octa[k+1])
            # only compute stats for non-silent periods
            tmp = self.get_mean_sd_spread(spectral_contrast[k,suv_vec>0],labl)
            spec_dict.update(tmp)

            if self.params["calc_voicedVsUn"]:
                scV = spectral_contrast[k, suv_vec == 2]
                tmp = self.get_mean_sd_spread(scV, labl)
                tmp = self.add_dpostfix(tmp, 'V')
                spec_dict.update(tmp)

                scU = spectral_contrast[k, suv_vec == 1]
                tmp = self.get_mean_sd_spread(scU, labl)
                tmp = self.add_dpostfix(tmp, 'U')
                spec_dict.update(tmp)

        return spec_dict

    def parse_voice_report(self, vreport,target_string):
        # split voice report into lines, then loop over the lines
        report_by_line = vreport.split('\n')
        for this_line in report_by_line:

            if this_line.find(target_string) != -1:
                # found the target! split the line at the :, convert value to numeric
                line_parts = this_line.split(':')
                last_part = line_parts[1]
                # this part of sting may contain multiple numbers - so split on spaces discard empty elements, take first.
                # there must be a better way, but this works...
                last_part_split = last_part.split(' ') # split on breaks
                last_part_split = [s for s in last_part_split if s] # discard empty bits
                val = float(last_part_split[0])

        return val

    def get_mean_sd_spread(self, data_vec, labl):
        # little utility to do the same stats over and over
        mn  = np.nanmean(data_vec)
        sd = np.nanstd(data_vec)
        iqr = np.nanpercentile(data_vec,75)-np.nanpercentile(data_vec,25)
        iqr_norm = iqr / np.nanmedian(data_vec)

        harm_dict = {labl+'_mean': mn, labl+'_sd': sd, labl+'_iqr': iqr,labl+'_iqrnorm': iqr_norm}
        return harm_dict

    def get_amplitude_stats(self, rmsdB, suv_vec):
        # in suv_vec, 0 means silent, 1 is unvoiced, 2 is voiced
        assert suv_vec.size == rmsdB.size, "suv size should match rms size"
        rmsNotSilent = rmsdB[suv_vec>0]
        rmsU= rmsdB[suv_vec == 1]
        rmsV = rmsdB[suv_vec == 2]
        # first, do all frames
        amp_dict = self.get_mean_sd_spread(rmsNotSilent,'rmsdB')
        #'intensity_mean': np.nanmean(rmsNotSilent), 'intensity_sd': np.nanstd(rmsNotSilent)}
        if self.params["calc_voicedVsUn"]:
            amp_dict.update(self.get_mean_sd_spread(rmsV, 'rmsdB_V'))
            #amp_dict.update({'intensity_V_mean': np.nanmean(rmsV), 'intensity_V_sd': np.nanstd(rmsV)})
            amp_dict.update(self.get_mean_sd_spread(rmsU, 'rmsdB_U'))
            #amp_dict.update({'intensity_U_mean': np.nanmean(rmsU), 'intensity_U_sd': np.nanstd(rmsU)})

        return amp_dict

    def get_harmonicity(self):
        # calculate HNR
        time_step = self.params["sgram_frame_sec"]/2
        harmonicity = self.pm_sound.to_harmonicity(time_step = time_step)
        mn_harm  = harmonicity.values[harmonicity.values != -200].mean()
        sd_harm  = harmonicity.values[harmonicity.values != -200].std()
        harm_dict = {'harmonicity_mean': mn_harm, 'harmonicity_sd': sd_harm}
        return harm_dict

    def get_cepstral_peak_prominence(self, show_cepstrum = False):
        # calculate CPP - see https://gitter.im/PraatParselmouth/Lobby?at=5ea5c81e61a0002f794e4622
        # for parameters, see https://www.fon.hum.uva.nl/praat/manual/Sound__To_PowerCepstrogram___.html
        power_cepstrogram = pm.praat.call(self.pm_sound, "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50)
        len_sec = self.pm_sound.get_total_duration()
        slice_step_sec = 0.1
        nslice = np.int32(len_sec / slice_step_sec)

        cpp_vec = []
        for sl in range(nslice):
            t_slice = (sl+1)*slice_step_sec
            slice = pm.praat.call(power_cepstrogram, "To PowerCepstrum (slice)", t_slice)
            cpp_slice = pm.praat.call(slice, "Get peak prominence",60.0, 330.0,
                         'Parabolic', 0.001, 0.0, 'Straight', 'Robust')
            cpp_vec.append(cpp_slice)

        # for parameters, see https: // www.fon.hum.uva.nl / praat / manual / PowerCepstrum__Get_peak_prominence___.html
        cpps_entire_waveform = pm.praat.call(power_cepstrogram, "Get CPPS", False, 0.01, 0.01, 60.0,
                                     330.0, 0.05, 'Parabolic', 0.001, 0.0, 'Straight', 'Robust')

        cep_dict = {'cpp_mean': np.nanmean(cpp_vec),
                    'cpp_sd': np.nanstd(cpp_vec),
                    'cpp_med': np.nanmedian(cpp_vec),
                    'cpp_iqr':np.nanpercentile(cpp_vec,75) - np.nanpercentile(cpp_vec,25),
                    'cpps_entirewv': cpps_entire_waveform
                    }  # save as dictionary for compatibility with other analyses

        # interesting: see http://www.homepages.ucl.ac.uk/~uclyyix/ProsodyPro/
        # appears we want to chop

        # cepstrum itself: unused but maybe interesting at some point?
        if show_cepstrum:
            cepstrum_values = pm.praat.call(power_cepstrogram, "To Matrix").values
            plt.figure()
            plt.imshow(np.log10(np.abs(cepstrum_values[1:200,:])),aspect="auto")
            plt.show()


        return(cep_dict)

    def get_dfa_alpha(self):
        y = self.pm_sound.values
        y = np.ndarray.flatten(y)  # flatten it to a true 1-d array
        scales, fluct, alpha = dfa.dfa(y)
        return alpha

        # PITCH-RELATED ##############
        # TODO consider vibratio measure:
        #  see https://github.com/Mak-Sim/Troparion/tree/5126f434b96e0c1a4a41fa99dd9148f3c959cfac/Perturbation_analysis

    def scrub_pitch_values(self, pitch_values, octave_thresh = 0.6):
        # removes suspicously large pitch fluctuatoins
        md_pitch = np.nanmedian(pitch_values)
        p_hi = md_pitch +  md_pitch * octave_thresh  # go up by less than double
        oct_down = md_pitch/2
        p_lo = md_pitch - oct_down * octave_thresh # drop less than an octavoe
        pitch_values_filt  = pitch_values[pitch_values > p_lo]
        pitch_values_filt = pitch_values_filt[pitch_values_filt < p_hi]
        return pitch_values_filt

    def get_ppe(self,extracted_pitch_values):
        """Compute pitch period entropy. Here is a reference MATLAB implementation:
        https://github.com/Mak-Sim/Troparion/blob/5126f434b96e0c1a4a41fa99dd9148f3c959cfac/Perturbation_analysis/pitch_period_entropy.m
        Note that computing the PPE relies on the existence of voiced portions in the F0 trajectory.

        (borrowed from Surfboard)

        Args:
            extracted_pitch_values (np.array): f0 voiced frames divided by f_min

        Returns:
            float: The pitch period entropy, as per http://www.maxlittle.net/students/thesis_tsanas.pdf
        """

        # take ratio as f0 divided by 10th percentile
        # as approximation of f0 voiced frames divided by f_min
        rat_f0 = extracted_pitch_values / np.nanpercentile(extracted_pitch_values, 5)
        semitone_f0 = np.log(rat_f0) / np.log(2 ** (1 / 12))

        # Whitening
        semitone_f0 = semitone_f0[~np.isnan(semitone_f0)] # drop nans first
        coefficients = librosa.core.lpc(semitone_f0, 2)
        semi_f0 = scipy.signal.lfilter(coefficients, [1], semitone_f0)# [0]
        # Filter to the [-1.5, 1.5] range.
        semi_f0 = semi_f0[np.where(semi_f0 > -1.5)]
        semi_f0 = semi_f0[np.where(semi_f0 < 1.5)]

        # shouldn't we always use same histogram bins?
        distrib = np.histogram(semi_f0, bins=30, density=True)[0]
        # Remove empty bins as these break the entropy calculation.
        distrib = distrib[distrib != 0]

        # Discrete probability distribution
        ppe = np.sum(-distrib * (np.log(distrib) / np.log(2)))
        return ppe

    def get_semitone_stats_from_pitch(self, extracted_pitch_values, ref_ptile = 10):

        # converts pitch to semitones
        # see http: // www.sengpielaudio.com / calculator - centsratio.htm
        # 2 possible approaches
        # 1) Nevler 2017 https://n.neurology.org/content/89/7/650 uses 10th ptile as reference, then uses 90th as range
        # 2) Max Little (and https://github.com/Mak-Sim/Troparion) uses median
        pitch_ref = np.percentile(extracted_pitch_values, ref_ptile)
        pitch_st = 12*np.log2(extracted_pitch_values / pitch_ref)
        p10 = np.percentile(pitch_st,10)
        p25 = np.percentile(pitch_st, 25)
        p50 = np.percentile(pitch_st, 50)
        p75 = np.percentile(pitch_st, 75)
        p90 = np.percentile(pitch_st, 90)

        return(p10,p25,p50,p75,p90, np.nanstd(pitch_st))

    def get_pitch_stats(self, pitch_vals, do_plot=False):

        # scrub the pitch...
        # TODO: is there any reason to not scrub upstream?
        pitch_ac_values = self.scrub_pitch_values(pitch_vals)

        # get basic stats
        mnpitch_ac = np.nanmean(pitch_ac_values)
        sdpitch_ac = np.nanstd(pitch_ac_values)
        if pitch_ac_values.size>0:
            pitchjmp_ac = np.nanmax(np.abs(np.diff(pitch_ac_values)))
            pitchjmp90ptile_ac = np.nanpercentile(np.abs(np.diff(pitch_ac_values)),90)
        else:
            pitchjmp_ac = np.nan; pitchjmp90ptile_ac = np.nan

        #mnpitch_cc = np.nanmean(pitch_cc_values)
        # store these...
        pitch_dict = {'mnpitch_ac' : mnpitch_ac, 'sdpitch_ac' : sdpitch_ac, 'pitchjmp_ac' : pitchjmp_ac,
                      'pitchjmp90ptile_ac': pitchjmp90ptile_ac}

        # compute values in semitones - referenced to self - and append
        # use Nevler method
        #p10n,p25n,p50n,p75n,p90nev = self.get_semitone_stats_from_pitch(pitch_ac_values, ref_ptile = 10)
        # note: p90nev = p90-p10 below.

        # use median - more typical?
        p10, p25, p50, p75, p90, sstd = self.get_semitone_stats_from_pitch(pitch_ac_values, ref_ptile=50)

        # get PPE, with error handling....
        try:
            ppe = self.get_ppe(pitch_ac_values)
        except:
            ppe = np.nan  # initialize to nan; fails if pitch_ac_values are all nan (or too few?)

        semitone_dict = {'semitone_med': p50, 'semitone_iqr': p75-p25,
                         'semitone_range': p90-p10,'semitoneSD':sstd,'PPE':ppe}

        pitch_dict.update(semitone_dict)

        return pitch_dict

    def get_pulsed_stats(self, do_plot=False):
        # get jitter and shimmer: thanks to Yannick
        pulses = pm.praat.call(self.pm_sound, "To PointProcess (periodic, cc)", self.params["pitchmin"], self.params["pitchmax"])

        # Since "To PointProcess (cc)" is currently not available in the Parselmouth Python API,
        # you need to revert to calling the Praat action like in a Praat script, so with the string "To PointProcess (cc)"

        # for args below, see http: // www.fon.hum.uva.nl / praat / manual / PointProcess__Get_jitter__local____.html
        # the 5 numbers input to jitter are: start and stop of time range (0 means take all),
        # shortest possible period, longest possible period, max period_factor (ratio between consecutive periods)
        jitter_loc = pm.praat.call(pulses, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        jitter_locabs= pm.praat.call(pulses, "Get jitter (local, absolute)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        jitter_rap= pm.praat.call(pulses, "Get jitter (rap)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        jitter_ppq5= pm.praat.call(pulses, "Get jitter (ppq5)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        jitter_ddp= pm.praat.call(pulses, "Get jitter (ddp)", 0.0, 0.0, 0.0001, 0.02, 1.3)

        pulse_dict = {'jit_local' : jitter_loc, 'jit_localabs' : jitter_locabs, 'jit_rap' : jitter_rap,
                       'jit_ppq5' : jitter_ppq5,  'jit_ddp' : jitter_ddp}

        # see https://osf.io/umrjq/ for project using parselmouth
        # In praat: select a sound object and a pulses object, then can select jitter.  This shows 6 parameters below:
        # first 5 parameters are same as jitter; last is 'max amplitude factor'
        shimmer_loc = pm.praat.call([self.pm_sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_loc_db = pm.praat.call([self.pm_sound, pulses], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = pm.praat.call([self.pm_sound, pulses], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_dda = pm.praat.call([self.pm_sound, pulses], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        shimmer_dict = {'sh_local' : shimmer_loc, 'sh_localdB' : shimmer_loc_db,
                        'sh_apq5' : shimmer_apq5, 'sh_dda' : shimmer_dda}
        pulse_dict.update(shimmer_dict)

        # overall voice report, as a string...
        pitch_ac = self.pm_sound.to_pitch_ac(pitch_floor=self.params["pitchmin"], pitch_ceiling=self.params["pitchmax"])

        vrep = pm.praat.call([pulses, self.pm_sound, pitch_ac], "Voice report", 0.0, 0.0, 75, 600, 1.3, 1.6, 0.03, 0.45)
        num_voice_breaks = self.parse_voice_report(vrep,'Number of voice breaks:')
        #frac_voice_breaks = parse_voice_report(vrep,'Fraction of locally unvoiced frames:')
        # add these to basic pitch dictionary
        pulse_dict.update({'num_voice_breaks': num_voice_breaks})

        if (do_plot == True):
            times = pitch_ac.xs(),
            pitch_ac_values = pitch_ac.selected_array['frequency']
            plt.plot(times, pitch_ac_values, 'o', markersize=5, color='b')
            plt.plot(times, pitch_ac_values, 'o', markersize=2)
            plt.grid(False)
            plt.ylim(0, pitch_ac.ceiling)
            plt.ylabel("fundamental frequency [Hz]")

        return pulse_dict


    def get_single_formant_info(self, frmt, formant_num):
        # called by formant_stats
        fmt_frq = []
        times = frmt.xs()
        for itm in range(frmt.nt):
            val = frmt.get_value_at_time(formant_number=formant_num, time=times[itm])
            fmt_frq.append(val)

        return times, fmt_frq

    ## DEPRECATED
    def get_formant_stats(self, do_plot=False):
        fmt_syl = self.pm_sound.to_formant_burg()
        tfmt, F1 = self.get_single_formant_info(fmt_syl, 1)
        tfmt, F2 = self.get_single_formant_info(fmt_syl, 2)

        # get basic stats
        mnF1 = np.nanmean(F1)
        sdF1 = np.nanstd(F1)
        mnF2 = np.nanmean(F2)
        sdF2 = np.nanstd(F2)

        if (do_plot == True):
            plt.plot(tfmt, F1, label='F1')
            plt.plot(tfmt, F2, label='F2')
            plt.xlabel('Time, sec')
            plt.ylabel('Freq, Hz')
            plt.legend()
            plt.show()

        return mnF1, mnF2, sdF1, sdF2

    def mfcc_wrapper(self, mfcc_all, suv_vec):
        assert suv_vec.size == mfcc_all.shape[1], "suv size should match mfcc size"
        mfccNotSilent = mfcc_all[:,suv_vec>0]
        mfccU = mfcc_all[:,suv_vec == 1]
        mfccV = mfcc_all[:,suv_vec == 2]
        # first, do all frames
        mfcc_dict = self.get_mfcc(mfccNotSilent)
        if self.params["calc_voicedVsUn_mfcc"]:
            tmp  = self.get_mfcc(mfccV)
            tmp = self.add_dpostfix(tmp,'V')
            mfcc_dict.update(tmp)

            tmp  = self.get_mfcc(mfccU)
            tmp = self.add_dpostfix(tmp,'U')
            mfcc_dict.update(tmp)
        return mfcc_dict

    def get_mfcc(self, mfcc_all):
        # get 'mean on empty slice' warning on line below of delta and delta2 not computed... but is ok
        mfcc_means = np.nanmean(mfcc_all, axis = 1)
        mfcc_sds = np.nanstd(mfcc_all, axis=1)

        # maybe use these later?
        # mfcc_p90 = np.percentile(mfcc_all, 90, axis=1)
        # mfcc_p10 = np.percentile(mfcc_all, 10, axis=1)

        mfcc_mn_keys = ['mfcc_mn_1','mfcc_mn_2','mfcc_mn_3','mfcc_mn_4','mfcc_mn_5','mfcc_mn_6','mfcc_mn_7',
                        'mfcc_mn_8','mfcc_mn_9','mfcc_mn_10','mfcc_mn_11','mfcc_mn_12','mfcc_mn_13',
                        'Dmfcc_mn_1', 'Dmfcc_mn_2', 'Dmfcc_mn_3', 'Dmfcc_mn_4', 'Dmfcc_mn_5', 'Dmfcc_mn_6', 'Dmfcc_mn_7',
                        'Dmfcc_mn_8', 'Dmfcc_mn_9', 'Dmfcc_mn_10', 'Dmfcc_mn_11', 'Dmfcc_mn_12', 'Dmfcc_mn_13',
                        'DDmfcc_mn_1', 'DDmfcc_mn_2', 'DDmfcc_mn_3', 'DDmfcc_mn_4', 'DDmfcc_mn_5', 'DDmfcc_mn_6', 'DDmfcc_mn_7',
                        'DDmfcc_mn_8', 'DDmfcc_mn_9', 'DDmfcc_mn_10', 'DDmfcc_mn_11', 'DDmfcc_mn_12', 'DDmfcc_mn_13']

        mfcc_mn_vals = mfcc_means.tolist()
        mfcc_mn_dict = dict(zip(mfcc_mn_keys, mfcc_mn_vals))

        mfcc_sd_keys = ['mfcc_sd_1','mfcc_sd_2','mfcc_sd_3','mfcc_sd_4','mfcc_sd_5','mfcc_sd_6','mfcc_sd_7',
                        'mfcc_sd_8','mfcc_sd_9','mfcc_sd_10','mfcc_sd_11','mfcc_sd_12','mfcc_sd_13']
        mfcc_sd_vals = mfcc_sds.tolist()
        mfcc_sd_dict = dict(zip(mfcc_sd_keys, mfcc_sd_vals))

        # now merge these dictionaries
        mfcc_dict  = {**mfcc_mn_dict, **mfcc_sd_dict}

        return mfcc_dict

