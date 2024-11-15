import librosa
import numpy as np
import scipy.signal


class Frames(object):
    def __init__(self, sound, segments=None, settings=None):
        self.time = None  # timing vector, t_vec
        self.pitch = None  # pitch values, pitch_values
        self.score = None  # silent, voiced, unvoiced, sil_voic_unvoic
        self.SGram = None
        self.rmsdB = None
        self.sample_rate = sound.sample_rate
        self.settings = None
        self.set_settings(settings)
        self._timealign_pitch_and_spec(sound)
        self.calculate_scores(segments)

    def set_settings(self, settings=None):
        if settings is None:
            settings = {
                    "sgram_frame_sec": 0.02, "pitchmin": 70.0, "pitchmax": 300.0,
                    "calc_voicedVsUn": True, "calc_voicedVsUn_mfcc": False}
        if not settings == self.settings:
            self.settings = settings

    def _timealign_pitch_and_spec(self, sound):
        self._calculate_spectra(sound.get_values())
        pitch, pitch_time = self._calculate_pitch(sound)
        self.pitch = np.interp(self.time, pitch_time, pitch, left = np.nan, right = np.nan)
        if np.nansum(self.pitch) == 0:
            print(" ** no voiced frames found after interp ** ")

    def _calculate_spectra(self, sound_values):
        nsamp = int(self.sample_rate * self.settings["sgram_frame_sec"])
        hop_samp = int(nsamp / 2)  # set pitch frames and spectrogram hoplength to be 2x as fast as spectrogram frames
        self._set_sgram(np.ndarray.flatten(sound_values), nsamp, hop_samp)
        self._set_time(nsamp, hop_samp)
        self._set_rms(nsamp, hop_samp)

    def _set_sgram(self, sound_array, nsamp, hop_samp):
        self.SGram = librosa.stft(y = sound_array, n_fft = nsamp, hop_length = hop_samp)

    def _set_time(self, nsamp, hop_samp):
        self.time = librosa.frames_to_time(
                np.arange(self.SGram.shape[1]), sr = self.sample_rate,
                n_fft = nsamp, hop_length = hop_samp)

    def _set_rms(self, nsamp, hop_samp):
        rms = librosa.feature.rms(S = self.SGram, frame_length = nsamp, hop_length = hop_samp)
        self.rmsdB = np.squeeze(librosa.amplitude_to_db(rms))

    def _calculate_pitch(self, sound):
        dt_pitch = self.settings["sgram_frame_sec"] / 5
        pitch = sound.to_pitch(
                time_step = dt_pitch, pitch_floor = self.settings["pitchmin"],
                pitch_ceiling = self.settings["pitchmax"])
        pitch = pitch.selected_array["frequency"]
        if np.sum(pitch) == 0:
            print("** no voiced frames found before interp **")
        pitch[pitch == 0] = np.nan  # replace unvoiced sapmles by NaN
        pitch_time = np.arange(pitch.size) * dt_pitch  # use relative time
        return pitch, pitch_time

    def get_pitch_stats(self):
        pitch = self._scrub_pitch(octave_threshold = 0.6)
        # get basic stats
        mnpitch_ac = np.nanmean(pitch)
        sdpitch_ac = np.nanstd(pitch)
        if pitch.size > 0:
            pitchjmp_ac = np.nanmax(np.abs(np.diff(pitch)))
            pitchjmp90ptile_ac = np.nanpercentile(np.abs(np.diff(pitch)), 90)
        else:
            pitchjmp_ac = np.nan
            pitchjmp90ptile_ac = np.nan

        #mnpitch_cc = np.nanmean(pitch_cc_values)
        # store these...
        pitch_dict = {
                'mnpitch_ac': mnpitch_ac, 'sdpitch_ac': sdpitch_ac,
                'pitchjmp_ac': pitchjmp_ac, 'pitchjmp90ptile_ac': pitchjmp90ptile_ac}

        # compute values in semitones - referenced to self - and append
        # use Nevler method
        # p10n,p25n,p50n,p75n,p90nev = self.get_semitone_stats_from_pitch(pitch, ref_ptile = 10)
        # note: p90nev = p90-p10 below.
        # use median - more typical?
        p10, p25, p50, p75, p90, sstd = self._get_semitone_stats_from_pitch(pitch, ref_ptile = 50)

        # get PPE, with error handling....
        try:
            ppe = self._get_ppe(pitch)
        except:
            ppe = np.nan  # initialize to nan; fails if pitch are all nan (or too few?)

        semitone_dict = {'semitone_med': p50, 'semitone_iqr': p75 - p25,
                         'semitone_range': p90 - p10,'semitoneSD': sstd, 'PPE': ppe}
        pitch_dict.update(semitone_dict)
        return pitch_dict

    def _scrub_pitch(self, octave_threshold):
        # removes suspicously large pitch fluctuatoins
        md_pitch = np.nanmedian(self.pitch)
        p_hi = md_pitch +  md_pitch * octave_threshold  # go up by less than double
        oct_down = md_pitch / 2
        p_lo = md_pitch - oct_down * octave_threshold  # drop less than an octave
        pitch_values_filt = self.pitch[self.pitch > p_lo]
        pitch_values_filt = pitch_values_filt[pitch_values_filt < p_hi]
        return pitch_values_filt

    def _get_semitone_stats_from_pitch(self, pitch, ref_ptile):
        # converts pitch to semitones
        # see http: // www.sengpielaudio.com / calculator - centsratio.htm
        # 2 possible approaches
        # 1) Nevler 2017 https://n.neurology.org/content/89/7/650 uses 10th ptile as reference, then uses 90th as range
        # 2) Max Little (and https://github.com/Mak-Sim/Troparion) uses median
        pitch_ref = np.percentile(pitch, ref_ptile)
        pitch_st = 12 * np.log2(pitch / pitch_ref)
        p10 = np.percentile(pitch_st, 10)
        p25 = np.percentile(pitch_st, 25)
        p50 = np.percentile(pitch_st, 50)
        p75 = np.percentile(pitch_st, 75)
        p90 = np.percentile(pitch_st, 90)

        return p10, p25, p50, p75, p90, np.nanstd(pitch_st)

    def _get_ppe(self, pitch):
        """Compute pitch period entropy. Here is a reference MATLAB implementation:
        https://github.com/Mak-Sim/Troparion/blob/5126f434b96e0c1a4a41fa99dd9148f3c959cfac/Perturbation_analysis/pitch_period_entropy.m
        Note that computing the PPE relies on the existence of voiced portions in the F0 trajectory.

        (borrowed from Surfboard)

        Args:
            pitch (np.array): f0 voiced frames divided by f_min

        Returns:
            float: The pitch period entropy, as per http://www.maxlittle.net/students/thesis_tsanas.pdf
        """
        # take ratio as f0 divided by 10th percentile
        # as approximation of f0 voiced frames divided by f_min
        rat_f0 = pitch / np.nanpercentile(pitch, 5)
        semitone_f0 = np.log(rat_f0) / np.log(2 ** (1 / 12))

        # Whitening
        semitone_f0 = semitone_f0[~np.isnan(semitone_f0)]  # drop nans first
        coefficients = librosa.core.lpc(semitone_f0, 2)
        semi_f0 = scipy.signal.lfilter(coefficients, [1], semitone_f0)# [0]
        # Filter to the [-1.5, 1.5] range.
        semi_f0 = semi_f0[np.where(semi_f0 > -1.5)]
        semi_f0 = semi_f0[np.where(semi_f0 < 1.5)]

        # shouldn't we always use same histogram bins?
        distrib = np.histogram(semi_f0, bins = 30, density = True)[0]
        # Remove empty bins as these break the entropy calculation.
        distrib = distrib[distrib != 0]

        # Discrete probability distribution
        ppe = np.sum(-distrib * (np.log(distrib) / np.log(2)))
        return ppe

    def calculate_scores(self, segments=None, silence_threshold_dB=25):
        if segments is None:
            self._calculate_scores_without_segments(silence_threshold_dB)
        else:
            self._calculate_scores_with_segments(segments)

    def _calculate_scores_without_segments(self, silence_threshold_dB):
        # estimates silent frames based on RMS intensity; uses pitch to find voiced vs unvoiced frames
        score = np.ones_like(self.rmsdB)
        # set threshold based on max VOICED region to avoid bangs, etc
        if np.sum(np.isfinite(self.pitch)) > 0:
            mxRms = np.max(self.rmsdB[np.isfinite(self.pitch)])
        else:
            mxRms = np.max(self.rmsdB)

        is_silent = self.rmsdB < (mxRms - silence_threshold_dB)
        score[is_silent] = 0
        score[np.isfinite(self.pitch)] = 2
        self.score = score

    def _calculate_scores_with_segments(self, segments):
        # estimates silent frames based on RMS intensity; uses pitch to find voiced vs unvoiced frames
        starts, ends = segments.get_segments()  # Offsets t0 to 0
        # 0) default to silence
        score = np.zeros_like(self.time)
        # 1) set regions in voiced regions to Unvoiced, by default
        dt = self.time[1] - self.time[0]  # this is half-width of librosa window
        for segment_index in range(len(starts)):
            ix = np.where(np.logical_and(
                    self.time >= starts[segment_index],
                    self.time <= ends[segment_index] + dt))
            ix = np.squeeze(ix)
            score[ix] = 1
        # 2) set voiced regions where pitch is finite
        score[np.isfinite(self.pitch)] = 2
        self.score = score

    def score_features(self):
        frac_voiced_of_nonsilent = np.sum(self.score == 2) / (np.sum(self.score > 0))
        frac_voiced = np.sum(self.score == 2) / len(self.score)
        frac_silent = np.sum(self.score == 0) / len(self.score)
        return {
                "frac_voiced": frac_voiced, "frac_silent": frac_silent, 
                "frac_voiced_of_nonsilent": frac_voiced_of_nonsilent}

    def get_amplitude_stats(self):
        # TODO: warn or just sore if self.score is None
        # in self.score, 0 means silent, 1 is unvoiced, 2 is voiced
        rmsNotSilent = self.rmsdB[self.score > 0]
        rmsU= self.rmsdB[self.score == 1]
        rmsV = self.rmsdB[self.score == 2]
        # first, do all frames
        amp_dict = _get_mean_sd_spread(rmsNotSilent, "rmsdB")
        #"intensity_mean": np.nanmean(rmsNotSilent), "intensity_sd": np.nanstd(rmsNotSilent)}
        if self.settings["calc_voicedVsUn"]:
            amp_dict.update(_get_mean_sd_spread(rmsV, "rmsdB_V"))
            #amp_dict.update({"intensity_V_mean": np.nanmean(rmsV), "intensity_V_sd": np.nanstd(rmsV)})
            amp_dict.update(_get_mean_sd_spread(rmsU, "rmsdB_U"))
            #amp_dict.update({"intensity_U_mean": np.nanmean(rmsU), "intensity_U_sd": np.nanstd(rmsU)})
        return amp_dict

    def to_spectral_features(self, settings=None):
        if settings is None:
            settings = self.settings
        return SpectralFeatures(self, settings)


class SpectralFeatures(object):
    def __init__(self, frames=None, settings=None):
        self.mfccs = None
        self.spec_contrast = None
        self.spec_flatness = None
        self.score = None
        self.set_settings(settings)
        if not frames is None:
            self.get_features_from_frames(frames)

    def set_settings(self, settings):
        if settings is None:
            self.settings = {
                    "calc_voicedVsUn": True, "calc_voicedVsUn_mfcc": False}
        else:
            self.settings = settings

    def get_features_from_frames(self, frames):
        Sm = librosa.feature.melspectrogram(  # BT may 26 bumped fmax from 8 to 12k
                S = np.abs(frames.SGram) ** 2, sr = frames.sample_rate,
                n_mels = 40, fmax = 12000)
        mfcc = librosa.feature.mfcc(S = librosa.power_to_db(Sm), n_mfcc = 13)
        sz = np.shape(mfcc)
        if sz[1] > 9: # by default, need
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order = 2)
        else:
            mfcc_delta = np.full([sz[0], sz[1]], np.nan)  # set up matrix of nan's
            mfcc_delta2 = np.full([sz[0], sz[1]], np.nan)  # set up matrix of nan's

        self.mfccs = np.concatenate((mfcc, mfcc_delta), axis = 0)
        self.spec_contrast = librosa.feature.spectral_contrast(S = np.abs(frames.SGram))
        self.spec_flatness = librosa.feature.spectral_flatness(S = np.abs(frames.SGram))
        self.score = frames.score

    def get_spec_contrast_and_flatness(self):
        # flatness is a 1-d array; only compute stats for non-silent periods
        spec_dict = _get_mean_sd_spread(self.spec_flatness[:, self.score > 0], "spec_flat")
        if self.settings["calc_voicedVsUn"]:
            sfU = self.spec_flatness[:, self.score == 1]
            tmp = _get_mean_sd_spread(sfU, "spec_flat")
            tmp = _add_dict_postfix(tmp, "U")
            spec_dict.update(tmp)
            sfV = self.spec_flatness[:, self.score == 2]
            tmp = _get_mean_sd_spread(sfV, "spec_flat")
            tmp = _add_dict_postfix(tmp, "V")
            spec_dict.update(tmp)

        # contrast is computed by octave band
        nb = self.spec_contrast.shape[0] # number of bands

        fmin = 100  # DANGER HARD-CODED!  this is the librosa default value
        octa = np.zeros(nb + 2)  # define bands - follows librosa code
        octa[1:] = fmin * (2.0 ** np.arange(0, nb + 1))

        for k in range(nb):
            labl = "contrst_%d_%d" % (octa[k], octa[k + 1])
            # only compute stats for non-silent periods
            tmp = _get_mean_sd_spread(self.spec_contrast[k, self.score > 0], labl)
            spec_dict.update(tmp)

            if self.settings["calc_voicedVsUn"]:
                scV = self.spec_contrast[k, self.score == 2]
                tmp = _get_mean_sd_spread(scV, labl)
                tmp = _add_dict_postfix(tmp, "V")
                spec_dict.update(tmp)

                scU = self.spec_contrast[k, self.score == 1]
                tmp = _get_mean_sd_spread(scU, labl)
                tmp = _add_dict_postfix(tmp, "U")
                spec_dict.update(tmp)
        return spec_dict

    def mfcc_wrapper(self):
        mfccNotSilent = self.mfccs[:, self.score > 0]
        mfccU = self.mfccs[:, self.score == 1]
        mfccV = self.mfccs[:, self.score == 2]
        # first, do all frames
        mfcc_dict = self._get_mfcc(mfccNotSilent)
        if self.settings["calc_voicedVsUn_mfcc"]:
            tmp = self._get_mfcc(mfccV)
            tmp = _add_dict_postfix(tmp, "V")
            mfcc_dict.update(tmp)
            tmp  = self._get_mfcc(mfccU)
            tmp = _add_dict_postfix(tmp, "U")
            mfcc_dict.update(tmp)
        return mfcc_dict

    def _get_mfcc(self, mfcc_all):
        # get 'mean on empty slice' warning on line below of delta and delta2 not computed... but is ok
        mfcc_means = np.nanmean(mfcc_all, axis = 1)
        mfcc_sds = np.nanstd(mfcc_all, axis = 1)

        # maybe use these later?
        # mfcc_p90 = np.percentile(mfcc_all, 90, axis=1)
        # mfcc_p10 = np.percentile(mfcc_all, 10, axis=1)

        mfcc_mn_keys = [
                'mfcc_mn_1', 'mfcc_mn_2', 'mfcc_mn_3', 'mfcc_mn_4', 'mfcc_mn_5',
                'mfcc_mn_6', 'mfcc_mn_7', 'mfcc_mn_8', 'mfcc_mn_9', 'mfcc_mn_10',
                'mfcc_mn_11', 'mfcc_mn_12', 'mfcc_mn_13', 'Dmfcc_mn_1',
                'Dmfcc_mn_2', 'Dmfcc_mn_3', 'Dmfcc_mn_4', 'Dmfcc_mn_5',
                'Dmfcc_mn_6', 'Dmfcc_mn_7', 'Dmfcc_mn_8', 'Dmfcc_mn_9',
                'Dmfcc_mn_10', 'Dmfcc_mn_11', 'Dmfcc_mn_12', 'Dmfcc_mn_13',
                'DDmfcc_mn_1', 'DDmfcc_mn_2', 'DDmfcc_mn_3', 'DDmfcc_mn_4',
                'DDmfcc_mn_5', 'DDmfcc_mn_6', 'DDmfcc_mn_7', 'DDmfcc_mn_8',
                'DDmfcc_mn_9', 'DDmfcc_mn_10', 'DDmfcc_mn_11', 'DDmfcc_mn_12',
                'DDmfcc_mn_13']

        mfcc_mn_vals = mfcc_means.tolist()
        mfcc_mn_dict = dict(zip(mfcc_mn_keys, mfcc_mn_vals))

        mfcc_sd_keys = [
                'mfcc_sd_1', 'mfcc_sd_2', 'mfcc_sd_3', 'mfcc_sd_4', 'mfcc_sd_5',
                'mfcc_sd_6', 'mfcc_sd_7', 'mfcc_sd_8', 'mfcc_sd_9', 'mfcc_sd_10',
                'mfcc_sd_11', 'mfcc_sd_12', 'mfcc_sd_13']
        mfcc_sd_vals = mfcc_sds.tolist()
        mfcc_sd_dict = dict(zip(mfcc_sd_keys, mfcc_sd_vals))

        # now merge these dictionaries
        mfcc_dict  = {**mfcc_mn_dict, **mfcc_sd_dict}

        return mfcc_dict


def _get_mean_sd_spread(data, label):
    mn = np.nanmean(data)
    sd = np.nanstd(data)
    iqr = np.nanpercentile(data, 75) - np.nanpercentile(data, 25)
    iqr_norm = iqr / np.nanmedian(data)
    return {
            label + "_mean": mn, label + "_sd": sd, label + "_iqr": iqr,
            label + "_iqrnorm": iqr_norm}

def _add_dict_postfix(d, postfix):
    return dict(("{}_{}".format(k, postfix), v) for k, v in d.items())

