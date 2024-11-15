import matplotlib.pyplot as plt
import numpy as np
import parselmouth as pm
import scipy.ndimage


class _Segments(object):
    def __init__(self, sound=None, settings=None):
        self._starts = []
        self._ends = []
        self.set_settings(settings)
        if sound is not None:
            self.segment_sound(sound, settings["plot_segments"])

    def _get_phonation_time(self):
        return np.sum(np.array(self._ends) - np.array(self._starts))

    def set_settings(self, settings=None):
        self.settings = settings

    def get_starts(self, offset=0):
        return [x - offset for x in self._starts]

    def get_ends(self, offset=0):
        return [x - offset for x in self._ends]

    def get_segments(self, offset=True):
        if offset and len(self._starts) > 0:
            offset = self._starts[0]
        else:
            offset = 0
        starts = self.get_starts(offset)
        ends = self.get_ends(offset)
        return starts, ends
 
    def segment_sound(self, sound, title_string=None):
        raise(NotImplemented)

    def plot_voicing(self, sound, title_string):
        intensity = sound.to_intensity(time_step = 0.010000)
        intensity = intensity.values.T  # dB for SPL, i.e. RMS amplitude
        voiced = sound.to_pitch(pitch_floor = 75.0, pitch_ceiling = 500.0)
        voiced = voiced.selected_array['frequency']
        voiced = voiced > 0
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(sound._pm_sound.xs(), sound._pm_sound.values.T)
        plt.title(title_string)
        plt.subplot(2, 1, 2)
        plt.plot(voiced)
        plt.plot(intensity / np.max(intensity))
        plt.savefig("{}.png".format(title_string))
        plt.close()


class PhonationSegments(_Segments):
    def __init__(self, sound=None, settings=None):
        self._short_end = []
        self._long_end = []
        super().__init__(sound, settings)

    def get_timing_features(self):
        return {"total_time": self._get_phonation_time()}

    def get_start(self, offset=0):  # Alias
        self.get_starts(offset)

    def get_end(self, offset=0):  # Alias
        self.get_ends(offset)

    def get_segment(self, offset=True):  # Alias
        self.get_segments(offset)

    def segment_sound(self, sound, title_string=None):
        pitch = sound.to_pitch(pitch_floor = 75.0, pitch_ceiling = 500.0)
        intensity = sound.to_intensity(time_step = 0.010000)
        voiced, voiced_raw, dt_voiced = self._find_voiced_regions(pitch, 0.1, 0.3)
        if sum(voiced == True) > 0:
            voiced = self._require_consistent_amplitude(
                    intensity, voiced, stdmult = 4, doplot = False, figname = title_string)
        starts, ends, _ = _find_biggest_voiced_seg(voiced)
        self._starts = [starts * dt_voiced]
        self._short_end = [ends * dt_voiced]
        self.use_short_end()
        self._filter_seg_timesV(pad_at_start = 0.75, len_to_keep = 3)

        # do a plot?
        if not (title_string == None):
            self.plot_voicing(sound, title_string)

    def use_short_end(self):
        self._ends = self._short_end

    def use_long_end(self):
        self._ends = self._long_end

    # No self
    def _find_voiced_regions(self, pitch_ac, closing_width_sec, opening_width_sec):
        # get pitch and use this to find time with voiced / unvoiced
        FMAX = 500.0
        pitch_ac_values = pitch_ac.selected_array['frequency']
        frame_voiced_raw = pitch_ac_values > 0
        frame_dt = pitch_ac.dt

        # fill in small holes
        frame_voiced1 = _fill_small_holes_via_closing(frame_voiced_raw, closing_width_sec, frame_dt)
        # set up structure elements for closing: operation fills in small holes
        # remove small segments via opening
        frame_voiced = _remove_small_objects_via_opening(frame_voiced1, opening_width_sec, frame_dt)

        return frame_voiced, frame_voiced_raw, frame_dt

    def _require_consistent_amplitude(
            self, intns, voiced_in, stdmult=4, doplot=False, figname="none"):
        BIGGEST_BELIEVABLE_SD = 3
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

        return voiced

    def _filter_seg_timesV(self, pad_at_start=0.5, len_to_keep=2.5):
        """
        do some filtering on the segments found to select part for analysis
        rule: find the first segment that is at least (pad_at_start+len_to_keep sec long.
        Discard the first pad_at_start sec, keep the next len_to_keep sec
        if no such segments, then return empty list

        returns sel_start, sel_end, sel_end_longer
        """
        sel_start = []
        sel_end = []
        sel_end_longer = []

        not_found = True
        for iseg in range(len(self._starts)):
            seg_dur = self._ends[iseg] - self._starts[iseg]
            if (not_found and (seg_dur > (pad_at_start + len_to_keep))):
                t_start = self._starts[iseg] + pad_at_start
                sel_start.append(t_start)
                sel_end.append(t_start + len_to_keep)
                sel_end_longer.append(max(
                        t_start + len_to_keep, self._ends[iseg] - pad_at_start))
                not_found = False

        self._starts = sel_start
        self._ends = sel_end  # Short by default
        self._short_end = sel_end
        self._long_end = sel_end_longer


def _fill_small_holes_via_closing(frames_in, width_sec, deltat):
    close_width = np.int(width_sec / deltat)
    if close_width > 2:  # becasue we will discard first, last
        strel_close = np.zeros(close_width)
        strel_close[1:-1] = 1
        frames_out = scipy.ndimage.binary_closing(frames_in, strel_close)
    else:
        frames_out = frames_in

    return frames_out

def _remove_small_objects_via_opening(frames_in, width_sec, deltat):
    # set up structure elements for opening: operation removes small objects
    open_width = np.int(width_sec / deltat)
    if open_width > 2:  # becasue we will discard first, last
        strel_open = np.zeros(open_width)
        strel_open[1:-1] = 1
        frames_out = scipy.ndimage.binary_opening(frames_in, strel_open)
    else:
        frames_out = frames_in

    return frames_out

def _find_biggest_voiced_seg(frame_voiced):
    # find start and end of largest voiced segment
    lab, nlab = scipy.ndimage.label(frame_voiced)
    lenmax = 0
    segmax = 0
    for iseg in range(1, nlab + 1):
        if np.sum(lab == iseg) > lenmax:  # or store array and use np.argmax
            lenmax = np.sum(lab == iseg)
            segmax = iseg

    ibiggest_samps = np.nonzero(lab == segmax)
    ibiggest_samps = np.squeeze(ibiggest_samps)

    return ibiggest_samps[0], ibiggest_samps[-1], ibiggest_samps


class DdkReadingSegments(_Segments):
    def __init__(self, sound=None, settings=None):
        self.syllable_nuclei = []
        self.pauses = []
        super().__init__(sound, settings)

    def get_timing_features(self, check_seg_change):
        timing_features = self._get_timing_features()
        timing_features.update(self._get_pause_features())
        timing_features.update(self._get_syllable_features(timing_features["total_time"], timing_features["phonationtime"], check_seg_change))
        return timing_features

    def _get_timing_features(self):
        pause_total_time = np.sum(np.array(self.pauses))
        phonationtime = self._get_phonation_time()
        total_time = phonationtime + pause_total_time
        pause_frac = pause_total_time / total_time
        return {"total_time": total_time, "phonationtime": phonationtime,
                "pause_total_time": pause_total_time, "pause_frac": pause_frac}

    def _get_pause_features(self):
        pauses = np.array(self.pauses)
        num_pauses = len(pauses)
        pause_len_mean = np.nan
        pause_len_std = np.nan
        pause_len_median = np.nan
        pause_iqr = np.nan

        if num_pauses > 0:
            pause_len_mean = np.mean(pauses)
            pause_len_std = np.std(pauses)
            pause_len_median = np.median(pauses)
            q75, q25 = np.percentile(pauses, [75, 25])
            pause_iqr = q75 - q25
        return {"num_pauses": num_pauses, "pause_len_mean": pause_len_mean,
                "pause_len_std": pause_len_std, "pause_median": pause_len_median,
                "pause_iqr": pause_iqr}

    def _get_syllable_features(self, total_time, phonationtime, check_seg_change=False):
        speech_starts_sec = np.array(self._starts)
        speech_stops_sec = np.array(self._ends)
        syl_nuclei_sec = np.array(self.syllable_nuclei)
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

        # TODO: include mean, std, and make function to calculate
        return {"num_syll": num_syll, "syl_speakingrate": syl_speakingrate,
                "syl_articulationrate": syl_articulationrate,
                "syl_avg_duration": syl_avg_duration,
                "syl_median_spacing": syl_median_spacing,
                "syl_iqr_spacing": syl_iqr_spacing}

        
class FeinbergSegments(DdkReadingSegments):
    def set_settings(self, settings):
        if settings is None:
            settings = {
                    "silence_db": -25, "mindip": 2, "minpause": 0.3,
                    "minpause2": 0.2, "maxpause": 2.0}
        self.settings = settings

    def get_timing_features(self, check_seg_change=True):
        return super().get_timing_features(check_seg_change)

    def segment_sound(self, sound, title_string=None):
        intensity = sound.to_intensity(50)
        threshold, silence_threshold = _find_silence_thresholds(intensity, self.settings["silence_db"])
        textgrid = pm.praat.call(intensity, "To TextGrid (silences)", silence_threshold, self.settings["minpause"], 0.1, "silent", "sounding")
        self._starts, self._ends = _find_speak_starts_and_stops(textgrid)

        sound_from_intensity = _filter_sound_from_intensity(intensity)
        timecorrection = _time_correction_for_converted_sound(sound, sound_from_intensity)
        timepeaks, first_valid = _estimate_peak_times(sound_from_intensity, threshold)
        validtime = _find_valid_peaks(intensity, timepeaks, self.settings["mindip"], first_valid)
        voicedpeak = _find_voiced_times(sound, validtime, textgrid)
        self.syllable_nuclei = np.asarray(voicedpeak) * timecorrection
        self._find_pauses()

        if not title_string is None:
            self.plot_voicing(sound, title_string)

    def _find_pauses(self):
        """
        gets a list of pauses long than min pause len sec
        returns pause_dur list
        """
        pause_dur = []
        for i in range(len(self._starts) - 1):
            this_pause = self._starts[i + 1] - self._ends[i]
            if this_pause > self.settings["minpause"]:
                pause_dur.append(this_pause)
        self.pauses = pause_dur
        self._discard_overlong_pauses()
        self._discard_tooshort_pauses()

    def _discard_overlong_pauses(self):
        self.pauses = list(filter(lambda x: x <= self.settings["maxpause"], self.pauses))

    def _discard_tooshort_pauses(self):
        self.pauses = list(filter(lambda x: x > self.settings["minpause2"], self.pauses))


def _find_silence_thresholds(intensity, silencedb):
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

def _filter_sound_from_intensity(intensity):
    intensity_matrix = pm.praat.call(intensity, "Down to Matrix")
    # sndintid = sound_from_intensity_matrix
    return pm.praat.call(intensity_matrix, "To Sound (slice)", 1)

def _time_correction_for_converted_sound(sound, sound_from_intensity):
    originaldur = sound.get_total_duration()
    # use total duration, not end time, to find out duration of intensity_duration
    # in order to allow nonzero starting times.
    # TODO: check... is this the same as sound_from_intensity.get_total_duration()
    intensity_duration = pm.praat.call(sound_from_intensity, "Get total duration")
    # calculate time correction due to shift in time for Sound object versus intensity object
    return originaldur / intensity_duration

def _estimate_peak_times(sound_from_intensity, threshold):
    point_process = pm.praat.call(sound_from_intensity, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
    # estimate peak positions (all peaks)
    numpeaks = pm.praat.call(point_process, "Get number of points")
    t = [pm.praat.call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]

    timepeaks = []
    first_valid = []
    for time in t:
        # TODO: check... is this the same as Sound.get_value(time)
        value = pm.praat.call(sound_from_intensity, "Get value at time", time, "Cubic")
        if value > threshold:
            first_valid.append(value)
            timepeaks.append(time)
    return timepeaks, first_valid[0]

def _find_valid_peaks(intensity, timepeaks, mindip, first_valid):
    # TODO: first time through the loop, old code uses sound_from_intensity to get this_intensity
    # Fill array with valid peaks: only intensity values if preceding dip in intensity is greater than mindip
    validtime = []
    for p, time in enumerate(timepeaks):
        if p == len(timepeaks) - 1:
            break  # Looking forward one, so don't process the last value
        if p == -1:
            this_intensity = first_valid
        else:
            this_intensity = pm.praat.call(intensity, "Get value at time", time, "Cubic")
        dip = pm.praat.call(intensity, "Get minimum", time, timepeaks[p + 1], "None")
        diffint = abs(this_intensity - dip)
        if diffint > mindip:
            validtime.append(time)
    return validtime

def _find_voiced_times(sound, validtime, textgrid):
    # Look for only voiced parts
    pitch = sound.to_pitch(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
    voicedpeak = []
    for time in validtime:
        whichinterval = pm.praat.call(textgrid, "Get interval at time", 1, time)
        whichlabel = pm.praat.call(textgrid, "Get label of interval", 1, whichinterval)
        value = pitch.get_value_at_time(time)
        if not np.isnan(value):
            if whichlabel == "sounding":
                voicedpeak.append(time)
    return voicedpeak

def _find_speak_starts_and_stops(textgrid):
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

