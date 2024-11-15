import matplotlib.pyplot as plt
import numpy as np
import parselmouth as pm

import dfa
import frames
import segments


class Sound(object):
    def __init__(self, waveform=None, sample_rate=None, num_clipped=0, settings=None):
        self._num_clipped = num_clipped
        self.sample_rate = sample_rate
        self.settings = None
        self.set_settings(settings)
        self._pm_sound = None
        self._load_sound(waveform)

    def set_settings(self, settings):
        if settings is None:
            self.settings = {
                    "pitchmin": 70.0, "pitchmax": 300.0,
                    "sgram_frame_sec": 0.02, "calc_voicedVsUn": True,
                    "calc_voicedVsUn_mfcc": False}
        else:
            self.settings = settings

    def _load_sound(self, waveform):
        if waveform is not None:
            self._pm_sound = pm.Sound(waveform, self.sample_rate)

    def num_clipped(self):
        return self._num_clipped

    def remove_regions(self, region_starts, region_ends):
        start = self._pm_sound.get_start_time()
        end = self._pm_sound.get_end_time()

        if len(region_starts) == 0 or len(region_ends) == 0:
            return
        save_from_beginning = start < region_starts[0]
        save_until_end = region_ends[-1] < end
        if save_from_beginning and save_until_end:
            # times to cut don't include start or end
            save_starts = [start] + region_ends
            save_ends = region_starts + [end]
        elif save_from_beginning and not save_until_end:
            #  interviewer/other is talking at end but not start
            save_starts = [start] + region_ends[:-1]  # drop last
            save_ends = region_starts
        elif not save_from_beginning and save_until_end:
            #  interviewer/other is talking at start but not end
            save_starts = region_ends
            save_ends = region_starts[1:] + [end]  # drop first value in int1
        else:
            # interviewer is talking at the start AND the end
            save_starts = region_ends[:-1]  # drop last
            save_ends = region_starts[1:]  # drop first

        print('  ')
        # basic case: interview not start or stop
        listy = []
        for seg in range(len(save_starts)):
            t1 = save_starts[seg]
            t2 = save_ends[seg]
            if (t2 - t1) > 0.5:  # don't save any little chunks shorter than 0.5 sec
                print('start and stop: ' + str(t1) + ' ' + str(t2))
                chunk = self._pm_sound.extract_part(from_time = t1, to_time = t2)
                listy.append(chunk)
        if len(listy):
            self._pm_sound = pm.Sound.concatenate(listy)
            return True
        return False

    def trim(self, segments):
        starts = segments.get_starts()
        ends = segments.get_ends()
        if len(starts) < 1 or len(ends) < 1:
            raise ValueError("")
        return self[slice(starts[0], ends[-1])]

    def get_SNR(self, noise_ptile=3, signal_ptile=95):
        # get stats to compute SNR.  Should be computed for entire recording
        intns = self.to_intensity()
        intensity_values_dB = intns.values.T  # dB for SPL, i.e. RMS amplitude
        # multiply by 2 below to go to power
        noise_pow_dB = 2 * np.percentile(intensity_values_dB, noise_ptile)
        sig_pow_dB = 2 * np.percentile(intensity_values_dB, signal_ptile)
        est_snr_power_dB = sig_pow_dB - noise_pow_dB
        return est_snr_power_dB, noise_pow_dB

    def get_total_duration(self):
        return self._pm_sound.get_total_duration()

    def get_pulsed_stats(self):
        # get jitter and shimmer: thanks to Yannick
        pulses = pm.praat.call(self._pm_sound, "To PointProcess (periodic, cc)", self.settings["pitchmin"], self.settings["pitchmax"])
        pulse_dict = self._get_jitter(pulses)
        pulse_dict.update(self._get_shimmer(pulses))
        # overall voice report, as a string...
        pitch_ac = self.to_pitch(
                pitch_floor = self.settings["pitchmin"],
                pitch_ceiling = self.settings["pitchmax"])
        vrep = pm.praat.call([pulses, self._pm_sound, pitch_ac], "Voice report", 0.0, 0.0, 75, 600, 1.3, 1.6, 0.03, 0.45)
        num_voice_breaks = self._parse_voice_report(vrep, 'Number of voice breaks:')
        pulse_dict.update({'num_voice_breaks': num_voice_breaks})
        if (self.settings["do_plot"]):
            times = pitch_ac.xs(),
            pitch_ac_values = pitch_ac.selected_array['frequency']
            plt.plot(times, pitch_ac_values, 'o', markersize=5, color='b')
            plt.plot(times, pitch_ac_values, 'o', markersize=2)
            plt.grid(False)
            plt.ylim(0, pitch_ac.ceiling)
            plt.ylabel("fundamental frequency [Hz]")
        return pulse_dict

    # No self
    def _get_jitter(self, pulses):
        # Since "To PointProcess (cc)" is currently not available in the Parselmouth Python API,
        # you need to revert to calling the Praat action like in a Praat script, so with the string "To PointProcess (cc)"
        # for args below, see http: // www.fon.hum.uva.nl / praat / manual / PointProcess__Get_jitter__local____.html
        # the 5 numbers input to jitter are: start and stop of time range (0 means take all),
        # shortest possible period, longest possible period, max period_factor (ratio between consecutive periods)
        jitter_loc = pm.praat.call(pulses, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        jitter_locabs = pm.praat.call(pulses, "Get jitter (local, absolute)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        jitter_rap = pm.praat.call(pulses, "Get jitter (rap)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        jitter_ppq5 = pm.praat.call(pulses, "Get jitter (ppq5)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        jitter_ddp = pm.praat.call(pulses, "Get jitter (ddp)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        return {'jit_local': jitter_loc, 'jit_localabs': jitter_locabs,
                'jit_rap': jitter_rap, 'jit_ppq5': jitter_ppq5,
                'jit_ddp': jitter_ddp}

    def _get_shimmer(self, pulses):
        # see https://osf.io/umrjq/ for project using parselmouth
        # In praat: select a sound object and a pulses object, then can select jitter.  This shows 6 parameters below:
        # first 5 parameters are same as jitter; last is 'max amplitude factor'
        shimmer_loc = pm.praat.call([self._pm_sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_loc_db = pm.praat.call([self._pm_sound, pulses], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = pm.praat.call([self._pm_sound, pulses], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_dda = pm.praat.call([self._pm_sound, pulses], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        return {'sh_local': shimmer_loc, 'sh_localdB': shimmer_loc_db,
                'sh_apq5': shimmer_apq5, 'sh_dda': shimmer_dda}

    # No self
    def _parse_voice_report(self, vreport, target_string):
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

    def get_harmonicity(self):
        # calculate HNR
        time_step = self.settings["sgram_frame_sec"] / 2
        harmonicity = self._pm_sound.to_harmonicity(time_step = time_step)
        mn_harm = harmonicity.values[harmonicity.values != -200].mean()
        sd_harm = harmonicity.values[harmonicity.values != -200].std()
        harm_dict = {'harmonicity_mean': mn_harm, 'harmonicity_sd': sd_harm}
        return harm_dict

    def get_cepstral_peak_prominence(self):
        # calculate CPP - see https://gitter.im/PraatParselmouth/Lobby?at=5ea5c81e61a0002f794e4622
        # for parameters, see https://www.fon.hum.uva.nl/praat/manual/Sound__To_PowerCepstrogram___.html
        power_cepstrogram = pm.praat.call(self._pm_sound, "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50)
        slice_step_sec = 0.1
        nslice = np.int32(self.get_total_duration() / slice_step_sec)

        # cepstrum itself: unused but maybe interesting at some point?
        if self.settings["do_plot"]:
            cepstrum_values = pm.praat.call(power_cepstrogram, "To Matrix").values
            plt.figure()
            plt.imshow(np.log10(np.abs(cepstrum_values[1:200,:])),aspect="auto")
            plt.show()

        cpp_vec = []
        for sl in range(nslice):
            t_slice = (sl + 1) * slice_step_sec
            cepstrogram_slice = pm.praat.call(power_cepstrogram, "To PowerCepstrum (slice)", t_slice)
            cpp_slice = pm.praat.call(cepstrogram_slice, "Get peak prominence", 60.0, 330.0, 'Parabolic', 0.001, 0.0, 'Straight', 'Robust')
            cpp_vec.append(cpp_slice)

        # for parameters, see https: // www.fon.hum.uva.nl / praat / manual / PowerCepstrum__Get_peak_prominence___.html
        cpps_entire_waveform = pm.praat.call(power_cepstrogram, "Get CPPS", False, 0.01, 0.01, 60.0, 330.0, 0.05, 'Parabolic', 0.001, 0.0, 'Straight', 'Robust')
        # interesting: see http://www.homepages.ucl.ac.uk/~uclyyix/ProsodyPro/
        # appears we want to chop
        # save as dictionary for compatibility with other analyses
        return {'cpp_mean': np.nanmean(cpp_vec), 'cpp_sd': np.nanstd(cpp_vec),
                'cpp_med': np.nanmedian(cpp_vec),
                'cpp_iqr':np.nanpercentile(cpp_vec, 75) - np.nanpercentile(cpp_vec, 25),
                'cpps_entirewv': cpps_entire_waveform}

    def get_dfa_alpha(self):
        y = self.get_values()
        y = np.ndarray.flatten(y)  # flatten it to a true 1-d array
        scales, fluct, alpha = dfa.dfa(y)
        return alpha

        # PITCH-RELATED ##############
        # TODO consider vibratio measure:
        #  see https://github.com/Mak-Sim/Troparion/tree/5126f434b96e0c1a4a41fa99dd9148f3c959cfac/Perturbation_analysis

    def get_values(self):
        return self._pm_sound.values

    def to_pitch(self, *args, **kwargs):
        return self._pm_sound.to_pitch_ac(*args, **kwargs)

    def to_intensity(self, *args, **kwargs):
        return self._pm_sound.to_intensity(*args, **kwargs)

    def praat_feinberg_segments(self, settings):
        return segments.FeinbergSegments(self, settings)

    def phonation_segments(self, settings):
        return segments.PhonationSegments(self, settings)

    def to_segments(self, segmenter, settings):
        use_segmenter = self._get_segmenter(segmenter)
        settings[segmenter]["plot_segments"] = settings["plot_segments"]
        return use_segmenter(self, settings[segmenter])

    def _get_segmenter(self, format):
        if format.lower() == "praat_feinberg":
            return segments.FeinbergSegments
        elif format.lower() == "phonation":
            return segments.PhonationSegments
        else:
            raise ValueError(format)

    def to_frames(self, segments=None, settings=None):
        if settings is None:
            settings = self.settings
        return frames.Frames(self, segments, settings)

    def __getitem__(self, key):
        # Only accept integer start and stop >= 0 for slice
        if key.start < 0 or key.stop < 0 or (key.step is not None and key.step != 1) or key.stop <= key.start:
            raise ValueError("Slice must be positive, stop must be greater than start, and step must be None or 1.")
        sound = self._pm_sound.extract_part(from_time = key.start, to_time = key.stop)
        new_sound = Sound()
        new_sound._num_clipped = self._num_clipped
        new_sound.sample_rate = self.sample_rate
        new_sound._pm_sound = sound
        new_sound.settings = self.settings
        return new_sound

