import contextlib
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import re
import scipy.io.wavfile
import scipy.signal
import wave

import sound


class WaveformExtractor(object):
    """
    extractor = WaveformExtractor(settings)
    extractor.load_waveform(file_name)
    extractor.set_settings(new_settings)
    extractor.to_sound()  # If interviewer file provided, will cut out times
    """
    def __init__(self, settings=None):
        self.file_name = None
        self.data = None
        self.sample_rate = None
        self.num_channels = None
        self.num_clipped = 0
        self.settings = None
        self.set_settings(settings)

    def set_settings(self, settings_dict):
        if settings_dict is None:
            settings_dict = {
                    "full_range_counts": 32768, "notch_filter_freq": -1,
                    "normalize_waveform": True, "interviewer_file": None}
        if not settings_dict == self.settings:
            self.settings = settings_dict
            if not self.file_name is None:
                self.load_waveform(self.file_name)

    def load_waveform(self, file_name):
        self.file_name = file_name
        self._extract_waveform()
        self._filter_waveform()
        self._normalize_waveform()

    def to_sound(self, settings=None):
        if self.data is None or self.sample_rate is None:
            print("Cannot convert to Sound. Data not yet loaded?")
            return
        this_sound = sound.Sound(self.data, self.sample_rate, settings = settings)
        if self.settings["interviewer_file"] is not None:
            interview_starts, interview_stops = self.get_interviewer_times()
            this_sound.remove_regions(interview_starts, interview_stops)
        return this_sound

    def get_interviewer_times(self):
        if self.file_name is None or self.settings["interviewer_file"] is None:
            print("Need to load a wave file and interviewer data to extract times.")
            return [], []
        wavbase = os.path.basename(self.file_name)
        wavbase = re.sub("\_enhanced", "", wavbase)
        wavbase = re.sub("\_raw", "", wavbase)
        wavbase = "".join(wavbase.split(".")[0:-1]) + ".txt"
        time_table = pd.read_csv(self.settings["interviewer_file"], header = None)
        time_table.columns = ["files", "t_start", "t_stop"]
        time_table = time_table[time_table["files"] == wavbase]
        if len(time_table.index) > 0:
            return time_table["t_start"].tolist(), time_table["t_stop"].tolist()
        return [], []

    def _extract_waveform(self):
        """Reads a .wav file.
        Takes the file name, and returns (PCM audio data, sample rate).
        """
        try:  # first try reading using 'wave' moddule
            self.__extract_direct()
        except:
            self.__extract_with_scipy()
        if isinstance(self.data, bytes):
            self._convert_waveform_from_bytes()
        elif not isinstance(self.data, np.ndarray):
            print("Unexpected waveform data format")
        if self.data is None or len(self.data) == 0:
            assert 2 == 1  # TODO: fix hack
            print("Waveform data failed to load")

    def __extract_direct(self):
        with contextlib.closing(wave.open(self.file_name, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels <= 2  # no more than stereo
            sample_width = wf.getsampwidth()
            assert sample_width == 2 #int16
            sample_rate = wf.getframerate()
            data = wf.readframes(wf.getnframes())
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels

    def __extract_with_scipy(self):
        sample_rate, data = scipy.io.wavfile.read(self.file_name)
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = 1  # Only extracts first channel

    def _convert_waveform_from_bytes(self):
        y = np.fromstring(self.data, dtype = np.int16)
        # if stereo, just grab channel 1
        y = y[::self.num_channels]
        # TODO consider allowing an input that would choose which channel
        self.num_clipped = np.sum(np.abs(y) > (self.settings["full_range_counts"] - 2))
        # De-mean the waveform to eliminate DC offset
        y = y - np.mean(y)
        self.data = y / self.settings["full_range_counts"]  # scale so stays in range -1 to 1

    def _filter_waveform(self):
        BANDWIDTH = 2.0
        if self.settings["notch_filter_freq"] <= 0:
            print("Notch filter is off. Data unfiltered")
            return

        print("Filtering waveform at {}, bandwidth {}.".format(
                self.settings["notch_filter_freq"], BANDWIDTH))
        # power line notch filter
        quality = notch_freq / notch_bw  # quality of 30 gives about 25 dB notch at 60 Hz
        b, a = scipy.signal.iirnotch(notch_freq, quality, self.sample_rate)
        yfilt = scipy.signal.lfilter(b, a, self.data)

        # remove any DC offset
        self.data = yfilt - np.mean(yfilt)

        debugit = False
        if debugit:
            freq, h = scipy.signal.freqz(b, a, fs = self.sample_rate, worN = 200000)
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

    def _normalize_waveform(self):
        if self.settings["normalize_waveform"]:
            self.data = (self.data - np.mean(self.data, axis = 0)) # / np.std(ynp, axis = 0)

