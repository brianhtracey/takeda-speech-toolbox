import glob
import numpy as np
import pandas as pd

import load_wave


# TODO: separate settings file
# Constants (paths to data)
TEST_FILES = "/home/sfabregas/voice/data/test-data/"
TEST_INTERVIEW = "/home/sfabregas/voice/all_interviewer.txt"
MIT_FILES = "/home/sfabregas/voice/data/denoised/"
MIT_INTERVIEW = "/home/sfabregas/voice/data/test-data/all_interviewer.txt"
REDEN_PICT = "/data/TAK-071-2002-Redenlab/pict_data_local/process/processed/"
REDEN_READ = "/data/TAK-071-2002-Redenlab/reading_data_local/process/processed/"


# Settings
file_settings = {
    "wavfile_path": REDEN_PICT}
sound_settings = {
            "pitchmin": 70.0,  # default 70.0
            "pitchmax": 300.0,  # default 300.0
            "sgram_frame_sec": 0.02,  # default 0.02
            "calc_voicedVsUn": True,  # default True
            "calc_voicedVsUn_mfcc": False,  # default False
            "do_plot": False}  # default False
current_settings = {
    "extractor": {
            "full_range_counts": 32768,  # default 32768
            "notch_filter_freq": -1,  # default -1
            "normalize_waveform": True,  # default True
            #"interviewer_file": TEST_INTERVIEW},  # default None
            "interviewer_file": None},
    "sound": sound_settings,
    "phonation": {},  # Default for phonation
    "praat_feinberg": {  # Defaults for praat_feinberg, not used here
            "silence_db": -25,
            "mindip": 2,
            "minpause": 0.2,  # Default is 0.3
            "minpause2": 0.2,
            "maxpause": 2.0},
    "frame": sound_settings,
    "spectra": sound_settings,
    "intervals": 2,
    "do_dfa": False}
debug_settings = {
    "plot": {
            "segments": True,
            "cepstral": False,
            "pulsed": False}}


def process_wav(wavfile, settings):
    # Setup & load data
    print("Loading data...")
    segmenter = "praat_feinberg"
    if "_AAAH" in wavfile:
        segmenter = "phonation"
    settings["segmenter"] = segmenter
    wav_extractor = load_wave.WaveformExtractor(settings["extractor"])
    wav_extractor.load_waveform(wavfile)
    sound = wav_extractor.to_sound(settings = settings["sound"])

    print("Setting file features...")
    # Extract whole sound features before trimming
    est_snr_dB, noise_dB = sound.get_SNR()  # noise_ptile, signal_ptile
    file_features = {  # Use basename of wavfile for file?
            "file": wavfile, "numClipped": sound.num_clipped(),
            "est_snr_dB": est_snr_dB, "noise_dB": noise_dB,
            "interval": "all"}

    print("Extracting segments and trimming sound...")
    # Convert to segments (also before trimming sound)
    segments = sound.to_segments(segmenter = settings["segmenter"], settings = settings)
    # Trim sound to remove silence at start and end
    try:
        sound = sound.trim(segments)
    except ValueError as e:
        # shortcut return if trim causes sound to be emtpy
        return None, settings["segmenter"]

    print("Extracting features...")
    all_features = extract_features(file_features, sound, segments, settings)
    print("Extracting {} invtervals...".format(settings["intervals"]))
    intervals = cut_intervals(all_features.loc[0, "total_time"], settings["intervals"])
    for i in range(len(intervals) - 1):
        print("Extracting interval {} features.".format(i + 1))
        interval = slice(intervals[i], intervals[i + 1])
        cut_sound = sound[interval]
        cut_segments = cut_sound.to_segments(segmenter = settings["segmenter"], settings = settings)
        file_features["interval"] = "[{}, {})".format(intervals[i], intervals[i + 1])
        interval_features = extract_features(file_features, cut_sound, cut_segments, settings)
        all_features = pd.concat([all_features, interval_features])

    return all_features, settings["segmenter"]


def extract_features(file_info, sound, segments, settings):
    # Extract timing features (based on segments before trimming)
    timing_feat = get_timing_features(segments, settings)
    # Extract sound features
    sound_feat = get_sound_features(sound, settings)
    # Score frames
    frames = sound.to_frames(segments, settings = settings["frame"])
    # Extract frame features
    frame_feat = get_frame_features(frames)
    # Calculate spectral features for frames
    spectra = frames.to_spectral_features(settings = settings["spectra"])
    spectra_feat = get_spectral_features(spectra)

    # Collate features
    features = {}
    features.update(file_info)
    features.update(timing_feat)
    features.update(sound_feat)
    features.update(frame_feat)
    features.update(spectra_feat)

    return pd.DataFrame([features])


def get_timing_features(segments, settings):
    # Extract timing features (based on segments before trimming)
    timing_feat = segments.get_timing_features()
    timing_feat["seg_method"] = settings["segmenter"]
    return timing_feat

def get_sound_features(sound, settings):
    sound_feat = {}
    sound_feat["dur"] = sound.get_total_duration()
    sound_feat.update(sound.get_pulsed_stats())
    sound_feat.update(sound.get_harmonicity())
    sound_feat.update(sound.get_cepstral_peak_prominence())
    if settings["do_dfa"]:
        dfa_alpha = sound.get_dfa_alpha()
    else:
        dfa_alpha = np.nan
    sound_feat["dfa_alpha"] = dfa_alpha
    return sound_feat

def get_frame_features(frames):
    frame_feat = {}
    frame_feat.update(frames.get_pitch_stats())
    frame_feat.update(frames.score_features())  # e.g., frac_voiced
    frame_feat.update(frames.get_amplitude_stats())
    return frame_feat

def get_spectral_features(spectra):
    spectra_feat = {}
    spectra_feat.update(spectra.get_spec_contrast_and_flatness())
    spectra_feat.update(spectra.mfcc_wrapper())
    return spectra_feat

def cut_intervals(total_time, number_intervals):
    if number_intervals < 1:
        raise ValueError("Interval must be a positive integer.")
    if number_intervals == 1:
        return [total_time]
    cuts =  [total_time * x / number_intervals for x in range(number_intervals + 1)]
    cuts[0] = 0  # Backstop in case of float issues
    cuts[-1] = total_time  # Backstop in case of float issues
    return cuts


def debug_setup(debug_settings, settings, wavfile):
    if debug_settings["plot"]["segments"]:
        title_string = wavfile.split("/")[-1].split(".")[0]
        settings["plot_segments"] = "{}_segments".format(title_string)
    else:
        setting["plot_segments"] = None
    return settings

phonations = []
praats = []
fails = []

wav_files = glob.glob("/data/TAK-071-2002-Redenlab/reading_data_local/process/processed/*.wav")
wav_files = glob.glob("{}/*.wav".format(file_settings["wavfile_path"]))
for wf in wav_files:
    print(wf)
    current_settings = debug_setup(debug_settings, current_settings, wf)
    try:
        output, method = process_wav(wf, current_settings)
    except Exception as e:
        output = wf
        method = "fail"
    if method == "praat_feinberg":
        praats.append(output)
    elif method == "phonation":
        phonations.append(output)
    else:
        fails.append(output)

if len(phonations) > 0:
    phonations = pd.concat(phonations)
    phonations.to_csv("tak071-2002_reading-processed-phonations_2023-07-10.csv", index = False)
else:
    print("No phonation data")

if len(praats) > 0:
    praats = pd.concat(praats)
    praats.to_csv("tak071-2002_reading-processed-utterances_2023-07-10.csv", index = False)
else:
    print("No praat-feinberg data")

print(fails)

