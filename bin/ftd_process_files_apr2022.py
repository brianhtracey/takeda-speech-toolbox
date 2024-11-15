
from datetime import date
import glob
import os

import feature_wrappers as wrap


# set up directory for outputs and QC plots
def set_output_directory(paths):
    today = date.today()
    thisdate = today.strftime("%B_%d_%y")
    outpath =  "{}{}/feat_{}".format(paths["basedir"], paths["outdir"], thisdate)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    return outpath

def set_qc_dir(paths):
    outpath = set_output_directory(paths)
    qc_path = outpath + "/plots"
    if not os.path.exists(qc_path):
        os.makedirs(qc_path)
    return qc_path

def get_transcription_directory(paths):
    return paths["basedir"] + paths["transcriptdir"]
 

def process_phonation_files(runpar, paths, doplot=False):
    phonationFiles = glob.glob("{}{}/*AAAH*.wav".format(paths["basedir"], paths["datadir"]))
    res_dir = set_output_directory(paths)
    qc_dir = set_qc_dir(paths)
    df_phon, _, _ = wrap.process_phonation_files(
            phonationFiles, qc_dir, runpar, doplot)
    # for "cut" files, strip _seg off end of file name
    df_phon["file"] = df_phon["file"].str.replace(r'_seg', "")
    # save out data for main (shorter) chunk
    df_phon.to_csv(os.path.join(res_dir, "processed_phonation_files.csv"), index = False)


def process_monologue_files(runpar, paths):
    monolFiles = glob.glob("{}{}/*MONL*.wav".format(paths["basedir"], paths["datadir"]))
    res_dir = set_output_directory(paths)
    transcript_dir = get_transcription_directory(paths)
    df_mono = wrap.process_ddk_or_reading(monolFiles, runpar, transcript_dir)

    if runpar["remove_interviewer_speech"]:
        df_mono.to_csv(os.path.join(res_dir, "processed_monologue_files_no_interviewer.csv"), index = False)
    else:
        df_mono.to_csv(os.path.join(res_dir, "processed_monologue_files.csv"), index = False)

def process_pataka_files(runpar, paths):
    ptakFiles = glob.glob("{}{}/*PTAK*.wav".format(paths["basedir"], paths["datadir"]))
    df_pataka = wrap.process_ddk_or_reading(ptakFiles, runpar)
    df_pataka.to_csv(os.path.join(res_dir, "processed_pataka_files.csv"), index = False)


def process_cookie_files(runpar, paths):
    cookFiles = glob.glob("{}{}/*COOK*.wav".format(paths["basedir"], paths["datadir"]))
    # TODO: check, always set runpar["remove_inverviewer_speech"] to False?
    df_cookie = wrap.process_ddk_or_reading(cookFiles, runpar)
    if runpar["remove_interviewer_speech"]:
        df_cookie.to_csv(os.path.join(res_dir, "processed_cookie_files_no_interviewer.csv"), index = False)
    else:
        df_cookie.to_csv(os.path.join(res_dir, "processed_cookie_files.csv"), index = False)


paths = {
    basedir: "/Users/ieu8424/",
    datadir: "Takeda - MIT-Takeda Program (Extranet) - cleaned_data/denoised_all_renamed_audio_files_with_gender/denoised",  # or 16k
    transcriptdir: "Documents/MIT_voice/Adam_FTD_and_Healthy_data/AWS_transcripts/all_monologue_AWS",
    outdir: "Documents/MIT_voice/Adam_FTD_and_Healthy_data/extracted_features"}

# set pre-processing and other params
runpar = {
        "notch_filter_freq": 50, "normalize_waveform": True,
        "voiced": True , # analyzed voiced vs. unvoiced?
        "do_dfa": False, "remove_interviewer_speech": True,
        "segmentation_method": "praat_feinberg",
        "min_pause_sec": 0.2, "max_pause_sec": 2.0,
        "debug": False}

# process_phonation_files(runpar, paths)
# process_monologue_files(runpar, paths)
# process_pataka_files(runpar, paths)
process_cookie_files(runpar, paths)

