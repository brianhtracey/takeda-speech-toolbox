
import contextlib
import matplotlib.pyplot as plt
import os.path
import parselmouth as pm
import scipy.io.wavfile
import wave


### loading utilities

def remove_contaminated_regions(sndObj, contam_start_sec, contam_end_sec):
    start = sndObj.get_start_time()
    end = sndObj.get_end_time()

    save_from_beginning = start < contam_start_sec[0]
    save_until_end = contam_end_sec[-1] < end
    if save_from_beginning and save_until_end:
        # times to cut don't include start or end
        save_starts = [start] + contam_end_sec
        save_ends = contam_start_sec + [end]
    elif save_from_beginning and not save_until_end:
        #  interviewer/other is talking at end but not start
        save_starts = [start] + contam_end_sec[:-1]  # drop last
        save_ends = contam_start_sec
    elif not save_from_beginning and save_until_end:
        #  interviewer/other is talking at start but not end
        save_starts = contam_end_sec
        save_ends = contam_start_sec[1:] + [end]  # drop first value in int1
    else:
        # interviewer is talking at the start AND the end
        save_starts = contam_end_sec[:-1]  # drop last
        save_ends = contam_start_sec[1:]  # drop first

    print('  ')
        # basic case: interview not start or stop
    listy = []
    for seg in range(len(save_starts)):
        t1 = save_starts[seg]
        t2 = save_ends[seg]
        if (t2-t1) > 0.5:  # don't save any little chunks shorter than 0.5 sec
        #if t1 != t2:  # this handles case that two segments marked for deletion may abut
            print('start and stop: ' + str(t1) + ' ' + str(t2))
            chunk = sndObj.extract_part(from_time = t1, to_time = t2)
            listy.append(chunk)

    littleSnd = pm.Sound.concatenate(listy)

    return littleSnd


## DEPECATED ??
def remove_prefix(text, prefix):
    # strip prefix off text
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def get_fileparts(fullname):
    path = os.path.dirname(fullname)
    basename = os.path.basename(fullname)
    base = os.path.splitext(basename)[0]
    ext = os.path.splitext(fullname)[1]
    return path, base, ext

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """

    try:  # first try reading using 'wave' moddule
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels <= 2  # no more than stereo
            sample_width = wf.getsampwidth()
            assert sample_width == 2 #int16
            sample_rate = wf.getframerate()
            # this doesn't matter as not using google VAD anymore
            #if not sample_rate in (8000, 16000, 32000, 48000):
            #    print('sample rate is not 8/16/32/49,000')
            data = wf.readframes(wf.getnframes())
    except:
        sample_rate,data = scipy.io.wavfile.read(path)
        num_channels = 1
    return data, sample_rate, num_channels

def plot_waveform_and_segmentation(wv_snd, seg_st, seg_end, sel_long_end, fname, savedir):
        # plot sound and segments
        plt.figure()
        plt.plot(wv_snd.xs(), wv_snd.values.T)
        plt.xlim([wv_snd.xmin - 0.1, wv_snd.xmax + 0.1])
        plt.xlabel("time [s]")
        plt.ylabel("amplitude")
        for k in range(len(seg_st)):
            plt.axvline(seg_st[k], color = 'g', linestyle = ':')
            plt.axvline(seg_end[k], color = 'r', linestyle = ':')
            plt.axvline(sel_long_end[k], color = 'k', linestyle = ':')
        # plt.show()  # need to comment this to run in loop; else, waits until this figure is closed
        plt.savefig("{x}/{y}.png".format(x = savedir, y = fname))
        plt.close()
        return
