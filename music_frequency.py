# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import wave;
import numpy as np;
import matplotlib.pyplot as plt;
from scipy.fft import *;
from scipy.io import wavfile;
import pandas as pd

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def freq(file, start_time, end_time):

    # Open the file and convert to mono
    framerate, data = wavfile.read(file)
    print('Parameters:', len(data))
    print("framerate: {} len(data) {} start {} -> end {}".format(framerate, len(data), int(start_time * framerate), (int(end_time * framerate) + 1)));
    if data.ndim > 1:
        data = data[:, 0]
    else:
        pass

    # Return a slice of the data from start_time to end_time
    dataToRead = data[int(start_time * framerate): int(end_time * framerate) + 1];
    print("len(dataToRead): ", len(dataToRead));

    # Fourier Transform
    N = len(dataToRead)
    yf = rfft(dataToRead)
    xf = rfftfreq(N, 1 / framerate)

    absy = np.abs(yf);
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)});
    print("absy: ", len(absy), absy);
    print("xf: ", len(xf), xf);

    # Uncomment these to see the frequency spectrum as a plot
    plt.plot(xf, absy)
    plt.show()

    # Get the most dominant frequency and return it
    df = pd.DataFrame()
    #bins = np.array([0, 1, 5, 25, 50, 150, 250, 1000, 5000, 10000])
    #df["bucket"] = pd.cut(df.data, bins)
    df["quantile"] = pd.qcut(xf, q=100)
    print(df["quantile"].value_counts());
    print(df.describe(include='category'));

    idx = np.argmax(absy)
    print("idx: ", idx);
    freq = xf[idx]
    return freq

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wav_obj = wave.open('/Users/abhirajk/Downloads/piano-G3.wav', 'rb');
    print('Parameters:', wav_obj.getparams())
    framerate = wav_obj.getframerate();
    print("framerate: ", framerate);
    n_frames = wav_obj.getnframes();
    print("n_frames: ", n_frames);
    duration = n_frames / framerate;
    print("duration: ", duration);
    #n_channels = wav_obj.getnchannels()
    #print("n_channels: ", n_channels);
    signal_wave = wav_obj.readframes(n_frames);
    signal_array = np.frombuffer(signal_wave, dtype=np.int16);
    times = np.linspace(0, n_frames / framerate, num=n_frames);
    print("len(times): ", len(times), times);

    # plt.figure(figsize=(15, 5))
    # plt.plot(times, signal_array)
    # plt.title('Left Channel')
    # plt.ylabel('Signal Value')
    # plt.xlabel('Time (s)')
    # plt.xlim(0, duration)
    # plt.show();
    #
    # plt.figure(figsize=(15, 5))
    # plt.specgram(signal_array, Fs=framerate, vmin=-20, vmax=50, cmap="rainbow")
    # plt.title('Channel')
    # plt.ylabel('Frequency (Hz)')
    # plt.xlabel('Time (s)')
    # plt.show()

    print(freq('/Users/abhirajk/Downloads/piano-G3.wav', 0, 3));



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
