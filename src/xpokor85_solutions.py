import math

import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import soundfile as sf
import IPython
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk

from scipy.io import wavfile
from scipy.signal import tf2zpk

s, fs = sf.read('xpokor85.wav')
print(min(s), max(s), fs)
# Vzorkovací frekvence je 16000 -> framerate
s = s[:250000]
t = np.arange(s.size) / fs
print(len(s))
print(len(s)/fs)
counter = 0
maxx = abs(max(s))
for i in s:
    s[counter] = i / maxx
    counter += 1

plt.figure(figsize=(6,3))
plt.plot(t, s)

# plt.gca() vraci handle na aktualni Axes objekt,
# ktery nam umozni kontrolovat ruzne vlastnosti aktualniho grafu
# napr. popisy os
# viz https://matplotlib.org/users/pyplot_tutorial.html#working-with-multiple-figures-and-axes
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Zvukový signál')

plt.tight_layout()
plt.show()

IPython.display.display(IPython.display.Audio(s, rate=fs))


def dft(s, fs):
    odkud_vzorky = 512      # začátek segmentu ve vzorcích
    pokud_vzorky = 512 + 1024             # konec segmentu ve vzorcích

    s_seg = s[odkud_vzorky:pokud_vzorky]
    N = s_seg.size

    s_seg_spec = np.fft.fft(s_seg)


    _, ax = plt.subplots(2,1)

    # np.arange(n) vytváří pole 0..n-1 podobně jako obyč Pythonovský range
    ax[0].plot(np.arange(s_seg.size) / fs, s_seg)
    ax[0].set_xlabel('$t[s]$')
    ax[0].set_title('Segment signalu $s$')
    ax[0].grid(alpha=0.5, linestyle='--')

    f = np.arange(s_seg_spec.size) / N * fs
#    zobrazujeme prvni pulku spektra
    ax[1].plot(f[:f.size//2+1], abs(np.real(s_seg_spec[:s_seg_spec.size//2+1])))
    ax[1].set_xlabel('$t[s]$')
    ax[1].set_title('FFT signálu')
    ax[1].grid(alpha=0.5, linestyle='--')
    # ax[1].plot(f[:f.size//2+1], G[:G.size//2+1])
    # ax[1].set_xlabel('$f[Hz]$')
    # ax[1].set_title('Spektralni hustota vykonu [dB]')
    # ax[1].grid(alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()


    N = len(s_seg)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, s_seg)

    f = np.arange(X.size) / N * fs

#    y = y/np.roots(s_seg.size)
    print(fs)
    plt.plot(f[:f.size//2+1], np.abs(np.real(X[:X.size//2+1])))
    plt.gca().set_xlabel('$Hz$')
    plt.gca().set_title('Vlastní DFT signálu')
    plt.grid(alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()

    # y = np.array([], dtype=complex)
    # y.
    ind = []
    val = []
    cnt = 0


    tmp = 10
    for i in X[:X.size//2+1]:
        if np.real(abs(i)) < 15:
            tmp = np.real(abs(i))
        else:
            ind.append(cnt)
            val.append(f[cnt])
            tmp = np.real(abs(i))
        cnt += 1

    print(ind)
    print(val)

    indices = [i for i, v in enumerate(np.abs(X)) if v > 15]

    freqs = [(i / N * fs) for i in indices]

    print (freqs)


def spektrum(s, fs):
    f, t, sgr = spectrogram(s, fs, nperseg=1024, noverlap=512)
    # prevod na PSD
    # (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)
    sgr_log = 10 * np.log10(sgr+1e-20)

    plt.figure(figsize=(9,3))
    plt.pcolormesh(t,f,sgr_log)
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvence [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()




def generate(s, fs):
    freq1 = 781.25
    time = np.arange(s.size) / fs
    signal1 = np.sin(2 * np.pi * freq1 * time)
    print(type(signal1))
    print(signal1)

    freq2 = 1578.125
    signal2 = np.sin(2 * np.pi * freq2 * time)

    freq3 = 2359.375
    signal3 = np.sin(2 * np.pi * freq3 * time)

    freq4 = 3140.625
    signal4 = np.sin(2 * np.pi * freq4 * time)

    signal = signal1 + signal2 + signal3 + signal4

    plt.plot(time, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

    scipy.io.wavfile.write("4cos.wav", fs, signal)
#    filter(signal, fs)

    spektrum(signal, fs)


def filter(s, fs):
    notch_freq = 781.25
    quality_factor = 5
    b_notch, a_notch = scipy.signal.iirnotch(notch_freq, quality_factor, fs)
    null_points(a_notch, b_notch)
    y_pure = s
    scipy.io.wavfile.write("ypure.wav", fs, y_pure)

    y_notched = scipy.signal.filtfilt(b_notch, a_notch, y_pure)


    notch_freq = 1578.125
    b_notch, a_notch = scipy.signal.iirnotch(notch_freq, quality_factor, fs)
    y_notched = scipy.signal.filtfilt(b_notch, a_notch, y_notched)
    null_points(a_notch, b_notch)

    notch_freq = 2359.375
    b_notch, a_notch = scipy.signal.iirnotch(notch_freq, quality_factor, fs)
    y_notched = scipy.signal.filtfilt(b_notch, a_notch, y_notched)
    null_points(a_notch, b_notch)

    notch_freq = 3140.625
    b_notch, a_notch = scipy.signal.iirnotch(notch_freq, quality_factor, fs)
    y_notched = scipy.signal.filtfilt(b_notch, a_notch, y_notched)
    null_points(a_notch, b_notch)

    scipy.io.wavfile.write("clean_bandstop.wav", fs, y_notched)



    spektrum(y_notched, fs)
    null_points(a_notch, b_notch)


def null_points(a, b):
    # [781.25, 1578.125, 2359.375, 3140.625]
 #   a2 = [781.25, 1578.125, 2359.375, 3140.625]
  #  b2 = [-781.25, -1578.125, -2359.375, -3140.625] #TODO dopočítat
    print(a)
    z2, p2, _ = tf2zpk(b, a)
    zplane(z2, p2)
    print(f'Nuly: {z2}')
    print(f'Póly: {p2}')

    _, ax = plt.subplots(1, 2, figsize=(8, 3))

    w, H = freqz(b, a)


    ax[0].plot(w / 2 / np.pi * fs, np.abs(H))
    ax[0].set_xlabel('Frekvence [Hz]')
    ax[0].set_title('Modul frekvenční charakteristiky $|H(e^{j\omega})|$')

    ax[1].plot(w / 2 / np.pi * fs, np.angle(H))
    ax[1].set_xlabel('Frekvence [Hz]')
    ax[1].set_title('Argument frekvenční charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')

    for ax1 in ax:
        ax1.grid(alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()


def impulse(fs):
    N_imp = 32
    imp = [1, *np.zeros(N_imp-1)]
    imp = np.array(imp)

    notch_freq = 781.25
    quality_factor = 5

    b_notch, a_notch = scipy.signal.iirnotch(notch_freq, quality_factor, fs)

    y_notched = scipy.signal.filtfilt(b_notch, a_notch, imp)
    signalplot(y_notched, fs)

    notch_freq = 1578.125
    b_notch, a_notch = scipy.signal.iirnotch(notch_freq, quality_factor, fs)
    y_notched = scipy.signal.filtfilt(b_notch, a_notch, imp)
    signalplot(y_notched, fs)

    notch_freq = 2359.375
    b_notch, a_notch = scipy.signal.iirnotch(notch_freq, quality_factor, fs)
    y_notched = scipy.signal.filtfilt(b_notch, a_notch, imp)
    signalplot(y_notched, fs)

    notch_freq = 3140.625
    b_notch, a_notch = scipy.signal.iirnotch(notch_freq, quality_factor, fs)
    y_notched = scipy.signal.filtfilt(b_notch, a_notch, imp)
    signalplot(y_notched, fs)


def signalplot(signal, fs):
    plt.plot(np.arange(signal.size) / fs, signal)
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Impulse respond')
    plt.grid(alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()

def centralize(s, fs):
    counter = 0
    sum = 0
    for i in s:
        sum += i
        counter += 1

    result = sum / s.size

    print(sum)
    print(abs(sum) / counter)
    print(s)
    counter = 0
    maxm = max(abs(s))
    for i in s:
        s[counter] = (i - result) / maxm
        counter += 1


    print(s)



    s = s[:250000]
    t = np.arange(s.size) / fs

    plt.figure(figsize=(6, 3))
    plt.plot(t, s)

    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Centralizovaný zvukový signál')

    plt.tight_layout()
    plt.show()

    IPython.display.display(IPython.display.Audio(s, rate=fs))

    #
    # i = 0
    # o = 0
    # y = []
    # while i <= s.size:
    #     i += 1024       #Jeden rámec má delku 1024
    #     y = s[o:i]
    #     if i >= s.size:
    #         break
    #
    #
    #
    #     _, ax = plt.subplots(2, 1)
    #
    #     ax[0].plot(np.arange(y.size) / fs, y)
    #     ax[0].set_xlabel('$t[s]$')
    #     ax[0].set_title('Segment signalu od: ' + str(o) + ' do: ' + str(i))
    #     ax[0].grid(alpha=0.5, linestyle='--')
    #     i -= 512  # překrytí 512
    #     o = i    #označuje začátek dělení původního signálu
    #     y_seg = y
    #     N = y_seg.size
    #
    #     y_seg_spec = np.fft.fft(y_seg)
    #     G = 10 * np.log10(1 / N * np.abs(y_seg_spec) ** 2)
    #
    #
    #
    #
    #     # np.arange(n) vytváří pole 0..n-1 podobně jako obyč Pythonovský range
    #     ax[1].plot(np.arange(y_seg.size) / fs + o, abs(np.real(y_seg_spec)))
    #     ax[1].set_xlabel('$t[s]$')
    #     ax[1].set_title('FFT signálu')
    #     ax[1].grid(alpha=0.5, linestyle='--')
    #
    #     plt.tight_layout()
    #     plt.show()


def zplane(b, a, filename=None):
    """Plot the complex z-plane given a transfer function.
    """
    from matplotlib import patches
    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0, 0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b / float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a / float(kd)
    else:
        kd = 1

    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn / float(kd)

    # Plot the zeros and set marker properties
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp(t1, markersize=10.0, markeredgewidth=1.0,
             markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp(t2, markersize=12.0, markeredgewidth=3.0,
             markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5;
    plt.axis('scaled');
    plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1];
    plt.xticks(ticks);
    plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    return z, p, k



centralize(s, fs)
dft(s, fs)
spektrum(s, fs)
generate(s, fs)
filter(s, fs)
impulse(fs)
#null_points()