''' 
Course: CST 205
Title: Wave File Fast Fourier Transform / Spectrogram
Abstract: This project is a website that allows the user to upload a 
.WAV file and it returns a graph showing the Fast Fourier Transform (FFT) 
of the wave as well as the Short-time Fourier transform (STFT)depicted 
as a spectrogram.
Author: Trent Taylor
Date: 5/15/25
'''

import os
from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap5
import numpy as np
import soundfile as sf
from scipy.fft import fft, fftfreq  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html
import matplotlib.pyplot as plt
import librosa
import librosa.display

app = Flask(__name__)
bootstrap = Bootstrap5(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['audio']
        if uploaded_file.filename.endswith('.wav'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(filepath)

            # Load signal using soundfile and librosa
            signal, sr = sf.read(filepath)  # https://python-soundfile.readthedocs.io/en/0.13.1/
            if signal.ndim > 1:
                signal = signal[:, 0]

            fs = sr  # `sr` is "sample rate" and `fs` is "sample frequency"
            t = np.linspace(0, len(signal) / fs, len(signal), endpoint=False)

            # FFT
            '''
            The Fast Fourier Transform takes a sound as input and decomposes it into its consitutent
            frequencies. It takes you from the time domain to the frequency domain.
            '''
            fft_vals = fft(signal)  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html
            fft_freqs = fftfreq(len(signal), 1 / fs)  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fftfreq.html

            # Spectrogram via STFT
            ''' STFT is a Fourier Transform which looks at the sounds in a wave 
            through a narrow window and applies the Fourier Transform to that window
            of sound, decomposing the sound into its consitutent sin & cos waves.

             '''
            y = signal.astype(np.float32)
            n_fft = 2048
            hop_length = 512
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)  # https://librosa.org/doc/main/generated/librosa.stft.html
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # https://librosa.org/doc/main/generated/librosa.amplitude_to_db.html

            fig, axs = plt.subplots(3, 1, figsize=(14, 12))

            # Time-domain
            axs[0].plot(t, signal)  
            axs[0].set_title("Time-Domain Signal")
            axs[0].set_xlabel("Time [s]")  
            axs[0].set_ylabel("Amplitude") 

            # FFT
            freq_limit = 4096
            mask = fft_freqs[:len(signal) // 2] <= freq_limit
            # limit the x-axis to 5000 hz because of freq limitations of guitar input signal
            axs[1].stem(
                fft_freqs[:len(signal) // 2][mask],
                np.abs(fft_vals)[:len(signal) // 2][mask],
                basefmt=" "
            )
            #axs[1].stem(fft_freqs[:len(signal) // 2], np.abs(fft_vals)[:len(signal) // 2], basefmt=" ")  
            axs[1].set_title("FFT - Frequency Spectrum")
            axs[1].set_xlabel("Frequency [Hz]")
            axs[1].set_ylabel("Magnitude")

            # Spectrogram
            img = librosa.display.specshow(  # https://librosa.org/doc/main/generated/librosa.display.specshow.html
                S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=axs[2]
            )
            axs[2].set_title("Spectrogram (Log Frequency)")
            fig.colorbar(img, ax=axs[2], format='%+2.0f dB')  # https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.colorbar

            plt.tight_layout()  
            output_path = 'static/output.png'
            plt.savefig(output_path)
            plt.close()  

            return render_template('upload.html', image_url='./static/output.png')

    return render_template('upload.html', image_url=None)
