import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


def generate_mel_spectrogram(audio_path, output_path):
    """
    Convert a WAV file into a Mel Spectrogram image.
    """

    # Load audio
    audio, sample_rate = librosa.load(audio_path, sr=16000)

    # Generate Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Save image
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        x_axis=None,
        y_axis=None
    )

    plt.axis("off")
    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()
