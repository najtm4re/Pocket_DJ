import pickle
import librosa
import pandas as pd
import numpy as np
from scipy import stats

class Features:
    
    def __init__(self, x, sr):
        self.features = pd.Series(index=self.columns(), dtype=np.float32)
        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        self.feature_stats('zcr', f)

        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                    n_bins=7*12, tuning=None))

        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        self.feature_stats('chroma_cqt', f)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        self.feature_stats('chroma_cens', f)
        f = librosa.feature.tonnetz(chroma=f)
        self.feature_stats('tonnetz', f)
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        self.feature_stats('chroma_stft', f)
        f = librosa.feature.spectral_centroid(S=stft)
        self.feature_stats('spectral_centroid', f)
        f = librosa.feature.spectral_bandwidth(S=stft)
        self.feature_stats('spectral_bandwidth', f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        self.feature_stats('spectral_contrast', f)
        f = librosa.feature.spectral_rolloff(S=stft)
        self.feature_stats('spectral_rolloff', f)
        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        self.feature_stats('mfcc', f)
        f = librosa.feature.rmse(S=stft)
        self.feature_stats('rmse', f)
        
    def columns(self):
        feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                            tonnetz=6, mfcc=20, rmse=1, zcr=1,
                            spectral_centroid=1, spectral_bandwidth=1,
                            spectral_contrast=7, spectral_rolloff=1)
        moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

        columns = []
        for name, size in feature_sizes.items():
            for moment in moments:
                it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
                columns.extend(it)

        names = ('feature', 'statistics', 'number')
        columns = pd.MultiIndex.from_tuples(columns, names=names)

        # More efficient to slice if indexes are sorted.
        return columns.sort_values()

    def feature_stats(self, name, values):
        self.features[name, 'mean'] = np.mean(values, axis=1)
        self.features[name, 'std'] = np.std(values, axis=1)
        self.features[name, 'skew'] = stats.skew(values, axis=1)
        self.features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        self.features[name, 'median'] = np.median(values, axis=1)
        self.features[name, 'min'] = np.min(values, axis=1)
        self.features[name, 'max'] = np.max(values, axis=1)

    def get_features(self):
        return self.features


x, sr = librosa.load('/home/anya/Downloads/Kurt Vile - Freeway.mp3', sr=None, mono=True)

#create object
song = Features(x, sr)

song.get_features().to_csv('ggg.csv')