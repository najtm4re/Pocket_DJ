#!/usr/bin/env python3

# FMA: A Dataset For Music Analysis
# Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson, EPFL LTS2.

# All features are extracted using [librosa](https://github.com/librosa/librosa).
# Alternatives:
# * [Essentia](http://essentia.upf.edu) (C++ with Python bindings)
# * [MARSYAS](https://github.com/marsyas/marsyas) (C++ with Python bindings)
# * [RP extract](http://www.ifs.tuwien.ac.at/mir/downloads.html) (Matlab, Java, Python)
# * [jMIR jAudio](http://jmir.sourceforge.net) (Java)
# * [MIRtoolbox](https://www.jyu.fi/hum/laitokset/musiikki/en/research/coe/materials/mirtoolbox) (Matlab)

import os
from pathlib import Path

import multiprocessing
import warnings
import numpy as np
from scipy import stats
import pandas as pd
import librosa
from tqdm import tqdm
from src.data import data_utils
from src.config import AUDIO_DIR, META_DIR

warnings.simplefilter(action="ignore", category=FutureWarning)

dir = os.getcwd()
# go one level higher and add it to path
parent_dir = str(Path(dir).parents[0])


def get_columns():
    feature_sizes = dict(
        chroma_stft=12,
        chroma_cqt=12,
        chroma_cens=12,
        tonnetz=6,
        mfcc=20,
        rmse=1,
        zcr=1,
        spectral_centroid=1,
        spectral_bandwidth=1,
        spectral_contrast=7,
        spectral_rolloff=1,
    )
    moments = ("mean", "std", "skew", "kurtosis", "median", "min", "max")

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, "{:02d}".format(i + 1)) for i in range(size))
            columns.extend(it)

    names = ("feature", "statistics", "number")
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()


def compute_features(track_id):
    features = pd.Series(
        index=get_columns(),
        dtype=np.float32,
        name=track_id.split("\\")[-1].split(".")[0],
    )

    # Catch warnings as exceptions (audioread leaks file descriptors).
    warnings.filterwarnings("error", module="librosa")

    def feature_stats(name, values):
        features[name, "mean"] = np.mean(values, axis=1)
        features[name, "std"] = np.std(values, axis=1)
        features[name, "skew"] = stats.skew(values, axis=1)
        features[name, "kurtosis"] = stats.kurtosis(values, axis=1)
        features[name, "median"] = np.median(values, axis=1)
        features[name, "min"] = np.min(values, axis=1)
        features[name, "max"] = np.max(values, axis=1)

    try:
        filepath = os.path.join(AUDIO_DIR, track_id)
        x, sr = librosa.load(filepath, sr=None, mono=True)  # kaiser_fast

        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        feature_stats("zcr", f)

        cqt = np.abs(
            librosa.cqt(
                x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12, tuning=None
            )
        )
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x) / 512) <= cqt.shape[1] <= np.ceil(len(x) / 512) + 1

        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats("chroma_cqt", f)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats("chroma_cens", f)
        f = librosa.feature.tonnetz(chroma=f)
        feature_stats("tonnetz", f)

        del cqt
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
        del x

        f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        feature_stats("chroma_stft", f)

        f = librosa.feature.rms(S=stft)
        feature_stats("rmse", f)

        f = librosa.feature.spectral_centroid(S=stft)
        feature_stats("spectral_centroid", f)
        f = librosa.feature.spectral_bandwidth(S=stft)
        feature_stats("spectral_bandwidth", f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feature_stats("spectral_contrast", f)
        f = librosa.feature.spectral_rolloff(S=stft)
        feature_stats("spectral_rolloff", f)

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats("mfcc", f)

    except Exception as e:
        print("{}: {}".format(track_id, repr(e)))

    return features


def get_features(size):
    print("Generating features")
    tracks = data_utils.load(META_DIR + "/tracks.csv")
    tracks = tracks[tracks["set", "subset"] <= size]
    features = pd.DataFrame(index=tracks.index, columns=get_columns(), dtype=np.float32)

    all_files = []
    for root, _, files in os.walk(AUDIO_DIR):
        for file in files:
            if "mp3" not in file:
                continue
            root_end = root.split("\\")[-1]
            all_files.append(os.path.join(root_end, file))
    for file in tqdm(all_files):
        track_features = compute_features(file)
        features.loc[int(track_features.name)] = track_features

    save(features, 10)


def save(features, ndigits):
    # Should be done already, just to be sure.
    # features = features.sort_index(axis=0)
    # features = features.sort_index(axis=1)
    features.to_csv(
        "../data/interim/features.csv", float_format="%.{}e".format(ndigits)
    )
