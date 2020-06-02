import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot,iplot
from scipy.stats import norm, kurtosis
import os
from scipy.signal import butter, lfilter, freqz
from scipy import signal
from sklearn.model_selection import train_test_split
from collections import Counter

import warnings
warnings.filterwarnings(action='once')
plt.rcParams["figure.figsize"] = 16,12


def create_labels():

    labels = pd.read_csv('../data/RawData/labels.txt', sep=" ", header=None)
    labels.columns = ['experiment','person','activity','start','end']

    return labels

def read_data():
    """Read all data to a dataframe"""

    list_df = [] #a list to collect the dataframes

    for i in range(1,62):

        if i < 10:
            i = '0' + str(i)
        else:
            i = str(i)

        for j in os.listdir('../data/RawData/'):

            if "acc_exp" + i in j:
                acc_path = "../data/RawData/" + j

            elif "gyro_exp" + i in j:
                gyro_path = "../data/RawData/" + j

        acc_df = pd.read_csv(acc_path, sep = " ", names=['acc_x','acc_y','acc_z'])
        gyro_df = pd.read_csv(gyro_path, sep = " ", names=['gyro_x','gyro_y','gyro_z'])

        exp_df = pd.concat([acc_df,gyro_df],1)
        exp_df["experiment"] = int(i) #keep track of the experiment

        list_df.append(exp_df)

    df = pd.concat(list_df)

    return df


def add_activity_label(df, labels):
    """Add activity labels form the labels dataframe"""

    df = df.reset_index()
    df = df.rename(columns={"index": "id"})

    df["activity"] = 0

    for index,row in labels.iterrows():

        df["activity"] = np.where((df.experiment == row["experiment"]) \
                                    & ((df.id >= row["start"]) & (df.id < row["end"])),
                                    row["activity"], df["activity"])

    return df

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def filter_acc(df, cutoff=10, fs=50, order=2):

    signals = ["acc_x","acc_y","acc_z"]

    new_df = pd.DataFrame(columns=signals)

    for experiment in df.experiment.unique():

        experiment_df = df[df.experiment == experiment]

        list_signals = []

        for j in signals:

            filtered_signal = butter_lowpass_filter(experiment_df[j], cutoff=cutoff, fs=fs, order=order)

            list_signals.append(filtered_signal)

        new_df = pd.concat([new_df, pd.DataFrame(np.array(list_signals).T,columns=signals)])

    return new_df


def filter_gyro(df, cutoff=10, fs=50, order=2):

    signals = ["gyro_x","gyro_y","gyro_z"]

    new_df = pd.DataFrame(columns=signals)

    for experiment in df.experiment.unique():

        experiment_df = df[df.experiment == experiment]

        list_signals = []

        for j in signals:

            filtered_signal = butter_lowpass_filter(experiment_df[j], cutoff=cutoff, fs=fs, order=order)

            list_signals.append(filtered_signal)

        new_df = pd.concat([new_df, pd.DataFrame(np.array(list_signals).T,columns=signals)])

    return new_df



def remake_df(filtered_df_acc, filtered_df_gyro, labeled_df):

    df =  pd.concat([labeled_df.drop(["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"],axis=1),
                           filtered_df_acc.reset_index(drop=True), filtered_df_gyro.reset_index(drop=True)],
                           axis=1)
    return df


def drop_unlabeled(labeled_df):

    """Drop unlabeld data from the original dataframe

    args:
        labeled_df: data frame with label features

    return:
        labeled_df
    """

    labeled_df["activity"] = np.where(labeled_df["activity"] == 0,np.NaN,labeled_df["activity"])
    labeled_df.dropna(inplace=True)

    labeled_df.reset_index(drop=True,inplace=True)

    return labeled_df


def renindex_df(labeled_df):
    """Reindex the dataframe to do the overlap"""

    labeled_df.reset_index(inplace=True)
    #labeled_df = labeled_df.sort_values(["activity","index"]).reset_index(drop=True)

    return labeled_df


def create_block_df(df, window_size, overlap):

    """Create a new df where each block get an id. The blocks are
    part of the original dataframe but each block has an overlap with the previous one."""

    k = 0
    overlap = 1 - overlap

    df["block"] = None
    df1 = pd.DataFrame()

    for activity in df.activity.unique():
        i = 0
        j = window_size

        activity_df = df[df.activity == activity]

        for _ in range(int(int(len(activity_df)/window_size)/overlap)):

            df2 = activity_df.iloc[int(i):int(j)].copy()
            df2["block"] = k

            new_df = pd.concat([df1,df2])

            df1 = new_df

            i += window_size*overlap
            j += window_size*overlap
            k += 1

    new_df.reset_index(drop=True,inplace=True)
    return new_df

def create_block_df_no_overlap(df,window_size):

    df = df.sort_values(["activity","index"]).reset_index(drop=True)

    df['flag'] = np.where(df['id'] % window_size == 0,1,0)
    df["block"] = df['flag'].cumsum()

    df.reset_index(drop=True,inplace=True)

    return df

def most_common(x):

    c = Counter(x)
    return c.most_common(1)[0][0]

def create_activity_labels(block_df):

    activity_labels = block_df.groupby("block").agg({'activity' : most_common})["activity"]

    return activity_labels.to_numpy()

def kurtosis_time(x):

    return kurtosis(x, fisher=True)

def rms_100(x):

    return np.sqrt(np.mean(x**2))

def crest(x):

    return max(abs(x))/np.sqrt(np.mean(x**2))

def create_aggregated(block_df):

    signals = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

    agg_df = block_df.groupby("block").agg({x: ["sum", "mean", "mad",
                                                "median", "min", "max",
                                                "std", "var", "sem",
                                                "skew", "quantile",
                                                kurtosis_time, rms_100,
                                                crest] for x in signals})

    return agg_df


def do_fft(df,nperseg):

    "Creat a new df with the frequency spectrum of each blocks"

    signals = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]
    #df.columns = ['index', 'id', 'experiment', 'activity'] + signals + ["block"]

    new_df = pd.DataFrame()

    for block in df.block.unique():

        fft_df = df[df["block"] == block]

        list_signals = []

        for j in signals:

            freq, amp = signal.welch(fft_df[j], 50, nperseg=nperseg)

            list_signals.append(amp)

        list_signals.append(np.full(len(freq), block))

        new_df = pd.concat([new_df, pd.DataFrame(np.array(list_signals).T,columns=[x+"_FFT" for x in signals]+["block"])])
        new_df.dropna(axis=1,inplace=True)


    #new_df["freq"] = np.tile(x,len(df.block.unique()))
    new_df["block"] = new_df["block"].astype('int32')

    return new_df



def peak_sum_all(x):

    peaks, _ = signal.find_peaks(x, height=0,)

    return sum(peaks)

def peak_mean_12(x):

    peaks, hight = signal.find_peaks(x, height=0,)
    hight["peak_heights"][::-1].sort()

    if len( hight["peak_heights"])>=12:

        return np.mean(hight["peak_heights"][:12])

def peak_mean_8(x):

    peaks, hight = signal.find_peaks(x, height=0,)
    hight["peak_heights"][::-1].sort()

    if len( hight["peak_heights"])>=8:

        return np.mean(hight["peak_heights"][:8])

def peak_mean_6(x):

    peaks, hight = signal.find_peaks(x, height=0,)
    hight["peak_heights"][::-1].sort()

    if len( hight["peak_heights"])>=6:

        return np.mean(hight["peak_heights"][:6])

def peak_mean_2(x):

    peaks, hight = signal.find_peaks(x, height=0,)
    hight["peak_heights"][::-1].sort()

    if len( hight["peak_heights"])>=2:

        return np.mean(hight["peak_heights"][:2])

def kurtosis_freq(x):

    return kurtosis(x, fisher=True)

def rms_10(x):

    y = x[:int(len(x)*0.1)]

    return np.sqrt(np.mean(y*2))
def rms_20(x):

    y = x[:int(len(x)*0.20)]

    return np.sqrt(np.mean(y**2))

def rms_50(x):

    y = x[:int(len(x)*0.50)]

    return np.sqrt(np.mean(y**2))

def rms_80(x):

    y = x[:int(len(x)*0.80)]

    return np.sqrt(np.mean(y**2))

def rms_100(x):

    return np.sqrt(np.mean(x**2))

def quad_sum(x):

    return np.sum(x**2)

def create_aggregated_freq(fft_df):

    signals = ['acc_x_FFT', 'acc_y_FFT', 'acc_z_FFT', 'gyro_x_FFT', 'gyro_y_FFT','gyro_z_FFT']

    fft_agg_df = fft_df.groupby("block").agg({x: ["sum", "mean", "mad",
                                                  "median", "min", "max",
                                              rms_80, rms_100, quad_sum] for x in signals })
    return fft_agg_df

def create_features(agg_df, fft_agg_df):

    features = agg_df.merge(fft_agg_df,on="block")

    return features


def find_na(df):

    features_to_drop = []

    for i in df.columns:

        if df[i].isna().sum() > len(df)*0.3:
            features_to_drop.append(i)

    print("features_to_drop saves to ../notebooks")
    pd.DataFrame(features_to_drop).to_csv("../notebooks/features_to_drop.txt")

    return features_to_drop

def drop_features(df,features_to_drop):

    df.drop(features_to_drop, axis = 1, inplace = True)

    return df.to_numpy()


def create_train_test(features, activity_labels):

    X_train, X_test, y_train, y_test = train_test_split(features, activity_labels,
                                                    random_state=42, test_size = 0.3, shuffle = True)

    return X_train, X_test, y_train, y_test
