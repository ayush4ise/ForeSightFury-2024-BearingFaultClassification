# Importing libraries 
import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import kurtosis, skew

# Extract time-domain features from the signal
def time_domain_features(signal, fs):
    # Time-domain features
    mean_value = np.mean(signal)
    std_dev = np.std(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)

    # Calculate second derivative
    time_axis = np.arange(len(signal)) / fs
    second_derivative = np.gradient(np.gradient(signal, time_axis), time_axis)

    # Calculate the features from second derivative
    mean_value_2 = np.mean(second_derivative)
    std_dev_2 = np.std(second_derivative)
    skewness_2 = skew(second_derivative)
    kurt_2 = kurtosis(second_derivative)

    return mean_value, std_dev, skewness, kurt , mean_value_2, std_dev_2, skewness_2, kurt_2

# Extract frequency-domain features from the signal
def frequency_domain_features(signal, fs):
    # Frequency-domain features using FFT
    fft_values = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(len(signal), 1/fs)

    # Calculate the features from FFT
    fft_absvalue = np.abs(fft_values)

    mean_value = np.mean(fft_absvalue)
    std_dev = np.std(fft_absvalue)
    skewness = skew(fft_absvalue)
    kurt = kurtosis(fft_absvalue)

    # Calculate second derivative
    second_derivative = np.gradient(np.gradient(fft_absvalue))

    # Calculate the features from second derivative
    mean_value_2 = np.mean(second_derivative)
    std_dev_2 = np.std(second_derivative)
    skewness_2 = skew(second_derivative)
    kurt_2 = kurtosis(second_derivative)

    return mean_value, std_dev, skewness, kurt , mean_value_2, std_dev_2, skewness_2, kurt_2

# Extract PSD features from the signal
def psd_features(signal, fs):
    # Calculate Power Spectral Density (PSD)
    f, psd = welch(signal, fs=fs, nperseg=len(signal)//2)

    # Calculate the features from PSD
    mean_value = np.mean(psd)
    std_dev = np.std(psd)
    skewness = skew(psd)
    kurt = kurtosis(psd)

    # Calculate second derivative
    time_axis = np.arange(len(psd)) / fs
    second_derivative = np.gradient(np.gradient(psd, time_axis), time_axis)

    # Calculate the features from second derivative
    mean_value_2 = np.mean(second_derivative)
    std_dev_2 = np.std(second_derivative)
    skewness_2 = skew(second_derivative)
    kurt_2 = kurtosis(second_derivative)

    return mean_value, std_dev, skewness, kurt , mean_value_2, std_dev_2, skewness_2, kurt_2

# Extract mixed features from a signal
def mix_features(signal, fs):
    # Time-domain features
    mean_value = np.mean(signal)
    std_dev = np.std(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)

    # Frequency-domain features using FFT
    fft_values = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(len(signal), 1/fs)
    dominant_frequency = np.abs(fft_freqs[np.argmax(np.abs(fft_values))])

    # Calculate Power Spectral Density (PSD)
    f, psd = welch(signal, fs=fs, nperseg=len(signal)//2)
    mean_psd = np.mean(psd)

    # Calculate second derivative
    # time_axis = np.arange(len(signal)) / fs
    # second_derivative = np.gradient(np.gradient(signal, time_axis), time_axis)

    return mean_value, std_dev, skewness, kurt, dominant_frequency, mean_psd# , second_derivative

#### Turn extracted features into a dataframe
def feature_df(file_path, fs, hp, target):
    # Load the data
    data = loadmat(file_path)
    vibration_signals = [x[0] for x in data[f'X{file_path[-7:-4]}_DE_time']]

    vibration_data = [vibration_signals[i:i+fs] for i in range(0, len(vibration_signals), fs)]
    features_list = [frequency_domain_features(signal, fs) for signal in vibration_data]

    # Create a dataframe with the features
    features_df = pd.DataFrame(features_list, columns=['mean_value', 'std_dev', 'skewness', 'kurt', 'mean_value_2', 'std_dev_2', 'skewness_2', 'kurt_2'])

    # Add the hp and target column
    features_df['hp'] = hp
    features_df['target'] = target
    try:
        features_df['defect_type'] = float(file_path[7:12])
    except:
        features_df['defect_type'] = 0

    return features_df

# Create a dataframe with all the features
# 0.007 inch defect
ball7_0hp = feature_df('./data/0.007 inch_0hp/Ball/118.mat', 12000, 0, 'ball')
ball7_1hp = feature_df('./data/0.007 inch_1hp/Ball/119.mat', 12000, 1, 'ball')
inner7_0hp = feature_df('./data/0.007 inch_0hp/Inner/105.mat', 12000, 0, 'inner')
inner7_1hp = feature_df('./data/0.007 inch_1hp/Inner/106.mat', 12000, 1, 'inner')
outer7_0hp = feature_df('./data/0.007 inch_0hp/Outer/130.mat', 12000, 0, 'outer')
outer7_1hp = feature_df('./data/0.007 inch_1hp/Outer/130.mat', 12000, 1, 'outer')

# 0.014 inch defect
ball14_0hp = feature_df('./data/0.014 inch_0hp/Ball/185.mat', 12000, 0, 'ball')
ball14_1hp = feature_df('./data/0.014 inch_1hp/Ball/186.mat', 12000, 1, 'ball')
inner14_0hp = feature_df('./data/0.014 inch_0hp/Inner/169.mat', 12000, 0, 'inner')
inner14_1hp = feature_df('./data/0.014 inch_1hp/Inner/170.mat', 12000, 1, 'inner')
outer14_0hp = feature_df('./data/0.014 inch_0hp/Outer/197.mat', 12000, 0, 'outer')
outer14_1hp = feature_df('./data/0.014 inch_1hp/Outer/198.mat', 12000, 1, 'outer')

# 0.021 inch defect
ball21_0hp = feature_df('./data/0.021 inch_0hp/Ball/222.mat', 12000, 0, 'ball')
ball21_1hp = feature_df('./data/0.021 inch_1hp/Ball/223.mat', 12000, 1, 'ball')
inner21_0hp = feature_df('./data/0.021 inch_0hp/Inner/209.mat', 12000, 0, 'inner')
inner21_1hp = feature_df('./data/0.021 inch_1hp/Inner/210.mat', 12000, 1, 'inner')
outer21_0hp = feature_df('./data/0.021 inch_0hp/Outer/234.mat', 12000, 0, 'outer')
outer21_1hp = feature_df('./data/0.021 inch_1hp/Outer/235.mat', 12000, 1, 'outer')

# normal data
normal_0hp = feature_df('./data/Normal/Normal_0_097.mat', 12000, 0, 'normal')
normal_1hp = feature_df('./data/Normal/Normal_1_098.mat', 12000, 1, 'normal')

# Combine all the dataframes
df = pd.concat([ball7_0hp, ball7_1hp, inner7_0hp, inner7_1hp, outer7_0hp, outer7_1hp,
                ball14_0hp, ball14_1hp, inner14_0hp, inner14_1hp, outer14_0hp, outer14_1hp,
                ball21_0hp, ball21_1hp, inner21_0hp, inner21_1hp, outer21_0hp, outer21_1hp,
                normal_0hp, normal_1hp])

# Save the dataframe
df.to_csv('frequency_domain_features.csv', index=False)