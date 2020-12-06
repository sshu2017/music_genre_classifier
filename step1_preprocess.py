import os
import librosa
import math
import json


DATASET_PATH = "data/genres/"
JSON_PATH = "data/json/"
OUTPUT_JSON_FILE_NAME = "marsyas.json"
SAMPLE_RATE = 22050
DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def process_one_wave_file(path_to_wave_file,
                          n_mfcc=12,
                          n_fft=2048, hop_length=512, num_segments=5):
    """Process one wave file
    Return a list of mfcc, which each is a list of floats itself, and a music genre label
    """
    print(path_to_wave_file)
    signal, sr = librosa.load(path_to_wave_file, sr=SAMPLE_RATE)

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    for s in range(num_segments):
        start_sample = num_samples_per_segment * s
        finish_sample = start_sample + num_samples_per_segment

        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                    sr=sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)

        mfcc = mfcc.T

        label = path_to_wave_file.split("/")[3]

        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
            return mfcc.tolist(), label
        else:
            pass


def save_mfcc(dataset_path, json_path, output_filename):
    data = {
        "mfcc": [],
        "labels": [],
    }

    for dirpath, dirnames, filenames in os.walk(dataset_path):
        if dirpath != dataset_path:
            for filename in filenames:
                wave_file = dirpath + "/" + filename
                mfcc_list, label = process_one_wave_file(wave_file)
                data['mfcc'].append(mfcc_list)
                data['labels'].append(label)

    # save data into a json file
    with open(json_path + output_filename, 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, OUTPUT_JSON_FILE_NAME)
