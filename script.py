import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models
from sklearn.metrics import accuracy_score

# Function to preprocess the data
def preprocess_data(data):
    # Assuming 'data' is a DataFrame with EEG data
    window_size = 2000  # Number of sampled data points over 8 seconds
    overlap = 500  # Number of sampled data points for 2-second overlap

    # Define sliding window parameters
    def create_sliding_windows_2d(data, window_size, overlap):
        windows = []
        start = 0
        end = window_size
        while end <= data.shape[0]:
            windows.append(data[start:end, :])
            start += window_size - overlap
            end = start + window_size
        return windows

    # Perform detrending and standardization
    scaler = StandardScaler()
    windows = create_sliding_windows_2d(data, window_size, overlap)
    preprocessed_windows = []
    for window in windows:
        # Detrending
        detrended_window = signal.detrend(window, axis=0)
        # Standardization
        standardized_window = scaler.fit_transform(detrended_window)
        preprocessed_windows.append(standardized_window)

    # Flatten the preprocessed windows
    X = np.array(preprocessed_windows)

    # Reshape sliding window data to match input shape
    num_channels = X.shape[2]
    X_reshaped = X.reshape((-1, window_size, num_channels, 1))

    return X_reshaped

# Load the saved model
model = models.load_model('/home/iiit-kottayam/new/final_model.h5')

# Function to evaluate the model
def evaluate_model(X, y):
    # Evaluate the model
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y, y_pred_classes)
    return accuracy

# Load the CSV file
def load_csv_file(csv_file):
    data = pd.read_csv(csv_file)
    return data.values  # Assuming EEG data is stored in the CSV file

# Main function
def main(csv_file):
    # Load CSV file
    data = load_csv_file(csv_file)

    # Preprocess data
    X_test = preprocess_data(data)

    # Extract label from the file name
    label = int(csv_file.split('/')[-1].split('.')[0].split('_')[-1])

    # Evaluate the model with known label
    y_test = np.array([label] * X_test.shape[0])  # Repeat the label for all samples
    accuracy = evaluate_model(X_test, y_test)
    print("Test Accuracy: {:.2f}%".format(accuracy * 100))

if __name__ == "__main__":
    # Replace 'your_csv_file.csv' with the path to your CSV file
    main('/home/iiit-kottayam/new/mdp/3_1.csv')
