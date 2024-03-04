import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models
import tensorflow as tf
import time

# Function to preprocess the data
def preprocess_data(data):
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
model = models.load_model('/home/iiit-kottayam/new/trained_model.h5')

# Function to evaluate the model for the entire subject's data
def evaluate_model_subject(X):
    y_pred = model.predict(X)
    # Take the majority vote of predictions for all windows
    subject_prediction = np.argmax(np.bincount(np.argmax(y_pred, axis=1)))
    return subject_prediction

# Function to predict using TFLite model
def predict_with_tflite(X, tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    for sample in X:
        input_data = np.expand_dims(sample, axis=0).astype(np.float32)  # Add batch dimension and convert to float32
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Append the predictions for the current sample to the list
        predictions.append(output_data)

    # Concatenate predictions along the batch dimension
    predictions = np.concatenate(predictions, axis=0)

    # Take the majority vote across all samples
    majority_vote = np.argmax(np.bincount(np.argmax(predictions, axis=1)))

    return majority_vote

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

    # Predict label for the entire subject using the Keras model
    start_time_keras = time.time()
    subject_prediction_keras = evaluate_model_subject(X_test)
    end_time_keras = time.time()

    # Predict label for the entire subject using the TFLite model
    start_time_tflite = time.time()
    tflite_model_path = '/home/iiit-kottayam/new/trained_model.tflite'
    subject_prediction_tflite = predict_with_tflite(X_test, tflite_model_path)
    end_time_tflite = time.time()

    # Print input data shapes
    #print("Input data shape of Keras model:", X_test.shape)
    input_shape_tflite = (X_test.shape[1], X_test.shape[2], X_test.shape[3])
    #print("Input data shape of TFLite model:", input_shape_tflite)

    print("Predicted label for the subject using Keras model:", subject_prediction_keras)
    print("Predicted label for the subject using TFLite model:", subject_prediction_tflite)
    print("0 = Non-Addict")
    print("1 = Addict")

    # Print time taken for prediction
    print("Time taken for prediction using Keras model: {:.2f} seconds".format(end_time_keras - start_time_keras))
    print("Time taken for prediction using TFLite model: {:.2f} seconds".format(end_time_tflite - start_time_tflite))

if __name__ == "__main__":
    main('/home/iiit-kottayam/new/mdp/24.csv')
