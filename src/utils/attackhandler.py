#Imports
import os
import pandas as pd
import tensorflow as tf
from keras import backend as K
import time
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle
from .modelgenerator import *

class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - self.epoch_start_time
        self.epoch_times.append(epoch_time)

    def get_training_times_df(self):
        total_training_time = time.time() - self.start_time
        average_epoch_times = [sum(self.epoch_times[:i+1]) / (i + 1) for i in range(len(self.epoch_times))]
        data = {
            'Epoch': list(range(1, len(self.epoch_times) + 1)),
            'Epoch Train_time': self.epoch_times,
            'Epoch Avg Train_time': average_epoch_times,
            'Total Training Time': total_training_time
        }
        return pd.DataFrame(data)

class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_times = []
        self.losses = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': []
        }

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - self.epoch_start_time
        self.epoch_times.append(epoch_time)

        self.losses['epoch'].append(epoch)
        self.losses['train_loss'].append(logs['loss'])
        self.losses['val_loss'].append(logs['val_loss'])

    def on_test_end(self, logs=None):
        self.losses['test_loss'].append(logs['loss'])

    def get_loss_df(self):
        total_training_time = time.time() - self.start_time
        average_epoch_times = [sum(self.epoch_times[:i+1]) / (i + 1) for i in range(len(self.epoch_times))]
        self.losses['avg_epoch_time'] = average_epoch_times
        self.losses['total_training_time'] = total_training_time
        return pd.DataFrame(self.losses)
    
# Data helper ----------------------------------------------------------------
def create_df_array(df, user_indices, datatype):
    return [df[[f'{datatype}_{idx}']] for idx in user_indices] #Only keep the datatype (pv) from selected users

def split_data(df_array, sequence_length, batch_size, dh):
    X_train, y_train, X_val, y_val, X_test, y_test = {}, {}, {}, {}, {}, {}

    for idx, df in enumerate(df_array):
        n = len(df)
        train_df = df[0:int(n * 0.7)]
        val_df = df[int(n * 0.7):int(n * 0.9)]
        test_df = df[int(n * 0.9):]

        # Min max scaling
        train_df = dh.min_max_scaling(train_df)
        val_df = dh.min_max_scaling(val_df)
        test_df = dh.min_max_scaling(test_df)

        # Sequencing
        train_sequences = dh.create_sequences(train_df, sequence_length)
        val_sequences = dh.create_sequences(val_df, sequence_length)
        test_sequences = dh.create_sequences(test_df, sequence_length)

        # Split into feature and label
        X_train[f'user{idx + 1}'], y_train[f'user{idx + 1}'] = dh.prepare_data(train_sequences, batch_size)
        X_val[f'user{idx + 1}'], y_val[f'user{idx + 1}'] = dh.prepare_data(val_sequences, batch_size)
        X_test[f'user{idx + 1}'], y_test[f'user{idx + 1}'] = dh.prepare_data(test_sequences, batch_size)

    return X_train, y_train, X_val, y_val, X_test, y_test

def get_callbacks():
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
    timing_callback = TimingCallback()
    custom_callback = CustomCallback()
    callbacks = [early_stopping, timing_callback, custom_callback]
    return callbacks

def load_and_prepare_data(file_path, user_indices, columns_filter_prefix, max_column_index=30):
    """
    Load and prepare the dataset by filtering the relevant columns and handling missing values.
    """
    # Load the dataset
    df = pd.read_csv(file_path, index_col='Date')
    df.index = pd.to_datetime(df.index)
    # Handle missing values
    df.fillna(0, inplace=True)

    if columns_filter_prefix == 'prosumption':
        for idx in range(0, max_column_index):
           df[f'prosumption_{idx+1}'] = df[f'load_{idx+1}'] - df[f'pv_{idx+1}']
        
    # Filter for the relevant columns
    columns_to_keep = [
        col for col in df.columns 
        if col.startswith(columns_filter_prefix) and int(col.split('_')[1]) <= max_column_index
    ]
    filtered_df = df[columns_to_keep]

    # Create DataFrame array for selected user indices
    df_array = create_df_array(filtered_df, user_indices, datatype=columns_filter_prefix)

    return df_array

# Local Learning ----------------------------------------------------------------

# Initialize result DataFrames
def initialize_result_dfs():
    return {
        "cnn_results": pd.DataFrame(columns=['train_time', 'avg_time_epoch', 'mse', 'mse_std', 'rmse','rmse_std','mae','mae_std']),
        "bilstm_results": pd.DataFrame(columns=['train_time', 'avg_time_epoch', 'mse', 'mse_std', 'rmse','rmse_std','mae','mae_std']),
        "softdense_results": pd.DataFrame(columns=['train_time', 'avg_time_epoch', 'mse', 'mse_std', 'rmse','rmse_std','mae','mae_std']),
        "softlstm_results": pd.DataFrame(columns=['train_time', 'avg_time_epoch', 'mse', 'mse_std', 'rmse','rmse_std','mae','mae_std']),
        "cnn_all_results": pd.DataFrame(columns=["user", "architecture", "train_time", "avg_time_epoch", "mse", "rmse", "mae"]),
        "bilstm_all_results": pd.DataFrame(columns=["user", "architecture", "train_time", "avg_time_epoch", "mse", "rmse", "mae"]),
        "softdense_all_results": pd.DataFrame(columns=["user", "architecture", "train_time", "avg_time_epoch", "mse", "rmse", "mae"]),
        "softlstm_all_results": pd.DataFrame(columns=["user", "architecture", "train_time", "avg_time_epoch", "mse", "rmse", "mae"]),
    }

def build_models(input_shape, m1):
    models = {
        "cnn_model": m1.build_cnn_model(input_shape),
        "bilstm_model": m1.build_bilstm_model(input_shape),
        "softdense_model": m1.build_soft_dense_moe_model(input_shape, m1),
        "softlstm_model": m1.build_soft_biLSTM_moe_model(input_shape, m1),
    }
    return models

def evaluate_models(models, X_train, y_train, X_val, y_val, X_test, y_test, callbacks, user_id, mh, max_epochs=100):
    results = {}
    results["cnn_user_results"] = mh.compile_fit_evaluate_model(models['cnn_model'], X_train, y_train, X_val, y_val, X_test, y_test, callbacks, user_id, "CNN", max_epochs=max_epochs)
    results["bilstm_user_results"] = mh.compile_fit_evaluate_model(models['bilstm_model'], X_train, y_train, X_val, y_val, X_test, y_test, callbacks, user_id, "BiLSTM", max_epochs=max_epochs)
    results["softdense_user_results"] = mh.compile_fit_evaluate_model(models['softdense_model'], X_train, y_train, X_val, y_val, X_test, y_test, callbacks, user_id, "Soft_dense", max_epochs=max_epochs)
    results["softlstm_user_results"] = mh.compile_fit_evaluate_model(models['softlstm_model'], X_train, y_train, X_val, y_val, X_test, y_test, callbacks, user_id, "Soft_lstm", max_epochs=max_epochs)
    return results

def merge_results(all_results, user_results):
    all_results['cnn_all_results'] = pd.merge(all_results['cnn_all_results'], user_results['cnn_user_results'], how='outer')
    all_results['bilstm_all_results'] = pd.merge(all_results['bilstm_all_results'], user_results['bilstm_user_results'], how='outer')
    all_results['softdense_all_results'] = pd.merge(all_results['softdense_all_results'], user_results['softdense_user_results'], how='outer')
    all_results['softlstm_all_results'] = pd.merge(all_results['softlstm_all_results'], user_results['softlstm_user_results'], how='outer')

    return {key: value for key, value in all_results.items() if '_all' in key}

def aggregate_results(df_array, all_results, mh):
    aggregated_results = {
        "cnn_results": mh.aggregate_user_results(df_array, all_results['cnn_all_results'], "CNN"),
        "bilstm_results": mh.aggregate_user_results(df_array, all_results['bilstm_all_results'], "BiLSTM"),
        "softdense_results": mh.aggregate_user_results(df_array, all_results['softdense_all_results'], "Soft_dense"),
        "softlstm_results": mh.aggregate_user_results(df_array, all_results['softlstm_all_results'], "Soft_lstm"),
    }
    return aggregated_results

# Main function to run the entire process
def run_local_learning(df_array, X_train, y_train, X_val, y_val, X_test, y_test, callbacks, m1, mh, max_epochs=100):
    all_results = initialize_result_dfs()

    # Initialize dictionary to store round-wise results
    round_results_df = pd.DataFrame(columns=['user', 'architecture', 'round', 'train_time', 'avg_time_epoch', 'mse', 'rmse', 'mae'])


    for idx in range(len(df_array)):
        user_id = f'user{idx+1}'
        
        for round in range(3):
            print("Building: ", idx+1, " - round ", round)
            # Build models
            models = build_models(X_train[user_id].shape, m1)

            # Evaluate models
            user_results = evaluate_models(models, X_train[user_id], y_train[user_id], X_val[user_id], y_val[user_id], X_test[user_id], y_test[user_id], callbacks, user_id, mh, max_epochs=max_epochs)

            # Merge results
            all_results = merge_results(all_results, user_results)

    # Aggregate results
    aggregated_results = aggregate_results(df_array, all_results, mh)

    return aggregated_results, all_results 

# Federated Learning with no attack ----------------------------------------------
def sum_weights(weight_list):
    """
    Return the sum of the listed weights. The is equivalent to avg of the weights
    """
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*weight_list):
        layer_mean = tf.math.reduce_mean(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

def initialize_global_models(X_train, user_id, m1):
    global_models = {
        "cnn_model": m1.build_cnn_model(X_train[user_id].shape),
        "bilstm_model": m1.build_bilstm_model(X_train[user_id].shape),
        "softdense_model": m1.build_soft_dense_moe_model(X_train[user_id].shape, m1),
        "softlstm_model": m1.build_soft_biLSTM_moe_model(X_train[user_id].shape, m1),
    }
    return global_models

def save_global_models(global_models, attack, round, cwd, fed_round=0):

    cnn_path = os.path.join(cwd, f"models/{attack}/CNN/FederatedRound_{fed_round}/round_{round}")
    bilstm_path = os.path.join(cwd, f"models/{attack}/BiLSTM/FederatedRound_{fed_round}/round_{round}")
    softdense_path = os.path.join(cwd, f"models/{attack}/SoftDense/FederatedRound_{fed_round}/round_{round}")
    softlstm_path = os.path.join(cwd, f"models/{attack}/SoftLSTM/FederatedRound_{fed_round}/round_{round}")
    
    # Create directories if they do not exist
    os.makedirs(cnn_path, exist_ok=True)
    os.makedirs(bilstm_path, exist_ok=True)
    os.makedirs(softdense_path, exist_ok=True)
    os.makedirs(softlstm_path, exist_ok=True)

    global_models["cnn_model"].save(cwd + f"/models/{attack}/CNN/FederatedRound_{fed_round}/round_{round}/model.keras")
    global_models["bilstm_model"].save(cwd + f"/models/{attack}/BiLSTM/FederatedRound_{fed_round}/round_{round}/model.keras")
    global_models["softdense_model"].save(cwd + f"/models/{attack}/SoftDense/FederatedRound_{fed_round}/round_{round}/model.keras")
    global_models["softlstm_model"].save(cwd + f"/models/{attack}/SoftLSTM/FederatedRound_{fed_round}/round_{round}/model.keras")

def load_global_models(attack, fed_round, round, cwd):
    custom_objects = {'StackExpertsLayer': StackExpertsLayer, 'MoEOutputLayer': MoEOutputLayer}
    global_models = {
        "cnn_model": keras.models.load_model(cwd + f"/models/{attack}/CNN/FederatedRound_{fed_round}/round_{round}/model.keras", custom_objects=custom_objects, compile=False),
        "bilstm_model": keras.models.load_model(cwd + f"/models/{attack}/BiLSTM/FederatedRound_{fed_round}/round_{round}/model.keras", custom_objects=custom_objects, compile=False),
        "softdense_model": keras.models.load_model(cwd + f"/models/{attack}/SoftDense/FederatedRound_{fed_round}/round_{round}/model.keras", custom_objects=custom_objects, compile=False),
        "softlstm_model": keras.models.load_model(cwd + f"/models/{attack}/SoftLSTM/FederatedRound_{fed_round}/round_{round}/model.keras", custom_objects=custom_objects, compile=False),
    }
    return global_models

def initialize_local_models(X_train, user_id, global_models, m1, loss, metrics):
    local_models = {
        "cnn_model": m1.build_cnn_model(X_train[user_id].shape),
        "bilstm_model": m1.build_bilstm_model(X_train[user_id].shape),
        "softdense_model": m1.build_soft_dense_moe_model(X_train[user_id].shape, m1),
        "softlstm_model": m1.build_soft_biLSTM_moe_model(X_train[user_id].shape, m1),
    }
    #Compile models
    local_models['cnn_model'].compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=metrics)
    local_models['bilstm_model'].compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=metrics)
    local_models['softdense_model'].compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=metrics)
    local_models['softlstm_model'].compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=metrics)
    
    # Set local model weights to the global model weights
    local_models['cnn_model'].set_weights(global_models['cnn_model'].get_weights())
    local_models['bilstm_model'].set_weights(global_models['bilstm_model'].get_weights())
    local_models['softdense_model'].set_weights(global_models['softdense_model'].get_weights())
    local_models['softlstm_model'].set_weights(global_models['softlstm_model'].get_weights())

    return local_models

def train_local_models(local_models, X_train, y_train, X_val, y_val, X_test, y_test, callbacks, user_id, mh, max_epochs=100):
    local_results = {
        "cnn_user_results": mh.compile_fit_evaluate_model(local_models['cnn_model'], X_train[user_id], y_train[user_id], X_val[user_id], y_val[user_id], X_test[user_id], y_test[user_id], callbacks, user_id, "CNN", max_epochs=max_epochs),
        "bilstm_user_results": mh.compile_fit_evaluate_model(local_models['bilstm_model'], X_train[user_id], y_train[user_id], X_val[user_id], y_val[user_id], X_test[user_id], y_test[user_id], callbacks, user_id, "BiLSTM", max_epochs=max_epochs),
        "softdense_user_results": mh.compile_fit_evaluate_model(local_models['softdense_model'], X_train[user_id], y_train[user_id], X_val[user_id], y_val[user_id], X_test[user_id], y_test[user_id], callbacks, user_id, "SoftDense", max_epochs=max_epochs),
        "softlstm_user_results": mh.compile_fit_evaluate_model(local_models['softlstm_model'], X_train[user_id], y_train[user_id], X_val[user_id], y_val[user_id], X_test[user_id], y_test[user_id], callbacks, user_id, "SoftLSTM", max_epochs=max_epochs),
    }
    return local_results

def initialize_local_weights_dic():
    return {
        "local_cnn_weight_list" : list(),
        "local_bilstm_weight_list" : list(),
        "local_softdense_weight_list" : list(),
        "local_softlstm_weight_list" : list(),
    }

def append_local_weights_dic(local_weights_dic, local_models):
    local_weights_dic["local_cnn_weight_list"].append(local_models['cnn_model'].get_weights())
    local_weights_dic["local_bilstm_weight_list"].append(local_models['bilstm_model'].get_weights())
    local_weights_dic["local_softdense_weight_list"].append(local_models['softdense_model'].get_weights())
    local_weights_dic["local_softlstm_weight_list"].append(local_models['softlstm_model'].get_weights())
    return local_weights_dic

def get_average_weights(local_weights_dic):
    return {
        "average_weights_cnn" : sum_weights(local_weights_dic["local_cnn_weight_list"]),
        "average_weights_bilstm" : sum_weights(local_weights_dic["local_bilstm_weight_list"]),
        "average_weights_softdense" : sum_weights(local_weights_dic["local_softdense_weight_list"]),
        "average_weights_softlstm" : sum_weights(local_weights_dic["local_softlstm_weight_list"]),
    }

def set_average_weights_to_global_models(global_models, average_weights_dic):
    global_models["cnn_model"].set_weights(average_weights_dic["average_weights_cnn"])
    global_models["bilstm_model"].set_weights(average_weights_dic["average_weights_bilstm"])
    global_models["softdense_model"].set_weights(average_weights_dic["average_weights_softdense"])
    global_models["softlstm_model"].set_weights(average_weights_dic["average_weights_softlstm"])
    return global_models

def run_federated_training(df_array, X_train, y_train, X_val, y_val, X_test, y_test, callbacks, m1, mh, attack, cwd, loss, metrics, rounds=3, fed_rounds=3, max_epochs=100):

    for round in range(rounds):
        
        user_id = next(iter(X_train))
        global_models = initialize_global_models(X_train, user_id, m1)
        save_global_models(global_models, attack, round, cwd)
        
        for fed_round in range(fed_rounds):

            global_models = load_global_models(attack, fed_round, round, cwd)
            local_weights_dic = initialize_local_weights_dic()

            for idx in range(len(df_array)):
                
                local_models = initialize_local_models(X_train, user_id, global_models, m1, loss, metrics)
                local_results = train_local_models(local_models, X_train, y_train, X_val, y_val, X_test, y_test, callbacks, user_id, mh, max_epochs=max_epochs)
                local_weights_dic = append_local_weights_dic(local_weights_dic, local_models)

                K.clear_session()

            average_weights_dic = get_average_weights(local_weights_dic)
            global_models = set_average_weights_to_global_models(global_models, average_weights_dic)

            save_global_models(global_models, attack, round, cwd, fed_round+1)
            print("FL: Saved Global models - round ", round, " (fed_round ", fed_round, ")")

def run_federated_local_evaluation(df_array, X_train, y_train, X_val, y_val, X_test, y_test, callbacks, m1, mh, attack, cwd, loss, metrics, rounds=3, fed_round=3, max_epochs=1):

    # Initialize result DataFrames
    all_results = initialize_result_dfs()

    for idx in range(len(df_array)):
        user_id = f'user{idx+1}'
        
        for round in range(rounds):
            print("Evaluation: Building: ", idx+1, " - round ", round)
            
            # Load global models
            global_models = load_global_models(attack, fed_round=fed_round, round=round, cwd=cwd)
            
            # Initialize local models
            local_models = initialize_local_models(X_train, user_id, global_models, m1, loss, metrics)
            
            # Evaluate models
            user_results = evaluate_models(local_models, X_train[user_id], y_train[user_id], X_val[user_id], y_val[user_id], X_test[user_id], y_test[user_id], callbacks, user_id, mh, max_epochs=max_epochs)
            
            # Merge the results into all_results
            all_results = merge_results(all_results, user_results)

    # Aggregate results
    aggregated_results = aggregate_results(df_array, all_results, mh)

    return aggregated_results, all_results

#Attacks ----------------------------------------------
def plot_impact_of_attack_noise(X_train_raw, X_train, user="user1"):
    
    plt.figure(figsize=(10,1))
    plt.plot(X_train_raw[user][0], label='Original Data', linestyle='--', color='blue')
    plt.plot(X_train[user][0], label='Poisend Data', linestyle='-', color='red')

    # Adding title, labels, legend, and grid
    plt.title(f'Original vs. Poisened Data for {user}')
    plt.xlabel('Time steps')
    plt.ylabel('kW')
    plt.legend()

    plt.show()

def save_dictionaries(data_to_save, folder_name="results/"):
    
    os.makedirs(folder_name, exist_ok=True)
    
    for filename, data in data_to_save:
        with open(os.path.join(folder_name, f"{filename}.pkl"), "wb") as f:
            pickle.dump(data, f)
    
    print("Dictionaries saved successfully!")