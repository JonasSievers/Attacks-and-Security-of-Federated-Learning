#Imports
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import pandas as pd
import numpy as np

from keras.callbacks import ModelCheckpoint

#The Modelhandler class contains usefull methods to compile, fit, evaluate, and plot models
class Modelhandler():

    #This method plots 1. training and validation loss & 2. prediction results
    def plot_model_predictions(self, model, history, y_test, X_test, batch_size, plt_length=200):
        # Plot training and validation loss
        plt.figure(figsize=(15, 3))
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Make predictions on the test set
        y_pred = model.predict(X_test, batch_size=batch_size)

        # Plot prediction results
        plt.figure(figsize=(10, 3))
        plt.plot(y_test[:plt_length], label='True')
        plt.plot(y_pred[:plt_length], label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Electricity consumption')
        plt.legend()
        plt.show()

    #This method compiles the model using Adam optimizer, fits the model, and evaluates it
    def compile_fit_evaluate_model(self, model, X_train, y_train, X_val, y_val, X_test, y_test, callbacks, user, architecture,
            loss=tf.keras.losses.MeanSquaredError(), metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()],
            max_epochs=100, batch_size=16):
        
        #Compile, fit, evaluate the model
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=metrics)
        history = model.fit(X_train, y_train, epochs=max_epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks, verbose=0,)
        test_loss = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

        # Store results in a DataFrame
        train_times = callbacks[1].get_training_times_df()
        total_train_time = train_times["Total Training Time"][0]
        avg_time_epoch = train_times["Epoch Avg Train_time"].iloc[-1]
        
        model_user_result = pd.DataFrame({
            "architecture": [architecture],
            "user": [user],
            "train_time": [total_train_time],
            "avg_time_epoch": [avg_time_epoch],
            "mse": [test_loss[0]],
            "rmse": [test_loss[1]],
            "mae": [test_loss[2]]
        })

        return model_user_result
    
    def aggregate_user_results(self, df_array, all_results, architecture):
        results = pd.DataFrame(columns=['architecture', 'train_time', 'avg_time_epoch', 'mse', 'mse_std', 'rmse', 'rmse_std', 'mae', 'mae_std'])

        for idx in range(len(df_array)):
            user_id = f'user{idx+1}'
            user_results = all_results[all_results["user"] == user_id]

            # Compute the aggregated statistics
            new_row = {
                'architecture' : architecture,
                'train_time': user_results["train_time"].mean(),
                'avg_time_epoch': user_results["avg_time_epoch"].mean(),
                'mse': user_results["mse"].mean(),
                'mse_std': user_results["mse"].std(),
                'rmse': user_results["rmse"].mean(),
                'rmse_std': user_results["rmse"].std(),
                'mae': user_results["mae"].mean(),
                'mae_std': user_results["mae"].std(),
            }

            # Append the new row to cnn_results
            results.loc[len(results)] = new_row
        
        return results
    
    def evaluate_ensemble(self, y_test, final_predictions, user, hyper, train_time, avg_time_epoch): 
        
        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, final_predictions)
        
        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)          

        # Calculate Mean Absolute Percentage Error (MAPE)
        epsilon = 1e-10  # Small epsilon to avoid division by zero
        mape = np.mean(np.abs((y_test - final_predictions) / (y_test + epsilon))) * 100

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, final_predictions)

        model_user_result = pd.DataFrame(
            data=[[user, hyper, train_time, avg_time_epoch, mse, rmse, mape, mae]], 
            columns=["user", "architecture", "train_time", "avg_time_epoch", "mse", "rmse", "mape", "mae"]
        )

        return model_user_result

    #This methods fits, predicts, and plots the results for sklearn models
    def statistical_model_compile_fit_evaluate(self, X_train, y_train, X_test, y_test, model):
        X_train_flattened = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
        model.fit(X_train_flattened, y_train)

        X_test_flattened = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
        y_pred = model.predict(X_test_flattened)

        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        # Plot the actual and predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Electricity consumption')
        plt.title('Model Prediction vs Actual')
        plt.legend()
        plt.show()

    