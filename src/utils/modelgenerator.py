#Imports
#Tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn import svm
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
import time

class StackExpertsLayer(layers.Layer):
  def call(self, experts):
      # Stack tensors along a new axis (axis=1)
      return tf.stack(experts, axis=1)

  def compute_output_shape(self, input_shape):
      # Calculate output shape based on the stacked expert tensors
      expert_shape = input_shape[0]  # All experts should have the same shape
      return (expert_shape[0], len(input_shape), expert_shape[1], expert_shape[2])
  
# Custom layer to compute MoE output
class MoEOutputLayer(layers.Layer):
  def call(self, inputs):
      routing_logits, expert_outputs = inputs
      # Compute the mixture of experts output using einsum
      return tf.einsum('bsn,bnse->bse', routing_logits, expert_outputs)

  def compute_output_shape(self, input_shape):
      routing_shape, expert_shape = input_shape
      return (routing_shape[0], expert_shape[-2], expert_shape[-1])

class ModelGenerator():
  
  #Builds the expert models for the MoE Layer
  def build_expert_network(self, expert_units):
      expert = keras.Sequential([
              layers.Dense(expert_units, activation="relu"), 
              ])
      return expert
  

  #Builds a MoE model with soft gating
  def build_soft_dense_moe_model(self, X_train_shape, m1, batch_size=16, horizon=1, dense_units=16,  expert_units=8, num_experts=4):
    #Input of shape (batch_size, sequence_length, features)
    inputs = layers.Input(shape=(X_train_shape[1], X_train_shape[2]), batch_size=batch_size, name='input_layer') 
    x = inputs
   
    #EMBEDDED MOE LAYER
    # Gating network (Routing Softmax)
    routing_logits = layers.Dense(num_experts, activation='softmax')(x)
    experts = [m1.build_expert_network(expert_units=expert_units)(x) for _ in range(num_experts)]
    expert_outputs = StackExpertsLayer()(experts)
    moe_output = MoEOutputLayer()([routing_logits, expert_outputs])
    #expert_outputs = layers.Lambda(lambda tensors: tf.stack(tensors, axis=1))(experts)
    #moe_output = layers.Lambda(lambda x: tf.einsum('bsn,bnse->bse', x[0], x[1]))([routing_logits, expert_outputs])
    #END MOE LAYER

    x = layers.Dense(dense_units, activation="relu")(moe_output)
    x = layers.Dense(dense_units, activation="relu")(x)

    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(horizon)(x)
    softgated_moe_model = models.Model(inputs=inputs, outputs=outputs, name="soft_dense_moe")

    return softgated_moe_model
  
     
  #Builds a MoE model with soft gating
  def build_soft_biLSTM_moe_model(self, X_train_shape, m1, batch_size=16, horizon=1, lstm_units=4, num_experts=4, expert_units=8):
    #Input of shape (batch_size, sequence_length, features)
    inputs = layers.Input(shape=(X_train_shape[1], X_train_shape[2]), batch_size=batch_size, name='input_layer') 
    x = inputs

    #EMBEDDED MOE LAYER
    # Gating network (Routing Softmax)
    routing_logits = layers.Dense(num_experts, activation='softmax')(x)
    experts = [m1.build_expert_network(expert_units=expert_units)(x) for _ in range(num_experts)]
    expert_outputs = StackExpertsLayer()(experts)
    moe_output = MoEOutputLayer()([routing_logits, expert_outputs])
    #END MOE LAYER

    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(moe_output)
    #x = layers.Dense(16)(moe_output)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(horizon)(x)
    softgated_moe_model = models.Model(inputs=inputs, outputs=outputs, name="soft_bilstm_moe")

    return softgated_moe_model
  

  def build_bilstm_model(self, X_train_shape, horizon=1, num_layers=2, units=8, batch_size=16):

    input_data = layers.Input(shape=(X_train_shape[1], X_train_shape[2]), batch_size=batch_size) 
    x =  layers.Bidirectional(layers.LSTM(units, return_sequences=True))(input_data)
    for _ in range(num_layers-1):
      x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    output = layers.Dense(horizon)(x) 

    bilstm_model = tf.keras.Model(inputs=input_data, outputs=output, name="lstm_model")
  
    return bilstm_model
  

  def build_cnn_model(self, X_train_shape, horizon=1, num_layers=4, filter=8, kernel_size=1, dense_units=16, batch_size=16):
      
      input_data = layers.Input(shape=(X_train_shape[1], X_train_shape[2]), batch_size=batch_size)

      x = layers.Conv1D(filters=filter, kernel_size=kernel_size)(input_data)
      for _ in range(num_layers-1):
          x = layers.Conv1D(filters=filter, kernel_size=kernel_size)(x)
      x = layers.Dropout(0.2)(x)
      x = layers.Dense(dense_units)(x)

      output = layers.Dense(horizon)(x)
      cnn_model = tf.keras.Model(inputs=input_data, outputs=output, name="cnn_model")

      return cnn_model
  


