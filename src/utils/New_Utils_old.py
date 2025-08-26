# ============================================================
# Imports
# ============================================================
import os
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K # type: ignore

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ============================================================
# Time features and utilities
# ============================================================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyc time-of-day features (scaled to [0,1] and rounded to 4 decimals)."""
    hours = df.index.hour
    minutes = df.index.minute
    hr = 2*np.pi*hours/24.0
    mn = 2*np.pi*minutes/60.0
    out = df.copy()
    out["hour_sin01"]   = np.round((np.sin(hr)+1)/2, 4)
    out["hour_cos01"]   = np.round((np.cos(hr)+1)/2, 4)
    out["minute_sin01"] = np.round((np.sin(mn)+1)/2, 4)
    out["minute_cos01"] = np.round((np.cos(mn)+1)/2, 4)
    return out

def infer_step_minutes_from_index(index: pd.DatetimeIndex) -> int:
    if len(index) < 2: return 30
    diffs = np.diff(index.view('i8')) # ns
    med_ns = np.median(diffs)
    return max(1, int(round(med_ns / 1e9 / 60.0)))

def hhmm_to_hour_min(hhmm: str):
    """parse a string like "10:30" into integers (10, 30)."""
    h, m = hhmm.split(":")
    return int(h), int(m)

def advance_time(h, m, step_minutes):
    total = (h*60 + m + step_minutes) % (24*60)
    return total//60, total%60

def cyc01_from_hm(h, m):
    hr = 2*np.pi*h/24.0
    mn = 2*np.pi*m/60.0
    return (np.round((np.sin(hr)+1)/2,4),
            np.round((np.cos(hr)+1)/2,4),
            np.round((np.sin(mn)+1)/2,4),
            np.round((np.cos(mn)+1)/2,4))

def hour_from_sincos01(sin01, cos01):
    # inverse mapping -> radians in [-pi, pi], then to hour in [0..23]
    sin = sin01*2-1
    cos = cos01*2-1
    radians = np.arctan2(sin, cos)
    hours = np.round((radians/(2*np.pi))*24).astype(int) % 24
    return hours

# ============================================================
# Data utilities
# ============================================================
def min_max_scale_selected(df: pd.DataFrame, cols_to_scale: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols_to_scale:
        vals = df[c].values.astype(float)
        vmin, vmax = np.min(vals), np.max(vals)
        denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
        df[c] = (vals - vmin) / denom
    return df

def create_sequences(df: pd.DataFrame, sequence_length: int) -> np.ndarray:
    X = []
    for i in range(len(df)-sequence_length+1):
        X.append(df.iloc[i:i+sequence_length,:].values)
    return np.asarray(X)

def prepare_data_with_label_time(sequences: np.ndarray, batch_size: int, sin_idx=1, cos_idx=2):
    """
    Return X, y, y_hour for each sequence (based on the label time step).
    X: seq[:-1,:], y: seq[-1,0], y_hour from seq[-1,sin/cos].
    Trim to full batches for static-batch models.
    """
    X = sequences[:, :-1, :].astype('float32')
    y = sequences[:, -1, 0].astype('float32')
    sin = sequences[:, -1, sin_idx]
    cos = sequences[:, -1, cos_idx]
    y_hour = hour_from_sincos01(sin, cos).astype(int)
    n = len(X) // batch_size
    return X[:n*batch_size], y[:n*batch_size], y_hour[:n*batch_size]

def load_and_prepare_data(file_path: str,user_indices: list,columns_filter_prefix: str = "load",
                          max_column_index: int | None = None):
    """
    Load the CSV, optionally derive 'prosumption_{i} = load_{i} - pv_{i}' for each building,
    then build per-user DataFrames with [main, hour_sin01, hour_cos01, minute_sin01, minute_cos01].
    """
    # Load
    df = pd.read_csv(file_path, index_col='Date')
    df.index = pd.to_datetime(df.index)
    df.fillna(0, inplace=True)

    # Determine index cap (if any)
    if max_column_index is None:
        max_column_index = max(user_indices)

    # If prosumption requested, create columns prosumption_{i} = load_{i} - pv_{i}
    if columns_filter_prefix.lower() == "prosumption":
        for idx in user_indices:  # only compute what we need
            load_col = f"load_{idx}"
            pv_col   = f"pv_{idx}"
            if load_col not in df.columns or pv_col not in df.columns:
                missing = [c for c in (load_col, pv_col) if c not in df.columns]
                raise ValueError(f"Cannot compute prosumption for building {idx}. Missing columns: {missing}")
            df[f"prosumption_{idx}"] = df[load_col] - df[pv_col]

    # Choose the main series columns to keep (load_*, pv_*, or prosumption_*)
    prefix = columns_filter_prefix.lower()
    valid_prefixes = {"load", "pv", "prosumption"}
    if prefix not in valid_prefixes:
        raise ValueError(f"columns_filter_prefix must be one of {valid_prefixes}, got '{columns_filter_prefix}'")

    cols = [
        c for c in df.columns if c.startswith(prefix + "_") and c.split('_')[1].isdigit() and int(c.split('_')[1]) <= max_column_index
    ]

    # Keep only those main series columns
    filtered = df[cols].copy()
    # Add time features
    tfdf = add_time_features(filtered)

    # Helper to build per-user frame
    def pick_user(tfdf_local: pd.DataFrame, idx: int) -> pd.DataFrame:
        main_col = f"{prefix}_{idx}"
        if main_col not in tfdf_local.columns:
            raise ValueError(f"Missing column {main_col} (did you request an index not present or above max_column_index?)")
        return tfdf_local[[main_col, "hour_sin01", "hour_cos01", "minute_sin01", "minute_cos01"]].copy()

    # Create one DataFrame per requested user
    df_array = [pick_user(tfdf, idx) for idx in user_indices]
    return df_array

def split_data(df_array, sequence_length, batch_size):
    """
    Returns:
      X_train/val/test, y_train/val/test, y_test_hour (for per-hour metrics)
    """
    X_train, y_train, X_val, y_val, X_test, y_test = {}, {}, {}, {}, {}, {}
    y_test_hour = {}

    for i, df in enumerate(df_array):
        n = len(df)
        train_df = df.iloc[0:int(0.7*n)]
        val_df   = df.iloc[int(0.7*n):int(0.9*n)]
        test_df  = df.iloc[int(0.9*n):]

        main_col = df.columns[0]
        train_df = min_max_scale_selected(train_df, [main_col])
        val_df   = min_max_scale_selected(val_df,   [main_col])
        test_df  = min_max_scale_selected(test_df,  [main_col])

        train_seq = create_sequences(train_df, sequence_length)
        val_seq   = create_sequences(val_df, sequence_length)
        test_seq  = create_sequences(test_df, sequence_length)

        ux = f"user{i+1}"
        X_train[ux], y_train[ux], _         = prepare_data_with_label_time(train_seq, batch_size)
        X_val[ux],   y_val[ux],   _         = prepare_data_with_label_time(val_seq,   batch_size)
        X_test[ux],  y_test[ux],  y_h       = prepare_data_with_label_time(test_seq,  batch_size)
        y_test_hour[ux] = y_h

    return X_train, y_train, X_val, y_val, X_test, y_test, y_test_hour

# ============================================================
# Models (BiLSTM, SoftDense-MoE, SoftLSTM-MoE) + Surrogate & Generator
# ============================================================
class StackExpertsLayer(layers.Layer):
    def call(self, experts):
        return tf.stack(experts, axis=1)

class MoEOutputLayer(layers.Layer):
    def call(self, inputs):
        routing_logits, expert_outputs = inputs
        return tf.einsum('bsn,bnse->bse', routing_logits, expert_outputs)

def _expert_dense(expert_units):
    return models.Sequential([layers.Dense(expert_units, activation="relu")])

def build_bilstm_model(input_shape, horizon=1, units=8, num_layers=2, dropout=0.2, batch_size=16, name="BiLSTM"):
    inp = layers.Input(shape=(input_shape[1], input_shape[2]), batch_size=batch_size)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(inp)
    for _ in range(num_layers-1):
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(horizon)(x)
    return models.Model(inp, out, name=name)

def build_soft_dense_moe_model(input_shape, num_experts=4, expert_units=8, dense_units=16, horizon=1, dropout=0.2, batch_size=16):
    inp = layers.Input(shape=(input_shape[1], input_shape[2]), batch_size=batch_size)
    x = inp
    routing = layers.Dense(num_experts, activation='softmax')(x)
    experts = [_expert_dense(expert_units)(x) for _ in range(num_experts)]
    expert_outputs = StackExpertsLayer()(experts)
    moe = MoEOutputLayer()([routing, expert_outputs])
    x = layers.Dense(dense_units, activation="relu")(moe)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Flatten()(x)
    out = layers.Dense(horizon)(x)
    return models.Model(inp, out, name="SoftDenseMoE")

def build_soft_bilstm_moe_model(input_shape, num_experts=4, expert_units=8, lstm_units=4, horizon=1, dropout=0.2, batch_size=16):
    inp = layers.Input(shape=(input_shape[1], input_shape[2]), batch_size=batch_size)
    x = inp
    routing = layers.Dense(num_experts, activation='softmax')(x)
    experts = [_expert_dense(expert_units)(x) for _ in range(num_experts)]
    expert_outputs = StackExpertsLayer()(experts)
    moe = MoEOutputLayer()([routing, expert_outputs])
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(moe)
    x = layers.Dropout(dropout)(x)
    x = layers.Flatten()(x)
    out = layers.Dense(horizon)(x)
    return models.Model(inp, out, name="SoftLSTMMoE")

# --- Surrogate forecaster used to guide the generator
def build_surrogate(input_shape, cfg):
    return build_bilstm_model(
        input_shape,horizon=1,units=cfg.get("units",8),num_layers=cfg.get("num_layers",1),
        dropout=cfg.get("dropout",0.0),batch_size=None,name="Surrogate")

# --- Perturbation generator (Conv1D -> tanh)
def build_perturbation_generator(input_shape):
    inp = layers.Input(shape=(input_shape[1], input_shape[2]))
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inp)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    # output per feature; we'll keep only channel 0 (main) and zero the rest
    out = layers.Conv1D(input_shape[2], 1, padding='same', activation='tanh')(x)
    return models.Model(inp, out, name="PerturbGen")

# ============================================================
# Training / Evaluation helpers
# ============================================================
class TimingCallback(callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_times = []
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_t0 = time.time()
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.time() - self.epoch_t0)
    def total_training_time(self):
        return time.time() - self.start_time
    def avg_epoch_time(self):
        return float(np.mean(self.epoch_times)) if self.epoch_times else 0.0

def compile_fit_eval(model, X_train, y_train, X_val, y_val, X_test, y_test,
                     max_epochs=100, batch_size=16, patience=10, lr=1e-3):
    model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
    tcb = TimingCallback()
    es = callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
    model.fit(X_train, y_train,validation_data=(X_val, y_val),
              epochs=max_epochs,batch_size=batch_size,callbacks=[es, tcb],verbose=0)
    loss, rmse, mae = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    return {
        "mse": float(loss), "rmse": float(rmse), "mae": float(mae),
        "train_time": float(tcb.total_training_time()), "avg_time_epoch": float(tcb.avg_epoch_time())
    }

def per_hour_metrics(y_true, y_pred, hours):
    df = pd.DataFrame({"hour": hours, "y": y_true, "yhat": y_pred})
    rows = []
    for h in range(24):
        d = df[df["hour"]==h]
        if len(d)==0: continue
        err = d["y"] - d["yhat"]
        mse = float(np.mean(err**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(err)))
        rows.append({"hour": h, "mse": mse, "rmse": rmse, "mae": mae})
    return pd.DataFrame(rows)

# ============================================================
# Federated learning (FedAvg)
# ============================================================
def sum_weights(weight_list):
    avg = []
    for layer_weights in zip(*weight_list):
        avg.append(np.mean(np.array(layer_weights, dtype=object), axis=0))
    return avg

def init_global_models(input_shape, models_to_run: list, model_cfg: dict, batch_size: int):
    builders = {
        "bilstm":   lambda: build_bilstm_model(input_shape,
                                               horizon=model_cfg["bilstm"]["horizon"],
                                               units=model_cfg["bilstm"]["units"],
                                               num_layers=model_cfg["bilstm"]["num_layers"],
                                               dropout=model_cfg["bilstm"]["dropout"],
                                               batch_size=batch_size),
        "softdense":lambda: build_soft_dense_moe_model(input_shape,
                                                       num_experts=model_cfg["softdense"]["num_experts"],
                                                       expert_units=model_cfg["softdense"]["expert_units"],
                                                       dense_units=model_cfg["softdense"]["dense_units"],
                                                       horizon=model_cfg["softdense"]["horizon"],
                                                       dropout=model_cfg["softdense"]["dropout"],
                                                       batch_size=batch_size),
        "softlstm": lambda: build_soft_bilstm_moe_model(input_shape,
                                                        num_experts=model_cfg["softlstm"]["num_experts"],
                                                        expert_units=model_cfg["softlstm"]["expert_units"],
                                                        lstm_units=model_cfg["softlstm"]["lstm_units"],
                                                        horizon=model_cfg["softlstm"]["horizon"],
                                                        dropout=model_cfg["softlstm"]["dropout"],
                                                        batch_size=batch_size),
    }
    global_models = {}
    for key in models_to_run:
        global_models[key] = builders[key]()
    return global_models

def clone_local_from_global(global_models, input_shape, models_to_run, model_cfg, batch_size):
    local = init_global_models(input_shape, models_to_run, model_cfg, batch_size)
    for k in local:
        local[k].set_weights(global_models[k].get_weights())
    return local

def federated_round(global_models, models_to_run, user_ids, data_dicts, train_cfg, model_cfg,
                    collect_per_hour: bool, y_test_hour_dict: dict):
    X_train, y_train, X_val, y_val, X_test, y_test = data_dicts
    per_user_results, per_hour_rows = [], []
    collected_weights = {k: [] for k in global_models.keys()}

    for uid in user_ids:
        input_shape = X_train[uid].shape
        local_models = clone_local_from_global(global_models, input_shape, models_to_run, model_cfg, train_cfg["batch_size"])

        for key, mdl in local_models.items():
            res = compile_fit_eval(
                mdl,
                X_train[uid], y_train[uid],
                X_val[uid],   y_val[uid],
                X_test[uid],  y_test[uid],
                max_epochs=train_cfg["max_epochs"],
                batch_size=train_cfg["batch_size"],
                patience=train_cfg["patience"],
                lr=train_cfg.get("learning_rate", 1e-3),
            )
            collected_weights[key].append(mdl.get_weights())
            nice = {"bilstm":"BiLSTM", "softdense":"SoftDenseMoE", "softlstm":"SoftLSTMMoE"}[key]
            row = {"user": uid, "architecture": nice, **res}
            per_user_results.append(row)

            if collect_per_hour:
                yhat = mdl.predict(X_test[uid], batch_size=train_cfg["batch_size"], verbose=0).squeeze()
                ph = per_hour_metrics(y_true=y_test[uid].squeeze(), y_pred=yhat, hours=y_test_hour_dict[uid])
                ph["user"] = uid
                ph["architecture"] = nice
                per_hour_rows.append(ph)

        K.clear_session()

    for key in global_models.keys():
        avg_w = sum_weights(collected_weights[key])
        global_models[key].set_weights(avg_w)

    per_hour_df = pd.concat(per_hour_rows, ignore_index=True) if per_hour_rows else pd.DataFrame()
    return per_user_results, per_hour_df

def run_federated_training(X_train, y_train, X_val, y_val, X_test, y_test,
                           models_to_run, rounds, fed_rounds, train_cfg, model_cfg,
                           collect_per_hour: bool, y_test_hour_dict: dict):
    user_ids = list(X_train.keys())
    input_shape = X_train[user_ids[0]].shape
    all_rows = []
    all_per_hour = []

    for r in range(rounds):
        global_models = init_global_models(input_shape, models_to_run, model_cfg, train_cfg["batch_size"])
        for f in range(fed_rounds):
            per_user, ph = federated_round(
                global_models,
                models_to_run,
                user_ids,
                (X_train, y_train, X_val, y_val, X_test, y_test),
                train_cfg,
                model_cfg,
                collect_per_hour=collect_per_hour,
                y_test_hour_dict=y_test_hour_dict
            )
            for row in per_user:
                row["round"] = r
                row["fed_round"] = f
            all_rows.extend(per_user)

            if collect_per_hour and not ph.empty:
                ph = ph.copy()
                ph["round"] = r
                ph["fed_round"] = f
                all_per_hour.append(ph)

            #print(f"FedAvg done: round {r}, fed_round {f}")

    res_df = pd.DataFrame(all_rows)
    ph_df = pd.concat(all_per_hour, ignore_index=True) if all_per_hour else pd.DataFrame()
    return res_df, ph_df

# ============================================================
# Clustering & noise attacks
# ============================================================
def make_random_clusters(nr_buildings: int, cluster_size: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    arr = np.arange(1, nr_buildings + 1)
    rng.shuffle(arr)
    return [list(arr[i:i+cluster_size]) for i in range(0, len(arr), cluster_size)]

def uniform_poison_user(X_train, user_key: str, scale: float = 0.2):
    X_train[user_key] = X_train[user_key] + np.random.uniform(
        low=-scale, high=scale, size=X_train[user_key].shape
    )

def gaussian_poison_user(X_train, user_key: str, scale: float = 0.2):
    X_train[user_key] = X_train[user_key] + np.random.normal(
        loc=0.0, scale=scale, size=X_train[user_key].shape
    )

# ============================================================
# GAN-style perturbation generator: training & application
# ============================================================
def build_time_mask_for_sequences(X, start_h, start_m, num_steps, step_minutes, atol=5e-4):
    """
    X shape: (N, T, F). time features at channels 1..4.
    Returns mask of shape (N, T, 1) with 1 where timestamp matches target hours.
    """
    N,T,F = X.shape
    mask = np.zeros((N, T, 1), dtype=np.float32)
    targets = []
    h, m = start_h, start_m
    for _ in range(num_steps):
        targets.append(cyc01_from_hm(h, m))
        h, m = advance_time(h, m, step_minutes)
    hs_list = [t[0] for t in targets]
    hc_list = [t[1] for t in targets]
    ms_list = [t[2] for t in targets]
    mc_list = [t[3] for t in targets]

    for i in range(N):
        # matches any of the target times
        cond_total = np.zeros((T,), dtype=bool)
        for hs,hc,ms,mc in zip(hs_list,hc_list,ms_list,mc_list):
            cond = (np.isclose(X[i,:,1], hs, atol=atol) &
                    np.isclose(X[i,:,2], hc, atol=atol) &
                    np.isclose(X[i,:,3], ms, atol=atol) &
                    np.isclose(X[i,:,4], mc, atol=atol))
            cond_total |= cond
        mask[i,:,0] = cond_total.astype(np.float32)
    return mask

def train_perturbation_generator(X, y, mask, input_shape, gan_cfg, surrogate_cfg):
    """
    Train generator G to maximize surrogate MSE on (X + delta(masked)), with L2 regularization.
    Returns the full perturbation tensor for X (same shape), to be applied on the main channel only.
    """
    # 1) Train surrogate on clean data
    surrogate = build_surrogate(input_shape, surrogate_cfg)
    surrogate.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(surrogate_cfg.get("lr",1e-3)))
    es = callbacks.EarlyStopping(monitor='loss', patience=surrogate_cfg.get("patience",3), restore_best_weights=True)
    surrogate.fit(X, y, epochs=surrogate_cfg.get("epochs",5), batch_size=gan_cfg.get("batch_size",64), verbose=0, callbacks=[es])

    # 2) Build generator
    G = build_perturbation_generator(input_shape)
    opt = tf.keras.optimizers.Adam(learning_rate=gan_cfg.get("gen_lr",1e-3))
    epsilon = float(gan_cfg.get("epsilon", 0.2))
    lam = float(gan_cfg.get("lambda_reg", 1e-3))
    bs = int(gan_cfg.get("batch_size", 64))
    steps = int(gan_cfg.get("steps", 200))

    # Dataset with mask
    ds = tf.data.Dataset.from_tensor_slices((X, y, mask)).shuffle(len(X)).batch(bs).repeat(1)

    for _ in range(steps):
        for Xb, yb, mb in ds:
            with tf.GradientTape() as tape:
                delta_full = G(Xb, training=True)                   # (B,T,F)
                delta_main = delta_full[:,:,0:1] * mb               # mask main channel
                delta_main = epsilon * tf.tanh(delta_main)          # bounded amplitude
                main_adv = tf.clip_by_value(Xb[:,:,0:1] + delta_main, 0.0, 1.0)
                Xadv = tf.concat([main_adv, Xb[:,:,1:]], axis=-1)   # leave time features intact

                yhat = surrogate(Xadv, training=False)
                mse = tf.reduce_mean(tf.square(yb - yhat))
                reg = tf.reduce_mean(tf.square(delta_main))
                # maximize mse -> minimize negative mse + lambda*reg
                loss = -mse + lam*reg

            grads = tape.gradient(loss, G.trainable_variables)
            opt.apply_gradients(zip(grads, G.trainable_variables))

    # 3) Produce final perturbations for full X
    delta_full = G.predict(X, batch_size=bs, verbose=0)
    delta_main = epsilon * np.tanh(delta_full[:,:,0:1])
    # apply mask (1 for poison, sparse for backdoor)
    delta_main = delta_main * mask
    # build full-chan perturbation (only main channel perturbed)
    delta = np.concatenate([delta_main, np.zeros_like(delta_full[:,:,1:])], axis=-1).astype(np.float32)
    return delta

def apply_delta_to_dataset(X, delta):
    """Apply delta to X (perturb only main channel and clip to [0,1])."""
    X_new = X.copy()
    adv_main = np.clip(X_new[:,:,0:1] + delta[:,:,0:1], 0.0, 1.0)
    X_new[:,:,0:1] = adv_main
    return X_new

# ============================================================
# MAIN
# ============================================================
def main(cfg: dict):
    set_seeds(cfg.get("seed", 42))

    # Paths / output
    results_dir = cfg.get("output", {}).get("results_dir", "results2")
    ensure_dir(results_dir) # Make sure directory exists
    exp_name = cfg.get("output", {}).get("experiment_name", "FL_Exp")

    # Data params
    file_path = cfg.get("data", {}).get("file_path", "../../data/3final_data/Final_Energy_dataset.csv")
    columns_filter_prefix = cfg.get("data", {}).get("columns_filter_prefix", "load")
    sequence_length = int(cfg.get("data", {}).get("sequence_length", 25))
    nr_buildings = int(cfg.get("data", {}).get("nr_buildings", 30))
    cluster_size = int(cfg.get("data", {}).get("cluster_size", 2))

    # Models & training params
    models_to_run = [m.lower() for m in cfg.get("models_to_run", ["bilstm","softdense","softlstm"])]
    model_cfg = cfg.get("model_hyperparams", {})
    train_cfg = {
        "max_epochs": int(cfg.get("train", {}).get("max_epochs", 100)),
        "batch_size": int(cfg.get("train", {}).get("batch_size", 16)),
        "patience": int(cfg.get("train", {}).get("patience", 10)),
        "learning_rate": float(cfg.get("train", {}).get("learning_rate", 1e-3)),
    }
    rounds = int(cfg.get("train", {}).get("rounds", 3))
    fed_rounds = int(cfg.get("train", {}).get("fed_rounds", 3))

    # Attack config
    attack_cfg = cfg.get("attack", {"enabled": False})
    gan_cfg = cfg.get("gan", {})

    # Make clusters and save mapping (keep for reproducibility)
    clusters = make_random_clusters(nr_buildings, cluster_size, seed=cfg.get("seed", 42))
    clusters_df = pd.DataFrame({
        "cluster_index": np.concatenate([[i+1]*len(c) for i,c in enumerate(clusters)]),
        "building": np.concatenate([c for c in clusters])
    })
    clusters_csv = os.path.join(results_dir, f"{exp_name}_clusters.csv")
    clusters_df.to_csv(clusters_csv, index=False)
    #print(f"Saved cluster assignment -> {clusters_csv}")

    # Collect only detailed results (all clusters pooled) and per-hour (optional)
    all_clusters_all_results = []
    per_cluster_ph = []

    for ci, buildings in enumerate(clusters, start=1):
        # Load & split per cluster
        df_array = load_and_prepare_data(file_path, buildings, columns_filter_prefix=columns_filter_prefix)
        step_minutes = infer_step_minutes_from_index(df_array[0].index)
        X_train, y_train, X_val, y_val, X_test, y_test, y_test_hour = split_data(
            df_array, sequence_length=sequence_length, batch_size=train_cfg["batch_size"]
        )
        
        # --- Attack on FIRST building (user1)
        poisoned_building = buildings[0]
        attack_type, attack_details = "none", "none"

        if attack_cfg.get("enabled", False):
            a_type = attack_cfg.get("type","poison").lower()
            mode   = attack_cfg.get("mode","noise").lower()

            if a_type == "poison":
                if mode == "noise":
                    dist = attack_cfg["poison"].get("distribution","uniform").lower()
                    scale = float(attack_cfg["poison"].get("scale",0.2))
                    if dist == "uniform":  uniform_poison_user(X_train, "user1", scale)
                    elif dist == "gaussian": gaussian_poison_user(X_train, "user1", scale)
                    else: raise ValueError("poison.distribution must be uniform|gaussian")
                    attack_type = "poison"
                    attack_details = f"mode=noise, distribution={dist}, scale={scale}"

                elif mode == "gan":
                    X = X_train["user1"]; y = y_train["user1"]
                    mask = np.ones((X.shape[0], X.shape[1], 1), dtype=np.float32)
                    delta = train_perturbation_generator(
                        X, y, mask, input_shape=X.shape, gan_cfg=gan_cfg, surrogate_cfg=gan_cfg.get("surrogate",{})
                    )
                    X_train["user1"] = apply_delta_to_dataset(X, delta)
                    attack_type = "poison"
                    attack_details = f"mode=gan, epsilon={gan_cfg.get('epsilon')}, lambda_reg={gan_cfg.get('lambda_reg')}, steps={gan_cfg.get('steps')}"
                else:
                    raise ValueError("attack.mode must be noise|gan")

            elif a_type == "backdoor":
                bd = attack_cfg.get("backdoor", {})
                start_h, start_m = hhmm_to_hour_min(bd.get("start_time","10:30"))
                num_steps = int(bd.get("num_steps",4))
                activate_in_test = bool(bd.get("activate_in_test", True))

                if mode == "noise":
                    noise_scale = float(bd.get("noise_scale",0.2))
                    X = X_train["user1"]
                    mask = build_time_mask_for_sequences(X, start_h, start_m, num_steps, step_minutes)
                    noise = np.random.normal(0.0, noise_scale, size=X[:,:,0:1].shape)
                    delta = np.concatenate([noise*mask, np.zeros((X.shape[0],X.shape[1],X.shape[2]-1))], axis=-1)
                    X_train["user1"] = apply_delta_to_dataset(X, delta)
                    if activate_in_test:
                        Xt = X_test["user1"]
                        mask_t = build_time_mask_for_sequences(Xt, start_h, start_m, num_steps, step_minutes)
                        noise_t = np.random.normal(0.0, noise_scale, size=Xt[:,:,0:1].shape)
                        delta_t = np.concatenate([noise_t*mask_t, np.zeros((Xt.shape[0],Xt.shape[1],Xt.shape[2]-1))], axis=-1)
                        X_test["user1"] = apply_delta_to_dataset(Xt, delta_t)
                    attack_type = "backdoor"
                    attack_details = f"mode=noise, start={bd.get('start_time')}, steps={num_steps}, noise_scale={bd.get('noise_scale')}, activate_in_test={activate_in_test}"

                elif mode == "gan":
                    X = X_train["user1"]; y = y_train["user1"]
                    mask = build_time_mask_for_sequences(X, start_h, start_m, num_steps, step_minutes)
                    delta = train_perturbation_generator(
                        X, y, mask, input_shape=X.shape, gan_cfg=gan_cfg, surrogate_cfg=gan_cfg.get("surrogate",{})
                    )
                    X_train["user1"] = apply_delta_to_dataset(X, delta)
                    if activate_in_test:
                        Xt = X_test["user1"]
                        mask_t = build_time_mask_for_sequences(Xt, start_h, start_m, num_steps, step_minutes)
                        delta_t = train_perturbation_generator(
                            Xt, y_test["user1"], mask_t, input_shape=Xt.shape, gan_cfg=gan_cfg, surrogate_cfg=gan_cfg.get("surrogate",{})
                        )
                        X_test["user1"] = apply_delta_to_dataset(Xt, delta_t)
                    attack_type = "backdoor"
                    attack_details = f"mode=gan, start={bd.get('start_time')}, steps={num_steps}, epsilon={gan_cfg.get('epsilon')}, lambda_reg={gan_cfg.get('lambda_reg')}, activate_in_test={activate_in_test}"
                else:
                    raise ValueError("attack.mode must be noise|gan")
            else:
                raise ValueError("attack.type must be poison|backdoor")

        attack_label = f"{attack_type}({attack_details})"
        print(f"[Cluster {ci}] buildings={buildings} | attack={attack_label}")

        # Train FL for this cluster
        collect_per_hour = (attack_type == "backdoor")
        cluster_results, cluster_per_hour = run_federated_training(
            X_train, y_train, X_val, y_val, X_test, y_test,
            models_to_run=models_to_run,
            rounds=rounds, fed_rounds=fed_rounds,
            train_cfg=train_cfg, model_cfg=model_cfg,
            collect_per_hour=collect_per_hour, y_test_hour_dict=y_test_hour
        )

        # Map 'user' -> building id and add metadata
        user_to_building = {f"user{k+1}": b for k,b in enumerate(buildings)}
        cluster_results = cluster_results.copy()
        cluster_results["building"] = cluster_results["user"].map(user_to_building)
        cluster_results["user_key"] = cluster_results["user"]
        cluster_results.drop(columns=["user"], inplace=True)

        cluster_id = f"C{ci:02d}"
        cluster_str = "-".join(str(b) for b in buildings)
        cluster_results["cluster_index"] = ci
        cluster_results["cluster_id"] = cluster_id
        cluster_results["cluster_buildings"] = cluster_str
        cluster_results["poisoned_building"] = poisoned_building
        cluster_results["attack_type"] = attack_type
        cluster_results["attack_details"] = attack_details
        cluster_results["attack"] = attack_label
        cluster_results["experiment"] = exp_name

        all_clusters_all_results.append(cluster_results)

        # keep per-hour only in memory; save once after all clusters
        if collect_per_hour and not cluster_per_hour.empty:
            cluster_per_hour = cluster_per_hour.copy()
            cluster_per_hour["cluster_index"] = ci
            cluster_per_hour["cluster_id"] = cluster_id
            cluster_per_hour["cluster_buildings"] = cluster_str
            cluster_per_hour["poisoned_building"] = poisoned_building
            cluster_per_hour["attack_type"] = attack_type
            cluster_per_hour["attack_details"] = attack_details
            cluster_per_hour["attack"] = attack_label
            cluster_per_hour["experiment"] = exp_name
            per_cluster_ph.append(cluster_per_hour)

    # --- Single detailed CSV across all clusters
    combined_all = pd.concat(all_clusters_all_results, ignore_index=True) if all_clusters_all_results else pd.DataFrame()
    combined_all_file = os.path.join(results_dir, f"{exp_name}_all_results.csv")
    combined_all.to_csv(combined_all_file, index=False)
    print(f"Saved -> {combined_all_file}")

    # --- Optional per-hour CSV across all clusters (if backdoor produced it)
    all_ph = None
    if per_cluster_ph:
        all_ph = pd.concat(per_cluster_ph, ignore_index=True)
        all_ph_file = os.path.join(results_dir, f"{exp_name}_per_hour_results.csv")
        all_ph.to_csv(all_ph_file, index=False)
        print(f"Saved -> {all_ph_file}")

    # return only the detailed frames you keep
    return combined_all, all_ph

def default_cfg():
    return {
    "seed": 42,
    "data": {
        "file_path": "../../data/3final_data/Final_Energy_dataset.csv",  # fallback: "../../data/3final_data/Final_Energy_dataset.csv"
        "columns_filter_prefix": "load",  # choices: "load" | "pv" | "prosumption" ; fallback: "load"
        "sequence_length": 25,            # fallback: 25
        "nr_buildings": 10,                # fallback: 30
        "cluster_size": 2                 # fallback: 2
    },
    "models_to_run": ["bilstm","softdense","softlstm"],  # choices: any subset of {"bilstm","softdense","softlstm"} ; fallback: ["bilstm","softdense","softlstm"]
    "model_hyperparams": {
        "bilstm":   { "horizon": 1, "units": 8, "num_layers": 2, "dropout": 0.2 },
        "softdense":{"horizon": 1, "num_experts": 4, "expert_units": 8, "dense_units": 16, "dropout": 0.2 },
        "softlstm": { "horizon": 1, "num_experts": 4, "expert_units": 8, "lstm_units": 4, "dropout": 0.2 }
    },
    "train": {
        "max_epochs": 50,            # fallback: 100
        "batch_size": 256,          # fallback: 16
        "patience": 10,             # fallback: 10
        "learning_rate": 1e-3,      # fallback: 1e-3
        "rounds": 3,                # fallback: 3
        "fed_rounds": 3             # fallback: 3
    },
    "attack": {
        "enabled": True,
        "type": "poison",            # "poison" or "backdoor"
        "mode": "noise",                 # "noise" or "gan"
        "poison": {                    # used when type="poison" and mode="noise"
            "distribution": "uniform", # "uniform" or "gaussian"
            "scale": 0.2
        },
        "backdoor": {                  # used when type="backdoor"
            "start_time": "10:30",
            "num_steps": 4,
            "noise_scale": 0.2,        # used only when mode="noise"
            "activate_in_test": False
        }
    },
    "gan": {
        "epsilon": 0.2,          # max per-step magnitude after tanh squashing
        "lambda_reg": 1e-4,      # perturbation magnitude regularizer
        "steps": 50,            # generator steps (epochs over batches)
        "batch_size": 64,        # generator mini-batch
        "gen_lr": 1e-3,
        "surrogate": {           # small forecaster used to guide the generator
            "epochs": 30,
            "units": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "lr": 1e-3,
            "patience": 3
        }
    },
    "output": {
        "results_dir": "results2",
        "experiment_name": "Tes"
    }
}

def make_cfg(columns="load", nr_buildings=10, cluster_size=2, experiment_name="Test",
    attack_type="poison",   # "poison" or "backdoor"
    attack_mode="noise",    # "noise" or "gan"
    scale=0.2,              # sets poison.scale, backdoor.noise_scale, GAN epsilon
):
    cfg = default_cfg() 

    cfg["data"]["columns_filter_prefix"] = columns
    cfg["data"]["nr_buildings"] = int(nr_buildings)
    cfg["data"]["cluster_size"] = int(cluster_size)

    cfg["output"]["experiment_name"] = experiment_name

    cfg["attack"]["type"] = attack_type
    cfg["attack"]["mode"] = attack_mode
    
    cfg["attack"]["poison"]   = {"scale": float(scale)}
    cfg["attack"]["backdoor"] = {"noise_scale": float(scale)}
    cfg["gan"]["epsilon"] = float(scale)

    return cfg