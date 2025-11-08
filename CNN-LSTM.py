import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D,concatenate, LeakyReLU,BatchNormalization,GlobalAveragePooling1D,Dropout,Input,Dense,MaxPooling1D
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
import os
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import argparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from tensorflow.keras.backend import int_shape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Huber

# 固定隨機種子
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# 讀取與處理資料
train_data = pd.read_csv('dataset\\EUA_Train.csv')
val_data = pd.read_csv('dataset\\EUA_Val.csv')
test_data = pd.read_csv('dataset\\EUA_Test.csv')
model_name="CNN-LSTM-base"
output_root='./finish'

def add_other_feature(df):
    df= df.copy()
    df["VIX_Open_lag1"] = df["VIX_Open"].shift(1)
    df["VIX_High_lag1"] = df["VIX_High"].shift(1)
    df["VIX_Low_lag1"] = df["VIX_Low"].shift(1)
    df["VIX_Close_lag1"] = df["VIX_Close"].shift(1)

    df["DJIA_Open_lag1"] = df["DJIA_Open"].shift(1)
    df["DJIA_Close_lag1"] = df["DJIA_Close"].shift(1)
    df["DJIA_High_lag1"] = df["DJIA_High"].shift(1)
    df["DJIA_Low_lag1"] = df["DJIA_Low"].shift(1)

    df["DJCIEN_Open_log"]=np.log1p(df["DJCIEN_Open"])
    df["DJCIEN_Close_log"]=np.log1p(df["DJCIEN_Close"])
    df["DJCIEN_Low_log"]=np.log1p(df["DJCIEN_Low"])
    df["DJCIEN_High_log"]=np.log1p(df["DJCIEN_High"])
    
    df["STOXX_Open_log"]=np.log1p(df["STOXX_Open"])
    df["STOXX_Close_log"]=np.log1p(df["STOXX_Close"])
    df["STOXX_High_log"]=np.log1p(df["STOXX_High"])
    df["STOXX_Low_log"]=np.log1p(df["STOXX_Low"])
    
    df["NG_Open_log"] = np.log1p(df["NG_Open"])
    df["NG_High_log"] = np.log1p(df["NG_High"])
    df["NG_Low_log"] = np.log1p(df["NG_Low"])
    df["NG_Close_log"] = np.log1p(df["NG_Close"])
    
    df["TTF_Open_log"] = np.log1p(df["TTF_Open"])
    df["TTF_High_log"] = np.log1p(df["TTF_High"])
    df["TTF_Low_log"] = np.log1p(df["TTF_Low"])
    df["TTF_Close_log"] = np.log1p(df["TTF_Close"])
    
    return df.dropna().reset_index(drop=True)

train_data = add_other_feature(train_data)
val_data   = add_other_feature(val_data)
test_data  = add_other_feature(test_data)

# 特徵選擇
feature_sets = {
    #  'test':['EUA_Open', 'EUA_High', 'EUA_Low','EUA_Close'],
    'EUA': ['EUA_Open', 'EUA_High', 'EUA_Low','EUA_Close'],
    'EUA+VIX': ['EUA_Open', 'EUA_High', 'EUA_Low', 'EUA_Close','VIX_Open_lag1','VIX_High_lag1','VIX_Low_lag1','VIX_Close_lag1',],
    'EUA+STOXX':['EUA_Open', 'EUA_High', 'EUA_Low','EUA_Close','STOXX_Open_log','STOXX_High_log','STOXX_Low_log','STOXX_Close_log'],
    'EUA+OIL':['EUA_Open', 'EUA_High', 'EUA_Low','EUA_Close','BRENT_Open','BRENT_High','BRENT_Low','BRENT_Close'],
    'EUA+GAS':['EUA_Open', 'EUA_High', 'EUA_Low','EUA_Close','TTF_Open_log','TTF_High_log','TTF_Low_log','TTF_Close_log'],
    'EUA+DJIA':['EUA_Open', 'EUA_High', 'EUA_Low','EUA_Close','DJIA_Open_lag1','DJIA_High_lag1','DJIA_Low_lag1','DJIA_Close_lag1'],
    'EUA+DJCIEN':['EUA_Open', 'EUA_High', 'EUA_Low','EUA_Close','DJCIEN_Open_log','DJCIEN_High_log','DJCIEN_Low_log','DJCIEN_Close_log'],
    'EUA+NG':['EUA_Open', 'EUA_High', 'EUA_Low','EUA_Close','NG_Open_log','NG_High_log','NG_Low_log','NG_Close_log'],
    'ALL':['EUA_Open', 'EUA_High', 'EUA_Low','EUA_Close',
           'BRENT_Open','BRENT_High','BRENT_Low','BRENT_Close',
           'TTF_Open_log','TTF_High_log','TTF_Low_log','TTF_Close_log',
           'VIX_Open_lag1','VIX_High_lag1','VIX_Low_lag1','VIX_Close_lag1',
           'DJCIEN_Open_log','DJCIEN_High_log','DJCIEN_Low_log','DJCIEN_Close_log',
           'DJIA_Open_lag1','DJIA_High_lag1','DJIA_Low_lag1','DJIA_Close_lag1',
            'STOXX_Open_log','STOXX_High_log','STOXX_Low_log','STOXX_Close_log',
            'NG_Open_log','NG_High_log','NG_Low_log','NG_Close_log'
           ],
    
    }



def create_sequences(data,target,time_steps):
    X,y = [],[]
    for i in range(len(data)-time_steps):
        X.append(data[i:i+time_steps, :])
        y.append(target[i+time_steps])

    return np.array(X),np.array(y)

def pad_sequences(sequences,dtype='float32',value=np.nan):
    max_len=max(len(seq) for seq in sequences)
    padded = np.full((len(sequences),max_len),value,dtype=dtype)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)]=seq
    return padded

# 超參數設置
units = 50
epoch = 150
batch = 16
loop_times = 5
dropout_rate = 0.25
learning_rate=0.001
start, end = 1,21

for dataset_name, features in feature_sets.items():
    print(f"\nRunning Feature set: {dataset_name}")

    base_dir = os.path.join(output_root, model_name, dataset_name)
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    csv_dir = os.path.join(base_dir, 'csv')
    pic_dir = os.path.join(base_dir, 'pic')
    for d in [checkpoint_dir, csv_dir, pic_dir]:
        os.makedirs(d, exist_ok=True)

    # 資料處理
    # train_features = train_data[features].values
    # train_target = train_data['EUA_Close'].values.reshape(-1, 1)
    # val_features = val_data[features].values
    # val_target = val_data['EUA_Close'].values.reshape(-1, 1)
    # test_features = test_data[features].values
    # test_target = test_data['EUA_Close'].values.reshape(-1, 1)

    train_features = train_data[features].values
    train_target = train_data['EUA_Open'].values.reshape(-1, 1)
    val_features = val_data[features].values
    val_target = val_data['EUA_Open'].values.reshape(-1, 1)
    test_features = test_data[features].values
    test_target = test_data['EUA_Open'].values.reshape(-1, 1)

    pipe_X = MinMaxScaler()
    pipe_y = MinMaxScaler()

    scaler_train_features = pipe_X.fit_transform(train_features)
    scaler_val_features = pipe_X.transform(val_features)
    scaler_test_features = pipe_X.transform(test_features)

    scaler_train_target = pipe_y.fit_transform(train_target)
    scaler_val_target = pipe_y.transform(val_target)
    scaler_test_target = pipe_y.transform(test_target)

    train_loss = {t: [] for t in range(start, end)}
    train_val_loss = {t: [] for t in range(start, end)}
    all_avg_predictions = {}
    actual_test_prices_dict = {}
    best_rmse = {t: float('inf') for t in range(start, end)}

    for t in range(start, end):
        print(f"  Time step: {t}")
        model_path = os.path.join(checkpoint_dir, f'best_model_{model_name}_t{t}.h5')
        metrics = {k: [] for k in ['val_rmse', 'val_mae', 'val_r2', 'val_mse', 'val_mape',''
                                   'test_rmse', 'test_mae', 'test_r2', 'test_mse', 'test_mape']}
        pred_list = []

        X_train, y_train = create_sequences(scaler_train_features, scaler_train_target, t)
        X_val, y_val = create_sequences(scaler_val_features, scaler_val_target, t)
        X_test, y_test = create_sequences(scaler_test_features, scaler_test_target, t)

        for i in range(loop_times):
            tf.keras.backend.clear_session()      # 釋放舊 graph
            tf.keras.utils.set_random_seed(42)    # 每回都用同一組種子
            input_layer=Input(shape=(X_train.shape[1], X_train.shape[2]), name='input_layer')
            x = Conv1D(32, 3, padding='causal', dilation_rate=1,activation='gelu')(input_layer)
            # x = Conv1D(64, 3, padding='causal', dilation_rate=2,activation='gelu')(x)
            # x = GlobalAveragePooling1D()(x)
            # x = tf.keras.layers.RepeatVector(1)(x)
            # GRU
            lstm_layer = LSTM(units, activation='tanh',  return_sequences=False, name='lstm_layer')(x)
            # dense=Dense(64)(lstm_layer)
            x = tf.keras.layers.ELU()(lstm_layer)
            output_layer = Dense(1, activation='linear', name='output_layer')(x)

            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=Huber(), metrics=['mse','mae'])
            early_stopping = EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True)
            # checkpoint=ModelCheckpoint(filepath=model_path,filename='best',monitor='val_loss',save_best_only=True,mode='min',verbose=1)
            history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch,
                                validation_data=(X_val, y_val), verbose=1,
                                callbacks=[early_stopping],
                                )

            train_loss[t].append(history.history['loss'])
            train_val_loss[t].append(history.history['val_loss'])

            val_pred = pipe_y.inverse_transform(model.predict(X_val))
            val_true = pipe_y.inverse_transform(y_val)

            val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
            val_mae = mean_absolute_error(val_true, val_pred)
            val_r2 = r2_score(val_true, val_pred)
            val_mse = mean_squared_error(val_true, val_pred)
            val_mape = np.mean(np.abs((val_true - val_pred) / val_true)) * 100

            test_pred = pipe_y.inverse_transform(model.predict(X_test))
            test_true = pipe_y.inverse_transform(y_test)

            test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
            test_mae = mean_absolute_error(test_true, test_pred)
            test_r2 = r2_score(test_true, test_pred)
            test_mse = mean_squared_error(test_true, test_pred)
            test_mape = np.mean(np.abs((test_true - test_pred) / test_true)) * 100

            pred_list.append(test_pred.flatten())

            for k, v in zip(['val_rmse', 'val_mae', 'val_r2', 'val_mse', 'val_mape',''
                             'test_rmse', 'test_mae', 'test_r2', 'test_mse', 'test_mape'],
                            [val_rmse, val_mae, val_r2, val_mse, val_mape,
                             test_rmse, test_mae, test_r2, test_mse, test_mape]):
                metrics[k].append(v)

        # 儲存平均預測與實際值
        avg_pred = np.mean(pred_list, axis=0)
        all_avg_predictions[t] = np.round(avg_pred, 2)
        actual_test_prices_dict[t] = test_true.flatten()

        # 儲存指標 CSV
        df = pd.DataFrame([[k] + v + [np.mean(v)] for k, v in metrics.items()],
                          columns=['Metric'] + [f'Loop_{i+1}' for i in range(loop_times)] + ['Avg'])
        df.to_csv(os.path.join(csv_dir, f'results_t{t}-{dataset_name}.csv'), index=False)

        prediction_df = pd.DataFrame({
            'Date': pd.to_datetime(test_data['Date'].values[-len(avg_pred):]),
            'Actual': actual_test_prices_dict[t],
            'Predicted': all_avg_predictions[t]
        })
        prediction_df.to_csv(os.path.join(csv_dir, f'prediction_t{t}-{dataset_name}.csv'), index=False)

    # loss 圖
    plt.figure(figsize=(18, 8))
    for t in train_loss:
        if train_loss[t]:
            pad = pad_sequences(train_loss[t])
            avg = np.nanmean(pad, axis=0)
            plt.plot(avg, label=f'Train t={t}')
    for t in train_val_loss:
        if train_val_loss[t]:
            pad = pad_sequences(train_val_loss[t])
            avg = np.nanmean(pad, axis=0)
            plt.plot(avg, linestyle='--', label=f'Val t={t}')
    plt.legend()
    plt.title(f'{dataset_name} Loss')
    plt.savefig(os.path.join(pic_dir, 'loss.png'))
    plt.close()

    # 預測圖
    for t, pred in all_avg_predictions.items():
        actual = actual_test_prices_dict[t]
        date = pd.to_datetime(test_data['Date'].values[-len(actual):])
        plt.figure(figsize=(20, 6))
        plt.plot(date, actual, label='Actual')
        plt.plot(date, pred, label='Predicted')
        plt.title(f'{dataset_name} - Time {t}')
        plt.legend()
        plt.savefig(os.path.join(pic_dir, f'pred_t{t}.png'))
        plt.close()

print("\n✅ 所有實驗與儲存完成！")