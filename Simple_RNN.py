
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from  tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,SimpleRNN
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import warnings
warnings.filterwarnings("ignore")
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



df = pd.read_csv('archive/international-airline-passengers.csv')
df.head()


df.columns
df.columns = ['Month', 'Passengers']

df.tail()

df.shape
df.dtypes

df.isnull().sum()



df= df[:144]
df.tail()

df.info()

df['Month']= pd.to_datetime(df['Month'])
df.info()



df.index = df['Month']
df.head()
df.drop('Month', axis=1, inplace=True)

result_df = df.copy()

df.plot(figsize=(14, 8), title='Monthly airline passengers');
plt.show()



data = df['Passengers'].values
data[0:5]



data = data.astype('float32')

data.shape



data = data.reshape(-1, 1)
data.shape



def split_data(dataframe, test_size):
    position = int(round(len(dataframe) * (1-test_size))) #
    train = dataframe[:position]
    test= dataframe[position:]
    return train,test,position

train, test, position = split_data(data,0.33)
train.shape
test.shape



scaler_train = MinMaxScaler(feature_range=(0, 1))
train = scaler_train.fit_transform(train)
scaler_test = MinMaxScaler(feature_range=(0, 1))
test = scaler_test.fit_transform(test)

test[0:5]
train[0:5]




def create_features(data, lookback):
    X, Y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        Y.append(data[i,0])
    return np.array(X), np.array(Y)

lookback = 1

X_train, y_train = create_features(train, lookback)


X_test, y_test = create_features(test, lookback)


X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

X_train.shape




model = Sequential()

model.add(SimpleRNN(units=50,
                    activation='relu',
                    input_shape=(X_train.shape[1], lookback)))

model.add(Dropout(0.2))


model.add(Dense(1))

model.summary()




model.compile(loss='mean_squared_error', optimizer='adam')

callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min'),
             ModelCheckpoint(filepath='mymodel.h5', monitor='val_loss', mode='min',
                             save_best_only=True, save_weights_only=False, verbose=1)]

history = model.fit(x = X_train,
                    y = y_train,
                    epochs=50,
                    batch_size=1,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    shuffle=False)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch', fontsize=16)
plt.xlabel('Loss', fontsize=16)
plt.ylim([0, max(plt.ylim())])
plt.title('Training and Validation Loss', fontsize=16)
plt.show()



loss = model.evaluate(X_test, y_test, batch_size=1)


train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler_train.inverse_transform(train_predict)
test_predict = scaler_test.inverse_transform(test_predict)

y_train = scaler_train.inverse_transform(y_train)
y_test = scaler_test.inverse_transform(y_test)


train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))


test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))

print(f'train rmse : {train_rmse}')
print(f'test rmse : {test_rmse}')

df.describe().T


train_prediction_df = result_df[lookback:position]
train_prediction_df['Predicted'] = train_predict
train_prediction_df.head()

test_prediction_df = result_df[position+lookback:]
test_prediction_df['Predicted'] = test_predict
test_prediction_df.head()

