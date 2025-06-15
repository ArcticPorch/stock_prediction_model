


df = pd.DataFrame(Data)
df1 =df.copy()
df
df.describe().T
df.info()






df






df['Date'] =df['date'].str.split(' ').str.get(0)
df






df.drop(columns=['date','symbol'],inplace=True)
df['Date']= pd.to_datetime(df['Date'])
df = df.set_index('Date')
df






font1 = {'family':'serif','size':18}
font2 = {'family':'serif','size':15}
font3 = {'family':'serif','size':13}






colors =['blue','Red', 'Yellow','turquoise','blue','Red', 'Yellow','turquoise', 'blue','Red', 'Yellow','turquoise']
colors= ['lightskyblue' , 'lightpink' , 'cadetblue','lightskyblue' , 'lightpink' , 'cadetblue','lightskyblue' , 'lightpink' , 'cadetblue','lightskyblue' , 'lightpink' , 'cadetblue']
f = plt.figure()
f.set_figwidth(20)
f.set_figheight(40)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.5)
i=1
for column in df.columns:
    plt.subplot(6,2,i)
    plt.plot(df[column], color=colors[i-1])
    plt.title(column,backgroundcolor='grey',color='white',fontdict=font1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Year',fontdict=font2,labelpad=15)
    plt.ylabel("Price",fontdict=font2,labelpad=15)
    plt.grid()
    i+=1


# 





df2 = df.drop(columns=['volume','divCash','splitFactor','adjVolume'])
df2






fig,ax = plt.subplots(figsize=(15,10))
df2.plot(ax = ax,alpha=0.5)
ax.set_title('Stock Price',backgroundcolor='grey',color='white',fontdict=font1)
ax.set_xlabel('Year',fontdict=font2,labelpad=15)
ax.set_ylabel("Price",fontdict=font2,labelpad=15)
ax.grid()






df1['Date'] =df1['date'].str.split(' ').str.get(0)
df1.drop(columns=['symbol','date','divCash','splitFactor'],inplace=True)
df_2016 = df1[(df1['Date']>='2016-01-01') &(df1['Date']<='2016-12-31')]
df_2016['Date']= pd.to_datetime(df_2016['Date'])
df_2016 = df_2016.set_index('Date')






df_2021 = df1[(df1['Date']>='2021-01-01') & (df1['Date'] <='2021-12-31')]
df_2021['Date'] = pd.to_datetime(df_2021['Date'])
df_2021 = df_2021.set_index('Date')






d2016 = df_2016.resample(rule ='MS').mean()
d2021 = df_2021.resample(rule ='MS').mean()






fig,(ax1,ax2) = plt.subplots(2,1,figsize=(15,10))
d2016[['close','high','low','open']].plot(ax=ax1)
ax1.set_title('Year 2016 Analysis ',backgroundcolor='grey',color='white',fontdict=font2)
d2021[['close','high','low','open']].plot(ax=ax2)
ax2.set_title('Year 2021 Analysis ',backgroundcolor='grey',color='white',fontdict=font2)

ax2.set_xlabel('Month',fontdict=font3)
ax1.set_xlabel('Month',fontdict=font3)






f = plt.figure()
f.set_figwidth(15)
f.set_figheight(40)
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.5)
i = 1
for column in df.drop(columns=['volume','divCash','splitFactor','adjVolume']).columns:
    plt.subplot(6,2,i)
    ax = df[column].resample('A').mean().plot.bar(color =['lightskyblue' , 'lightpink' , 'cadetblue','lightskyblue' , 'lightpink' , 'cadetblue'])
    plt.xticks(rotation = 45,fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.title(f'Yearly end Mean {column} Price',backgroundcolor='grey',color='white',fontdict=font1)
    plt.xlabel('Date',fontdict=font2,labelpad=15)
    ax.yaxis.grid(True)
    i+=1






W6 = df.rolling(window=6).mean()
W30 = df.rolling(window=30).mean()
W60 = df.rolling(window=60).mean()
plt.figure(figsize=(12, 9))
df['close'].plot(label='Close Price').autoscale(axis='x',tight=True)
W6['close'].plot(label='Business days rolling').autoscale(axis='x',tight=True)
W30['close'].plot(label='30 Days rolling').autoscale(axis='x',tight=True)
W60['close'].plot(label='60 Days rolling').autoscale(axis='x',tight=True)
plt.legend()
plt.title('Moving Averages Analysis',backgroundcolor='grey',color='white',fontdict=font2, fontweight='bold')
plt.xlabel('Date',fontdict=font3,labelpad=15)
plt.ylabel('Price',fontdict=font3,labelpad=15)
plt.grid(True)
plt.tight_layout()
plt.show()






DF = df[['close','high','low','open']]






scaler = MinMaxScaler()
DF[DF.columns]= scaler.fit_transform(DF)
DF.shape






training_size = round(len(DF)*0.80)
train_data = DF.iloc[:training_size,0:4]
test_data = DF.iloc[training_size:,0:4]
print(train_data.shape,test_data.shape)






def prepare_time_series_data(Data,window_size):
  sequences =[]
  labels =[]
  i = 0
  for j   in range(window_size,len(Data)):
    sequences.append(Data.iloc[i:j])
    labels.append(Data.iloc[j])
    i+=1
  return np.array(sequences),np.array(labels)






X_train ,y_train = prepare_time_series_data(train_data,60)
X_test ,y_test = prepare_time_series_data(test_data,60)
X_train.shape,y_train.shape,X_test.shape,y_test.shape






length = 60
LSTM1  = Sequential()
LSTM1.add(LSTM(100,return_sequences=True,input_shape=(length,X_train.shape[2])))
LSTM1.add(Dropout(0.2))
LSTM1.add(LSTM(100,return_sequences = False,input_shape=(length,X_train.shape[2])))
LSTM1.add(Dropout(0.2))
LSTM1.add(Dense(4))
LSTM1.compile(optimizer='adam',loss='mse',metrics=['mae'])
LSTM1.summary()







early_stop = EarlyStopping(monitor='loss',patience=5)






LSTM1.fit(X_train,y_train,epochs=30,validation_data =(X_test,y_test),batch_size=32,callbacks=[early_stop])






LSTM1.history.history.keys()






title = 'Loss and Mean_absolute-error over Epochs'
xlabel='Epochs'
LSTM1_losses = pd.DataFrame(LSTM1.history.history)
ax = LSTM1_losses.plot(figsize=(10,6),title =title)
ax.autoscale(axis = 'x',tight=True)
ax.set(xlabel=xlabel);






def highlight_best(data):
  data_highlighted= data.copy()
  min_loss = data_highlighted['loss'].min()
  min_mae = data_highlighted['mae'].min
  min_val_loss = data_highlighted['val_loss'].min()
  min_val_mae = data_highlighted['val_mae'].min()
  min_loass = data_highlighted['loss']==min_loss
  min_mae = data_highlighted['mae']==min_mae
  min_val_loss = data_highlighted['val_loss']==min_val_loss
  min_val_mae = data_highlighted['val_mae']==min_val_mae

  data_highlighted.style.apply(lambda x: ['background: yellow' if v else '' for v in min_loss], subset=['loss']) \
                  .apply(lambda x: ['background: yellow' if v else '' for v in min_mae], subset=['mae']) \
                  .apply(lambda x: ['background: yellow' if v else '' for v in min_val_loss], subset=['val_loss']) \
                  .apply(lambda x: ['background: yellow' if v else '' for v in min_val_mae], subset=['val_mae'])
  return data_highlighted






def highlight_best(data):
    data_highlighted = data.copy()

    min_loss = data_highlighted['loss'].min()
    min_mae = data_highlighted['mae'].min()
    min_val_loss = data_highlighted['val_loss'].min()
    min_val_mae = data_highlighted['val_mae'].min()

    data_highlighted['loss_highlight'] = data_highlighted['loss'] == min_loss
    data_highlighted['mae_highlight'] = data_highlighted['mae'] == min_mae
    data_highlighted['val_loss_highlight'] = data_highlighted['val_loss'] == min_val_loss
    data_highlighted['val_mae_highlight'] = data_highlighted['val_mae'] == min_val_mae

    # Apply styling based on boolean masks
    data_highlighted = data_highlighted.style.applymap(lambda v: 'background: yellow' if v else '', subset=['loss_highlight', 'mae_highlight', 'val_loss_highlight', 'val_mae_highlight'])

    return data_highlighted






highlighted_LSTM1_losses = highlight_best(LSTM1_losses)
highlighted_LSTM1_losses






def predict_and_inverse_transform(DF,X_test,model,scaler):
  test = DF.iloc[-len(X_test):].copy()
  predictions = model.predict(X_test)
  inverse_predictions = scaler.inverse_transform(predictions)
  inverse_predictions = pd.DataFrame(inverse_predictions,columns=['Predicted Close','Predicted High','Predicted Low','Predicted Open'],index= DF.iloc[-len(X_test):].index)
  test_df =pd.concat([test.copy(),inverse_predictions],axis = 1)
  test_df[['close','high','low','open']] = scaler.inverse_transform(test_df[['close','high','low','open']])
  return test_df






test_df = predict_and_inverse_transform(DF,X_test,LSTM1,scaler)






plt.figure(figsize=(10,6))
test_df['close'].plot(label='Close Price').autoscale(axis='x',tight=True)
test_df['Predicted Close'].plot(label='Predicted Close Price').autoscale(axis='x',tight=True)
plt.legend()
plt.title('Comparison of Actual and Predicted Close Prices ',backgroundcolor='grey',color='white',fontdict=font2, fontweight='bold')
plt.xlabel('Date',fontdict=font3,labelpad=15)
plt.ylabel('Price',fontdict=font3,labelpad=15)
plt.tight_layout()
plt.grid(True)






LSTM2 = Sequential()
LSTM2.add(LSTM(150,input_shape=(length,X_train.shape[2]),return_sequences=True))
LSTM2.add(Dropout(0.2))
LSTM2.add(LSTM(100,input_shape=(length,X_train.shape[2]),return_sequences=True))
LSTM2.add(Dropout(0.2))
LSTM2.add(LSTM(100,input_shape=(length,X_train.shape[2]),return_sequences=False))
LSTM2.add(Dropout(0.2))
LSTM2.add(Dense(units=50))
LSTM2.add(Dense(units=5))
LSTM2.add(Dense(X_train.shape[2]))
LSTM2.compile(optimizer='adam',loss='mse',metrics=['mae'])
LSTM2.summary()






LSTM2.fit(X_train, y_train,epochs=30,validation_data=(X_test, y_test),batch_size = 32,callbacks=[early_stop],verbose=1)






title=' Loss and Mean Absolute Error vs. Epochs '
xlabel=' Epochs '
LSTM2_losses = pd.DataFrame(LSTM2.history.history)

ax = LSTM2_losses.plot(figsize=(10,6),title=title)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel);






highlighted_LSTM2_losses = highlight_best(LSTM2_losses)
highlighted_LSTM2_losses






test_df2 = predict_and_inverse_transform(DF,X_test,LSTM2,scaler)






plt.figure(figsize=(10, 6))
test_df2['close'].plot(label='Close Price').autoscale(axis='x',tight=True)
test_df2['Predicted Close'].plot(label='Predicted Close Price').autoscale(axis='x',tight=True)

plt.legend()
plt.title('Comparison of Actual and Predicted Close Prices',backgroundcolor='grey',color='white',fontdict=font2, fontweight='bold')
plt.xlabel('Date',fontdict=font3,labelpad=15)
plt.ylabel('Price',fontdict=font3,labelpad=15)
plt.grid(True)
plt.tight_layout()
plt.show()






GRU_Model = Sequential()
GRU_Model.add(GRU(128,input_shape=(length,X_train.shape[2]),activation='tanh'))
GRU_Model.add(Dense(X_train.shape[2]))
GRU_Model.compile(optimizer='adam',loss='mse',metrics=['mae'])
GRU_Model.summary()






GRU_Model.fit(X_train, y_train, epochs=30,validation_data=(X_test, y_test),batch_size = 32,callbacks=[early_stop],verbose=1)






title=' Loss and Mean Absolute Error vs. Epochs '
xlabel=' Epochs '
GRU_losses = pd.DataFrame(GRU_Model.history.history)

ax = GRU_losses.plot(figsize=(10,6),title=title)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel);






highlighted_GRU_losses = highlight_best(GRU_losses)
highlighted_GRU_losses






test_df3 = predict_and_inverse_transform(DF, X_test, GRU_Model, scaler)






plt.figure(figsize=(10, 6))
test_df3['close'].plot(label='Close Price').autoscale(axis='x',tight=True)
test_df3['Predicted Close'].plot(label='Predicted Close Price').autoscale(axis='x',tight=True)

plt.legend()
plt.title('Comparison of Actual and Predicted Close Prices',backgroundcolor='grey',color='white',fontdict=font2, fontweight='bold')
plt.xlabel('Date',fontdict=font3,labelpad=15)
plt.ylabel('Price',fontdict=font3,labelpad=15)
plt.grid(True)
plt.tight_layout()






plt.figure(figsize=(10, 6))
test_df3['high'].plot(label='High Price').autoscale(axis='x',tight=True)
test_df3['Predicted High'].plot(label='Predicted High Price').autoscale(axis='x',tight=True)

plt.legend()
plt.title('Comparison of Actual and Predicted High Prices',backgroundcolor='grey',color='white',fontdict=font2, fontweight='bold')
plt.xlabel('Date',fontdict=font3,labelpad=15)
plt.ylabel('Price',fontdict=font3,labelpad=15)
plt.grid(True)
plt.tight_layout()






plt.figure(figsize=(10, 6))
test_df3['low'].plot(label='Low Price').autoscale(axis='x',tight=True)
test_df3['Predicted Low'].plot(label='Predicted Low Price').autoscale(axis='x',tight=True)

plt.legend()
plt.title('Comparison of Actual and Predicted Low Prices',backgroundcolor='grey',color='white',fontdict=font2, fontweight='bold')
plt.xlabel('Date',fontdict=font3,labelpad=15)
plt.ylabel('Price',fontdict=font3,labelpad=15)
plt.grid(True)
plt.tight_layout()






plt.figure(figsize=(10, 6))
test_df3['open'].plot(label='Open Price').autoscale(axis='x',tight=True)
test_df3['Predicted Open'].plot(label='Predicted Open Price').autoscale(axis='x',tight=True)

plt.legend()
plt.title('Comparison of Actual and Predicted Open Prices',backgroundcolor='grey',color='white',fontdict=font2, fontweight='bold')
plt.xlabel('Date',fontdict=font3,labelpad=15)
plt.ylabel('Price',fontdict=font3,labelpad=15)
plt.grid(True)
plt.tight_layout()


# **NEXT**

# NEW **MODEL**





import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pi
import plotly.express as px
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta






daily_changes = df['close'].diff()
fig = px.histogram(daily_changes, nbins=50, title='Histogram of Daily Price Changes')
fig.update_xaxes(title='Daily Price Change')
fig.update_yaxes(title='Frequency')
fig.update_layout(template='plotly_dark')
fig.show()






df






df['20_day_MA']= df['close'].rolling(window=20).mean()
fig = go.Figure(data=[go.Candlestick(x=df.index,open=df['open'],high=df['high'],low=df['low'],close=df['close'],name="Candlesticks",increasing_line_color='green',decreasing_line_color='red',line=dict(width=1),showlegend=False)])
fig.add_trace(go.Scatter(x=df.index,y=df['20_day_MA'],mode ='lines',name='20 day MA',line=dict(color='rgba(255,255,0,0.3)')))
fig.update_layout(xaxis_title="Date",yaxis_title="Price",title="Google Stock Price Analysis",template='plotly_dark')
fig.show()






fig = go.Figure()

# Candlestick chart
fig.add_trace(go.Candlestick(
    x=df.index,open=df['open'],high=df['high'],low=df['low'],close=df['close'],name="Candlesticks",increasing_line_color='green',decreasing_line_color='red',line=dict(width=1),showlegend=False
))

# 20-day Moving Average line
fig.add_trace(go.Scatter(x=df.index,y=df['20_day_MA'],mode='lines',name='20 day MA',line=dict(color='rgba(255,255,0,0.3)')
))

# Update layout
fig.update_layout(xaxis_title="Date",yaxis_title="Price",title="Google Stock Price Analysis",template='plotly_dark'
)

# Show the figure
fig.show()






df=df.drop('20_day_MA',axis=1)






df['date'] = pd.to_datetime(df.index)
df=df[['date','close','high','low','open','volume']]






scaler =MinMaxScaler()
normalized_data = df[['open','high','low','close','volume']].copy()
normalized_data = scaler.fit_transform(normalized_data)






train_data,test_data = train_test_split(normalized_data,train_size=0.8,shuffle=False)
train_df = pd.DataFrame(train_data, columns=['open', 'high', 'low', 'volume', 'close'])
test_df = pd.DataFrame(test_data, columns=['open', 'high', 'low', 'volume', 'close'])













def generate_sequences(df, seq_length=50):
    X = df[['open', 'high', 'low', 'volume', 'close']].reset_index(drop=True)
    y = df[['open', 'high', 'low', 'volume', 'close']].reset_index(drop=True)

    sequences = []
    labels = []

    for index in range(len(X) - seq_length + 1):
        sequences.append(X.iloc[index : index + seq_length].values)
        labels.append(y.iloc[index + seq_length - 1].values)

    sequences = np.array(sequences)
    labels = np.array(labels)

    return sequences, labels






train_sequences, train_labels = generate_sequences(train_df)
test_sequences, test_labels = generate_sequences(test_df)






model = Sequential([
    LSTM(units= 50,return_sequences=True,input_shape=(50,5)),
    Dropout(0.2),
    LSTM(units=50,return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=5)
    ])






model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])
model.summary()






epochs = 200
batch_size= 32
history = model.fit(
    train_sequences,
    train_labels,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = (test_sequences,test_labels),
    verbose = 1
)






train_predictions = model.predict(train_sequences)
test_predictions = model.predict(test_sequences)






fig = make_subplots(rows =1 ,cols=1,subplot_titles=('Close Prediction'))
train_close_pred = train_predictions[:,0]
train_close_actual= train_labels[:,0]






fig.add_trace(go.Scatter(x=np.arange(len(train_close_actual)), y=train_close_actual, mode='lines', name='Actual', opacity=0.9))
fig.add_trace(go.Scatter(x=np.arange(len(train_close_pred)), y=train_close_pred, mode='lines', name='Predicted', opacity=0.6))

fig.update_layout(title='Close Predictions on Train Data', template='plotly_dark')
fig.show()






latest_prediction = []
last_seq = test_sequences[:-1]

for _ in range(10):
    prediction = model.predict(last_seq)
    latest_prediction.append(prediction)






pi.templates.default = "plotly_dark"

predicted_data_next = np.array(latest_prediction).reshape(-1, 5)
last_date = df['date'].max()
next_10_days = [last_date + timedelta(days=i) for i in range(1, 11)]

for i, feature_name in enumerate(['open', 'high', 'low', 'volume', 'close']):
    if feature_name in ['volume', 'close']:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=next_10_days, y=predicted_data_next[:, i],
                                 mode='lines+markers', name=f'Predicted {feature_name.capitalize()} Prices'))

        fig.update_layout(title=f'Predicted {feature_name.capitalize()} Prices for the Next 10 Days',
                          xaxis_title='Date', yaxis_title=f'{feature_name.capitalize()} Price')

        fig.show()

