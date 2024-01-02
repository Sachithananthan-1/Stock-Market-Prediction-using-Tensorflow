import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense
import yfinance as yf

def download_stock_data(symbol, startdate, enddate):
    data = yf.download(symbol,start=startdate, end=enddate, progress=False)
    data['Date'] = data.index
    return data

def predict_next_pts_with_dates(last_sequence, lastdate,time_steps, interval_hours, num_predictions):
    predictions = []
    prediction_dates = []

    for _ in range(num_predictions):
        y_pred = model.predict(last_sequence)
        predictions.append(y_pred.flatten()[0])
        last_sequence = np.roll(last_sequence,shift=-1,axis =1)
        last_sequence[0,-1,0] = y_pred.flatten()[0]

    #Calculating the next date based on the last predicted date and the interval
        lastdate = lastdate + pd.DateOffset(hours=interval_hours)
        prediction_dates.append(lastdate)
    return predictions, prediction_dates


    #Streamlit app
st.title("Stock Market Prediction")

#User input for stock symbol, start date,and end date
stock_symbol =st.text_input("Enter stock symbol(e.g., RELIANCE.NS):","RELIANCE.NS")
startdate = st.date_input("Enter start date:", pd.to_datetime("2022-01-01"))
enddate = st.date_input("Enter end date:", pd.to_datetime("2023-01-01"))

#User input for prediction interval and number of predicted points
interval_type = st.selectbox("Select prediction interval type:", ["Hour","Day","Minute"])
interval_value = st.number_input("Enter prediction interval value:",min_value=1,value=1)
num_predictions = st.number_input("Enter number of predicted points:",min_value=1,value=10)

#Convert interval type to hours
if interval_type == "Hour":
    interval_hours = interval_value
elif interval_type == "Day":
    interval_hours = interval_value *24
elif interval_type == "Minute":
    interval_hours = interval_value/60

#Download Stock Data
df = download_stock_data(stock_symbol,startdate,enddate)
df=df.dropna()

st.subheader(f"Downloaded Stock Data for {stock_symbol}:")
st.write(df)

#Training the data
if st.button("Train Model"):
    st.info("Training the model.Please wait...")

    #Extract input (X) and output (Y) variables
    X = df[['Open','High','Low']].values
    Y = df['Close'].values

    #Normalize the data using the scaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    Y = scaler.fit_transform(Y.reshape(-1,1))

    #Prepare the data for Simple RNN
    time_steps = 24
    x_rnn, y_rnn = [],[]

    for i in range(len(X) - time_steps):
        x_rnn.append(X[i:(i+time_steps), :])
        y_rnn.append(Y[i + time_steps])

    x_rnn, y_rnn = np.array(x_rnn), np.array(y_rnn)

    # Build and train Simple RNN model

    model = Sequential()
    model.add(SimpleRNN(units=50, activation='relu',input_shape=(x_rnn.shape[1], x_rnn.shape[2])))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_rnn,y_rnn, epochs=10,batch_size=1, verbose=2)

    #Store the trained model in st.session_state
    st.session_state.model = model
    st.success("Model trained successfully!")


if 'model' in st.session_state:
    #Use the trained model to predict the next sequence
    x_pred = x_rnn[-1].reshape((1,time_steps,x_rnn.shape[2]))
    last_date = df['Date'].iloc[-1]
    predictions, predictions_dates = predict_next_pts_with_dates(x_pred,last_date,time_steps,interval_hours,num_predictions)

    #Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))

    #Display predicted values with dates
    st.subheader("Predicted Next Close Prices:")
    predicted_df = pd.DataFrame({"Date":predictions_dates, "Predicted Close":predictions.flatten()})
    st.write(predicted_df)

    #Visualize the predicted sequence alonng with the actual data
    st.line_chart(df.set_index('Date')['Close'].rename("Actual Close Price"))
    st.line_chart(predicted_df.set_index('Date'))
    st.line_chart(pd.concat([df.set_index('Date')['Close'].rename("Actual Close Price"),predicted_df.set_index('Date')['Predicted Close']],axis=1))

else:
    st.warning("Train the model to make predictions.")