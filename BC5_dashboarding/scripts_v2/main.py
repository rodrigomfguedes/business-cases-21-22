import pandas as pd
import numpy as np
import tweepy
from PIL import Image
import config
import plotly.graph_objs as go
from functools import reduce
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
tf.random.set_seed(1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import cryptoModelsParams
import datetime
from copy import deepcopy
from io import BytesIO

#Setting up the twitter API


auth = tweepy.OAuthHandler(config.TWITTER_CONSUMER_KEY, config.TWITTER_CONSUME_SECRET)
auth.set_access_token(config.TWITTER_ACCESS_TOKEN, config.TWITTER_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

#fic DB
#connection = psycopg2.connect(host=config.DB_HOST, database=config.DB_NAME, user=config.DB_USER, password=config.DB_PASS)
#cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

url = https://raw.githubusercontent.com/rodrigomfguedes/business-cases-21-22/main/BC5_dashboarding/scripts_v2/assets/aa_finance.jpeg
response = requests.get(url)
image = Image.open(BytesIO(response.content)
#Sidebar config
st.sidebar.markdown("<h1 style='text-align: center; color: black;'>AA Finance</h1>", unsafe_allow_html=True)
#image = Image.open('assets/aa_finance.jpeg')
st.sidebar.image(image, caption="Providing investment insights for Investments4Some")

option = st.sidebar.selectbox("Choose a dashboard:",{'Candlecharts','Compare Assets','Tweets','Crypto Predictions'})

st.header(option)

if option == 'Crypto Predictions':
    crypto = st.sidebar.selectbox("Choose a cryptocurrency:",
                                 {'ADA', 'ATOM', 'AVAX', 'AXS','BTC','ETH','LINK','LUNA1','MATIC','SOL'})

    n_days = st.sidebar.slider('Number of days ahead to predict:', min_value=1, max_value=5)

    coin_name = crypto + '-USD'

    # Get the data frame
    df = yf.Ticker(coin_name).history(period='5y')[['Close']]
    df['Date'] = df.index
    df.reset_index(drop=True, inplace=True)
    df_complete = df.copy(deep=True)

    # FILTER DATASET
    df = df.loc[df['Date'] >= '2021-11-01']
    df.reset_index(drop=True, inplace=True)


    # Create a DF (windowed_df) where the middle columns will correspond to the close values of X days before the target date
    # and the final column will correspond to the close value of the target date. Use these values for prediction and play
    # with the value of X

    def get_windowed_df(X, df):
        start_Date = df['Date'] + pd.Timedelta(days=X)

        perm = np.zeros((1, X + 1))

        # Get labels for DataFrame
        j = 1
        labels = []

        while j <= X:
            label = 'closeValue_' + str(j) + 'daysBefore'
            labels.append(label)

            j += 1

        labels.append('closeValue')

        for i in range(X, df.shape[0]):
            temp = np.zeros((1, X + 1))

            # Date for i-th day
            # temp[0,0] = df.iloc[i]['Date']

            # Close values for k days before
            for k in range(X):
                temp[0, k] = df.iloc[i - k - 1, 0]

            # Close value for i-th date
            temp[0, -1] = df.iloc[i, 0]

            # Add values to the permanent frame
            perm = np.vstack((perm, temp))

            # Get the array in dataframe form
            windowed_df = pd.DataFrame(perm[1:, :], columns=labels)

        return windowed_df

    #Specific model parameters
    X_win = cryptoModelsParams.X_win[crypto]
    lr = cryptoModelsParams.learningRate[crypto]
    bs = cryptoModelsParams.batchSize[crypto]
    epochs = cryptoModelsParams.nEpochs[crypto]

    # Get the dataframe and append the dates
    windowed_df = get_windowed_df(X_win, df)
    windowed_df['Date'] = df.iloc[X_win:]['Date'].reset_index(drop=True)


    # Get the X,y and dates into a numpy array to apply on a model

    def windowed_df_to_date_X_y(windowed_dataframe):
        df_as_np = windowed_dataframe.to_numpy()

        dates = df_as_np[:, -1]

        middle_matrix = df_as_np[:, 0:-2]
        X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

        Y = df_as_np[:, -2]

        return dates, X.astype(np.float32), Y.astype(np.float32)


    dates, X, y = windowed_df_to_date_X_y(windowed_df)

    # Partition for train, validation and test

    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

    fig, axs = plt.subplots(1, 1, figsize=(12, 5))

    # Plot the partitions
    axs.plot(dates_train, y_train)
    axs.plot(dates_val, y_val)
    axs.plot(dates_test, y_test)

    axs.legend(['Train', 'Validation', 'Test'])

    st.subheader('Train/Test split for '+crypto)
    #Plot split
    st.pyplot(fig, use_container_width=True)

    #Train model with data
    model = Sequential([layers.Input((X_win, 1)),
                        layers.LSTM(64),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(1)])

    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=lr),
                  metrics=['mean_absolute_error'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, shuffle=False, batch_size=bs, verbose=2)

    # PREDICT THE VALUES USING THE MODEL
    train_predictions = model.predict(X_train).flatten()
    val_predictions = model.predict(X_val).flatten()
    test_predictions = model.predict(X_test).flatten()

    fig, axs = plt.subplots(3, 1, figsize=(14, 14))

    axs[0].plot(dates_train, train_predictions)
    axs[0].plot(dates_train, y_train)
    axs[0].legend(['Training Predictions', 'Training Observations'])

    axs[1].plot(dates_val, val_predictions)
    axs[1].plot(dates_val, y_val)
    axs[1].legend(['Validation Predictions', 'Validation Observations'])

    axs[2].plot(dates_test, test_predictions)
    axs[2].plot(dates_test, y_test)
    axs[2].legend(['Testing Predictions', 'Testing Observations'])

    # Plot model training predictions
    st.subheader('Model predictions for training')
    st.pyplot(fig, use_container_width=True)


    #Recursive predictions for n_days

    # Get prediction for future dates recursively based on the previous existing information. Then update the window of days upon
    # which the predictions are made

    recursive_predictions = []
    recursive_dates = np.concatenate([dates_test])

    extra_dates = np.array([df.iloc[-1]['Date']+datetime.timedelta(days=1+i) for i in range(n_days)])
    recursive_dates = np.append(recursive_dates, extra_dates)

    last_window = deepcopy(X_train[-1])

    for target_date in recursive_dates:
        next_prediction = model.predict(np.array([last_window])).flatten()
        recursive_predictions.append(next_prediction)
        last_window = np.insert(last_window, 0, next_prediction)[:-1]

    #Plot the results of the predictions
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))

    axs[0].plot(dates_train, train_predictions)
    axs[0].plot(dates_train, y_train)
    axs[0].plot(dates_val, val_predictions)
    axs[0].plot(dates_val, y_val)
    axs[0].plot(dates_test, test_predictions)
    axs[0].plot(dates_test, y_test)
    axs[0].plot(recursive_dates, recursive_predictions)
    axs[0].legend(['Training Predictions',
                   'Training Observations',
                   'Validation Predictions',
                   'Validation Observations',
                   'Testing Predictions',
                   'Testing Observations',
                   'Recursive Predictions'])

    axs[1].plot(dates_test, y_test)
    axs[1].plot(recursive_dates, recursive_predictions)
    axs[1].legend(['Testing Observations',
                   'Recursive Predictions'])

    # Plot model training predictions
    st.subheader('Predicted value of ' + crypto + ' in the next ' + str(n_days) + ' days')
    st.pyplot(fig, use_container_width=True)
    st.write(recursive_predictions[-n_days:])


if option == 'Tweets':
    # List to append temporary user profiles that will be shown in the dashboard
    twitter_temp = []

    st.subheader('Twitter Dashboard Logic')
    asset = st.sidebar.text_input("Search by asset:", value='', max_chars=None, key=None, type='default')

    new_user = st.sidebar.text_input("Add new twitter account:", value='', max_chars=None, key=None, type='default')
    if new_user:
        twitter_temp.append(new_user)
    #Display latest 3 tweets of each user
    if not asset:
        for username in config.TWITTER_USERNAMES+twitter_temp:
            user = api.get_user(screen_name=username)
            tweets = api.user_timeline(screen_name=username)

            st.image(user.profile_image_url)
            st.subheader(username)

            for tweet in tweets[0:3]:
                if '$' in tweet.text:
                    words = tweet.text.split(' ')
                    for word in words:
                        if word.startswith('$') and word[1:].isalpha():
                            symbol = word[1:]
                            st.write(symbol)
                            st.write(tweet.text)
                            st.image(f"https://finviz.com/chart.ashx?t={symbol}")
                else:
                    st.write(tweet.text)


    #Display comments for specific asset
    else:
        for username in config.TWITTER_USERNAMES+twitter_temp:
            user = api.get_user(screen_name=username)
            tweets = api.user_timeline(screen_name=username)

            for tweet in tweets:
                if asset.upper() in tweet.text.upper():
                    st.image(user.profile_image_url)
                    st.subheader(username)

                    st.write(asset)
                    st.write(tweet.text)
                    st.image(f"https://finviz.com/chart.ashx?t={asset}")

if option == 'Candlecharts':
    symbol = st.sidebar.text_input("Manually input Asset:", value='AAPL', max_chars=None, key=None, type='default')

    st.subheader(symbol.upper() + ' asset history')

    #Get asset and data
    asset = yf.Ticker(symbol)
    data = asset.history(period='5y')

    #Get date in date format
    a = data.index
    date = []
    for datetime in a:
        date.append(datetime.date())

    data['Date'] = date
    data.set_index('Date', drop=True, inplace=True)

    #Plot Candlestick
    fig = go.Figure(data = [go.Candlestick(x=data.index,
                    open = data['Open'],
                    high = data['High'],
                    low = data['Low'],
                    close = data['Close'],
                    name = symbol)])
    fig.update_xaxes(type = 'category')
    fig.update_layout(title= symbol + " historical data in the previous 5 years",
                      autosize = False,
                      height = 800,
                      yaxis=dict(
                          title_text="Asset value in USD",
                          titlefont=dict(size=15),
                      )
                      )



    st.plotly_chart(fig, use_container_width=True)
    st.write(data)

if option == 'Compare Assets':
    st.subheader('Analyze multiple assets based on their 5 year history')
    assets = st.sidebar.text_input("List of assets to compare (separated by ,) :", value='AAPL,MCD', max_chars=None, key=None, type='default')

    assets = assets.split(',')

    #Get data from each asset
    data = []
    df_list = []
    i = 0
    for asset in assets:
        i += 1
        ticker = yf.Ticker(asset)
        str_df = "df_" + str(i)

        #Rename DF's by order (df_1, df_2...)
        locals()[str_df] = ticker.history(period='5y')[['Close']]
        locals()[str_df].rename(columns={'Close': 'Close' + '_' + asset}, inplace=True)

        #Get date in column
        locals()[str_df]['Date'] = locals()[str_df].index.date
        locals()[str_df].reset_index(drop=True, inplace=True)

        #Store df name in list
        df_list.append(locals()[str_df])

    #Merge df's
    df_merged = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), df_list)
    #Create df with column labels
    #df = pd.DataFrame(data=[data[i] for i in range(len(assets))]).transpose()
    #df_merged.set_index(df_merged['Date'], inplace=True, drop=True)
    #df.columns = assets

    # Create traces
    fig = go.Figure()
    asset_list = list(df_merged.columns)
    asset_list.remove('Date')

    for column in asset_list:
        fig.add_trace(go.Scatter(x=df_merged['Date'], y=df_merged[column],mode='lines',name=column))

    st.plotly_chart(fig, use_container_width=True)
    st.write(df_merged)
