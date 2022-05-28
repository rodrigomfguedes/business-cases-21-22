import pandas as pd
import numpy as np
import streamlit as st
import tweepy
from PIL import Image
import config
import plotly.graph_objs as go
from functools import reduce
import plotly.express as px
import psycopg2
import yfinance as yf

#Setting up the twitter API
auth = tweepy.OAuthHandler(config.TWITTER_CONSUMER_KEY, config.TWITTER_CONSUME_SECRET)
auth.set_access_token(config.TWITTER_ACCESS_TOKEN, config.TWITTER_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

#fic DB
#connection = psycopg2.connect(host=config.DB_HOST, database=config.DB_NAME, user=config.DB_USER, password=config.DB_PASS)
#cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)


#Sidebar config
st.sidebar.markdown("<h1 style='text-align: center; color: black;'>AA Finance</h1>", unsafe_allow_html=True)
image = Image.open('assets/aa_finance.jpeg')
st.sidebar.image(image, caption="Providing investment insights for Investments4Some")

option = st.sidebar.selectbox("Choose a dashboard:",{'Candlecharts','Compare Assets','Tweets','Patterns'})

st.header(option)


#List to append temporary user profiles that will be shown in the dashboard
twitter_temp = []

if option == 'Tweets':
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
