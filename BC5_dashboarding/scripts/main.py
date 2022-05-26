import pandas as pd
import numpy as np
import streamlit as st
import tweepy
from PIL import Image
import config
import plotly.graph_objs as go
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

option = st.sidebar.selectbox("Choose a dashboard:",{'Candlecharts','Tweets','Patterns'})




st.header(option)

if option == 'Tweets':
    st.subheader('Twitter Dashboard Logic')
    for username in config.TWITTER_USERNAMES:
        user = api.get_user(screen_name=username)
        tweets = api.user_timeline(screen_name=username)

        st.image(user.profile_image_url)

        for tweet in tweets:
            if '$' in tweet.text:
                words = tweet.text.split(' ')
                for word in words:
                    if word.startswith('$') and word[1:].isalpha():
                        symbol = word[1:]
                        st.write(symbol)
                        st.write(tweet.text)
                        st.image(f"https://finviz.com/chart.ashx?t={symbol}")


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
    fig.update_layout(height = 800)

    st.plotly_chart(fig, use_container_width=True)
    st.write(data)
