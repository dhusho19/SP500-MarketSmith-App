import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.title('SP500 & MarketSmith Analysis')
st.markdown("""Welcome to this SP500 & MarketSmith Analysis.
            Here we will be analysing Industry Group Ranking(s) and their Simple & Exponential Moving Averages,
            using a Crossover Strategy to determine buy & sell triggers.""")

def main():

    # allow user to upload their own file through a streamlit sile uploader
    uploaded_file = st.file_uploader("Please Upload a CSV File",type=['csv'])
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        df = pd.read_csv(uploaded_file, parse_dates=['Date Stamp'])

        # extract the date attribute from the datetime object, in order to allow filtering of df between two dates
        df['Date'] = df['Date Stamp'].dt.date
        # Now we have the date from this column we can drop it and set it as the index
        df.drop('Date Stamp', axis=1, inplace=True)
        df.set_index('Date', inplace=True)

        # sidebar filters
        industry = sorted(df['Name'].unique().tolist())
        st.sidebar.header('Industries')
        selected_industry = st.sidebar.selectbox('SP500 Industries', industry)

        mv_options = ['SMA','EMA']
        st.sidebar.header('Simple Moving or Exponential Moving Average')
        sma_ema = st.sidebar.radio('',mv_options)
        short_term = st.sidebar.slider('Short-Term Moving Average', min_value=1,
                                                                    max_value=5,
                                                                    value=4)

        long_term = st.sidebar.slider('Long-Term Moving Average',  min_value=10,
                                                                   max_value=50,
                                                                   value=10)
        min_date = df.index.min()
        st.sidebar.header("Date Range")
        start_date = st.sidebar.date_input('Start Date',min_date)
        end_date = st.sidebar.date_input('End Date')

        # display filtered data
        df_selected_industry = df.loc[(df['Name'] == selected_industry) & (df.index >= start_date) & (df.index <= end_date)]

        if sma_ema == 'SMA':
            # create Simple Moving Average column
            df_selected_industry[sma_ema + '_' + str(short_term)] = df_selected_industry['Ind Group Rank'].rolling(window=short_term, min_periods=1).mean()
            df_selected_industry[sma_ema + '_' + str(long_term)] = df_selected_industry['Ind Group Rank'].rolling(window=long_term, min_periods=1).mean()

        elif sma_ema == 'EMA':
             # create Exponential Moving Average columns
            df_selected_industry[sma_ema + '_' + str(short_term)] = df_selected_industry['Ind Group Rank'].ewm(span=short_term, adjust=False).mean()
            df_selected_industry[sma_ema + '_' + str(long_term)] =  df_selected_industry['Ind Group Rank'].ewm(span=long_term, adjust=False).mean()


        df_selected_industry['alert'] = 0.0
        df_selected_industry['alert'] = np.where(df_selected_industry[sma_ema + '_' + str(short_term)]>df_selected_industry[sma_ema + '_' + str(long_term)], 1.0, 0.0)
        df_selected_industry['position'] = df_selected_industry['alert'].diff()


        # enable toggle to view & unview the dataset
        if st.checkbox('Show File Details & Dataframe'):
            st.write(file_details)
            st.markdown('** Original Dataset:**')
            st.write('>Data Dimension: ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.')
            st.write(df)

        # filtered data to create plots
        if st.button('Show Industry Group Ranking Plots'):
            st.header('SP500 Industry Rankings')

            fig, ax = plt.subplots(figsize=(16, 8))
            df_selected_industry['Ind Group Rank'].plot(label='IND GROUP RANK',style='k--')
            df_selected_industry[sma_ema + '_' + str(short_term)].plot(color='b')
            df_selected_industry[sma_ema + '_' + str(long_term)].plot(color='m')

            # buy alerts
            plt.plot(df_selected_industry[df_selected_industry['position'] == 1].index,
                    df_selected_industry[sma_ema + '_' + str(short_term)][df_selected_industry['position'] == 1],
                    'v', markersize = 15, color = 'r', label = 'BUY')

            # sell alerts
            plt.plot(df_selected_industry[df_selected_industry['position'] == -1].index,
                    df_selected_industry[sma_ema + '_' + str(short_term)][df_selected_industry['position'] == -1],
                    '^', markersize = 15, color = 'g', label = 'SELL')

            # reserve y axis as the lower Industry Group Rank the better
            plt.gca().invert_yaxis()
            plt.title(selected_industry,fontsize=25)
            plt.xlabel('Date',fontsize=20)
            plt.ylabel('Industry Group Rank',fontsize=20)
            plt.grid()
            plt.legend();
            return st.pyplot(fig)

    else:
        st.subheader("About")
        st.info("Built with Streamlit")
        st.info("hushon.d@googlemail.com")
        st.text("Donovan Hushon")

if __name__ == '__main__':
    main()







