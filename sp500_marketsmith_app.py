import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

st.title('MarketSmith 197 Industry Groups')


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
        st.sidebar.header('197 Industry Groups')

        # list of all Industries in the dataset to populate the filter with.
        industry = sorted(df['Name'].unique().tolist())

        # store users Industry selection
        selected_industry = st.sidebar.selectbox('', industry)

        # find the symbol related to the Industry selection
        symbol_filtered = df.loc[df['Name'] == selected_industry, 'Symbol'].unique()
        # create a filter to populate the related symbol.
        selected_symbol = st.sidebar.selectbox('',symbol_filtered)

        # Short and Long Term Moving Average filters
        mv_options = ['SMA','EMA']
        sma_ema = st.sidebar.radio('',mv_options)
        short_term = st.sidebar.slider('ST', min_value=1,
                                             max_value=10,
                                             value=4)

        long_term = st.sidebar.slider('LT',  min_value=2,
                                             max_value=40,
                                             value=10)

        # date picker filters, find the minimum date in the dataset and use that as the start date
        min_date = df.index.min()
        st.sidebar.header("Date Range")
        start_date = st.sidebar.date_input('Begin',min_date)
        end_date = st.sidebar.date_input('End')

        # filtered data
        df_selected = df.loc[(df['Name'] == selected_industry) & (df.index >= start_date) & (df.index <= end_date)]


        if sma_ema == 'SMA':
            # create Simple Moving Average column
            df_selected[sma_ema + '_' + str(short_term)] = df_selected['Ind Group Rank'].rolling(window=short_term, min_periods=1).mean()
            df_selected[sma_ema + '_' + str(long_term)] = df_selected['Ind Group Rank'].rolling(window=long_term, min_periods=1).mean()

        elif sma_ema == 'EMA':
             # create Exponential Moving Average columns
            df_selected[sma_ema + '_' + str(short_term)] = df_selected['Ind Group Rank'].ewm(span=short_term, adjust=False).mean()
            df_selected[sma_ema + '_' + str(long_term)] =  df_selected['Ind Group Rank'].ewm(span=long_term, adjust=False).mean()

        # signal alerts for crossover strategy
        df_selected['alert'] = 0.0
        df_selected['alert'] = np.where(df_selected[sma_ema + '_' + str(short_term)]>df_selected[sma_ema + '_' + str(long_term)], 1.0, 0.0)
        df_selected['position'] = df_selected['alert'].diff()


        # enable toggle to view & unview the dataset
        if st.checkbox('Show File Details & Dataframe'):
            st.write(file_details)
            st.markdown('** Original Dataset:**')
            st.write('>Data Dimension: ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.')
            st.write(df)

        # filtered data to create plots
        if st.button('Plot IG Ranking Graph'):
            st.header('IBD Industry Group Ranking')

            fig = px.line(df_selected, x=df_selected.index, y=['Ind Group Rank',df_selected[sma_ema + '_' + str(short_term)],df_selected[sma_ema + '_' + str(long_term)]],
                          hover_name='Name')

            fig.add_scatter(x=df_selected[df_selected['position'] == -1].index,
                            y=df_selected[sma_ema + '_' + str(short_term)][df_selected['position'] == -1],
                            name= 'Buy',
                            mode='markers',
                            marker_symbol='star-triangle-up',
                            marker_color='green', marker_size=15)

            fig.add_scatter(x=df_selected[df_selected['position'] == 1].index,
                            y=df_selected[sma_ema + '_' + str(short_term)][df_selected['position'] == 1],
                            name= 'Sell',
                            mode='markers',
                            marker_symbol='star-triangle-down',
                            marker_color='red', marker_size=15)

            fig.update_layout(
                            title=selected_industry,
                            xaxis_title="Date",
                            yaxis_title="IG Ranking",
                            legend_title ='')

            fig.update_yaxes(autorange="reversed")
            return st.plotly_chart(fig)

    else:
        st.subheader("About")
        st.info("Built with Streamlit")
        st.info("hushon.d@googlemail.com")
        st.text("Donovan Hushon")

if __name__ == '__main__':
    main()







