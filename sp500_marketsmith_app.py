import pandas as pd
import numpy as np
import base64
import streamlit as st
import plotly.express as px
import datetime

st.title('MS Sector & Industry Group Rotation')

st.sidebar.header('Moving Averages')
# Short and Long Term Moving Average filters
mv_options = ['SMA','EMA']
sma_ema = st.sidebar.radio('',mv_options)
short_term = st.sidebar.slider('ST', min_value=1,
                                        max_value=10,
                                        value=10)

mid_term = st.sidebar.slider('IT', min_value=0,
                                        max_value=50,
                                        value=21)

long_term = st.sidebar.slider('LT',  min_value=2,
                                        max_value=40,
                                        value=50)

# column names for long and short moving average columns
short_term_col = sma_ema + '_' + str(short_term)
mid_term_col = sma_ema + '_' + str(mid_term)
long_term_col = sma_ema + '_' + str(long_term)



def app():

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
        st.sidebar.header('Sector & Industry Groups')
        # load list of Sector values & create filter
        sector = sorted(df['Sector'].unique().tolist())
        selected_sector = st.sidebar.selectbox('', sector)

        # list of all relevant Industries based on the Sector filter selection
        industry = df.loc[df['Sector'] == selected_sector, 'Name'].unique()

        # store users Industry selection
        selected_industry = st.sidebar.selectbox('', industry)

        # date picker filters, find the minimum date in the dataset and use that as the start date
        min_date = df.index.min()
        st.sidebar.header("Date Range")
        start_date = st.sidebar.date_input('Begin',min_date)
        end_date = st.sidebar.date_input('End')

        # enable toggle to view & unview the dataset
        if st.checkbox('Show File Details & Dataframe'):
            st.write(file_details)
            st.markdown('** Original Dataset:**')
            st.write('>Data Dimension: ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.')
            st.write(df)

        # filtered sector & industry data
        df_selected_sector = df.loc[(df['Sector'] == selected_sector) & (df.index >= start_date) & (df.index <= end_date)]
        df_selected_industry = df.loc[(df['Name'] == selected_industry) & (df.index >= start_date) & (df.index <= end_date)]

        # create a total market value to utilise for weighted sector rank, then add this to the dataframe
        total_mkt_val = df_selected_sector.groupby([df_selected_sector.index,'Sector'])['Ind Mkt Val (bil)'].transform('sum')
        df_selected_sector['total_mkt_val'] = total_mkt_val

        # calculate a percentage weight for each industry
        df_selected_sector['weight'] = df_selected_sector['Ind Mkt Val (bil)']/df_selected_sector['total_mkt_val']
        # use the newly created weight column to calculate sector rank
        df_selected_sector['Sector Rank'] = df_selected_sector['weight']*df_selected_sector['Ind Group Rank']

        df_sector_rank = df_selected_sector.groupby([df_selected_sector.index,'Sector'])['Sector Rank'].sum().reset_index()
        df_sector_rank.set_index('Date',inplace=True)

        st.markdown("""---""")
        # function calls
        sector_crossover_strategy(df_sector_rank)
        industry_crossover_strategy(df_selected_industry)
        plotting(df_sector_rank,df_selected_industry,selected_sector,selected_industry)
        st.markdown("""---""")
        # weekly_summary(df)

    else:
        st.subheader("About")
        st.info("Built with Streamlit")
        st.info("hushon.d@googlemail.com")
        st.text("Donovan Hushon")

def sector_crossover_strategy(df):
    if sma_ema == 'SMA':
        # Sector
        df[short_term_col] = df['Sector Rank'].rolling(window=short_term, min_periods=1).mean()
        df[mid_term_col] = df['Sector Rank'].rolling(window=mid_term, min_periods=1).mean()
        df[long_term_col] = df['Sector Rank'].rolling(window=long_term, min_periods=1).mean()

    elif sma_ema == 'EMA':
        # Sector
        df[short_term_col] = df['Sector Rank'].ewm(span=short_term, adjust=False).mean()
        df[mid_term_col] = df['Sector Rank'].ewm(span=mid_term, adjust=False).mean()
        df[long_term_col] =  df['Sector Rank'].ewm(span=long_term, adjust=False).mean()

    # signal alerts for crossover strategy Sector
    df['alert'] = 0.0
    df['alert'] = np.where(df[short_term_col]>df[long_term_col], 1.0, 0.0)
    # create a new column 'Position' which is a day-to-day difference of the alert column.
    df['position'] = df['alert'].diff()

def industry_crossover_strategy(df):
    if sma_ema == 'SMA':
        # Industry
        df[short_term_col] = df['Ind Group Rank'].rolling(window=short_term, min_periods=1).mean()
        df[mid_term_col] = df['Ind Group Rank'].rolling(window=mid_term, min_periods=1).mean()
        df[long_term_col] = df['Ind Group Rank'].rolling(window=long_term, min_periods=1).mean()

    elif sma_ema == 'EMA':
        # Industry
        df[short_term_col] = df['Ind Group Rank'].ewm(span=short_term, adjust=False).mean()
        df[mid_term_col] = df['Ind Group Rank'].ewm(span=mid_term, adjust=False).mean()
        df[long_term_col] =  df['Ind Group Rank'].ewm(span=long_term, adjust=False).mean()

    # signal alerts for crossover strategy Industry
    df['alert'] = 0.0
    df['alert'] = np.where(df[short_term_col]>df[long_term_col], 1.0, 0.0)
    # create a new column 'Position' which is a day-to-day difference of the alert column.
    df['position'] = df['alert'].diff()

def plotting(df_sector_rank, df_selected_industry,selected_sector,selected_industry):

    st.header('Graphing Sectors & IG')
    # filtered data to create plots
    if st.checkbox('Plot Sector Ranking Graph'):
        st.subheader('IBD Sector Ranking')

        fig = px.line(df_sector_rank, x=df_sector_rank.index, y=['Sector Rank',df_sector_rank[short_term_col],df_sector_rank[mid_term_col],df_sector_rank[long_term_col]],
                        hover_name='Sector', template = 'plotly_dark',
                        color_discrete_map={'Sector Rank':'white', short_term_col:'green',mid_term_col:'yellow',long_term_col:'red'}
                         )

        fig.add_scatter(x=df_sector_rank[df_sector_rank['position'] == -1].index,
                        y=df_sector_rank[short_term_col][df_sector_rank['position'] == -1],
                        name= 'Buy',
                        mode='markers',
                        marker_symbol='star-triangle-up',
                        marker_color='green', marker_size=15)

        fig.add_scatter(x=df_sector_rank[df_sector_rank['position'] == 1].index,
                        y=df_sector_rank[short_term_col][df_sector_rank['position'] == 1],
                        name= 'Sell',
                        mode='markers',
                        marker_symbol='star-triangle-down',
                        marker_color='red', marker_size=15)

        fig.update_layout(title=selected_sector,
                        xaxis_title="Date",
                        yaxis_title="Sector Ranking",
                        legend_title ='')

        fig.update_yaxes(autorange="reversed")

        # checkbox to hide and show the buy & sell dataframe
        if st.checkbox('Buy & Sell Sector Data'):
            st.subheader('Buy & Sell DataFrame for ' + selected_sector)
            # create buy and sell column, to easily identify the triggers
            df_sector_rank['buy_sell'] = np.where(df_sector_rank['position'] == -1,'BUY','SELL')
            # sort df desc order
            sorted_sector_df = df_sector_rank.sort_index(ascending=False)
            sorted_sector_df.reset_index(inplace=True)
            # call download function, with a subset of the data. Only looking at rows for buy and sell triggers
            st.markdown(filedownload(sorted_sector_df.loc[:,['Date','Sector','Sector Rank',short_term_col,long_term_col,'buy_sell']].loc[(sorted_sector_df['position'].isin([-1,1]))],selected_sector), unsafe_allow_html=True)
            # write df to streamlit app
            st.write(sorted_sector_df.loc[:,['Date','Sector','Sector Rank','buy_sell']].loc[(sorted_sector_df['position'].isin([-1,1]))].head(3))

        return st.plotly_chart(fig)

    if st.checkbox('Plot IG Ranking Graph'):
        st.subheader('IBD Industry Group Ranking')

        fig = px.line(df_selected_industry, x=df_selected_industry.index, y=['Ind Group Rank',df_selected_industry[short_term_col],df_sector_rank[mid_term_col],df_selected_industry[long_term_col]],
                        hover_name='Name',template = 'plotly_dark',
                        color_discrete_map={'Sector Rank':'white',short_term_col:'green',mid_term_col:'yellow',long_term_col:'red'}
                        )

        fig.add_scatter(x=df_selected_industry[df_selected_industry['position'] == -1].index,
                        y=df_selected_industry[short_term_col][df_selected_industry['position'] == -1],
                        name= 'Buy',
                        mode='markers',
                        marker_symbol='star-triangle-up',
                        marker_color='green', marker_size=15)

        fig.add_scatter(x=df_selected_industry[df_selected_industry['position'] == 1].index,
                        y=df_selected_industry[short_term_col][df_selected_industry['position'] == 1],
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

        # checkbox to hide and show the buy & sell dataframe
        if st.checkbox('Buy & Sell IG Data'):
            st.subheader('Buy & Sell DataFrame for ' + selected_industry)
            # create buy and sell column, to easily identify the triggers
            df_selected_industry['buy_sell'] = np.where(df_selected_industry['position'] == -1,'BUY','SELL')
            # sort df desc order
            sorted_industry_df = df_selected_industry.sort_index(ascending=False)
            sorted_industry_df.reset_index(inplace=True)
            # call download function, with a subset of the data. Only looking at rows for buy and sell triggers
            st.markdown(filedownload(sorted_industry_df.loc[:,['Date','Symbol','Sector','Name','Ind Group Rank',short_term_col,long_term_col,'buy_sell']].loc[(sorted_industry_df['position'].isin([-1,1]))],selected_industry), unsafe_allow_html=True)
            # write df to streamlit app
            st.write(sorted_industry_df.loc[:,['Date','Sector','Name','Ind Group Rank','buy_sell']].loc[(sorted_industry_df['position'].isin([-1,1]))].head(3))

        return st.plotly_chart(fig)

# download buy and sell data
def filedownload(df, sector_ig):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="Buy Sell Triggers {sector_ig}.csv">Download to CSV File</a>'
    return href

if __name__ == '__main__':
    app()