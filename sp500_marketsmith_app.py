import pandas as pd
import numpy as np
import base64
import streamlit as st
import plotly.express as px
import datetime

st.title('MS Sector & Industry Group Rotation')

st.sidebar.header('Moving Averages')
# Short and Long Term Moving Average filters
mv_options = ['EMA','SMA']
sma_ema = st.sidebar.radio('',mv_options)
short_term = st.sidebar.slider('ST', min_value=1,
                                        max_value=30,
                                        value=10)

mid_term = st.sidebar.slider('IT', min_value=0,
                                        max_value=50,
                                        value=21)

long_term = st.sidebar.slider('LT',  min_value=21,
                                        max_value=200,
                                        value=100)

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
        df_selected_sector['Sector Rank'] = df_selected_sector['Sector Rank'].astype('float32').round(2).astype('int')

        df_sector_rank = df_selected_sector.groupby([df_selected_sector.index,'Sector'])['Sector Rank'].sum().reset_index()
        df_sector_rank.set_index('Date',inplace=True)

        st.markdown("""---""")
        # function calls
        sector_crossover_strategy(df_sector_rank)
        industry_crossover_strategy(df_selected_industry)
        #summary(df)
        plotting(df_sector_rank,df_selected_industry,selected_sector,selected_industry)
        st.markdown("""---""")
        df_daily_changes = summary(df)
        st.markdown("""---""")
        daily_signal_changes(df_daily_changes)
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
    df['alert_st'] = 0.0
    df['alert_st'] = np.where(df[short_term_col]>df[mid_term_col], 1.0, 0.0)
    df['alert_lt'] = 0.0
    df['alert_lt'] = np.where(df[mid_term_col]>df[long_term_col], 1.0, 0.0)
    # create a new column 'Position' which is a day-to-day difference of the alert column.
    df['position_st'] = df['alert_st'].diff() # 1 is BUY
    df['position_lt'] = df['alert_lt'].diff()

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
    df['alert_st'] = 0.0
    df['alert_st'] = np.where(df[short_term_col]>df[mid_term_col], 1.0, 0.0)
    df['alert_lt'] = 0.0
    df['alert_lt'] = np.where(df[mid_term_col]>df[long_term_col], 1.0, 0.0)
    # create a new column 'Position' which is a day-to-day difference of the alert column.
    df['position_st'] = df['alert_st'].diff()
    df['position_lt'] = df['alert_lt'].diff()

def plotting(df_sector_rank, df_selected_industry,selected_sector,selected_industry):

    st.header('Graphing Sectors & IG')
    # filtered data to create plots
    if st.checkbox('Plot Sector Ranking Graph'):
        st.subheader('IBD Sector Ranking')

        fig = px.line(df_sector_rank, x=df_sector_rank.index, y=['Sector Rank',df_sector_rank[short_term_col],df_sector_rank[mid_term_col],df_sector_rank[long_term_col]],
                        hover_name='Sector', template = 'plotly_dark',
                        color_discrete_map={'Sector Rank':'white', short_term_col:'green',mid_term_col:'yellow',long_term_col:'red'}
                         )

        fig.add_scatter(x=df_sector_rank.loc[df_sector_rank['position_st'] == -1].index,
                        y=df_sector_rank[short_term_col][df_sector_rank['position_st'] == -1],
                        name= 'ST Buy',
                        mode='markers',
                        marker_symbol='star-triangle-up',
                        marker_color='green', marker_size=15)

        fig.add_scatter(x=df_sector_rank.loc[df_sector_rank['position_st'] == 1].index,
                        y=df_sector_rank[short_term_col][df_sector_rank['position_st'] == 1],
                        name= 'ST Sell',
                        mode='markers',
                        marker_symbol='star-triangle-down',
                        marker_color='red', marker_size=15)

        fig.add_scatter(x=df_sector_rank.loc[df_sector_rank['position_lt'] == -1].index,
                        y=df_sector_rank[mid_term_col][df_sector_rank['position_lt'] == -1],
                        name= 'LT Buy',
                        mode='markers',
                        marker_symbol='star-triangle-up',
                        marker_color='blue', marker_size=15)

        fig.add_scatter(x=df_sector_rank.loc[df_sector_rank['position_lt'] == 1].index,
                        y=df_sector_rank[mid_term_col][df_sector_rank['position_lt'] == 1],
                        name= 'LT Sell',
                        mode='markers',
                        marker_symbol='star-triangle-down',
                        marker_color='orange', marker_size=15)

        fig.update_layout(title=selected_sector,
                        xaxis_title="Date",
                        yaxis_title="Sector Ranking",
                        legend_title ='')

        fig.update_yaxes(autorange="reversed")
        fig.update_traces(patch={"line": {"dash": 'dot'}}, selector={"legendgroup": "Sector Rank"})

        # checkbox to hide and show the buy & sell dataframe
        if st.checkbox('Buy & Sell Sector Data'):
            st.subheader('Buy & Sell DataFrame for ' + selected_sector)
            # create buy and sell column, to easily identify the triggers
            df_sector_rank['Buy Sell ST'] = np.where(df_sector_rank['position_st'] == -1,'BUY','SELL')
            df_sector_rank['Buy Sell LT'] = np.where(df_sector_rank['position_lt'] == -1,'BUY','SELL')
            # sort df desc order
            sorted_sector_df = df_sector_rank.sort_index(ascending=False)
            sorted_sector_df.reset_index(inplace=True)

            # call download function, with a subset of the data. Only looking at rows for buy and sell triggers
            csv = convert_df(sorted_sector_df.loc[:,['Date','Sector','Sector Rank',short_term_col,long_term_col,'Buy Sell ST']].loc[(sorted_sector_df['position_st'].isin([-1,1]))])
            st.download_button(label="Download data as CSV",
                               data=csv,
                               file_name='Sector_Latest_Signals.csv',
                               mime='text/csv')

            # write df to streamlit app
            st.write(sorted_sector_df.loc[:,['Date','Sector','Sector Rank','Buy Sell ST']].loc[(sorted_sector_df['position_st'].isin([-1,1]))].head(3))
            #st.write(sorted_sector_df.loc[:,['Date','Sector','Sector Rank','buy_sell']].loc[(sorted_sector_df['position_lt'].isin([-1,1]))].head(3))

        return st.plotly_chart(fig)

    if st.checkbox('Plot IG Ranking Graph'):
        st.subheader('IBD Industry Group Ranking')

        fig = px.line(df_selected_industry, x=df_selected_industry.index, y=['Ind Group Rank',df_selected_industry[short_term_col],df_selected_industry[mid_term_col],df_selected_industry[long_term_col]],
                        hover_name='Name',template = 'plotly_dark',
                        color_discrete_map={'Ind Group Rank':'white',short_term_col:'green',mid_term_col:'yellow',long_term_col:'red'}
                        )

        fig.add_scatter(x=df_selected_industry.loc[df_selected_industry['position_st'] == -1].index,
                        y=df_selected_industry[short_term_col][df_selected_industry['position_st'] == -1],
                        name= 'ST Buy',
                        mode='markers',
                        marker_symbol='star-triangle-up',
                        marker_color='green', marker_size=15)

        fig.add_scatter(x=df_selected_industry.loc[df_selected_industry['position_st'] == 1].index,
                        y=df_selected_industry[short_term_col][df_selected_industry['position_st'] == 1],
                        name= 'ST Sell',
                        mode='markers',
                        marker_symbol='star-triangle-down',
                        marker_color='red', marker_size=15)

        fig.add_scatter(x=df_selected_industry.loc[df_selected_industry['position_lt'] == -1].index,
                        y=df_selected_industry[mid_term_col][df_selected_industry['position_lt'] == -1],
                        name= 'LT Buy',
                        mode='markers',
                        marker_symbol='star-triangle-up',
                        marker_color='blue', marker_size=15)

        fig.add_scatter(x=df_selected_industry.loc[df_selected_industry['position_lt'] == 1].index,
                        y=df_selected_industry[mid_term_col][df_selected_industry['position_lt'] == 1],
                        name= 'LT Sell',
                        mode='markers',
                        marker_symbol='star-triangle-down',
                        marker_color='orange', marker_size=15)

        fig.update_layout(
                        title=selected_industry,
                        xaxis_title="Date",
                        yaxis_title="IG Ranking",
                        legend_title ='')

        fig.update_yaxes(autorange="reversed")
        fig.update_traces(patch={"line": {"dash": 'dot'}}, selector={"legendgroup": "Ind Group Rank"})

        # checkbox to hide and show the buy & sell dataframe
        if st.checkbox('Buy & Sell IG Data'):
            st.subheader('Buy & Sell DataFrame for ' + selected_industry)
            # create buy and sell column, to easily identify the triggers
            df_selected_industry['Buy Sell ST'] = np.where(df_selected_industry['position_st'] == -1,'BUY','SELL')
            df_selected_industry['Buy Sell LT'] = np.where(df_selected_industry['position_lt'] == -1,'BUY','SELL')
            # sort df desc order
            sorted_industry_df = df_selected_industry.sort_index(ascending=False)
            sorted_industry_df.reset_index(inplace=True)

            # Rounding formatting
            sorted_industry_df[short_term_col] = sorted_industry_df[short_term_col].astype('float32').round(2).astype('int')
            sorted_industry_df[mid_term_col] = sorted_industry_df[mid_term_col].astype('float32').round(2).astype('int')
            sorted_industry_df[long_term_col] = sorted_industry_df[long_term_col].astype('float32').round(2).astype('int')

            # call download function, with a subset of the data. Only looking at rows for buy and sell triggers
            csv = convert_df(sorted_industry_df.loc[:,['Date','Symbol','Sector','Name','Ind Group Rank',short_term_col,mid_term_col,long_term_col,'Buy Sell ST']].loc[(sorted_industry_df['position_st'].isin([-1,1]))])
            st.download_button(label="Download data as CSV",
                               data=csv,
                               file_name='IG_Latest_Signals.csv',
                               mime='text/csv')
            # write df to streamlit app
            st.write(sorted_industry_df.loc[:,['Date','Sector','Name','Ind Group Rank',short_term_col,mid_term_col,long_term_col,'position_st','position_lt','Buy Sell ST','Buy Sell LT']].loc[(sorted_industry_df['position_st'].isin([-1,1]))].head(3))

        return st.plotly_chart(fig)


def summary(df):
    st.header('IG Summary')
    if st.checkbox('IG'):
        df.reset_index(inplace=True)
        # Create a IG list of all unqiue IG's
        industry_lst = sorted(df['Name'].unique().tolist())

        # Iterate through each IG and load into a list & conver to an NumPy array
        lst = []
        for i in industry_lst:
            df_industry = df.loc[(df['Name'] == i)]
            industry_crossover_strategy(df_industry)
            lst.append(df_industry)
        arr = np.asarray(lst)

        # Load the array which is storing the data into a DataFrame
        df1 = pd.DataFrame(arr.reshape(-1, 13), columns=['Date','Symbol','Name','Sector','Ind Group Rank','Ind Mkt Val (bil)',short_term_col,mid_term_col,long_term_col,'alert_st','alert_lt','position_st','position_lt'])
        df1.index = np.repeat(np.arange(arr.shape[0]), arr.shape[1]) + 1
        df1.index.name = 'id'
        # create buy and sell column, to easily identify the triggers
        df1['Buy Sell ST'] = np.where(df1['alert_st'] == 0,'BUY','SELL')
        df1['Buy Sell LT'] = np.where(df1['alert_lt'] == 0,'BUY','SELL')

        # Filter Dataframes to only look at rows which are signals
        df_st = df1.loc[df1['position_st'].isin([-1,1])]
        df_lt = df1.loc[df1['position_lt'].isin([-1,1])]
        frames = [df_st, df_lt]
        df_final = pd.concat(frames)
        df_final.sort_values(by=['Date'], ascending=True, inplace=True)

        # Pull back the latest two signals per IG
        df_final = df_final.groupby('Name').tail(1).reset_index(drop=True)
        df_final.drop(['alert_st','alert_lt','position_st','position_lt'], axis=1, inplace=True)

        # Rounding formatting
        df_final[short_term_col] = df_final[short_term_col].astype('float32').round(2).astype('int')
        df_final[mid_term_col] = df_final[mid_term_col].astype('float32').round(2).astype('int')
        df_final[long_term_col] = df_final[long_term_col].astype('float32').round(2).astype('int')

        # Sort DataFrame and reshape it to merge each IG onto the one row
        df_final.sort_values(by=['Name','Date'], ascending=True, inplace=True)
        # df_final = pd.DataFrame(df_final.values.reshape(-1, df_final.shape[1] * 2),
        #                   columns=['Date_2','Symbol_2','Name_2','Sector_2','Ind Group Rank_2','Ind Mkt Val (bil)_2','short_term_col_2','mid_term_col_2','long_term_col_2','Buy Sell ST_2','Buy Sell LT_2','Date_1','Symbol_1','Name_1','Sector_1','Ind Group Rank_1','Ind Mkt Val (bil)_1',short_term_col,mid_term_col,long_term_col,'Buy Sell ST_1','Buy Sell LT_1'])

        # Re order the column structure
        # df_final = df_final.reindex(columns=['Date_1','Symbol_1','Name_1','Sector_1','Ind Group Rank_1','Ind Mkt Val (bil)_1',short_term_col,mid_term_col,long_term_col,'Buy Sell ST_1','Buy Sell LT_1','Date_2','Symbol_2','Name_2','Sector_2','Ind Group Rank_2','Ind Mkt Val (bil)_2','short_term_col_2','mid_term_col_2','long_term_col_2','Buy Sell ST_2','Buy Sell LT_2'])
        unique_sectors = sorted(df_final['Sector'].unique().tolist())
        sector_options = st.multiselect('Sectors of Interest',unique_sectors, default=['ENERGY','SOFTWARE','MEDICAL'
                                                                                      ,'TELECOM','BANKS','CHIPS'
                                                                                      ,'RETAIL','CONSUMER'])
        df_final = df_final.loc[df_final['Sector'].isin(sector_options)]
        st.write(df_final)

        # Call download function
        csv = convert_df(df_final)
        st.download_button(label="Download data as CSV",
            data=csv,
            file_name='IG_Latest_Signals.csv',
            mime='text/csv')

        return df_final


def daily_signal_changes(df):
    """
    Look max date in dataframe which is filter on signals only, then compare this when them IG previous signal.
    """
    st.header('IG Daily Changes')
    if st.checkbox('IG Signal Changes'):
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        max_date = df['Date'].max()
        df_daily = df.loc[(df['Date'] == max_date)]
        st.write(df_daily)


def filedownload(df, sector_ig):
    """
    Download buy and sell data
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="Buy Sell Triggers {sector_ig}.csv">Download to CSV File</a>'
    return href

def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')


if __name__ == '__main__':
    app()