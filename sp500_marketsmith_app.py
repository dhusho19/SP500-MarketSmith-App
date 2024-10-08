import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import base64
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.dates import date2num
from scipy.stats import linregress
from datetime import datetime
import webbrowser
import tempfile
from streamlit.components.v1 import html
from functools import reduce


st.set_page_config(layout="wide")

tab_main, tab_signal = st.tabs(['📈 Main', '📈 Sector & IG Signals'])

with tab_main:

    st.sidebar.header('Sector & Industry Groups')
    # Create placeholders for dynamic filters (Sector and Industry)
    dynamic_sector_placeholder = st.sidebar.empty()
    dynamic_industry_placeholder = st.sidebar.empty()

    # date picker filters, find the minimum date in the dataset and use that as the start date
    date_string = '2021-05-19'
    date_format = '%Y-%m-%d'
    min_date = datetime.strptime(date_string, date_format)

    st.sidebar.header("Date Range")
    start_date = st.sidebar.date_input('Begin', min_date)
    end_date = st.sidebar.date_input('End')

    st.sidebar.header('Moving Averages')
    # Short and Long Term Moving Average filters
    mv_options = ['EMA','SMA']
    sma_ema = st.sidebar.radio('',mv_options,key=5)
    short_term = st.sidebar.slider('ST', min_value=1,
                                            max_value=30,
                                            value=10)

    mid_term = st.sidebar.slider('IT', min_value=1,
                                            max_value=50,
                                            value=20)

    long_term = st.sidebar.slider('LT',  min_value=1,
                                            max_value=200,
                                            value=30)

    # column names for long and short moving average columns
    short_term_col = sma_ema + '_' + str(short_term)
    mid_term_col = sma_ema + '_' + str(mid_term)
    long_term_col = sma_ema + '_' + str(long_term)


    def app():

        # allow user to upload their own file through a streamlit sile uploader
        uploaded_file = st.file_uploader("Please Upload a CSV File",type=['csv'],key=1)
        if uploaded_file is not None:
            file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size,"Key":21}
            df = pd.read_csv(uploaded_file, parse_dates=['Date Stamp'])

            # extract the date attribute from the datetime object, in order to allow filtering of df between two dates
            df['Date'] = df['Date Stamp'].dt.date
            # Now we have the date from this column we can drop it and set it as the index
            df.drop('Date Stamp', axis=1, inplace=True)
            df.set_index('Date', inplace=True)

            # Values to exclude
            exclude_igs = ['Energy-Coal','Finance-Blank Check','Finance-ETF / ETN','Finance-Publ Inv Fd-Bal','Finance-Publ Inv Fd-Bond','Finance-Publ Inv Fd-Eqt',
                           'Finance-Publ Inv Fd-Glbl','Finance-Savings & Loan','Food-Dairy Products','Media-Periodicals','Office Supplies Mfg',
                           'Oil&Gas-Royalty Trust','Retail-Mail Order&Direct','Retail/Whlsle-Jewelry','Retail/Whlsle-Office Sup','Tobacco']

            df = df[~df['Name'].isin(exclude_igs)]
            # sidebar filters
            # load list of Sector values & create filter
            sector = sorted(df['Sector'].unique().tolist())
            selected_sector = dynamic_sector_placeholder.selectbox('Select Sector', sector)

            # list of all relevant Industries based on the Sector filter selection
            industry = df.loc[df['Sector'] == selected_sector, 'Name'].unique()
            # store users Industry selection
            selected_industry = dynamic_industry_placeholder.selectbox('Select Industry', industry)

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
            # take the average of the weighted values by date by sector
            df_selected_sector['Sector Rank Avg'] = df_selected_sector.groupby([df_selected_sector.index,'Sector'])['Sector Rank'].transform('mean')
            # the sum of the sector rank by date by sector
            df_sector_rank = df_selected_sector.groupby([df_selected_sector.index,'Sector'])['Sector Rank'].sum().reset_index()
            df_sector_rank_avg = df_selected_sector.groupby([df_selected_sector.index,'Sector'])['Sector Rank Avg'].mean().reset_index()

            # df formatting & merge
            df_sector_final = pd.merge(df_sector_rank, df_sector_rank_avg, how='left', on=['Date', 'Sector'])
            df_sector_final.set_index('Date',inplace=True)
            st.markdown("""---""")
            # function calls
            data_col, chart_col = st.columns(2)
            crossover_strategy(df_sector_final, 'Sector Rank Avg')
            crossover_strategy(df_selected_industry, 'Ind Group Rank')
            with chart_col:
                st.header('Graphing Sectors & IG')
                if st.checkbox('Plot Regression'):
                    plotting_regression(df_sector_final,df_selected_industry,selected_sector,selected_industry)
                else:
                    plotting(df_sector_final,df_selected_industry,selected_sector,selected_industry)
            with data_col:
                df_sector_daily_changes = summary_sector(df)
                df_daily_changes = summary(df)
            st.markdown("""---""")
            daily_sector_signal_changes(df_sector_daily_changes)
            daily_signal_changes(df_daily_changes)
        else:
            st.subheader("About")
            st.info("Built with Streamlit")
            st.info("hushon.d@googlemail.com")
            st.text("Donovan Hushon")


    def crossover_strategy(df, rank_col):

        if sma_ema == 'SMA':
            # Sector/Industry
            df[short_term_col] = df[rank_col].rolling(window=short_term[0], min_periods=1).mean()
            df[mid_term_col] = df[rank_col].rolling(window=mid_term[1], min_periods=1).mean()
            df[long_term_col] = df[rank_col].rolling(window=long_term[2], min_periods=1).mean()

        elif sma_ema == 'EMA':
            # Sector/Industry
            df[short_term_col] = df[rank_col].ewm(span=short_term, adjust=False).mean()
            df[mid_term_col] = df[rank_col].ewm(span=mid_term, adjust=False).mean()
            df[long_term_col] =  df[rank_col].ewm(span=long_term, adjust=False).mean()


            # signal alerts for crossover strategy Sector
            df['alert_st'] = 0.0
            df['alert_st'] = np.where(df[short_term_col]>df[mid_term_col], 1.0, 0.0)
            df['alert_lt'] = 0.0
            df['alert_lt'] = np.where(df[mid_term_col]>df[long_term_col], 1.0, 0.0)
            # create a new column 'Position' which is a day-to-day difference of the alert column.
            df['position_st'] = df['alert_st'].diff() # 1 is BUY
            df['position_lt'] = df['alert_lt'].diff()

            return df

    def crossover_marketing_performance(df, rank_col):
        df = df.copy()

        if sma_ema == 'SMA':
            # Sector/Industry
            df[short_term_col] = df[rank_col].rolling(window=short_term[0], min_periods=1).mean()
            df[long_term_col] = df[rank_col].rolling(window=long_term[2], min_periods=1).mean()

        elif sma_ema == 'EMA':
            # Sector/Industry
            df[short_term_col] = df[rank_col].ewm(span=short_term, adjust=False).mean()
            df[long_term_col] =  df[rank_col].ewm(span=long_term, adjust=False).mean()


            # signal alerts for crossover strategy Sector
            df['alert'] = 0.0
            df['alert'] = np.where(df[short_term_col]>df[long_term_col], 1.0, 0.0)

            # create a new column 'Position' which is a day-to-day difference of the alert column.
            df['position'] = df['alert'].diff() # 1 is BUY

            return df

        # define function to open chart in new browser window
    def open_chart(fig, selected_industry):
        industry = selected_industry.replace("/", "_") + "_" # Messes up the file path

        # convert Plotly figure to HTML and save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, prefix=industry, suffix='.html') as f:
            f.write(fig.to_html(include_plotlyjs='cdn', full_html=True))
            url = 'file://' + f.name  # constr  uct URL to temporary file

        # open URL in new browser window
        webbrowser.open(url, new=2)

    def plotting_regression(df_sector_rank, df_selected_industry, selected_sector, selected_industry):

        # filtered data to create plots
        if st.checkbox('Plot Sector Ranking Graph'):
            st.subheader('IBD Sector Ranking')

            fig = px.line(df_sector_rank, x=df_sector_rank.index,
                        y=['Sector Rank Avg', df_sector_rank[short_term_col], df_sector_rank[mid_term_col], df_sector_rank[long_term_col]],
                        hover_name='Sector', template='plotly_dark',
                        color_discrete_map={'Sector Rank Avg': 'light blue', short_term_col: 'green', mid_term_col: 'yellow', long_term_col: 'red'}
                        )
            add_regression_and_std_lines(fig, df_sector_rank.index, df_sector_rank['Sector Rank Avg'], '', 'red', z_scores=[1, 2])

            fig.add_scatter(x=df_sector_rank.loc[df_sector_rank['position_st'] == -1].index,
                            y=df_sector_rank[short_term_col][df_sector_rank['position_st'] == -1],
                            name='ST Buy',
                            mode='markers',
                            marker_symbol='star-triangle-up',
                            marker_color='green', marker_size=15)

            fig.add_scatter(x=df_sector_rank.loc[df_sector_rank['position_st'] == 1].index,
                            y=df_sector_rank[short_term_col][df_sector_rank['position_st'] == 1],
                            name='ST Sell',
                            mode='markers',
                            marker_symbol='star-triangle-down',
                            marker_color='red', marker_size=15)

            fig.add_scatter(x=df_sector_rank.loc[df_sector_rank['position_lt'] == -1].index,
                            y=df_sector_rank[mid_term_col][df_sector_rank['position_lt'] == -1],
                            name='LT Buy',
                            mode='markers',
                            marker_symbol='star-triangle-up',
                            marker_color='blue', marker_size=15)

            fig.add_scatter(x=df_sector_rank.loc[df_sector_rank['position_lt'] == 1].index,
                            y=df_sector_rank[mid_term_col][df_sector_rank['position_lt'] == 1],
                            name='LT Sell',
                            mode='markers',
                            marker_symbol='star-triangle-down',
                            marker_color='orange', marker_size=15)

            fig.update_layout(title=selected_sector,
                            xaxis_title="Date",
                            yaxis_title="Sector Ranking",
                            legend_title=''
                            )

            fig.update_yaxes(autorange="reversed")
            fig.update_traces(patch={"line": {"dash": 'dot'}}, selector={"legendgroup": "Sector Rank Avg"})


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
                csv = convert_df(sorted_sector_df.loc[:,['Date','Sector','Sector Rank Avg',short_term_col,long_term_col,'Buy Sell ST']].loc[(sorted_sector_df['position_st'].isin([-1,1]))])
                st.download_button(label="Download data as CSV",
                                data=csv,
                                file_name='Sector_Latest_Signals.csv',
                                mime='text/csv')

                # write df to streamlit app
                st.write(sorted_sector_df.loc[:,['Date','Sector','Sector Rank Avg','Buy Sell ST']].loc[(sorted_sector_df['position_st'].isin([-1,1]))].head(3))

            # create button to open chart in new window
            if st.button('Open chart'):
                open_chart(fig, selected_industry)
                # display Plotly Express chart
                st.plotly_chart(fig)
            else:
                return st.plotly_chart(fig)

        if st.checkbox('Plot IG Ranking Graph'):
            st.subheader('IBD Industry Group Ranking')

            # Ensure that the DataFrame is sorted by the datetime index
            df_selected_industry = df_selected_industry.sort_index()

            fig = px.line(df_selected_industry, x=df_selected_industry.index, y=['Ind Group Rank',df_selected_industry[short_term_col],df_selected_industry[mid_term_col],df_selected_industry[long_term_col]],
                            hover_name='Name',template = 'plotly_dark',
                            color_discrete_map={'Ind Group Rank':'light blue',short_term_col:'green',mid_term_col:'yellow',long_term_col:'red'}
                            )
            add_regression_and_std_lines(fig, df_selected_industry.index, df_selected_industry['Ind Group Rank'], '', 'red', z_scores=[1, 2])

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
                st.write(sorted_industry_df.loc[:,['Date','Sector','Name','Ind Group Rank','Buy Sell ST']].loc[(sorted_industry_df['position_st'].isin([-1,1]))].head(3))

            # create button to open chart in new window
            if st.button('Open chart'):
                open_chart(fig, selected_industry)
                # display Plotly Express chart
                st.plotly_chart(fig)
            else:
                return st.plotly_chart(fig)

        if st.checkbox('Plot IG Industry Market Value'):
            st.subheader('IBD Industry Group Market Value')

            # Calculate regression line and standard deviation lines
            slope, intercept, _, _, _ = linregress(date2num(df_selected_industry.index), df_selected_industry['Ind Mkt Val (bil)'])
            regression_line = intercept + slope * date2num(df_selected_industry.index)
            std_dev = np.std(df_selected_industry['Ind Mkt Val (bil)'])
            z_scores = [1, 2]
            std_colors = ['green', 'orange']

            upper_std_lines = [regression_line + z_score * std_dev for z_score in z_scores]
            lower_std_lines = [regression_line - z_score * std_dev for z_score in z_scores]

            # Create scatter plot for 'Ind Mkt Val (bil)' with a dashed line
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_selected_industry.index, y=df_selected_industry['Ind Mkt Val (bil)'],
                                    mode='lines', line=dict(color='lightskyblue', dash='dash'),
                                    name='Ind Mkt Val (bil)'))

            # Add regression line and standard deviation lines as separate traces
            fig.add_trace(go.Scatter(x=df_selected_industry.index, y=regression_line, mode='lines',
                                    name='Regression Line', line=dict(color='red', dash='dot')))
            for i, z_score in enumerate(z_scores):
                fig.add_trace(go.Scatter(x=df_selected_industry.index, y=upper_std_lines[i], mode='lines',
                                        name=f'Upper Std Dev ({z_score})', line=dict(color=std_colors[i], dash='dash')))
                fig.add_trace(go.Scatter(x=df_selected_industry.index, y=lower_std_lines[i], mode='lines',
                                        name=f'Lower Std Dev ({z_score})', line=dict(color=std_colors[i], dash='dash')))

            # Customize layout
            fig.update_layout(
                title=selected_industry,
                xaxis_title='Date',
                yaxis_title='Ind Mkt Val (bil)',
                legend_title=''
            )
            # create button to open chart in new window
            if st.button('Open chart'):
                open_chart(fig, selected_industry)
                # display Plotly Express chart
                st.plotly_chart(fig)
            else:
                return st.plotly_chart(fig)

    def plotting(df_sector_rank, df_selected_industry, selected_sector, selected_industry):
        # filtered data to create plots
        if st.checkbox('Plot Sector Ranking Graph'):
            st.subheader('IBD Sector Ranking')

            fig = px.line(df_sector_rank, x=df_sector_rank.index,
                        y=['Sector Rank Avg', df_sector_rank[short_term_col], df_sector_rank[mid_term_col], df_sector_rank[long_term_col]],
                        hover_name='Sector', template='plotly_dark',
                        color_discrete_map={'Sector Rank Avg': 'light blue', short_term_col: 'green', mid_term_col: 'yellow', long_term_col: 'red'}
                        # trendline="ols"  # Add this line for regression line
                        )

            fig.add_scatter(x=df_sector_rank.loc[df_sector_rank['position_st'] == -1].index,
                            y=df_sector_rank[short_term_col][df_sector_rank['position_st'] == -1],
                            name='ST Buy',
                            mode='markers',
                            marker_symbol='star-triangle-up',
                            marker_color='green', marker_size=15)

            fig.add_scatter(x=df_sector_rank.loc[df_sector_rank['position_st'] == 1].index,
                            y=df_sector_rank[short_term_col][df_sector_rank['position_st'] == 1],
                            name='ST Sell',
                            mode='markers',
                            marker_symbol='star-triangle-down',
                            marker_color='red', marker_size=15)

            fig.add_scatter(x=df_sector_rank.loc[df_sector_rank['position_lt'] == -1].index,
                            y=df_sector_rank[mid_term_col][df_sector_rank['position_lt'] == -1],
                            name='LT Buy',
                            mode='markers',
                            marker_symbol='star-triangle-up',
                            marker_color='blue', marker_size=15)

            fig.add_scatter(x=df_sector_rank.loc[df_sector_rank['position_lt'] == 1].index,
                            y=df_sector_rank[mid_term_col][df_sector_rank['position_lt'] == 1],
                            name='LT Sell',
                            mode='markers',
                            marker_symbol='star-triangle-down',
                            marker_color='orange', marker_size=15)

            fig.update_layout(title=selected_sector,
                            xaxis_title="Date",
                            yaxis_title="Sector Ranking",
                            legend_title=''
                            )

            fig.update_yaxes(autorange="reversed")
            fig.update_traces(patch={"line": {"dash": 'dot'}}, selector={"legendgroup": "Sector Rank Avg"})


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
                csv = convert_df(sorted_sector_df.loc[:,['Date','Sector','Sector Rank Avg',short_term_col,long_term_col,'Buy Sell ST']].loc[(sorted_sector_df['position_st'].isin([-1,1]))])
                st.download_button(label="Download data as CSV",
                                data=csv,
                                file_name='Sector_Latest_Signals.csv',
                                mime='text/csv')

                # write df to streamlit app
                st.write(sorted_sector_df.loc[:,['Date','Sector','Sector Rank Avg','Buy Sell ST']].loc[(sorted_sector_df['position_st'].isin([-1,1]))].head(3))

            # create button to open chart in new window
            if st.button('Open chart'):
                open_chart(fig, selected_industry)
                # display Plotly Express chart
                st.plotly_chart(fig)
            else:
                return st.plotly_chart(fig)

        if st.checkbox('Plot IG Ranking Graph'):
            st.subheader('IBD Industry Group Ranking')

            # Ensure that the DataFrame is sorted by the datetime index
            df_selected_industry = df_selected_industry.sort_index()

            fig = px.line(df_selected_industry, x=df_selected_industry.index, y=['Ind Group Rank',df_selected_industry[short_term_col],df_selected_industry[mid_term_col],df_selected_industry[long_term_col]],
                            hover_name='Name',template = 'plotly_dark',
                            color_discrete_map={'Ind Group Rank':'light blue',short_term_col:'green',mid_term_col:'yellow',long_term_col:'red'}
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
                            xaxis_title='Date',
                            yaxis_title='IG Ranking',
                            legend_title ='')

            fig.update_yaxes(autorange='reversed')
            fig.update_traces(patch={'line': {'dash': 'dot'}}, selector={'legendgroup': 'Ind Group Rank'})

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
                st.write(sorted_industry_df.loc[:,['Date','Sector','Name','Ind Group Rank','Buy Sell ST']].loc[(sorted_industry_df['position_st'].isin([-1,1]))].head(3))

            # create button to open chart in new window
            if st.button('Open chart'):
                open_chart(fig, selected_industry)
                # display Plotly Express chart
                st.plotly_chart(fig)
            else:
                return st.plotly_chart(fig)

        if st.checkbox('Plot IG Industry Market Value'):
            st.subheader('IBD Industry Group Market Value')

            # Create scatter plot for 'Ind Mkt Val (bil)' with a dashed line
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_selected_industry.index, y=df_selected_industry['Ind Mkt Val (bil)'],
                                    mode='lines', line=dict(color='lightskyblue', dash='dash'),
                                    name='Ind Mkt Val (bil)'))

            # Customize layout
            fig.update_layout(
                title=selected_industry,
                xaxis_title='Date',
                yaxis_title='Ind Mkt Val (bil)',
                legend_title=''
            )

            fig.update_traces(showlegend=True)
            # create button to open chart in new window
            if st.button('Open chart'):
                open_chart(fig, selected_industry)
                # display Plotly Express chart
                st.plotly_chart(fig)
            else:
                return st.plotly_chart(fig)

    # Function to calculate regression line and standard deviation lines
    def add_regression_and_std_lines(fig, x_values, y_values, line_name, color, z_scores=[1]):
        # Hardcoded colors for standard deviation lines
        std_colors = ['green', 'orange']

        slope, intercept, _, _, _ = linregress(date2num(x_values), y_values)
        regression_line = intercept + slope * date2num(x_values)

        # Add trendline
        trendline = go.Scatter(x=x_values,
                            y=regression_line,
                            mode='lines',
                            name=f'{line_name} Regression Line',
                            line=dict(color=color, dash='dot'))

        fig.add_trace(trendline)

        # Calculate standard deviation lines
        for i, z_score in enumerate(z_scores):
            std_dev = np.std(y_values)
            upper_std_line = regression_line + z_score * std_dev
            lower_std_line = regression_line - z_score * std_dev

            # Add standard deviation lines
            upper_std_trace = go.Scatter(x=x_values,
                                        y=upper_std_line,
                                        mode='lines',
                                        name=f'{line_name} Upper Std Dev ({z_score})',
                                        line=dict(color=std_colors[i], dash='dash'))
            lower_std_trace = go.Scatter(x=x_values,
                                        y=lower_std_line,
                                        mode='lines',
                                        name=f'{line_name} Lower Std Dev ({z_score})',
                                        line=dict(color=std_colors[i], dash='dash'))
            fig.add_trace(upper_std_trace)
            fig.add_trace(lower_std_trace)


    def summary_sector(df):
        st.header('Sector & IG Summary')
        if st.checkbox('Sector'):
            df.reset_index(inplace=True)

            # Create a IG list of all unqiue IG's
            sector_lst = sorted(df['Sector'].unique().tolist())

            # Iterate through each IG and load into a list & convert to an NumPy array
            lst = []
            for i in sector_lst:
                df_sector = df.loc[(df['Sector'] == i)]
                df_sector_ranking = sector_ranking(df_sector)
                crossover_strategy(df_sector_ranking, 'Sector Rank Avg')
                lst.append(df_sector_ranking)

            # create an empty list to store all dataframes, then concatenate them at the end of the iteration
            dfs = []
            for x in lst:
                dfs.append(x)
            df1_sector = pd.concat(dfs, ignore_index=True)

            # create buy and sell column, to easily identify the triggers
            df1_sector['Buy Sell ST'] = np.where(df1_sector['alert_st'] == 0,'BUY','SELL')
            df1_sector['Buy Sell LT'] = np.where(df1_sector['alert_lt'] == 0,'BUY','SELL')
            df1_sector.sort_values(by=['Sector','Date'], ascending=True, inplace=True)

            # Filter Dataframes to only look at rows which are signals
            df_sector_st = df1_sector.loc[df1_sector['position_st'].isin([-1,1])]
            df_sector_lt = df1_sector.loc[df1_sector['position_lt'].isin([-1,1])]

            sector_frames = [df_sector_st, df_sector_lt]
            df_sector_frames = pd.concat(sector_frames)

            df_sector_frames.sort_values(by=['Sector','Date'], ascending=True, inplace=True)

            # Pull back the latest two signals per IG
            df_sector_final = df_sector_frames.groupby('Sector').tail(1).reset_index(drop=True)
            df_sector_final.drop(['Sector Rank','alert_st','alert_lt','position_st','position_lt'], axis=1, inplace=True)

            # Rounding formatting
            df_sector_final[short_term_col] = df_sector_final[short_term_col].astype('float32').round(2).astype('int')
            df_sector_final[mid_term_col] = df_sector_final[mid_term_col].astype('float32').round(2).astype('int')
            df_sector_final[long_term_col] = df_sector_final[long_term_col].astype('float32').round(2).astype('int')

            # Sort DataFrame and reshape it to merge each IG onto the one row
            df_sector_final.sort_values(by=['Sector','Date'], ascending=True, inplace=True)

            # Selection filter for sectors
            unique_sectors = sorted(df_sector_final['Sector'].unique().tolist())
            sector_options = st.multiselect('Sectors of Interest',unique_sectors, default=unique_sectors, key="10")

            # Short-Term signal filter
            st_signal = ['BUY', 'SELL']
            st_signal_options = st.multiselect('Short-Term Buy & Sell Signal',st_signal, default=st_signal, key="11")

            # Long-Term signal filter
            lt_signal = ['BUY', 'SELL']
            lt_signal_options = st.multiselect('Long-Term Buy & Sell Signal',lt_signal, default=lt_signal, key="12")

            # Find the latest Sector Rank and pull it  through to the summary
            max_date = df1_sector['Date'].max()
            df_sector_latest = df1_sector.loc[df1_sector['Sector'].isin(sector_options) & (df1_sector['Date'] == max_date)]

            df_sector_final = df_sector_final.loc[df_sector_final['Sector'].isin(sector_options)]
            df_sector_final = pd.merge(df_sector_final, df_sector_latest, on=['Sector'], how='left')

            # Rename the columns were two instances occur, validation the data is correct pulling through date
            df_sector_final.rename(columns = {'Date_x':'Date','Sector Rank Avg_x':'Sector Rank Avg Old', short_term_col+'_x':short_term_col, mid_term_col+'_x':mid_term_col,long_term_col+'_x':long_term_col,'Buy Sell ST_x':'Buy Sell ST','Buy Sell LT_x':'Buy Sell LT','Sector Rank Avg_y':'Sector Rank Avg'},inplace=True)

            # Drop the latest date column & dropped the instances of Sector Rnk when the signal occurred.
            df_sector_final.drop(['Sector Rank Avg Old','Date_y','Sector Rank', short_term_col+'_y', mid_term_col+'_y', long_term_col+'_y', 'alert_st', 'alert_lt', 'position_st', 'position_lt','Buy Sell ST_y', 'Buy Sell LT_y'], axis=1, inplace=True)

            df_sector_final['Sector Rank Avg'] = df_sector_final['Sector Rank Avg'].astype('float32').round(2).astype('int')
            #df_sector_final['Date'] = df_sector_final['Date'].astype('float32').round(2).astype('int')
            df_sector_final['Date'] = pd.to_datetime(df_sector_final['Date'])
            # Restructure columns in dataframe
            df_sector_final = df_sector_final.reindex(columns=['Date','Sector','Sector Rank Avg',short_term_col,mid_term_col,long_term_col,'Buy Sell ST','Buy Sell LT'])

            # Filter dataframe
            df_sector_final = df_sector_final.loc[df_sector_final['Sector'].isin(sector_options) & df_sector_final['Buy Sell ST'].isin(st_signal_options) & df_sector_final['Buy Sell LT'].isin(lt_signal_options)]
            st.write(df_sector_final)

            # Call download function
            csv = convert_df(df1_sector)
            st.download_button(label="Download full dataset as CSV",
                data=csv,
                file_name='Sector_Signals.csv',
                mime='text/csv')

            # Call download function
            csv = convert_df(df_sector_final)
            st.download_button(label="Download data as CSV",
                data=csv,
                file_name='Sector_Latest_Signals.csv',
                mime='text/csv')

            return df_sector_final


    def summary(df):
        if st.checkbox('IG'):
            df.reset_index(inplace=True)
            # an index column appears if you select the sector's summary first, which affects the reshaping. Therefore, need to drop the column if it exists.
            if 'index' in df.columns:
                df.drop('index', axis=1, inplace=True)

            # Create a IG list of all unqiue IG's
            industry_lst = sorted(df['Name'].unique().tolist())

            # Iterate through each IG and load into a list & convert to an NumPy array
            ig_lst = []
            for ig in industry_lst:
                df_industry = df.loc[(df['Name'] == ig)]
                crossover_strategy(df_industry, 'Ind Group Rank')
                ig_lst.append(df_industry)
            arr = np.asarray(ig_lst)

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
            df_final1 = df_final.groupby('Name').tail(1).reset_index(drop=True)
            df_final1.drop(['alert_st','alert_lt','position_st','position_lt'], axis=1, inplace=True)

            # Rounding formatting
            df_final1[short_term_col] = df_final1[short_term_col].astype('float32').round(2).astype('int')
            df_final1[mid_term_col] = df_final1[mid_term_col].astype('float32').round(2).astype('int')
            df_final1[long_term_col] = df_final1[long_term_col].astype('float32').round(2).astype('int')

            # Sort DataFrame and reshape it to merge each IG onto the one row
            df_final1.sort_values(by=['Name','Date'], ascending=True, inplace=True)

            # IG Filter
            unique_sectors = sorted(df_final1['Sector'].unique().tolist())
            sector_options = st.multiselect('Sectors of Interest',unique_sectors, default=unique_sectors,key="14")
            # Short-Term signal filter
            ig_st_signal = ['BUY', 'SELL']
            ig_st_signal_options = st.multiselect('Short-Term Buy & Sell Signal',ig_st_signal, default=ig_st_signal, key="15")
            # Long-Term signal filter
            ig_lt_signal = ['BUY', 'SELL']
            ig_lt_signal_options = st.multiselect('Long-Term Buy & Sell Signal',ig_lt_signal, default=ig_lt_signal, key="16")

            # Find the latest Ind Group Rank / Mkt Val and pull it  through to the summary
            max_date = df['Date'].max()
            df_latest = df.loc[df['Sector'].isin(sector_options) & (df['Date'] == max_date)]

            # Filter dataframe
            df_final1 = df_final1.loc[df_final1['Sector'].isin(sector_options) & df_final1['Buy Sell ST'].isin(ig_st_signal_options) & df_final1['Buy Sell LT'].isin(ig_lt_signal_options)]
            df_final2 = pd.merge(df_final1, df_latest, on=['Symbol','Name','Sector'], how='left')
            # Rename the columns were two instances occur, validation the data is correct pulling through date
            df_final2.rename(columns = {'Date_x':'Date', 'Date_y':'Latest Date','Ind Group Rank_y':'Ind Group Rank', 'Ind Mkt Val (bil)_y':'Ind Mkt Val (bil)'},inplace=True)
            # Drop the latest date column & dropped the instances of IG Rnk / Mkt Val when the signal occurred.
            df_final2.drop(['Ind Group Rank_x','Ind Mkt Val (bil)_x', 'Latest Date'], axis=1, inplace=True)
            # Re-order the dataframe
            df_final2 = df_final2.reindex(columns=['Date','Symbol','Name','Sector','Ind Group Rank','Ind Mkt Val (bil)',short_term_col,mid_term_col,long_term_col,'Buy Sell ST','Buy Sell LT'])
            st.write(df_final2)


            # Call download function
            csv = convert_df(df1)
            st.download_button(label="Download full dataset as CSV",
                data=csv,
                file_name='IG_Signals.csv',
                mime='text/csv')

            # Call download function
            csv = convert_df(df_final2)
            st.download_button(label="Download data as CSV",
                data=csv,
                file_name='IG_Latest_Signals.csv',
                mime='text/csv')

            return df_final2


    def sector_ranking(df):
        # create a total market value to utilise for weighted sector rank, then add this to the dataframe
        total_mkt_val = df.groupby(['Date','Sector'])['Ind Mkt Val (bil)'].transform('sum')
        df['total_mkt_val'] = total_mkt_val

        # calculate a percentage weight for each industry
        df['weight'] = df['Ind Mkt Val (bil)']/df['total_mkt_val']
        # use the newly created weight column to calculate sector rank
        df['Sector Rank'] = df['weight']*df['Ind Group Rank']
        #df['Sector Rank'] = df['Sector Rank'].astype('float32').round(2).astype('int')
        df['Sector Rank Avg'] = df.groupby(['Date','Sector'])['Sector Rank'].transform('mean')
        df_sector_rank = df.groupby(['Date','Sector'])['Sector Rank'].sum().reset_index()
        df_sector_rank_avg = df.groupby(['Date','Sector'])['Sector Rank Avg'].mean().reset_index()

        df_sector_rank.set_index('Date',inplace=True)
        df_sector_final = pd.merge(df_sector_rank, df_sector_rank_avg, how='left', on=['Date', 'Sector'])

        return df_sector_final


    def daily_sector_signal_changes(df):
        """
        Look max date in dataframe which is filter on signals only, then compare this when them IG previous signal.
        """
        st.header('Sector & IG Daily Changes')
        if st.checkbox('Sector Signal Changes'):
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            max_date = df['Date'].max()
            df_sector_daily = df.loc[(df['Date'] == max_date)]
            st.write(df_sector_daily)


    def daily_signal_changes(df):
        """
        Look max date in dataframe which is filter on signals only, then compare this when them IG previous signal.
        """
        if st.checkbox('IG Signal Changes'):
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            max_date = df['Date'].max()
            df_daily = df.loc[(df['Date'] == max_date)]
            st.write(df_daily)


    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

with tab_signal:
    st.title('Market Signal Performance')
    def app_signals():
        # allow user to upload their own file through a streamlit sile uploader
        uploaded_file = st.file_uploader("Please Upload a CSV File",type=['csv'],key=3)
        if uploaded_file is not None:
            file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size,"Key":23}
            df = pd.read_csv(uploaded_file, parse_dates=['Date'])

            df['Date'] = df['Date'].dt.date

            df_filtered = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

            # Now we have the date from this column we can drop it and set it as the index
            if 'Unnamed: 0' in df.columns:
                df.drop('Unnamed: 0', axis=1, inplace=True)
            else:
                df.drop('id', axis=1, inplace=True)
            df.set_index('Date', inplace=True)

            # enable toggle to view & unview the dataset
            if st.checkbox('Show File Details & DataFrame'):
                st.write(file_details)
                st.markdown('** Original Dataset:**')
                st.write('>Data Dimension: ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.')
                st.write(df)

            col_data, col_chart = st.columns(2)
            with col_data:
                # size() method to count the number of occurrences of each signal type
                # Short Term Buy Signals
                df_st_buy_cnt = df_filtered[df_filtered['Buy Sell ST'] == 'BUY'].groupby('Date').size().reset_index(name='ST BUY')
                df_st_buy_cnt.rename(columns = {'Buy Sell ST':'ST BUY'},inplace=True)

                # Short Term Sell Signals
                df_st_sell_cnt = df_filtered[df_filtered['Buy Sell ST'] == 'SELL'].groupby('Date').size().reset_index(name='ST SELL')
                df_st_sell_cnt.rename(columns = {'Buy Sell ST':'ST SELL'},inplace=True)

                # Long Term Buy Signals
                df_lt_buy_cnt = df_filtered[df_filtered['Buy Sell LT'] == 'BUY'].groupby('Date').size().reset_index(name='LT BUY')
                df_lt_buy_cnt.rename(columns = {'Buy Sell LT':'LT BUY'},inplace=True)

                # Long Term Sell Signals
                df_lt_sell_cnt = df_filtered[df_filtered['Buy Sell LT'] == 'SELL'].groupby('Date').size().reset_index(name='LT SELL')
                df_lt_sell_cnt.rename(columns = {'Buy Sell LT':'LT SELL'},inplace=True)

                # Merge datasets
                df_st_cnt = pd.merge(df_st_buy_cnt, df_st_sell_cnt, on=['Date'], how='left')
                df_lt_cnt = pd.merge(df_lt_buy_cnt, df_lt_sell_cnt, on=['Date'], how='left')
                df_final_cnt = pd.merge(df_st_cnt, df_lt_cnt, on=['Date'], how='left')

                if 'IG_Signals' in file_details['FileName']:
                    denominator = 197
                elif 'Sector_Signals' in file_details['FileName']:
                    denominator = 33
                else:
                    # Handle the case where neither 'IG_Signals' nor 'Sector_Signals' is in the file name
                    denominator = 1

                df_final_cnt[['ST Buy %', 'ST Sell %', 'LT Buy %', 'LT Sell %']] = df_final_cnt[['ST BUY', 'ST SELL', 'LT BUY', 'LT SELL']] / denominator * 100

                df_final_cnt[['ST Buy %', 'ST Sell %', 'LT Buy %', 'LT Sell %']] = (df_final_cnt[['ST Buy %', 'ST Sell %', 'LT Buy %', 'LT Sell %']]
                                                                                    .astype('float32')
                                                                                    .round(0)
                                                                                    .astype('int'))
                df_final_cnt.set_index('Date', inplace=True)

                # Display the filtered dataframe
                st.write(df_final_cnt)
                st.markdown("""---""")
                # Create an empty dataframe to store the intermediate results
                df_final_overall = df_final_cnt.copy()

                column_mapping = {
                    'ST Buy %': '_ST_Buy_%',
                    'ST Sell %': '_ST_Sell_%',
                    'LT Buy %': '_LT_Buy_%',
                    'LT Sell %': '_LT_Sell_%',
                }

                dataframes = []

                for col in column_mapping:
                    df = crossover_marketing_performance(df_final_overall, col)
                    df['Buy Sell'] = np.where(df['alert'] == 0, 'BUY', 'SELL')
                    suffix = column_mapping[col]
                    df.rename(columns={
                        short_term_col: short_term_col + suffix,
                        long_term_col: long_term_col + suffix,
                        'Buy Sell': 'Buy Sell ' + col,
                        'position': 'position' + suffix
                    }, inplace=True)
                    df.drop(['alert'], axis=1, inplace=True)
                    dataframes.append(df)


                df1, df2, df3, df4 = dataframes

                dfs = [df1, df2, df3, df4]

                df_combined = reduce(lambda  left,right: pd.merge(left,right,on=['Date', 'ST BUY', 'ST SELL',
                                                                                 'LT BUY', 'LT SELL',
                                                                                 'ST Buy %', 'ST Sell %',
                                                                                 'LT Buy %', 'LT Sell %'],
                                                                                  how='outer'), dfs)

                # Rounding formatting
                columns = [short_term_col, long_term_col]

                for col in columns:
                    df_combined[col+'_ST_Buy_%'] = df_combined[col+'_ST_Buy_%'].astype('float32').round(2).astype('int')
                    df_combined[col+'_ST_Sell_%'] = df_combined[col+'_ST_Sell_%'].astype('float32').round(2).astype('int')
                    df_combined[col+'_LT_Buy_%'] = df_combined[col+'_LT_Buy_%'].astype('float32').round(2).astype('int')
                    df_combined[col+'_LT_Sell_%'] = df_combined[col+'_LT_Sell_%'].astype('float32').round(2).astype('int')


                # Define the desired order of columns in the legend
                column_order = ['ST Buy %', short_term_col+'_ST_Buy_%', long_term_col+'_ST_Buy_%', 'Buy_ST_Buy_%', 'Sell_ST_Buy_%',
                                'ST Sell %', short_term_col+'_ST_Sell_%', long_term_col+'_ST_Sell_%', 'Buy_ST_Sell_%', 'Sell_ST_Sell_%',
                                'LT Buy %', short_term_col+'_LT_Buy_%', long_term_col+'_LT_Buy_%', 'Buy_LT_Buy_%', 'Sell_LT_Buy_%'
                                'LT Sell %', short_term_col+'_LT_Sell_%', long_term_col+'_LT_Sell_%', 'Buy_LT_Sell_%', 'Sell_LT_Sell_%',
                               ]


                # Reorder columns in df_combined
                df6 = df_combined.loc[:,['ST Buy %', short_term_col+'_ST_Buy_%', long_term_col+'_ST_Buy_%',
                                'ST Sell %', short_term_col+'_ST_Sell_%', long_term_col+'_ST_Sell_%',
                                'LT Buy %', short_term_col+'_LT_Buy_%', long_term_col+'_LT_Buy_%',
                                'LT Sell %', short_term_col+'_LT_Sell_%', long_term_col+'_LT_Sell_%',
                                ]]

            with col_chart:
                if st.checkbox('Plot Regression '):
                    # original overview plot
                    fig_signals = px.line(df_final_cnt, x=df_final_cnt.index,
                                        y=[df_final_cnt['ST Buy %'],df_final_cnt['ST Sell %'],df_final_cnt['LT Buy %'],df_final_cnt['LT Sell %']],
                                        template = 'plotly_dark',
                                        color_discrete_map={'ST Buy %':'green','ST Sell %':'yellow','LT Sell %':'red','LT Buy %':'blue'}
                                        )

                    fig_signals.update_layout(title='Market Performance',
                                        xaxis_title="Date",
                                        yaxis_title="Signal Count",
                                        legend_title=''
                                        )
                    fig_signals.update_traces(patch={"line": {"dash": 'dot'}}, selector={"legendgroup": 'ST Sell %'}).update_traces(patch={"line": {"dash": 'dot'}}, selector={"legendgroup": 'ST Buy %'}).update_traces(patch={"line": {"dash": 'dot'}}, selector={"legendgroup": 'LT Sell %'}).update_traces(patch={"line": {"dash": 'dot'}}, selector={"legendgroup": 'LT Buy %'})

                    # The below adds vertical and horziontal lines as the cursor to the plot
                    fig_signals.update_yaxes(showgrid=False, zeroline=False, showticklabels=True,
                                    showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid')

                    fig_signals.update_xaxes(showgrid=False, zeroline=False, rangeslider_visible=False, showticklabels=True,
                                    showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='solid')

                    # Add regression and multiple standard deviation lines for each line in the plot
                    add_regression_and_std_lines(fig_signals, df_final_cnt.index, df_final_cnt['ST Buy %'], 'ST Buy %', 'green', z_scores=[1, 2])
                    add_regression_and_std_lines(fig_signals, df_final_cnt.index, df_final_cnt['ST Sell %'], 'ST Sell %', 'yellow', z_scores=[1, 2])
                    add_regression_and_std_lines(fig_signals, df_final_cnt.index, df_final_cnt['LT Buy %'], 'LT Buy %', 'blue', z_scores=[1, 2])
                    add_regression_and_std_lines(fig_signals, df_final_cnt.index, df_final_cnt['LT Sell %'], 'LT Sell %', 'red', z_scores=[1, 2])

                    st.plotly_chart(fig_signals)

                else:
                    # original overview plot
                    fig_signals = px.line(df_final_cnt, x=df_final_cnt.index,
                                        y=[df_final_cnt['ST Buy %'],df_final_cnt['ST Sell %'],df_final_cnt['LT Buy %'],df_final_cnt['LT Sell %']],
                                        template = 'plotly_dark',
                                        color_discrete_map={'ST Buy %':'green','ST Sell %':'yellow','LT Sell %':'red','LT Buy %':'blue'}
                                        )

                    fig_signals.update_layout(title='Market Performance',
                                        xaxis_title="Date",
                                        yaxis_title="Signal Count",
                                        legend_title=''
                                        )
                    fig_signals.update_traces(patch={"line": {"dash": 'dot'}}, selector={"legendgroup": 'ST Sell %'}).update_traces(patch={"line": {"dash": 'dot'}}, selector={"legendgroup": 'ST Buy %'}).update_traces(patch={"line": {"dash": 'dot'}}, selector={"legendgroup": 'LT Sell %'}).update_traces(patch={"line": {"dash": 'dot'}}, selector={"legendgroup": 'LT Buy %'})

                    # The below adds vertical and horziontal lines as the cursor to the plot
                    fig_signals.update_yaxes(showgrid=False, zeroline=False, showticklabels=True,
                                    showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid')

                    fig_signals.update_xaxes(showgrid=False, zeroline=False, rangeslider_visible=False, showticklabels=True,
                                    showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='solid')

                    st.plotly_chart(fig_signals)

                # averages plot
            fig_average_signals = px.line(df_combined, x=df_combined.index,
                                y=df6.columns,
                                category_orders={'legendgroup': column_order},
                                template = 'plotly_dark',
                                color_discrete_map={
                                                    'ST Buy %':'green', short_term_col+'_ST_Buy_%':'green', long_term_col+'_ST_Buy_%':'teal',
                                                    'ST Sell %':'yellow', short_term_col+'_ST_Sell_%':'yellow', long_term_col+'_ST_Sell_%':'orange',
                                                    'LT Buy %':'blue', short_term_col+'_LT_Buy_%':'blue', long_term_col+'_LT_Buy_%':'orange',
                                                    'LT Sell %':'red', short_term_col+'_LT_Sell_%':'red', long_term_col+'_LT_Sell_%':'grey'
                                                    })

            fig_average_signals.add_scatter(x=df_combined.loc[df_combined['position'+'_ST_Sell_%'] == -1].index,
                                y=df_combined[short_term_col+'_ST_Sell_%'][df_combined['position'+'_ST_Sell_%'] == -1],
                                name= 'Buy_ST_Sell_%',
                                mode='markers',
                                marker_symbol='star-triangle-up',
                                legendgroup='Buy_ST_Sell_%',
                                marker_color='green', marker_size=15)

            fig_average_signals.add_scatter(x=df_combined.loc[df_combined['position'+'_ST_Sell_%'] == 1].index,
                            y=df_combined[short_term_col+'_ST_Sell_%'][df_combined['position'+'_ST_Sell_%'] == 1],
                            name= 'Sell_ST_Sell_%',
                            mode='markers',
                            marker_symbol='star-triangle-down',
                            legendgroup='Sell_ST_Sell_%',
                            marker_color='red', marker_size=15)

            fig_average_signals.add_scatter(x=df_combined.loc[df_combined['position'+'_ST_Buy_%'] == 1].index,
                            y=df_combined[short_term_col+'_ST_Buy_%'][df_combined['position'+'_ST_Buy_%'] == 1],
                            name= 'Buy_ST_Buy_%',
                            mode='markers',
                            marker_symbol='star-triangle-up',
                            legendgroup='Buy_ST_Buy_%',
                            marker_color='green', marker_size=15)

            fig_average_signals.add_scatter(x=df_combined.loc[df_combined['position'+'_ST_Buy_%'] == -1].index,
                            y=df_combined[short_term_col+'_ST_Buy_%'][df_combined['position'+'_ST_Buy_%'] == -1],
                            name= 'Sell_ST_Buy_%',
                            mode='markers',
                            marker_symbol='star-triangle-down',
                            legendgroup='Sell_ST_Buy_%',
                            marker_color='red', marker_size=15)

            fig_average_signals.add_scatter(x=df_combined.loc[df_combined['position'+'_LT_Sell_%'] == -1].index,
                            y=df_combined[short_term_col+'_LT_Sell_%'][df_combined['position'+'_LT_Sell_%'] == -1],
                            name= 'Buy_LT_Sell_%',
                            mode='markers',
                            marker_symbol='star-triangle-up',
                            legendgroup='Buy_LT_Sell_%',
                            marker_color='green', marker_size=15)

            fig_average_signals.add_scatter(x=df_combined.loc[df_combined['position'+'_LT_Sell_%'] == 1].index,
                            y=df_combined[short_term_col+'_LT_Sell_%'][df_combined['position'+'_LT_Sell_%'] == 1],
                            name= 'Sell_LT_Sell_%',
                            mode='markers',
                            marker_symbol='star-triangle-down',
                            legendgroup='Buy_LT_Sell_%',
                            marker_color='red', marker_size=15)


            fig_average_signals.add_scatter(x=df_combined.loc[df_combined['position'+'_LT_Buy_%'] == 1].index,
                            y=df_combined[short_term_col+'_LT_Buy_%'][df_combined['position'+'_LT_Buy_%'] == 1],
                            name= 'Buy_LT_Buy_%',
                            mode='markers',
                            marker_symbol='star-triangle-up',
                            legendgroup='Buy_LT_Buy_%',
                            marker_color='green', marker_size=15)

            fig_average_signals.add_scatter(x=df_combined.loc[df_combined['position'+'_LT_Buy_%'] == -1].index,
                            y=df_combined[short_term_col+'_LT_Buy_%'][df_combined['position'+'_LT_Buy_%'] == -1],
                            name= 'Sell_LT_Buy_%',
                            mode='markers',
                            marker_symbol='star-triangle-down',
                            legendgroup='Sell_LT_Buy_%',
                            marker_color='red', marker_size=15)


            fig_average_signals.update_layout(title='Market Signal Performance',
                                xaxis_title="Date",
                                yaxis_title="Signal Count",
                                legend_title=''
                                )

            fig_average_signals.update_traces(line_dash='dot', selector=dict(name='ST Buy %'))
            fig_average_signals.update_traces(line_dash='dash', selector=dict(name=short_term_col+'_ST_Buy_%'))
            fig_average_signals.update_traces(line_dash='dot', selector=dict(name='ST Sell %'))
            fig_average_signals.update_traces(line_dash='dash', selector=dict(name=short_term_col+'_ST_Sell_%'))

            fig_average_signals.update_traces(line_dash='dot', selector=dict(name='LT Buy %'))
            fig_average_signals.update_traces(line_dash='dash', selector=dict(name=short_term_col+'_LT_Buy_%'))
            fig_average_signals.update_traces(line_dash='dot', selector=dict(name='LT Sell %'))
            fig_average_signals.update_traces(line_dash='dash', selector=dict(name=short_term_col+'_LT_Sell_%'))

            # The below adds vertical and horziontal lines as the cursor to the plot
            fig_average_signals.update_yaxes(showgrid=False, zeroline=False, showticklabels=True,
                            showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid')

            fig_average_signals.update_xaxes(showgrid=False, zeroline=False, rangeslider_visible=False, showticklabels=True,
                            showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='solid')

            st.plotly_chart(fig_average_signals)


if __name__ == '__main__':
    with tab_main:
        app()
    with tab_signal:
        app_signals()
