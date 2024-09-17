import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import altair as alt

alt.themes.enable("dark")

# Page config
st.set_page_config(
    page_title="Global Leptospirosis Cases",
    layout='wide',
    initial_sidebar_state='expanded'
)
df = pd.read_csv('leptospirosis_cases_latilong.csv')

# Ensure the year columns are strings
year_columns = [str(year) for year in range(2007, 2023)]

# Reshape the dataframe to long format
df_long = pd.melt(df, id_vars=['Region', 'LAT', 'LONG'], value_vars=year_columns, var_name='Year', value_name='Cases')

# Convert the 'Year' column to numeric
df_long['Year'] = df_long['Year'].astype(int)

# Ensure the 'Cases' column is numeric (in case it was treated as strings)
df_long['Cases'] = pd.to_numeric(df_long['Cases'], errors='coerce')

# Title
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)

# Title with icon
st.markdown("""
    <h1 style="display: flex; align-items: center;">
        Global Leptospirosis Cases
        <i class="fas fa-bacteria" style="margin-left: 10px; color: #007bff;"></i>
    </h1>
""", unsafe_allow_html=True)
coln1, coln2 = st.columns([1, 1]) 
with coln1:
     # HTML and CSS to style the text with a box around it
    st.markdown("""
        <div style="border: 2px solid #e0e0e0; padding: 10px; border-radius: 5px; background-color: #333; color: #e0e0e0;">
            Leptospirosis occurs worldwide but is most common in tropical and subtropical areas with high rainfall. The disease is found mainly wherever humans come into contact with the urine of infected animals or a urine-polluted environment.
        </div>
    """, unsafe_allow_html=True)
with coln2:
    st.image('giphy.gif', use_column_width=True,)
selected_region = st.selectbox('Select a Region',options=['Europe','USA'])

if(selected_region=='Europe'):
    kpi1,kpi2,kpi3=st.columns(3)
    with kpi1:
        selected_year = st.selectbox('Select a Year', sorted(df_long['Year'].unique()))

    with kpi2:
        st.markdown("Maximum Number of Reported Cases(for selected year):")
        year_data = df_long[df_long['Year'] == selected_year]
        max_cases=year_data.loc[year_data['Cases'].idxmax(), 'Cases']
    # Create two columns
        st.markdown(max_cases)
    with kpi3:
        country = st.selectbox('Select a Country', df['Region'].unique())

    # Filter data for the selected country
    country_data = df_long[df_long['Region'] == country]


    # Identify the country with the highest number of cases for the selected year
    max_cases_country = year_data.loc[year_data['Cases'].idxmax(), 'Region']

    col1, col2 = st.columns([1, 1])  # Adjust column width ratios if necessary

    # Plotting the time series for the selected country in the first column
    with col2:
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('black') 
        ax.set_facecolor('black')
        ax.plot(country_data['Year'], country_data['Cases'], marker='o', linestyle='-', color='aquamarine')
        ax.set_title(f'Leptospirosis Cases in {country} (2007-2022)').set_color('white')
        ax.set_xlabel('Year').set_color('white')
        ax.set_ylabel('Number of Cases').set_color('white')
        ax.grid(True,color='gainsboro',alpha=0.2)
        ax.tick_params(axis='x', colors='white')  # Change x-axis tick color
        ax.tick_params(axis='y', colors='white')  
        
        # Display the time-series plot
        st.pyplot(fig)


    # Filter the dataframe for the selected year

    # Map visualization using pydeck
    # Define a function to assign red color to the country with the highest cases
    def get_fill_color(row):
        if row['Region'] == max_cases_country:
            return [255, 0, 0, 140]  # Red color with transparency for highest case country
        else:
            return [0, 0, 255, 140]  # Blue color with transparency for others

    # Apply the color function
    year_data['color'] = year_data.apply(get_fill_color, axis=1)

    # Create the pydeck layer with dynamic colors
    scatter_layer = pdk.Layer(
        'ScatterplotLayer',
        data=year_data,
        get_position=['LONG', 'LAT'],
        get_radius=50000,  # Adjust the radius size as needed
        get_fill_color='color',  # Use the color column for the fill color
        pickable=True,
        auto_highlight=True
    )

    # Set up the deck.gl view
    view_state = pdk.ViewState(
        latitude=np.mean(df['LAT']),
        longitude=np.mean(df['LONG']),
        zoom=1,
        pitch=50
    )
    def lstm_model(trainX, trainY):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(trainX.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Render the map in the second column
    with col1:
        r = pdk.Deck(layers=[scatter_layer], initial_view_state=view_state, tooltip={"text": "{Region}\nCases: {Cases}"})
        st.pydeck_chart(r)


    # Filter data for the selected year
    year_bar_data = df_long[df_long['Year'] == selected_year]
    with col1:
        # Plotting the cases for each country
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('black') 
        ax.set_facecolor('black')
        year_bar_data_sorted = year_bar_data.sort_values(by='Cases', ascending=False)

        ax.barh(year_bar_data_sorted['Region'], year_bar_data_sorted['Cases'], color='aquamarine')
        ax.set_xlabel('Number of Cases').set_color('white')
        ax.set_ylabel('Country').set_color('white')
        ax.set_title(f'Leptospirosis Cases by Country in {selected_year}').set_color('white')
        ax.grid(True,alpha=0.1)
        ax.tick_params(axis='x', colors='white')  # Change x-axis tick color
        ax.tick_params(axis='y', colors='white')

        # Display the bar plot
        st.pyplot(fig)
        import plotly.graph_objects as go
    with col1:
        year_bar_data_sorted = year_bar_data.sort_values(by='Cases', ascending=False)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=year_bar_data_sorted['Region'],
            x=year_bar_data_sorted['Cases'],
            orientation='h',
            marker=dict(color='aquamarine'),# Position of the text
            hoverinfo='x+y'  # Hover information
        ))

        fig.update_layout(
            title=f'Leptospirosis Cases by Country in {selected_year}',
            xaxis_title='Number of Cases',
            yaxis_title='Country',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.3)'),  # Grid color with alpha
            
            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.2)')  # Grid color with alpha)
        )

        # Display the interactive bar plot
        st.plotly_chart(fig)

    from sklearn.model_selection import train_test_split

    # # Forecasting button
    if st.button('Generate Forecast'):
         if len(country_data) < 2:
             st.warning("Not enough data to perform forecasting for this country.")
         else:
             # Prepare data for forecasting
             X = country_data['Year'].values.reshape(-1, 1)  # Features (Years)
             y = country_data['Cases'].values  # Target (Cases)
             st.write(X)
             st.write(y)
             #Train-Test Split
             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

             # Scaling
             scaler = MinMaxScaler(feature_range=(0, 1))
             scaled_train_y = scaler.fit_transform(y_train.reshape(-1, 1))
             scaled_test_y = scaler.transform(y_test.reshape(-1, 1))

             # Prepare data for LSTM
             def create_lstm_dataset(data, time_step=1):
                 X, Y = [], []
                 for i in range(len(data) - time_step):
                     X.append(data[i:(i + time_step), 0])
                     Y.append(data[i + time_step, 0])
                 return np.array(X), np.array(Y)

             time_step = 3
             trainX, trainY = create_lstm_dataset(scaled_train_y, time_step)

             # Reshaping for LSTM input (samples, time steps, features)
             trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

             # Create and fit the LSTM model
             lstm = lstm_model(trainX,trainY)
             lstm.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

             # Make predictions
             lstm_predictions = lstm.predict(trainX)
             lstm_predictions = scaler.inverse_transform(lstm_predictions)  # Rescale back to original values
             # Prepare test data for LSTM
             testX, testY = create_lstm_dataset(scaled_test_y, time_step)
             testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
         # Make predictions on the test set
             lstm_test_predictions = lstm.predict(testX)
             lstm_test_predictions = scaler.inverse_transform(lstm_test_predictions)  
             # Modify the LSTM prediction loop
             future_cases = []
             input_data = scaled_train_y[-time_step:].reshape(1, time_step, 1)  # Reshape to 3D

             for i in range(5):  # Forecasting the next 5 years
                 prediction = lstm.predict(input_data)  # Predict using the reshaped input
                 future_cases.append(scaler.inverse_transform(prediction)[0][0])  # Append the predicted value
                 # Append the prediction to input_data and reshape it for the next step
                 input_data = np.append(input_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

             future_years = np.array([year for year in range(2023, 2028)])
             plt.figure(figsize=(12, 6))
             plt.plot(country_data['Year'], country_data['Cases'], label='Actual Cases', color='green')
             plt.plot(country_data['Year'][:len(lstm_predictions)], lstm_predictions, label='Training Predictions', color='blue')
             plt.plot(country_data['Year'][len(lstm_predictions):len(lstm_predictions) + len(lstm_test_predictions)], lstm_test_predictions, label='Test Predictions', color='orange')
             plt.plot(future_years, future_cases, label='Future Predictions', color='red', marker='o')
             plt.xlabel('Year')
             plt.ylabel('Cases')
             plt.title('LSTM Predictions vs Actual Data')
             plt.legend()
             plt.show()
             # --- Linear Regression ---
             lr_model = LinearRegression()
             lr_model.fit(X_train, y_train)
             lr_predictions = lr_model.predict(X_test)  # Test predictions
             lr_future_years = np.array([[year] for year in range(2023, 2028)])
             lr_forecast = lr_model.predict(lr_future_years)
             lr_mae = mean_absolute_error(y_test, lr_predictions)

             arima_model = ARIMA(y_train, order=(1, 1, 1))  # Example order, adjust as necessary
             arima_model_fit = arima_model.fit()
             arima_forecast = arima_model_fit.forecast(steps=5)
             arima_in_sample_predictions = arima_model_fit.predict(start=0, end=len(y_train)-1, dynamic=False)
             arima_mae = mean_absolute_error(y_train, arima_in_sample_predictions)
             arima_mse = mean_squared_error(y_train, arima_in_sample_predictions)
             arima_rmse = np.sqrt(arima_mse)

             # Create a DataFrame for the forecasted values
             forecast_df = pd.DataFrame({
                 'Year': lr_future_years.flatten(),
                 'Linear Regression': lr_forecast,
                 'ARIMA': arima_forecast,
                 'LSTM Forecast': future_cases
             })

             # Display the forecasted values in a table
             st.subheader('Forecasted Leptospirosis Cases')
             st.table(forecast_df)

             # Display the Mean Absolute Error for each model
             st.subheader('Model Accuracy')
             st.write(f'Linear Regression MAE: {lr_mae:.2f}')
             st.write(f'ARIMA MAE: {arima_mae:.2f}')
            
             with col1:
                 fig, ax = plt.subplots()
                 ax.plot(country_data['Year'], country_data['Cases'], marker='o', linestyle='-', color='b', label='Historical Cases')
                 ax.plot(forecast_df['Year'], forecast_df['LSTM Forecast'], marker='o', linestyle='--', color='orange', label='LSTM Forecast')
                 ax.plot(forecast_df['Year'], forecast_df['Linear Regression'], marker='o', linestyle='--', color='r', label='LR Forecast')
                 ax.plot(forecast_df['Year'], forecast_df['ARIMA'], marker='o', linestyle='--', color='green', label='ARIMA Forecast')
                
                 # Set the title and labels
                 ax.set_title(f'Leptospirosis Cases in {country} (2007-2027)')
                 ax.set_xlabel('Year')
                 ax.set_ylabel('Number of Cases')
                 ax.grid(True)
                 ax.legend()

                 # Set x-ticks to be integers
                 ax.set_xticks(np.arange(2007, 2028))  # Set ticks from 2007 to 2027
                 ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))  # Format ticks as integers
                 plt.xticks(rotation=90)  # Rotate x-axis labels

                 # Display the plot
                 st.pyplot(fig)
            
elif(selected_region=='USA'):
    dfusa=pd.read_csv("america_coord.csv")
    # st.write(dfusa)
    # Ensure the year columns are strings
    year_columns = [str(year) for year in range(2014, 2021)]

    # # # Reshape the dataframe to long format
    # # df_long = pd.melt(dfusa, id_vars=['Region', 'LAT', 'LONG'], value_vars=year_columns, var_name='Year', value_name='Cases')

    # # Convert the 'Year' column to numeric
    # df_long['Year'] = df_long['Year'].astype(int)

    # # Ensure the 'Cases' column is numeric (in case it was treated as strings)
    dfusa['Cases'] = pd.to_numeric(dfusa['Cases'], errors='coerce')
    dfusa['Year'] = pd.to_numeric(dfusa['Year'], errors='coerce')

    kpi1,kpi2,kpi3=st.columns(3)
    with kpi1:
        selected_year = st.selectbox('Select a Year', sorted(dfusa['Year'].unique()))

    with kpi2:
        st.markdown("Maximum Number of Reported Cases(for selected year):")
        year_data = dfusa[dfusa['Year'] == selected_year]
        max_cases=year_data.loc[year_data['Cases'].idxmax(), 'Cases']
    # Create two columns
        st.markdown(max_cases)
    with kpi3:
        country = st.selectbox('Select a Country',dfusa['Regions'].unique(),index=1)

    # Filter data for the selected country
    country_data = dfusa[dfusa['Regions'] == country]
    


    # Identify the country with the highest number of cases for the selected year
    max_cases_country = year_data.loc[year_data['Cases'].idxmax(), 'Regions']

    col1, col2 = st.columns([1, 1])  # Adjust column width ratios if necessary

    # Plotting the time series for the selected country in the first column
    with col2:
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('black') 
        ax.set_facecolor('black')
        ax.plot(country_data['Year'], country_data['Cases'], marker='o', linestyle='-', color='aquamarine')
        ax.set_title(f'Leptospirosis Cases in {country} (2014-2021)').set_color('white')
        ax.set_xlabel('Year').set_color('white')
        ax.set_ylabel('Number of Cases').set_color('white')
        ax.grid(True,color='gainsboro',alpha=0.2)
        ax.tick_params(axis='x', colors='white')  # Change x-axis tick color
        ax.tick_params(axis='y', colors='white')  
        
        # Display the time-series plot
        st.pyplot(fig)


    # Filter the dataframe for the selected year

    # Map visualization using pydeck
    # Define a function to assign red color to the country with the highest cases
    def get_fill_color(row):
        if row['Regions'] == max_cases_country:
            return [255, 0, 0, 140]  # Red color with transparency for highest case country
        else:
            return [0, 0, 255, 140]  # Blue color with transparency for others

    # Apply the color function
    year_data['color'] = year_data.apply(get_fill_color, axis=1)

    # Create the pydeck layer with dynamic colors
    scatter_layer = pdk.Layer(
        'ScatterplotLayer',
        data=year_data,
        get_position=['LONG', 'LAT'],
        get_radius=50000,  # Adjust the radius size as needed
        get_fill_color='color',  # Use the color column for the fill color
        pickable=True,
        auto_highlight=True
    )

    # Set up the deck.gl view
    view_state = pdk.ViewState(
        latitude=np.mean(df['LAT']),
        longitude=np.mean(df['LONG']),
        zoom=1,
        pitch=50
    )
    
    # Render the map in the second column
    with col1:
        r = pdk.Deck(layers=[scatter_layer], initial_view_state=view_state, tooltip={"text": "{Regions}\nCases: {Cases}"})
        st.pydeck_chart(r)

    import plotly.express as px
    import plotly.io as pio
    import plotly.graph_objects as go
    # Filter data for the selected year
    year_bar_data = dfusa[dfusa['Year'] == selected_year]

    with col1:
        year_bar_data_sorted = year_bar_data.sort_values(by='Cases', ascending=False)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=year_bar_data_sorted['Regions'],
            x=year_bar_data_sorted['Cases'],
            orientation='h',
            marker=dict(color='aquamarine'),# Position of the text
            hoverinfo='x+y'  # Hover information
        ))

        fig.update_layout(
            title=f'Leptospirosis Cases by Country in {selected_year}',
            xaxis_title='Number of Cases',
            yaxis_title='Country',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.3)'),  # Grid color with alpha
            
            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.2)')  # Grid color with alpha)
        )

        # Display the interactive bar plot
        st.plotly_chart(fig)
