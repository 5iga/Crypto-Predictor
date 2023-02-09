import requests
import pandas as pd
import numpy as np
import datetime
import json
from config import settings
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

class CoinAPI:
    def __init__(self, api_key):
        self.__api_key = api_key
        

    def get_daily(self, ticker, startdate, period, limit=100):
    
        """Get daily time series of an equity from AlphaVantage API.

        Parameters
        ----------
        ticker : str
            The ticker symbol of the equity.
        startdate : str
            Date to begin collecting records from
        period : str
            Time period for candlesticks
        limit: str
            Maximim number of entries to read in

        Returns
        -------
        pd.DataFrame
            Columns are 'open', 'high', 'low', 'close', 'volume' and 'trades'.
            All columns are numeric.
        """       

        # Send request to API
        url = (
                "https://rest.coinapi.io/v1/ohlcv/"
                f"BITSTAMP_SPOT_{ticker}_USD/history?"
                f"period_id={period}&"
                f"limit={limit}&"
                f"time_start={startdate}T00:00:00"
            )

        headers = {'X-CoinAPI-Key' : f'{self.__api_key}'}
        response = requests.get(url, headers=headers)
        response_data = response.json()
        print(response)

        # Check if there's been an error
        

        # Clean results
        df = (pd.DataFrame.from_dict(response_data, dtype=float)
            .rename(columns={'time_period_start': 'date',
             'price_open':'open',
             'price_high':'high',
             'price_low':'low',
             'price_close':'close',
             'volume_traded':'volume',
             'trades_count':'trades' })
            .drop(columns=["time_period_end", "time_open", "time_close"])
            .set_index("date")
)

        df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')

        # Return results
        return df

coin = CoinAPI(api_key='')
df_btc = coin.get_daily("BTC", "2022-01-01", "1DAY", "1000")

print(df_btc.shape)
print(df_btc.head())

#Split into feature matrix and target vector
y = df_btc['close']
X = df_btc.drop(columns='close')

#second split, allocating 80% of data for training and 20% of data for testing
cutoff = int(len(X) * 0.8)
X_train, y_train = X.iloc[:cutoff], y.iloc[:cutoff]
X_test, y_test = X.iloc[cutoff:], y.iloc[cutoff:]

#calculate baseline and mae
y_pred_baseline = [y_train.mean()] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)



#instantiate and fit model
model = LinearRegression()
model.fit(X_train, y_train)

training_mae = mean_absolute_error(y_train, model.predict(X_train))
test_mae = mean_absolute_error(y_test, model.predict(X_test))
print("Training MAE:", round(training_mae, 2))
print("Test MAE:", round(test_mae, 2))

#check model accuracy
print(model.score(X_test, y_test))

#make predictions
predictions = model.predict(X_test)

#make line plots of actual prices and predicted prices [USD] against time
plt.figure()
plt.plot(y_test)
plt.plot(predictions)
plt.legend(['predicted_price','actual_price']) 
plt.ylabel("BTC Price [USD]") 
plt.xlabel("Date")
plt.title("Plot of Predicted BTC prices vs Actual prices")
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.gcf().autofmt_xdate() # Rotation
plt.show()