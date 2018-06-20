The attached dataset comprises of daily price data for the ASX200 and Dow Jone Industrial Average. We also use the daily trading volumes for the ASX200.
The raw dataset includes trading days from 1/1/2014 to 23/3/2018.

In prediction we assume data daily collection. When Monday is over you have all of the data for that day. You can use the data from Monday to predict Tuesday price movement. 

From each dataset only the closing price was used in our study - where there are multiple attributes. 

For the DJIA and volumne data we used log returns and five days of moving averages. For the ASX200 we calculated the following technical indicators: 
	Descriptive statistics used (per http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155133):
        * BIAS(n): Measures the divergence of the current log return from an n-day moving average of log returns. We let n = 6. 
        * PSY(n): Psychological line is a proxy for market sentiment.
        * ASY(n): The average return in the last n days.
        * OBV: The average return in the last n days.
