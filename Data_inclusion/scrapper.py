from datetime import date
import pandas as pd
from nsepy import get_history


k = pd.read_csv('companies.csv') 
s = k['Symbol']
for i in range(len(s)):
    if k['Industry'][i]=='IT':
        df = get_history(symbol= s[i],start=date(2015,1,1),end=date(2019,2,1))
        df.to_csv('data.csv', mode='a', header=False)
