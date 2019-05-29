import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.dates as dates

plt.style.use('classic')


df =pd.read_csv('Data/data.csv')
g = df.groupby('Symbol')

count = 1

for Symbol , Symbol_df in g:
    plt.figure(count)
    plt.plot(Symbol_df['Date'],Symbol_df['Close'],scalex=True,scaley=True,data=None)
    plt.gcf().autofmt_xdate()
    plt.title(Symbol)
    plt.savefig('Plots/'+str(Symbol)+'.png')
    count = count + 1

#plt.show()
