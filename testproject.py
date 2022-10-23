import TheoreticallyOptimalStrategy as tos
import indicators
import datetime as dt

df_trades = tos.testPolicy('JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000)
tos.report()
indicators.run()



