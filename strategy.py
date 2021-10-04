import numpy as np
import pandas as pd
from multiprocessing import Pool

class Position:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.position_type = None
        self.profit = 0
        self.trades_table = pd.DataFrame(dict(zip(
            ['date_open', 'date_close',
             'position_type',
             'open_equity', 'close_equity',
             'open_price', 'close_price',
             'volume'], [[None] for _ in range(8)]
        )))

    def open_position(self, date, price, position_type):
        self.position_type = position_type
        # create new row for new trade record
        self.trades_table.loc[len(self.trades_table)] = [None for _ in range(8)]
        # record trade data

        self.trades_table.iloc[-1]['position_type'] = position_type
        self.trades_table.iloc[-1]['date_open'] = date
        self.trades_table.iloc[-1]['open_price'] = price
        self.trades_table.iloc[-1]['volume'] = self.equity // price
        self.trades_table.iloc[-1]['open_equity'] = self.equity

    def close_position(self, date, price):
        # if we have no opened position (at the beginning)
        if not self.position_type:
            pass
        else:
            # if we already have a position
            volume = self.trades_table['volume'].iloc[-1]
            open_price = self.trades_table['open_price'].iloc[-1]

            if self.position_type == 'long':
                self.profit = volume * (price - open_price)
            elif self.position_type == 'short':
                self.profit = volume * (open_price - price)

            self.equity += self.profit

            # add records to trades table
            self.trades_table['close_equity'].iloc[-1] = self.equity
            self.trades_table['date_close'].iloc[-1] = date
            self.trades_table['close_price'].iloc[-1] = price

    def compute_performance(self, data, **kwargs):
        # total return
        total_ret = self.trades_table['close_equity'].iloc[-1]/self.initial_capital-1
        # Sharpe ratio
        strategy_rets = pd.concat((self.trades_table['open_equity'], 
                                   self.trades_table['close_equity'])).\
                                   drop_duplicates().interpolate().pct_change().iloc[1:]
        sharpe_ratio = strategy_rets.mean()/strategy_rets.std()
        # max drawdown
        cumulative = (strategy_rets + 1).cumprod()
        peaks = cumulative.cummax()
        drawdowns = cumulative / peaks - 1
        max_drawdown = drawdowns.min()
        # buy & hold strategy
        buy_hold = round(data.iloc[-1]/data.iloc[0] - 1, 4)
        metrics = {'total return': total_ret,
                   'sharpe ratio': sharpe_ratio,
                   'max drawdown': max_drawdown,
                   'buy & hold return': buy_hold}
        
        self.metrics = pd.Series({**kwargs, **metrics})
        
        


def EMA(data: pd.Series, window: int):
    """ Returns EMA from given time series """
    return data.ewm(span=window, min_periods=0, adjust=False, ignore_na=False).mean()


def test_EMA(data: pd.Series,
             slowEMA: int,
             fastEMA: int,
             verySlowEMA: int,
             initial_capital: int):
    """"""
    # calculate MA and add to data
    df = pd.DataFrame({'close': data.values,
                       'slow EMA': EMA(data, slowEMA),
                       'fast EMA': EMA(data, fastEMA),
                       'very slow EMA': EMA(data, verySlowEMA)})
    # empty position
    pos = Position(initial_capital)

    for i in range(df.shape[0]):

        # first trading day: do nothing
        if i == 0:
            pass
        # second trading day: buy or sell
        elif i == 1:
            if df.iloc[i]['fast EMA'] > df.iloc[i]['slow EMA']:
                print((df.index[i], df.iloc[i]['close']))
                pos.open_position(date=df.index[i],
                                  price=df.iloc[i]['close'],
                                  position_type='long')
            else:
                pos.open_position(date=df.index[i],
                                  price=df.iloc[i]['close'],
                                  position_type='short')

        # not first and not second trading day
        else:

            longCondition = (df.iloc[i]['fast EMA'] > df.iloc[i]['slow EMA']) \
                            and \
                            (df.iloc[i - 1]['fast EMA'] < df.iloc[i - 1]['slow EMA'])
            
            shortCondition = (df.iloc[i]['fast EMA'] < df.iloc[i]['very slow EMA']) \
                             and \
                             (df.iloc[i - 1]['fast EMA'] > df.iloc[i - 1]['very slow EMA'])

            if longCondition or shortCondition: 

                # first, close existing position
                pos.close_position(date=df.index[i],
                                   price=df.iloc[i]['close'])

                # than open new long or short
                if longCondition:
                    pos.open_position(date=df.index[i],
                                      price=df.iloc[i]['close'],
                                      position_type='long')
                elif shortCondition:
                    pos.open_position(date=df.index[i],
                                      price=df.iloc[i]['close'],
                                      position_type='short')

    # at the last date of trading sample close opened position
    pos.close_position(date=df.index[i],
                       price=df.iloc[-1]['close'])
    
    # prettify trades table
   
    pos.trades_table.dropna(inplace=True)
    # calculate metrics
    pos.compute_performance(data=data,
                            **{'slow EMA': slowEMA,
                               'fast EMA': fastEMA,
                               'very slow EMA': verySlowEMA})

    return pos

def wrapper(x):
    return test_EMA(**x)

def parallel_test(args):
    with Pool() as p:    
        results = list(p.map(wrapper, args))
        results = [x.metrics for x in results]
        results = pd.concat(results, axis=1).T.sort_values(by='total return', 
            ascending=False)

        return results