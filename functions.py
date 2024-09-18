import doupand as dp
from quant_stock import config
import pandas as pd
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数
from matplotlib import pyplot as plt
import os
import ast
import glob
import re
import pandas_ta
import matplotlib.font_manager as fm

# Ensure Matplotlib can use a font that supports Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use a font that supports Chinese, like SimHei
plt.rcParams['axes.unicode_minus'] = False

def import_stock_data(stock_code):
    print(config.input_data_path)

    df = pd.read_csv(config.input_data_path + "/" + stock_code + ".csv", skiprows=1, encoding='gbk')

    df["previous close"] = df["收盘价"].shift(1)
    df["涨跌幅"] = df['收盘价'] / df["previous close"] - 1
    df.sort_values(by=['交易日期'], inplace=True)
    return df

def import_stock_data_new(stock_code, other_columns =[]):

    df = pd.read_csv(config.full_trading_data_path + "/" + "stock data" + "/" + stock_code + ".csv")
    df.rename(columns={'股票代码': 'code', '股票名称': 'stock', '交易日期': 'date', '开盘价': 'open'
                       ,'最高价': 'high', '最低价': 'low','收盘价': 'close',
                       '成交量': 'volume', '涨跌幅': 'change'}, inplace=True)

    df = df[[ 'code', 'date', 'open', 'high','low', 'close', 'volume', 'change'] + other_columns]
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['date'], inplace=True, ignore_index=True)

    df["previous_close"] = df["close"].shift(1)
    df["change"] = df['close'] / df["previous_close"] - 1

    df = df.drop(df.index[0]).reset_index()

    #forward_adjusted_price
    df = forward_adjusted_price_1(df)

    return df


#################################################################doupand
def import_stock_data_doupand(stock_code, other_columns =[]):

    df = pd.read_csv(config.doupand_daliy_price_load +  "/" + stock_code + ".csv")


    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.sort_values(by=['date'], inplace=True, ignore_index=True)

    df["previous_close"] = df["close"].shift(1)
    df["change"] = df['close'] / df["previous_close"] - 1


    df = df.drop(df.index[0]).reset_index()

    #backward_adjusted_price
    df = backforwad_adjusted_price_doupand(df)

    return df







def doupand_load_all_daliy_price_data(token, start_date, end_date):
    #start_date ="20050101"
    #end_date = "20240630"

    dp.set_token(token)
    dr = dp.data_reader()

    #extract all codes
    code_list = []
    for i in ['SSE', 'SZSE', 'BSE']:
        codes = dr.ashare_description(exch_market=i)
        codes = list(codes["dp_code"])
        code_list = code_list + codes
    print(f"1. The total number of stocks: {len(code_list)}")

    ####################################
    # 指定文件夹路径
    folder_path = 'D:\investment\doupand_stock_data\daliy_price'  # 请替换为你的文件夹路径
    # 获取文件夹中所有 CSV 文件的名称及完整路径
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    csv_files = [file.replace('.csv', "") for file in csv_files]
    print(f"The total number of stocks we have in the local: {len(csv_files)}")
    # 使用集合进行差集运算
    set_a = set(code_list)
    set_b = set(csv_files)
    code_list = list(set_a - set_b)
    print(f"2. The total number of stocks: {len(code_list)}")
    ##########################

    #load stock to local machine
    i = 0
    for code in code_list:
        i=i+1
        print(f"counting: {i} and code: {code}")

        try:
            price = dr.ashare_eod_price(dp_code=code, start_date=start_date, end_date=end_date)
            # price = pd.DataFrame(price_data)
            price = price[
                ['dp_code', 'trade_date', 'preclose', 'open', 'high', 'low', 'close', 'volume', 'amount', 'adj_close']]
            price.rename(columns={"dp_code": "code", "trade_date": "date"}, inplace=True)

            indi = dr.ashare_eod_indicator(dp_code=code, start_date=start_date, end_date=end_date)
            indi.rename(columns={"dp_code": "code", "trade_date": "date"}, inplace=True)
            # indi =pd.DataFrame(indi_data)

            df = pd.merge(price, indi, on=['code', 'date'], how='left')
            df.to_csv(config.doupand_daliy_price_load + f"/{code}.csv", index=False)

        except:
            print(f"----------------------参数dp_code {code} 的格式不符合要求！")
            continue


def doupand_load_industry_class(token):
    dp.set_token(token)
    dr = dp.data_reader()

    code_list = []
    for i in ['SSE', 'SZSE', 'BSE']:
        codes = dr.ashare_description(exch_market=i)
        codes = list(codes["dp_code"])
        code_list = code_list + codes
    print(len(code_list))

    indu_data = pd.DataFrame()
    i = 0
    for code in code_list:

        i = i + 1
        print(f"counting: {i} and code: {code}")
        try:
            indu = dr.sw_industry_class(dp_code=code)
            indu.rename(columns={"dp_code": "code"}, inplace=True)
            indu = indu.groupby('code').agg({'industry_name': list,
                                             'industry_level': list,
                                             'industry_code': list}).reset_index()
            indu_data = pd.concat([indu_data, indu], ignore_index=True)

        except:
            print(f"----------------------参数dp_code {code} might have sth wrong！")
            continue

    indu_data.to_csv(config.doupand_industry_class_load + "industry_class.csv")



def doupand_load_industry_daliy_price(token, start_date, end_date):
    #start_date ="20050101"
    #end_date = "20240630"

    dp.set_token(token)
    dr = dp.data_reader()

    #extract all codes
    industry_class = pd.read_csv(config.doupand_industry_class_load + "/industry_class.csv", usecols=['industry_code'])
    industry_class['industry_code'] = industry_class['industry_code'].apply(ast.literal_eval)
    code_list = industry_class['industry_code'].explode().tolist()
    code_list = list(set(code_list))

    print(f"1. The total number of industry_code: {len(code_list)}")

    ####################################
    # 指定文件夹路径
    folder_path = config.doupand_industry_daliy_price_load # 请替换为你的文件夹路径
    # 获取文件夹中所有 CSV 文件的名称及完整路径
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    csv_files = [file.replace('.csv', "") for file in csv_files]
    print(f"The total number of industry_code we have in the local: {len(csv_files)}")
    # 使用集合进行差集运算
    set_a = set(code_list)
    set_b = set(csv_files)
    code_list = list(set_a - set_b)
    print(f"2. The total number of industry_code: {len(code_list)}")
    ##########################

    #load stock to local machine
    i = 0
    for code in code_list:

        i=i+1
        print(f"counting: {i} and industry_code: {code}")

        try:
            price = dr.sw_eod_price(industry_code=code, start_date=start_date, end_date=end_date)
            price.rename(columns={"dp_code": "code", "trade_date": "date"}, inplace=True)

            price.to_csv(config.doupand_industry_daliy_price_load + f"/{code}.csv", index=False)

        except:
            print(f"----------------------参数dp_code {code} 的格式不符合要求！")
            continue


def doupand_load_index_daliy_price(token, start_date, end_date):
    #start_date ="20050101"
    #end_date = "20240630"

    dp.set_token(token)
    dr = dp.data_reader()

    #extract all codes
    code_list = []
    for i in ['SSE', 'SZSE', 'CSI']: #交易场所，SSE:上交所; SZSE:深交所; CSI:中证
        codes = dr.ashare_index_description(exch_market=i)
        codes = list(codes["dp_code"])
        code_list = code_list + codes
    print(f"1. The total number of index: {len(code_list)}")

    ####################################
    # 指定文件夹路径
    folder_path = config.doupand_index_daliy_price_load   # 请替换为你的文件夹路径
    # 获取文件夹中所有 CSV 文件的名称及完整路径
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    csv_files = [file.replace('.csv', "") for file in csv_files]
    print(f"The total number of index we have in the local: {len(csv_files)}")
    # 使用集合进行差集运算
    set_a = set(code_list)
    set_b = set(csv_files)
    code_list = list(set_a - set_b)
    print(f"2. The total number of index: {len(code_list)}")
    ##########################

    #load stock to local machine
    i = 0
    for code in code_list:
        i=i+1
        print(f"counting: {i} and index: {code}")

        try:
            price = dr.aindex_eod_price(dp_code=code, start_date=start_date, end_date=end_date)
            price.rename(columns={"dp_code": "code", "trade_date": "date"}, inplace=True)

            indi = dr.ashare_index_description(dp_code=code)
            indi = indi[["dp_code", 'name']]
            indi.rename(columns={"dp_code": "code"}, inplace=True)


            df = pd.merge(price, indi, on=['code'], how='left')
            df.to_csv(config.doupand_index_daliy_price_load  + f"/{code}.csv", index=False)

        except:
            print(f"----------------------参数dp_code {code} 的格式不符合要求！")
            continue


def doupand_covert_period(df, period):

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.set_index("date", inplace=True)
    new_df = df.resample(rule=period).last()

    new_df['open'] = df['open'].resample(period).first()
    new_df['high'] = df['high'].resample(period).max()
    new_df['low'] = df['low'].resample(period).min()
    new_df['volume'] = df['volume'].resample(period).sum()
    new_df['turnover'] = df['turnover'].resample(period).sum()

    new_df['total_mv'] = df['total_mv'].resample(period).last()

    new_df['close'] = df['close'].resample(period).last()
    new_df = new_df.dropna(subset=['open', 'close', 'high', 'low', 'volume', 'turnover'], how="all")

    new_df["preclose"] = new_df["close"].shift(1)
    new_df["change"] = new_df['close'] / new_df["preclose"] - 1

    new_df["trading_days"] = df['close'].resample(period).size()

    if period == 'm':
        new_df["daliy_close_price"] = df['close'].resample(period).apply(list)
        new_df["daliy_volume"] = df['volume'].resample(period).apply(list)
        new_df["daliy_turnover"] = df['turnover'].resample(period).apply(list)
    elif period == 'y':
        # Resample by month within each year
        new_df["monthly_close_price"] = df['close'].resample('M').apply(list).resample('Y').apply(list)
        new_df["momthly_volume"] = df['volume'].resample('M').apply(list).resample('Y').apply(list)
        new_df["monthly_turnover"] = df['turnover'].resample('M').apply(list).resample('Y').apply(list)

    #new_df["daily_equity"] = df['change'].resample(period).apply(lambda  x: list((x+1).cumprod()))
    #new_df["last_day_change"] = df['change'].resample(period).apply(last)

    new_df = new_df.reset_index()

    return new_df

def rsi_6(df):
    # Set the parameter n1 (you can change this value as needed)
    n1 = 6

    # Calculate lc (previous day's closing price)
    df['lc'] = df['close'].shift(1)

    # Calculate max(close - lc, 0)
    df['max_diff'] = (df['close'] - df['lc']).clip(lower=0)

    # Calculate abs(close - lc)
    df['abs_diff'] = (df['close'] - df['lc']).abs()

    # Calculate the simple moving average (SMA) for the max_diff and abs_diff
    df['sma_max_diff'] = df['max_diff'].rolling(window=n1, min_periods=1).mean()
    df['sma_abs_diff'] = df['abs_diff'].rolling(window=n1, min_periods=1).mean()

    # Calculate RSI-like value (rsi1)
    df['rsi1'] = (df['sma_max_diff'] / df['sma_abs_diff']) * 100

    return df

def doupand_agg_stock_daliy_price(period, date):

    csv_files = glob.glob(config.doupand_daliy_price_load + '\*.csv')
    all_data = pd.DataFrame()
    i = 1
    for file in csv_files:
        print(i)
        i = i + 1

        pattern = re.compile(r'\\([^\\]+)\.csv$')
        code = pattern.search(file).group(1)

        df = pd.read_csv(config.doupand_daliy_price_load + "/" + code + ".csv")
        df['close'] = df['adj_close']
        df["change"] = df['close'] / df["preclose"] - 1
        df = df[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'change', 'turnover', 'total_mv']]

        if period != 'd':
            df = doupand_covert_period(df, period)
        # consider the stock is older than 1 year

        ########### extre columns
        df['high_price_before'] = df['close'].expanding(min_periods=1).max().shift(1)
        df['rsi'] = pandas_ta.rsi(df['close'], length=6)

        all_data = pd.concat([all_data, df], ignore_index=True)

    all_data.sort_values(['date', 'code'], inplace=True)
    all_data.reset_index(inplace=True, drop=True)
    print(all_data.shape)

    all_data.to_hdf(
        config.output_data_path + "/stock_data_all_H5_all_" + period + '_' + date + ".h5",
        key="all_data",
        mode="w"
    )

def doupand_agg_industry_daliy_price(period, date):

    csv_files = glob.glob(config.doupand_industry_daliy_price_load + '\*.csv')
    all_data = pd.DataFrame()
    i = 1
    for file in csv_files:
        print(i)
        i = i + 1

        pattern = re.compile(r'\\([^\\]+)\.csv$')
        code = pattern.search(file).group(1)

        df = pd.read_csv(config.doupand_industry_daliy_price_load + "/" + code + ".csv")
        #df.rename(columns={"industry_code": "code"}, inplace=True)


        df["change"] = df['close'] / df["preclose"] - 1
        df = df[['industry_code', 'date', 'open', 'high', 'low', 'close', 'volume', 'change']]

        df = doupand_covert_period(df, period)
        # consider the stock is older than 1 year

        all_data = pd.concat([all_data, df], ignore_index=True)

    all_data.sort_values(['date', 'industry_code'], inplace=True)
    all_data.reset_index(inplace=True, drop=True)
    print(all_data.shape)

    all_data.to_hdf(
        config.output_data_path + "/industry_data_all_H5_all_" + period + '_' + date + ".h5",
        key="all_data",
        mode="w"
    )


def doupand_import_000001sh_data(other_columns =[]):
    df = pd.read_csv(config.doupand_index_daliy_price_load + "/"  + "000001.SH" + ".csv" )

    df["change"] = df['close'] / df["preclose"] - 1
    df = df[["code","date", "change"]]
    df.rename(columns={"code": "index_code"}, inplace=True)
    df.rename(columns={"change": "index_change"}, inplace=True)

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.sort_values(by=['date'], inplace=True, ignore_index=True)

    return df


def forward_adjusted_price(df):
    df["adjustment_factor"] = (1 + df['change']).cumprod()  # Adjustment factor
    df['fa_close'] = df["adjustment_factor"] * (df.iloc[-1]['close'] / df.iloc[-1]["adjustment_factor"])
    df['fa_open'] = df['open'] / df['close'] * df['fa_close']
    df['fa_high'] = df['high'] / df['close'] * df['fa_close']
    df['fa_low'] = df['low'] / df['close'] * df['fa_close']


    return df



def forward_adjusted_price_1(df):

    df["adjustment_factor"] = (1 + df['change']).cumprod()  # Adjustment factor
    df['fa_close'] = df["adjustment_factor"] * (df.iloc[-1]['close'] / df.iloc[-1]["adjustment_factor"])
    df['fa_open'] = df['open'] / df['close'] * df['fa_close']
    df['fa_high'] = df['high'] / df['close'] * df['fa_close']
    df['fa_low'] = df['low'] / df['close'] * df['fa_close']

    df['close'] = df['fa_close']
    df['low'] = df['fa_low']
    df['high'] = df['fa_high']
    df['open'] = df['fa_open']

    df.drop(columns=['fa_close', 'fa_low', 'fa_open','fa_high'], inplace=True)



    return df

def backforwad_adjusted_price_doupand(df):

    df.rename(columns={"adj_close": "ba_close"}, inplace=True)

    df['ba_open'] = df['open'] / df['close'] * df['ba_close']
    df['ba_high'] = df['high'] / df['close'] * df['ba_close']
    df['ba_low'] = df['low'] / df['close'] * df['ba_close']

    df['close'] = df['ba_close']
    df['low'] = df['ba_low']
    df['high'] = df['ba_high']
    df['open'] = df['ba_open']

    df.drop(columns=['ba_close', 'ba_low', 'ba_open', 'ba_high'], inplace=True)

    return df

# import sh000001 data

def import_sh000001_data(other_columns =[]):
    df = pd.read_csv(config.full_trading_data_path + "/" + "index data" + "/" + "sh000001" + ".csv" )
    df = df[["index_code","date"] + other_columns]

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.sort_values(by=['date'], inplace=True, ignore_index=True)
    df.rename(columns={"change": "index_change"}, inplace=True)
    return df

# deal with 停牌 trading halt
def deal_with_trading_halt(df):

    df_index = import_sh000001_data()

    earliest_date = df['date'].min()

    new_df_index = df_index[df_index['date'] >= earliest_date ]
    df_new = pd.merge(left = df, right = new_df_index, on=["date"], how = "outer"
                      , sort = True, suffixes=["_stock", "_index"], indicator=True)

    df_new[['code', 'close']] = df_new[['code', 'close']].fillna(method="ffill")
    df_new['open'] = df_new['open'].fillna(value=df_new["close"])
    df_new['high']= df_new['high'].fillna(value=df_new["close"])
    df_new['low'] = df_new['low'].fillna(value=df_new["close"])
    df_new[['volume', 'change']] = df_new[['volume', 'change']].fillna(value=0)

    df_new.loc[df_new[df_new["volume"] != 0].index, "if_trade"] = 1
    df_new['if_trade'].fillna(value=0, inplace=True)

    df_new.drop(columns=['index_code', '_merge', "index"], inplace=True)

    return df_new


# covert to month
def covert_period(df, period):

    df.set_index("date", inplace=True)
    new_df = df.resample(rule=period).last()

    new_df['open'] = df['open'].resample(period).first()
    new_df['high'] = df['high'].resample(period).max()
    new_df['low'] = df['low'].resample(period).min()
    new_df['volume'] = df['volume'].resample(period).sum()
    new_df = new_df.dropna(subset=['open', 'close', 'high', 'low'], how="all")

    new_df["previous_close"] = new_df["close"].shift(1)
    new_df["change"] = new_df['close'] / new_df["previous_close"] - 1
    new_df["trading_days"] = df['close'].resample(period).size()

    new_df["daily_equity"] = df['change'].resample(period).apply(lambda  x: list((x+1).cumprod()))
    new_df["last_day_change"] = df['change'].resample(period).apply(last)

    if "money" in df.columns:
        new_df["money"] = df["money"].resample(period).last()
    if "if_trade" in df.columns:
        new_df["if_trade"] = df["if_trade"].resample(period).apply(last)

    new_df = new_df.reset_index()


    return new_df




def position(df, ):
    initial_money = 1000000
    slippage = 0.01
    c_rate = 5.0 / 10000  # commission fee
    t_rate = 1.0 / 1000  # tax

    # 第一天
    df.at[0, "hold_num"] = 0  # 持有股票数量
    df.at[0, 'stock_value'] = 0  # 持有股票市直
    df.at[0, "actual_pos"] = 0  # 实际仓位
    df.at[0, 'cash'] = initial_money
    df.at[0, 'equity'] = initial_money  # 总资产 = stock_value + cash

    for i in range(1, df.shape[0]):

        hold_num = df.at[i - 1, "hold_num"]
        # see if today I need to adjust my position
        if df.at[i, 'pos'] != df.at[i - 1, 'pos']:

            theory_num = df.at[i - 1, 'equity'] * df.at[i, 'pos'] / df.at[i, "fa_open"]
            theory_num = int(theory_num)

            # 今天和昨天的持有股票相比较，判断加仓还是减仓
            if theory_num >= hold_num:
                buy_num = theory_num - hold_num
                buy_num = int(buy_num / 100) * 100

                # buy cash
                buy_cash = buy_num * (df.at[i, "fa_open"] + slippage)
                # commission fee
                commission = round(buy_cash * c_rate, 2)
                if commission < 5 and commission != 0:
                    commission = 5

                # check if the cost is higher than cash we have
                cost = buy_cash + commission
                # print("cost: " + str(cost))
                # print("max_cash: " + str(df.at[i - 1, 'cash']))
                while cost > df.at[i - 1, 'cash']:
                    buy_num = buy_num - 100
                    # buy cash
                    buy_cash = buy_num * (df.at[i, "fa_open"] + slippage)
                    # commission fee
                    commission = round(buy_cash * c_rate, 2)
                    if commission < 5 and commission != 0:
                        commission = 5
                    cost = buy_cash + commission

                # calcuate today hold_num and cash we have
                df.at[i, "commission"] = commission
                df.at[i, "hold_num"] = hold_num + buy_num
                df.at[i, "cash"] = df.at[i - 1, "cash"] - buy_cash - commission


            else:
                # 计算卖出股票数量，卖出股票可以不是整数，不需要取整百
                sell_num = hold_num - theory_num

                sell_cash = sell_num * (df.at[i, "fa_open"] - slippage)
                # 不足五元， 按五元收
                commission = round(max(sell_cash * c_rate, 5), 2)
                df.at[i, "commission"] = commission

                #
                tax = round(sell_cash * t_rate, 2)
                df.at[i, "tax"] = tax

                # calucate  today hold_num and cash we have
                df.at[i, "hold_num"] = hold_num - sell_num
                df.at[i, "cash"] = df.at[i - 1, "cash"] + sell_cash - commission - tax

        # on need to adjust position
        else:
            df.at[i, "hold_num"] = hold_num
            df.at[i, "cash"] = df.at[i - 1, "cash"]

        df.at[i, "stock_value"] = df.at[i, "hold_num"] * df.at[i, "fa_close"]
        df.at[i, "equity"] = df.at[i, 'cash'] + df.at[i, "stock_value"]
        df.at[i, "actual_pos"] = df.at[i, "stock_value"] / df.at[i, "equity"]

    return df


def back_testing_select_stocks(select_stock):

    #import index_df
    index_df = import_sh000001_data(['change'])
    index_df = index_df[index_df['date'] > pd.to_datetime('20051230')].reset_index()

    #import daliy_df
    daliy_df = pd.read_hdf(config.output_data_path + "/stock_data_all_H5_all_daily_04082024.h5", mode='r')
    daliy_df = daliy_df[['date', 'code', 'change']]
    daliy_df['date'] = pd.to_datetime(daliy_df['date'])
    daliy_df['YearMonth'] = daliy_df['date'].dt.to_period('M')


    # table: select_stock - calculate the mean change next month for the selected stocks
    select_stock['YearMonth'] = select_stock['date'].dt.to_period('M')

    # move codes to the next months
    select_stock["codes"] = select_stock["codes"].shift()
    select_stock_exploded = select_stock.explode('codes')
    select_stock_exploded.rename(columns={"codes": "code"}, inplace=True)
    daliy_df = pd.merge(select_stock_exploded[['YearMonth', 'code']], daliy_df, on=['YearMonth', 'code'])

    equity = daliy_df.groupby('date').agg(codes=('code', lambda x: list(x)),
                                                 mean_change=('change', 'mean')).reset_index()
    #######################################################################################################要改成right join
    equity = equity.merge(index_df[['date', 'index_change']], how="right", on='date')
    equity['mean_change'] = equity['mean_change'].fillna(0)


    equity['equity_curve'] = (1 + equity['mean_change']).cumprod()
    equity['benchmark'] = (equity["index_change"] + 1).cumprod()


    # 策略评价指标
    equity['date'] = pd.to_datetime(equity['date'])
    # Total return
    total_return = (equity.iloc[-1]['equity_curve'] / 1) - 1
    total_return_text = "Total Return: {:.2f}%".format(total_return)

    # Annualized return
    trading_days = (equity['date'].iloc[-1] - equity["date"].iloc[0]).days + 1
    annual_return = pow(total_return, 365.0 / trading_days) - 1
    annual_return_text = "Annualized Return: {:.2f}%".format(round(annual_return * 100, 2))

    # Daily change standard deviation
    daily_std = equity['mean_change'].std()
    daily_std_text = "Daily Change Std Dev: {:.5f}".format(daily_std)

    # Maximum drawdown
    equity['max2here'] = equity['equity_curve'].expanding().max()
    equity['dd2here'] = equity['equity_curve'] / equity['max2here'] - 1
    end_date, max_draw_down = tuple(equity.sort_values(by=['dd2here']).iloc[0][['date', 'dd2here']])
    max_draw_down_text = "Maximum Drawdown: {:.2f}%".format(max_draw_down * 100)

    # Maximum drawdown start date
    start_date = equity[equity['date'] <= end_date].sort_values(by='equity_curve', ascending=False).iloc[0]['date']
    drawdown_period_text = "Max Drawdown Period: {} to {}".format(start_date.date(), end_date.date())
    '''
    # Visualization
    equity.set_index('date', inplace=True)
    plt.figure(figsize=(14, 7))
    plt.plot(equity['equity_curve'], label='Equity Curve')
    plt.plot(equity['benchmark'], label='Benchmark')
    plt.title('Equity Curve vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(loc='upper center')
    plt.grid(True)

    # Adding text to the plot
    plt.text(equity.index[0], equity['equity_curve'].max(), total_return_text, fontsize=12, verticalalignment='top')
    plt.text(equity.index[0], equity['equity_curve'].max() * 0.95, annual_return_text, fontsize=12,
             verticalalignment='top')
    plt.text(equity.index[0], equity['equity_curve'].max() * 0.90, daily_std_text, fontsize=12, verticalalignment='top')
    plt.text(equity.index[0], equity['equity_curve'].max() * 0.85, max_draw_down_text, fontsize=12,
             verticalalignment='top')
    plt.text(equity.index[0], equity['equity_curve'].max() * 0.80, drawdown_period_text, fontsize=12,
             verticalalignment='top')

    plt.show()
    '''

    # Maximum drawdown start date
    start_date = equity[equity['date'] <= end_date].sort_values(by='equity_curve', ascending=False).iloc[0]['date']
    drawdown_period_text = "Max Drawdown Period: {} to {}".format(start_date.date(), end_date.date())

    # Visualization
    equity.set_index('date', inplace=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14))

    # Plotting the equity curve and benchmark on the same y-axis (ax1)
    ax1.plot(equity['equity_curve'], label='Equity Curve', color='b')
    ax1.plot(equity['benchmark'], label='Benchmark', color='orange')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper center')
    ax1.set_title('Equity Curve and Benchmark on the Same Y-axis')
    #ax1.grid(True)

    # Adding text to ax1 plot
    text_x = equity.index[0]
    text_y = equity['equity_curve'].max()

    ax1.text(text_x, text_y, total_return_text, fontsize=8, verticalalignment='top')
    ax1.text(text_x, text_y * 0.95, annual_return_text, fontsize=8, verticalalignment='top')
    ax1.text(text_x, text_y * 0.90, daily_std_text, fontsize=8, verticalalignment='top')
    ax1.text(text_x, text_y * 0.85, max_draw_down_text, fontsize=8, verticalalignment='top')
    ax1.text(text_x, text_y * 0.80, drawdown_period_text, fontsize=8, verticalalignment='top')

    # Plotting the equity curve on the primary y-axis and benchmark on the secondary y-axis (ax2)
    ax2.plot(equity['equity_curve'], label='Equity Curve', color='b')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Equity Curve', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    ax3 = ax2.twinx()
    ax3.plot(equity['benchmark'], label='Benchmark', color='orange')
    ax3.set_ylabel('Benchmark', color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')

    # Positioning the combined legend for ax2 and ax3
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center')
    plt.show()


def doupand_back_testing_select_stocks(select_stock):

    #import index_df
    index_df = doupand_import_000001sh_data()
    index_df['date'] = pd.to_datetime(index_df['date'], format='%Y%m%d')
    index_df = index_df[index_df['date'] > pd.to_datetime('20050228')].reset_index()

    #import daliy_df
    daliy_df = pd.read_hdf(config.output_data_path + "/stock_data_all_H5_all_d_19082024.h5", mode='r')
    daliy_df['date'] = pd.to_datetime(daliy_df['date'], format='%Y%m%d' )
    daliy_df = daliy_df[daliy_df['date'] > pd.to_datetime('20050228')].reset_index()
    daliy_df = daliy_df[['date', 'code', 'change']]
    daliy_df['date'] = pd.to_datetime(daliy_df['date'])
    daliy_df['YearMonth'] = daliy_df['date'].dt.to_period('M')


    # table: select_stock - calculate the mean change next month for the selected stocks
    select_stock['YearMonth'] = select_stock['date'].dt.to_period('M')

    # move codes to the next months
    select_stock["codes"] = select_stock["codes"].shift()
    select_stock_exploded = select_stock.explode('codes')
    select_stock_exploded.rename(columns={"codes": "code"}, inplace=True)

    daliy_df = daliy_df.merge(select_stock_exploded[['YearMonth', 'code']], on=['YearMonth', 'code'], how ="inner")
    #daliy_df = pd.merge(select_stock_exploded[['YearMonth', 'code']], daliy_df, on=['YearMonth', 'code'])

    equity = daliy_df.groupby('date').agg(codes=('code', lambda x: list(x)),
                                                 mean_change=('change', 'mean')).reset_index()
    #######################################################################################################要改成right join
    equity = equity.merge(index_df[['date', 'index_change']], how="right", on='date')
    equity['mean_change'] = equity['mean_change'].fillna(0)


    equity['equity_curve'] = (1 + equity['mean_change']).cumprod()
    equity['benchmark'] = (equity["index_change"] + 1).cumprod()

    #### extract to csv file
    equity['year_month'] = equity['date'].dt.to_period('M')
    equity['codes'] = equity['codes'].astype(str)
    equity.to_csv("equity.csv", index=False)
    monthly_code_summary = equity.groupby(['year_month', 'codes']).mean().reset_index()
    monthly_code_summary.to_csv("monthly_code_summary.csv", index=False)


    # 策略评价指标
    equity['date'] = pd.to_datetime(equity['date'])
    # Total return
    total_return = (equity.iloc[-1]['equity_curve'] / 1) - 1
    total_return_text = "总收益: {:.2f}倍".format(total_return)

    # Annualized return
    trading_days = (equity['date'].iloc[-1] - equity["date"].iloc[0]).days + 1
    annual_return = pow(total_return, 365.0 / trading_days) - 1
    annual_return_text = "年化收益: {:.2f}%".format(round(annual_return * 100, 2))

    # Daily change standard deviation
    daily_std = equity['mean_change'].std()
    daily_std_text = "Daily Change Std Dev: {:.5f}".format(daily_std)

    # Maximum drawdown
    equity['max2here'] = equity['equity_curve'].expanding().max()
    equity['dd2here'] = equity['equity_curve'] / equity['max2here'] - 1
    end_date, max_draw_down = tuple(equity.sort_values(by=['dd2here']).iloc[0][['date', 'dd2here']])
    max_draw_down_text = "最大回撤: {:.2f}%".format(max_draw_down * 100)

    # Maximum drawdown start date
    start_date = equity[equity['date'] <= end_date].sort_values(by='equity_curve', ascending=False).iloc[0]['date']
    drawdown_period_text = "最大回撤期间: {} to {}".format(start_date.date(), end_date.date())
    '''
    # Visualization
    equity.set_index('date', inplace=True)
    plt.figure(figsize=(14, 7))
    plt.plot(equity['equity_curve'], label='Equity Curve')
    plt.plot(equity['benchmark'], label='Benchmark')
    plt.title('Equity Curve vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(loc='upper center')
    plt.grid(True)

    # Adding text to the plot
    plt.text(equity.index[0], equity['equity_curve'].max(), total_return_text, fontsize=12, verticalalignment='top')
    plt.text(equity.index[0], equity['equity_curve'].max() * 0.95, annual_return_text, fontsize=12,
             verticalalignment='top')
    plt.text(equity.index[0], equity['equity_curve'].max() * 0.90, daily_std_text, fontsize=12, verticalalignment='top')
    plt.text(equity.index[0], equity['equity_curve'].max() * 0.85, max_draw_down_text, fontsize=12,
             verticalalignment='top')
    plt.text(equity.index[0], equity['equity_curve'].max() * 0.80, drawdown_period_text, fontsize=12,
             verticalalignment='top')

    plt.show()
    '''



    # Visualization
    equity.set_index('date', inplace=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14))

    # Plotting the equity curve and benchmark on the same y-axis (ax1)
    ax1.plot(equity['equity_curve'], label='资金曲线', color='b')
    ax1.plot(equity['benchmark'], label='上证指数', color='orange')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper center')
    ax1.set_title('Equity Curve and Benchmark on the Same Y-axis')
    #ax1.grid(True)

    # Adding text to ax1 plot
    text_x = equity.index[0]
    text_y = equity['equity_curve'].max()

    ax1.text(text_x, text_y, total_return_text, fontsize=8, verticalalignment='top')
    ax1.text(text_x, text_y * 0.95, annual_return_text, fontsize=8, verticalalignment='top')
    ax1.text(text_x, text_y * 0.90, daily_std_text, fontsize=8, verticalalignment='top')
    ax1.text(text_x, text_y * 0.85, max_draw_down_text, fontsize=8, verticalalignment='top')
    ax1.text(text_x, text_y * 0.80, drawdown_period_text, fontsize=8, verticalalignment='top')

    # Plotting the equity curve on the primary y-axis and benchmark on the secondary y-axis (ax2)
    ax2.plot(equity['equity_curve'], label='Equity Curve', color='b')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Equity Curve', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    ax3 = ax2.twinx()
    ax3.plot(equity['benchmark'], label='Benchmark', color='orange')
    ax3.set_ylabel('Benchmark', color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')

    # Positioning the combined legend for ax2 and ax3
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center')
    plt.show()

###### small functions

# Custom function to get the first and last values
def first_and_last(series):

    cleaned_series = series[pd.notna(series)]
    if len(series) > 2:
        return [series.iloc[0], series.iloc[-1]]
    elif len(series) == 1:
        return [series.iloc[0]]
    else:
        return []
def last(series):
    cleaned_series = series[pd.notna(series)]
    return series.iloc[-1]

#### indicator


def holding_duration(value, df):
    # Calculate the group size dynamically based on the value
    group_size = value
    # Group the values based on the group size and take the first element of each group
    df['codes'] = df.groupby(df.index // group_size)['codes'].transform('first')
    return df
