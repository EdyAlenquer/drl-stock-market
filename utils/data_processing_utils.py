
import pandas as pd
import itertools

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader



def get_data(
    start_date,
    end_date, 
    tickers, 
    tech_indicators,
    use_vix=True,
    use_turbulence=True
):

    df = YahooDownloader(
        start_date, 
        end_date,
        tickers
    ).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list = tech_indicators,
        use_vix=use_vix,
        use_turbulence=use_turbulence,
        user_defined_feature = False
    )

    df = fe.preprocess_data(df)

    list_ticker = df["tic"].unique().tolist()
    list_date = list(pd.date_range(df['date'].min(),df['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(df,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(df['date'])]
    processed_full = processed_full.sort_values(['date','tic'])
    
    print("Valores nulos por coluna:")
    print(df.isna().sum())

    df = processed_full.dropna()

    print('\nValores nulos excluídos')
    return df