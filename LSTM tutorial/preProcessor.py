import pandas as pd
import numpy as np
from datetime import datetime

def renameCols(df):
    d = {
        "p (mbar)": 'pressure',
        'T (degC)': 'temp',
        'Tdew (degC)': 'dewPoint',
        'rh (%)': 'relHumidity',
        'VPdef (mbar)': 'vaporPressure'
    }
    return df.rename(columns=d)
         
def sampleData(df):
    ''' Data is sampled at 10 minute intervals which is too much for a tutorial so 
    sample the data in larger intervals'''
    before = df.shape[0]
    df = df.loc[5::6]
    print("{:,} rows removed due to sampling".format(before-df.shape[0]))
    return df

def formatDate(df):
    df.loc[:, 'date'] = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    return df

def fixWind(df):
    # Wind velocity has some negative numbers; set them to 0
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    df.loc[bad_wv, 'wv (m/s)'] = 0.0
    #wv = df['max. wv (m/s)']
    #bad_wv = wv == -9999.0
    #df.loc[bad_wv, 'max. wv (m/s)'] = 0.0
    return df

def wind2vec(df):
    # Convert wind speeds and direction to a vector
    wv = df.pop('wv (m/s)')
    #max_wv = df.pop('max. wv (m/s)')
    # Convert to radians.
    wd_rad = df.pop('wd (deg)')*np.pi / 180
    # Calculate the wind x and y components.
    df.loc[:, 'Wx'] = wv*np.cos(wd_rad)
    df.loc[:, 'Wy'] = wv*np.sin(wd_rad)
    # Calculate the max wind x and y components.
    #df.loc[:, 'max Wx'] = max_wv*np.cos(wd_rad)
    #df.loc[:, 'max Wy'] = max_wv*np.sin(wd_rad)
    return df

def prepTime(df):
    # Timestamp by itself doesn't show cycles so convert to sin/cos
    timestamp_s = df['date'].map(datetime.timestamp)
    day = 24*60*60
    year = (365.2425)*day
    
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    del df['date']
    return df

def reorderCols(df):
    cols = list(df.columns)
    cols.remove('temp')
    cols.append('temp')
    return df[cols]

def preProcess(df, cfg):
    df = renameCols(df)
    df = sampleData(df)
    df = formatDate(df)
    df = fixWind(df)
    df = wind2vec(df)
    df = prepTime(df)
    df = reorderCols(df)
    df = df.astype(np.float32)
    return df.reset_index(drop=True)