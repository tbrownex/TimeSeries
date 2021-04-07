import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plotWind(df):
    plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
    plt.colorbar()
    plt.xlabel('Wind X [m/s]')
    plt.ylabel('Wind Y [m/s]')
    ax = plt.gca()
    ax.axis('tight')
    return

def plotTime(df):
    plt.plot(np.array(df['Day sin'])[:25])
    plt.plot(np.array(df['Day cos'])[:25])
    plt.xlabel('Time [h]')
    plt.title('Time of day signal')
    return

def plotDistribution(df, train):
    mu = train.mean()
    std = train.std()
    df_std = (df - mu) / std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)
    return