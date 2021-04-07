''' A dictionary object holds key parameters such as:
    - the location and name of the input data file
    - the location and name of the log file
    - the default logging level
    - the number of months in the "lookback window" for identifying "wins"
    - the number of months in the "lookback window" for identifying "losses"
    - how many months to use for Test
    - an indicator allowing execution in Test mode'''

__author__ = "The Hackett Group"

def getConfig():
    d = {}
    d["dataLoc"]    = "/home/tbrownex/data/LSTM/"
    d["fileName"]  = "jena_climate.csv"
    d["trainPct"] = 0.7
    d["valPct"] = 0.2
    d["testPct"] = 0.1
    '''d['sequence_length'] = 50
    d['normalise'] = True
    d["batchSize"] = 32
    d["epochs"] = 3
    d["labelColumn"] = "MeanRunTime"'''
    d["logLoc"]     = "/home/tbrownex/"
    d["logFile"]    = "weather.log"
    d["logDefault"] = "info"
    
    assert abs(d['trainPct']+d['valPct']+d['testPct']-1.0) < 1e-3, "percentages must sum to 1.0"
    return d