import wfdb

wfdb.dl_database('mitdb', dl_dir='data/mitdb')

record = wfdb.rdrecord('data/mitdb/100')
annotation = wfdb.rdann('data/mitdb/100', 'atr')