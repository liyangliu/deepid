import pandas as pd

idx = pd.read_csv('train.txt', sep = ' ')
ali = pd.read_csv('train_ali.txt', sep = ' ')
id_ali = pd.merge(idx, ali, on = 'img_id')
id_ali.to_csv('train_id_ali.txt', sep = ' ', index = False)

idx = pd.read_csv('val.txt', sep = ' ')
ali = pd.read_csv('val_ali.txt', sep = ' ')
id_ali = pd.merge(idx, ali, on = 'img_id')
id_ali.to_csv('val_id_ali.txt', sep = ' ', index = False)

idx = pd.read_csv('test.txt', sep = ' ')
ali = pd.read_csv('test_ali.txt', sep = ' ')
id_ali = pd.merge(idx, ali, on = 'img_id')
id_ali.to_csv('test_id_ali.txt', sep = ' ', index = False)
