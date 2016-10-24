import pandas as pd
import numpy as np

file_nums = pd.read_csv('file_nums.txt', sep = ' ')
sorted_nums = file_nums.sort(columns = 'num')
sorted_nums.to_csv('sorted_nums.txt', sep = ' ', index = False)
