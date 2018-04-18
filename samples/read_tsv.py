# /usr/bin/python
# coding=utf-8
"""functions for reading tsv tables
"""

from __future__ import print_function

import pandas as pd
import os
import os.path

# Change this to where data is stored
data_dir = '../data' 

def read_tables(folder):
    """reads all tables found in directory and concatenates them into one big
    table
    """
    common_table = pd.DataFrame()
    empty_files = []
    nparsed = 0

    for root, dirs, files in os.walk(folder):
        for f in [f for f in files if os.path.splitext(f)[-1] == '.tsv']:
            # print('scanning file', f, '...', end=' ')
            try:
                # Основной вызов в чтении. delimiter='\t', чтобы считать табы разделителями,
                # skiprows=[1], чтобы пропустить строку с типами данных.
                table = pd.read_csv(os.path.join(root, f), delimiter='\t', skiprows=[1])
            except pd.errors.EmptyDataError as ede:
                # print('Empty file!!')
                empty_files.append(f)
            finally:
                # print(table.shape)
                common_table = common_table.append(table)

            nparsed += 1

    print('parsed', nparsed, 'files. Rows: ', common_table.shape[0], ' empty files number:', len(empty_files))
    print('empty files list: [', ', '.join(empty_files), ']\n')
    
    # чтобы установить сквозную нумерацию во всех прочитанных таблицах
    common_table = common_table.reset_index(drop=True)
    return common_table

if __name__ == '__main__':
    table = read_tables(data_dir)
    print(table.head())