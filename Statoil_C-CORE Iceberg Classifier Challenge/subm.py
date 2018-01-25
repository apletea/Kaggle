
import os
import numpy as np
import pandas as pd
from subprocess import check_output
#
def extract_ans():

    print(check_output(["ls", "./subm"]).decode("utf8"))
    sub_path = './subm'
    all_files = os.listdir(sub_path)

    outs = [pd.read_csv(os.path.join(sub_path,f),index_col = 0) for f in all_files]
    concat_sub = pd.concat(outs, axis=1)

    cols = list(map(lambda x: "is_iceberg_" + str(x), range(len(concat_sub.columns))))

    concat_sub.columns = cols

    concat_sub.reset_index(inplace=True)

    print concat_sub.head()
    concat_sub['is_iceberg_max'] = concat_sub.iloc[:, 1:4].max(axis=1)
    concat_sub['is_iceberg_min'] = concat_sub.iloc[:, 1:4].min(axis=1)
    concat_sub['is_iceberg_mean'] = concat_sub.iloc[:, 1:4].mean(axis=1)
    concat_sub['is_iceberg_median'] = concat_sub.iloc[:, 1:4].median(axis=1)
    cutoff_lo = 0.67
    cutoff_hi = 0.37
    print concat_sub
    sub_base = pd.read_csv('ens_ice_98989898989898989.csv')

    concat_sub['is_iceberg_base'] = sub_base['is_iceberg']
    concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:4] > cutoff_lo, axis=1),
                                    concat_sub['is_iceberg_max'],
                                    np.where(np.all(concat_sub.iloc[:,1:4] < cutoff_hi, axis=1),
                                             concat_sub['is_iceberg_min'],
                                             concat_sub['is_iceberg_base']))

    concat_sub[['id', 'is_iceberg']].to_csv('stack_minmax_bestbase.csv',
                                        index=False, float_format='%.6f')

    concat_sub['is_iceberg_base'] = sub_base['is_iceberg']
    concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:, 1:4] > cutoff_lo, axis=1),
                                        concat_sub['is_iceberg_max'],
                                        np.where(np.all(concat_sub.iloc[:, 1:4] < cutoff_hi, axis=1),
                                                 concat_sub['is_iceberg_min'],
                                                 concat_sub['is_iceberg_base']))
    concat_sub['is_iceberg'] = np.clip(concat_sub['is_iceberg'].values, 0.001, 0.999)
    concat_sub[['id', 'is_iceberg']].to_csv('submissionok14.csv',
                                            index=False, float_format='%.6f')
