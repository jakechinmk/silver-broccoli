"""
Created by Jake on 2023-01-08 (yyyy-mm-dd)
Objective: To conduct eda using dataprep for classification
Design: Basic OOP style
"""

import numpy as np
import pandas as pd

from dataprep.eda import create_report, plot_diff
from dataprep.clean import clean_headers

from eurybia import SmartDrift

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Automated EDA',
        description='Automated exploratory analysis with Dataprep'
    )

    parser.add_argument(
        '--train', 
        type=str, 
        help='File Path: Train Dataset with Label in csv format'
        )

    parser.add_argument(
        '--test',
        type=str,
        help='File Path: Test Dataset with Label in csv format'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        help='Target column str'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help='File Directory: Output Folder'
    )

    args = parser.parse_args()
    
    assert os.path.exists(args.train), 'Provide the correct train path'
    assert os.path.exists(args.test), 'Provide the correct test path'
    assert os.path.exists(args.valid), 'Provide the correct valid path'

    assert '.csv' in args.train, 'Provide train dataset in csv file format'
    assert '.csv' in args.test, 'Provide test dataset in csv file format'

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    assert args.target in train_df.columns, f'Provided data does not have {args.target} column.'
    assert sorted(test_df.columns.tolist()) == sorted(test_df.columns.tolist()), 'train df and test df does not have the same columns.'

    train_df = clean_headers(train_df)
    test_df = clean_headers(test_df)

    create_report(train_df, title='Train Dataset with Label').save(filename='train_df_report', to=args.output_dir)
    
    create_report(test_df, title='Test Dataset with Label').save(filename='test_df_report', to=args.output_dir)
    
    plot_diff([train_df, test_df], config={
        'diff.label':['train', 'test']
    }).save(filename='diff_report_01_numeric', to=args.output_dir)

    plot_diff([train_df, test_df], config={
        'diff.label':['train', 'test'],
        'diff.density':True,
    }).save(filename='diff_report_02_density', to=args.output_dir)

    sd = SmartDrift(df_current=test_df, df_baseline=train_df)
    sd.generate_report(output_file=f'{args.output_dir}/drift_report.html', title_story='Data drift')

