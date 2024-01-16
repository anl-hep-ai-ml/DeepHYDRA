import sys
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

plt.rcParams['figure.constrained_layout.use'] = True


def legend_without_duplicate_labels(ax, loc):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc=loc)


if __name__ == '__main__':

    parameters = pd.read_csv('data/parameters.csv', index_col=0)
    activations = pd.read_csv('data/activations.csv', index_col=0)
    flops = pd.read_csv('data/flops.csv', index_col=0)
    auc_roc = pd.read_csv('data/auc_roc.csv', sep='\t', index_col=0)

    unreduced_data_regex = r'\(Un|STRADA|T-DBSCAN'

    sizes = parameters['Parameters'].add(activations['Activations'], fill_value=0)

    sizes.rename('Size', inplace=True)

    flops_strada_reduction =  flops.loc['Reduction'] + flops.loc['T-DBSCAN']

    flops.loc['Informer-MSE (Unreduced)'] = flops.loc['Informer-MSE (Reduced)'] + flops.loc['Reduction']
    flops.loc['Informer-SMSE (Unreduced)'] = flops.loc['Informer-SMSE (Reduced)'] + flops.loc['Reduction']
    flops.loc['TranAD (Unreduced)'] = flops.loc['TranAD (Reduced)'] + flops.loc['Reduction']
    flops.loc['DAGMM (Unreduced)'] = flops.loc['DAGMM (Reduced)'] + flops.loc['Reduction']
    flops.loc['USAD (Unreduced)'] = flops.loc['USAD (Reduced)'] + flops.loc['Reduction']
    flops.loc['OmniAnomaly (Unreduced)'] = flops.loc['OmniAnomaly (Reduced)'] + flops.loc['Reduction']

    flops.loc['STRADA-MSE'] = flops.loc['Informer-MSE (Reduced)'] + flops_strada_reduction
    flops.loc['STRADA-SMSE'] = flops.loc['Informer-SMSE (Reduced)'] + flops_strada_reduction
    flops.loc['STRADA-TranAD'] = flops.loc['TranAD (Reduced)'] + flops_strada_reduction
    flops.loc['T-DBSCAN/DAGMM'] = flops.loc['DAGMM (Reduced)'] + flops_strada_reduction
    flops.loc['T-DBSCAN/USAD'] = flops.loc['USAD (Reduced)'] + flops_strada_reduction
    flops.loc['T-DBSCAN/OmniAnomaly'] = flops.loc['OmniAnomaly (Reduced)'] + flops_strada_reduction

    sizes_reduced = sizes.loc[~(sizes.index.str.contains(unreduced_data_regex))]
    flops_reduced = flops.loc[~(flops.index.str.contains(unreduced_data_regex))]
    auc_roc_reduced = auc_roc.loc[~(auc_roc.index.str.contains(unreduced_data_regex))]

    sizes_unreduced = sizes.loc[(sizes.index.str.contains(unreduced_data_regex))]
    flops_unreduced = flops.loc[(flops.index.str.contains(unreduced_data_regex))]
    auc_roc_unreduced = auc_roc.loc[(auc_roc.index.str.contains(unreduced_data_regex))]

    # These colors are specifically chosen to improve
    # accessibility for readers with colorblindness

    data_reduced = pd.concat((sizes_reduced,
                                flops_reduced,
                                auc_roc_reduced), axis=1)
    
    data_reduced.drop('Reduction', inplace=True)
    
    data_unreduced = pd.concat((sizes_unreduced,
                                    flops_unreduced,
                                    auc_roc_unreduced), axis=1)

    colors = {  '1L-Method 3 (Reduced)': '#D81B60',
                '1L-Method 3 (Unreduced)': '#D81B60',
                '1L-Method 4 (Reduced)': '#1E88E5',
                '1L-Method 4 (Unreduced)': '#1E88E5',
                'MERLIN (Reduced)': '#FFC107',
                'MERLIN (Unreduced)': '#FFC107',
                'T-DBSCAN': '#000000',
                'Informer-MSE (Reduced)': '#1CB2C5',
                'Informer-MSE (Unreduced)': '#1CB2C5',
                'STRADA-MSE': '#1CB2C5',
                'Informer-SMSE (Reduced)': '#6F8098',
                'Informer-SMSE (Unreduced)': '#6F8098',
                'STRADA-SMSE': '#6F8098',
                'TranAD (Reduced)': '#D4FC14',
                'TranAD (Unreduced)': '#D4FC14',
                'STRADA-TranAD': '#D4FC14',
                'DAGMM (Reduced)': '#004D40',
                'DAGMM (Unreduced)': '#004D40',
                'T-DBSCAN/DAGMM': '#004D40',
                'USAD (Reduced)': '#C43F42',
                'USAD (Unreduced)': '#C43F42',
                'T-DBSCAN/USAD': '#C43F42',
                'OmniAnomaly (Reduced)': '#1164B3',
                'OmniAnomaly (Unreduced)': '#1164B3',
                'T-DBSCAN/OmniAnomaly': '##1164B3',}
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ax.set_xscale('log')
    ax.set_xlim(1, 5e16)

    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('Model')

    for model_name, colors in colors.items():

        if model_name != 'MERLIN':

            results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}

            left = 0

            for operator, flops in results.items():

                label, color = get_label_and_color(operator)

                ax.barh(model_name,
                                flops,
                                color=color,
                                height=0.8,
                                left=left,
                                label=label)

                left += flops

        else:
            results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}

            left = 0

            for operator, flops in results.items():

                ax.barh(model_name,
                                flops,
                                color='#000000',
                                fill=False,
                                hatch='/',
                                height=0.8,
                                left=left,
                                label='ESTIMATED')

                left += flops

    legend_without_duplicate_labels(ax, 'upper right')

    plt.savefig(f'plots/computational_intensity_comparison_by_operator.png')