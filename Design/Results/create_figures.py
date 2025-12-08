# %%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
def readable_run_name(run, varying_keys):
    parts = run.split('_')
    readable = []
    for key in varying_keys:
        if key in run:
            value = run.split(key)[1].split('_')[1]
            value = value.replace('p', '.')
            # Color: green for '0' or '0.0', red otherwise
            if value in ['0', '0.0']:
                color_value = f"\033[92m{value}\033[0m"  # Green
            else:
                color_value = f"\033[91m{value}\033[0m"  # Red
            readable.append(f"{key}: {color_value}")
    return "\n".join(readable)
# conda activate designer_3.12 ; python '/u/home/z/zeh/rwollman/zeh/Repos/Design/Design/create_figures.py' 
if __name__ == '__main__':
    output = {}
    for notebook_name in ['Run14']: # 88
        # notebook_name = f'Run{notebook_name}'

        base_path = f"/u/home/z/zeh/rwollman/zeh/Projects/Design/Runs/{notebook_name}/"
        design_results = os.path.join(base_path,'design_results')
        if not os.path.exists(design_results):
            continue
        for run in os.listdir(design_results):
            # first check if the run was started 
            if 'used_user_parameters.csv' in os.listdir(os.path.join(design_results,run)):
                parameters = pd.read_csv(os.path.join(design_results,run,'used_user_parameters.csv'),index_col=0)
                parameters = parameters.to_dict()['values']
                # print(f"Run {run} was started")
            else:
                continue
            if not os.path.exists(os.path.join(design_results,run,'results')):
                continue
            # next check if the run is done
            if 'Results.csv' in os.listdir(os.path.join(design_results,run,'results')):
                results = pd.read_csv(os.path.join(design_results,run,'results','Results.csv'),index_col=0)
                # print(f"Run {run} was completed")
            else:
                continue
            results = results.loc[['Number of Probes (Constrained)','No Noise Accuracy', 'No Noise Separation','No Noise Dynamic Range']]
            results.index = ['Probes','Accuracy','Separation','Dynamic Range']
            results = results.to_dict()['values']
            # parameters = parameters.to_dict()['values']
            output[run] = {'parameters': parameters, 'results': results}
        # break
    len(output.keys())

    param_values = {}
    for run in output.keys():
        param_dict = output[run]['parameters']
        for key in param_dict:
            if not key in parameters.keys():
                continue
            value = param_dict[key]
            if key not in param_values:
                param_values[key] = []
            param_values[key].append(value)
    # del param_values['output']
    varying_params = {}
    for key, values in param_values.items():
        if not key in parameters.keys():
            continue
        if key == 'output':
            continue
        unique_values = np.unique(values).tolist()
        if (len(unique_values) > 1) or (key == 'fig'):
            # sort the values
            try:
                unique_values = sorted(unique_values, key=float)
            except:
                unique_values = sorted(unique_values)
            varying_params[key] = unique_values
    figures = varying_params['fig']


    from re import X
    from tkinter import Y
    from matplotlib.ticker import ScalarFormatter
    for figure in figures:
        print(f'Plotting {figure}')
        runs = [run for run in output.keys() if output[run]['parameters']['fig']==figure]
        param_values = {}
        for run in runs:
            param_dict = output[run]['parameters']
            for key in param_dict:
                if not key in parameters.keys():
                    continue
                value = param_dict[key]
                if key not in param_values:
                    param_values[key] = []
                param_values[key].append(value)
        del param_values['output']
        varying_params = {}
        for key, values in param_values.items():
            if not key in parameters.keys():
                continue
            unique_values = np.unique(values).tolist()
            if len(unique_values) > 1:
                # sort the values
                try:
                    unique_values = sorted(unique_values, key=float)
                except:
                    unique_values = sorted(unique_values)
                varying_params[key] = unique_values
        # print("Varying parameter keys and their unique values:")
        subplot_parameter = 'decoder_n_lyr'
        if not subplot_parameter in varying_params:
            print(f'No subplot parameter {subplot_parameter} found for figure {figure}')
            continue
        subplot_values = varying_params[subplot_parameter]
        x_axis_parameter = [i for i in varying_params.keys() if i not in [subplot_parameter]][0]
        x_axis_values = varying_params[x_axis_parameter]
        if x_axis_parameter =='brightness':
            x_axis_values = [i for i in x_axis_values if not i in ['5.5','6']]
        y_axis_parameters = ['Probes','Accuracy','Separation','Dynamic Range']
        subplot_dfs_means = {}
        subplot_dfs_stds = {}
        # axs = axs.ravel()
        for i,subplot_value in enumerate(subplot_values):
            subplot_dfs_means[subplot_value] = pd.DataFrame(index=pd.Index(x_axis_values), columns=pd.Index(y_axis_parameters))
            subplot_dfs_stds[subplot_value] = pd.DataFrame(index=pd.Index(x_axis_values), columns=pd.Index(y_axis_parameters))
            for x_axis_value in x_axis_values:
                run = [run for run in runs if output[run]['parameters'][x_axis_parameter]==x_axis_value and output[run]['parameters'][subplot_parameter]==subplot_value]
                for y_axis_parameter in y_axis_parameters:
                    if len(run) == 0:
                        continue
                    if len(run) ==1:
                        mu = output[run[0]]['results'][y_axis_parameter]
                        std = 0#mu*0.1#0
                    else:
                        mu = np.mean([output[run]['results'][y_axis_parameter] for run in run])
                        std = np.std([output[run]['results'][y_axis_parameter] for run in run])
                    subplot_dfs_means[subplot_value].loc[x_axis_value,y_axis_parameter] = mu
                    subplot_dfs_stds[subplot_value].loc[x_axis_value,y_axis_parameter] = std
            # subplot_dfs[subplot_value].plot(kind='bar',ax=ax,width=0.8)
            color_mapper = {
                'Probes': 'black',
                'Accuracy': 'orange',
                'Separation': 'purple',
                'Dynamic Range': 'cyan'
            }
            x_axis_label_mapper = {'n_bit':'Number of Bits',
            'separation_wt':'Separation Weight',
            'brightness':'Brightness (Log10)',
            'n_probes': 'Number of Encoding Probes'}
            if not x_axis_parameter in x_axis_label_mapper:
                x_axis_label_mapper[x_axis_parameter] = x_axis_parameter.replace('_',' ').title()
            total_bar_width = 0.8
            single_bar_width = total_bar_width / len(y_axis_parameters)
            x_pos = np.arange(len(x_axis_values))
            line_mapper = {'0':'solid','1':'dotted','2':'dashed','3':'dashdot'}
            legend_mapper = {'0':'Linear Decoder N=0',
            '1':'Non Linear Decoder N=1',
            '2':'Non Linear Decoder N=2',
            '3':'Non Linear Decoder N=3'}
        fig,axs = plt.subplots(1,1,figsize=(6,4),dpi=500)
        plt.subplots_adjust(wspace=0.3, hspace=0.6)
        plt.suptitle(f'{figure}')
        main_ax = axs
        for j,y_axis_parameter in enumerate(['Dynamic Range','Accuracy','Probes','Separation']):
            if j == 0: # left y axis, left ticks, left labels
                ax = main_ax
                ax.yaxis.set_ticks_position('left')
                ax.yaxis.set_label_position('left')
                ax.tick_params(axis='y', labelcolor=color_mapper[y_axis_parameter], pad=1.5)
                ax.set_ylabel(y_axis_parameter,color=color_mapper[y_axis_parameter], labelpad=0, va='bottom')
                # if i ==0:
                #     ax.set_title(f'Linear Decoder N=0')
                # else:
                #     ax.set_title('Non Linear Decoder N=1')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_axis_values)
                ax.set_xlim(min(x_pos)-0.75,max(x_pos)+0.75)
                ax.set_xlabel(x_axis_label_mapper[x_axis_parameter], labelpad=10, va='bottom')
                # if x_axis_parameter == 'n_probes':
                #     ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                #     ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            else:
                ax = main_ax.twinx()
            if j==1: # left y axis, right ticks, right labels
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_label_position('right')
                ax.spines['right'].set_position(('axes', 0))
                ax.tick_params(axis='y', labelcolor=color_mapper[y_axis_parameter], pad=1.5)
                ax.set_ylabel(y_axis_parameter,color=color_mapper[y_axis_parameter], labelpad=10, va='bottom')
            elif j==2: # right y axis, left ticks, left labels
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_label_position('right')
                ax.tick_params(axis='y', labelcolor=color_mapper[y_axis_parameter], direction='in', pad=-20)
                ax.set_ylabel(y_axis_parameter,color=color_mapper[y_axis_parameter], labelpad=-20, va='bottom')
            elif j==3: # right y axis, right ticks, right labels
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_label_position('right')
                ax.spines['left'].set_position(('axes', 0))
                ax.tick_params(axis='y', labelcolor=color_mapper[y_axis_parameter], pad=1.5)
                ax.set_ylabel(y_axis_parameter,color=color_mapper[y_axis_parameter], labelpad=10, va='bottom')
            if y_axis_parameter in ['Probes','Dynamic Range']:
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            if y_axis_parameter == 'Accuracy':
                ax.set_ylim(0,1)
            else:
                values = [(subplot_dfs_means[subplot_value][y_axis_parameter] + subplot_dfs_stds[subplot_value][y_axis_parameter]).max() for subplot_value in subplot_values]
                y_max = max(values) * 1.1
                y_min = 0
                ax.set_ylim(y_min,y_max)
            bar_position = x_pos + j*total_bar_width/len(y_axis_parameters) - 0.4 + 0.8/len(y_axis_parameters)/2
            # bar_container = ax.bar(bar_position, subplot_dfs[subplot_value][y_axis_parameter], width=single_bar_width, color=color_mapper[y_axis_parameter])
            for i in range(len(subplot_values)):
                ax.plot(x_pos, subplot_dfs_means[subplot_values[i]][y_axis_parameter], color=color_mapper[y_axis_parameter], linestyle=line_mapper[str(i)], label=legend_mapper[str(i)])
                # error bars
                ax.errorbar(x_pos, subplot_dfs_means[subplot_values[i]][y_axis_parameter], yerr=subplot_dfs_stds[subplot_values[i]][y_axis_parameter], color=color_mapper[y_axis_parameter], linestyle=line_mapper[str(i)], label=legend_mapper[str(i)])
                # ax.fill_between(x_pos, subplot_dfs_means[subplot_values[i]][y_axis_parameter] - subplot_dfs_stds[subplot_values[i]][y_axis_parameter], subplot_dfs_means[subplot_values[i]][y_axis_parameter] + subplot_dfs_stds[subplot_values[i]][y_axis_parameter], color=color_mapper[y_axis_parameter], alpha=0.2)
        # ax.legend()
        plt.tight_layout()
        # set background to transparent
        # fig.patch.set_facecolor('none')
        for char in [' ','(',')','=','+','-','*','/','^','%']:
            figure = figure.replace(char,'_')
        plt.savefig(f'/u/home/z/zeh/project-rwollman/Projects/Design/Figures/{figure}.pdf',dpi=300)

        plt.close()