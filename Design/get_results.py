# %%
import os
import pandas as pd
import json
# conda activate designer_3.12 ; python '/u/home/z/zeh/rwollman/zeh/Repos/Design/Design/get_results.py' 

if __name__ == '__main__':
    base_path = f"/u/home/z/zeh/rwollman/zeh/Projects/Design/Runs/"
    full_results = {}
    for run in os.listdir(base_path):
        if not os.path.exists(os.path.join(base_path,run,'design_results')):
            print(f"Run {run} does not have a design_results directory")
            continue
        designs = os.listdir(os.path.join(base_path,run,'design_results'))
        for design in designs:
            if not os.path.exists(os.path.join(base_path,run,'design_results',design,'results')):
                print(f"Design {design} does not have a results directory")
                continue
            print(f"Loading Design {design}")
            try:
                results = pd.read_csv(os.path.join(base_path,run,'design_results',design,'results','Results.csv'),index_col=0)
                results = results.loc[['Number of Probes (Constrained)','No Noise Accuracy', 'No Noise Separation','No Noise Dynamic Range']]
                results.index = ['Probes','Accuracy','Separation','Dynamic Range']
                results = results.to_dict()['values']
            except Exception as e:
                print(f"Error loading Results.csv for Design {design}: {e}")
                continue
            try:
                parameters = pd.read_csv(os.path.join(base_path,run,'design_results',design,'used_user_parameters.csv'),index_col=0)
                parameters = parameters.to_dict()['values']
            except Exception as e:
                print(f"Error loading used_user_parameters.csv for Design {design}: {e}")
                continue
            full_results[design] = {'results': results, 'parameters': parameters}

    # save full_results to json
    with open(os.path.join(base_path,'results.json'),'w') as f:
        json.dump(full_results,f)

# %%