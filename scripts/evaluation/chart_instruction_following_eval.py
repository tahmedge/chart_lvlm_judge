import pandas as pd
import glob
import os
import json
directory_path = 'chart_instruction_following_outputs'
csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
df_list = []
for file in csv_files:
    df = pd.read_csv(file)
    df_list.append((os.path.basename(file), df))

print("Model", '|', "Accuracy")
for df_name, df in df_list:
    success = 0
    failure = 0
    for index, row in df.iterrows():
        response = str(row['response'])
        response = response.replace("```","")
        response = response.replace("json","")
        try:
            json_response = json.loads(response)
            if 'Model A' in json_response['Model']:
                failure += 1
            elif 'Model B' in json_response['Model'] or 'B' in json_response['Model']:
                success += 1
            else:
                failure += 1
        except:
            response = response.split("Model:")[-1].strip()
            response = response.replace("\"","")
            response = response.split("Explanation:")[0].split("Model:")[-1].strip()

            if 'Model A' in response or '\"Model A\"' in response:
                failure += 1
            elif 'Model B' in response or response=='B':
                success += 1
            elif '\"Model B\"' in response:
                success += 1
            elif 'Model B\'s' in response:
                success += 1
            else:
                failure += 1
    print(df_name, '|', 100*(success/(success + failure)))
