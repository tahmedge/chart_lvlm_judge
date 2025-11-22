import os
import json
import pandas as pd, re

DIR1 = "CLOSED_SOURCE_LLM_RESPONSE_DATASET_DIRECTORY"
DIR2 = "OPEN_SOURCE_LLM_RESPONSE_DATASET_DIRECTORY"

def find_common_subdirs(a: str, b: str):
    sub_a = {d for d in os.listdir(a) if os.path.isdir(os.path.join(a, d))}
    sub_b = {d for d in os.listdir(b) if os.path.isdir(os.path.join(b, d))}
    return sorted(sub_a & sub_b)

def list_files(d: str):
    return sorted([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])

def main(dir1: str, dir2: str):
    common_subdirs = find_common_subdirs(dir1, dir2)
    if not common_subdirs:
        print("No common subdirectories found.")
        return

    for sub in common_subdirs:
        if 'pointwise' not in sub:
            continue
        print("Dataset: ",sub)
        path1 = os.path.join(dir1, sub)
        path2 = os.path.join(dir2, sub)
        files1 = set(list_files(path1))
        files2 = set(list_files(path2))
        print()
        for f1 in files1:

            print()
            print("Judge: ", f1.replace(".csv",""))
            print("\nModel | Error Distance | Format Following Success Rate")
            for f2 in files2:

                df1 = pd.read_csv(path1 + "/" + f1)
                df2 = pd.read_csv(path2 + "/" + f2)
                df2 = df2[df2['prompt'].isin(df1['prompt'])]

                assert len(df1) == len(df2)

                difference = 0
                success = 0
                failure = 0

                for (index1, row1), (index2, row2) in zip(df1.iterrows(), df2.iterrows()):
                    temp = 1
                    res1 = str(row1['response'])
                    res2 = str(row2['response'])

                    reference = res1

                    text = str(res2)
                    text = text.replace("```", "")
                    text = text.replace("json", "")

                    if text.strip() and text.strip()[0] != "{" and "{" in text:  # address additional text at the beginning of response
                        text = text.split("{")[1]
                        text = "{" + text

                    text = text.replace("\\n", " ")
                    text = text.replace("\n", " ")
                    text = text.replace("\'", " ")
                    text = text.replace("[", " ")
                    text = text.replace("]", " ")

                    if text.strip() and "}" not in text and text.strip()[-1] != "}":  # address missing ending bracket

                        text = text + "}"
                        if "{" not in text:
                            text = "{" + text
                    elif "}" in text and text.strip()[-1] != "}":  # address additional text at the end of response
                        text = text.split("}")[0] + "}"
                        if "{" not in text:
                            text = "{" + text
                    text = re.sub(r' +', ' ', text)
                    text = text.replace("{ ", "{")
                    text = text.replace(" }", "}")
                    text = text.replace("\\", " ")

                    try:
                        text = json.loads(text)
                        reference = reference.replace("```", "")
                        reference = reference.replace("json", "")
                        reference = reference.replace("\'", " ")
                        reference = json.loads(reference)

                        if 'score' in text:
                            text['Score'] = text['score']
                            del text['score']

                        difference += abs(reference['Score'] - min(5, text['Score']))
                        temp = 0
                        success += 1

                    except:

                        if temp!=0:
                            failure += 1

                        difference += 5

                # model name, error distance, format following success rate
                print(
                    f"{f2.replace('.csv', '')} | {difference / (success + failure):.2f} | {100 * (success / (success + failure)):.2f}"
                )

if __name__ == "__main__":
    main(DIR1, DIR2)
