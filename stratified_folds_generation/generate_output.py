import pandas as pd

from assignment2.stratified_folds_generation.stratified_cross_folding import generate_stratified_folds

data_set = pd.read_csv("../data/pima-indians-diabetes.csv", header=None)

folds = generate_stratified_folds(data_set, 10)

result = ""
for i in range(10):
    result += f"fold{i + 1}" + "\n"
    x = folds[i].to_string(header=False,
                           index=False,
                           index_names=False).split('\n')
    rows = [','.join(ele.split()) for ele in x]
    for row in rows:
        result += row + "\n"
    result += "\n"


with open("output.txt", "w") as text_file:
    text_file.write(result)
