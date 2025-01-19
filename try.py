import sys
import pandas as pd
import numpy as np

def check_format(inp, wt, imp):
    try:
        df = pd.read_csv(inp)
    except FileNotFoundError:
        print(f"Error: File '{inp}' not found.")
        sys.exit()
    if df.shape[1] < 3:
        print("Error: Input file must have at least 3 columns.")
        sys.exit()
    try:
        for col in df.columns[1:]:
            pd.to_numeric(df[col], errors='raise')
    except ValueError:
        print("Error: All columns from the 2nd to the last must contain numeric data.")
        sys.exit()
    
    n_col = df.shape[1] - 1 
    if len(wt) != n_col or len(imp) != n_col:
        print("Error: The number of weights and impacts must match the number of numeric columns.")
        sys.exit()

    if not all(i in ['+', '-'] for i in imp):
        print("Error: All impacts must be either '+' or '-'.")
        sys.exit()
    
    return df

def topsis(inp, weights, impacts, op):
    df = check_format(inp, weights, impacts)
    matrix = df.iloc[:, 1:].values
    n_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
    wt = np.array(weights, dtype=float)
    if n_matrix.shape[1] != len(wt):
        print("Error: The number of weights does not match the number of numeric columns.")
        sys.exit()
    w_matrix = n_matrix * wt
    ideal_best = []
    ideal_worst = []
    for i, impact in enumerate(impacts):
        col = w_matrix[:, i]
        if impact == '+':
            ideal_best.append(max(col))
            ideal_worst.append(min(col))
        else:
            ideal_best.append(min(col))
            ideal_worst.append(max(col))
    sep_best = np.sqrt(((w_matrix - ideal_best) ** 2).sum(axis=1))
    sep_worst = np.sqrt(((w_matrix - ideal_worst) ** 2).sum(axis=1))
    scores = sep_worst / (sep_worst + sep_best)
    df['Topsis Score'] = scores
    df['Rank'] = scores.argsort()[::-1].argsort() + 1
    df.to_csv(op, index=False)
    print(f"Result saved to '{op}'")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit()

    ip = sys.argv[1]
    wt = list(map(float, sys.argv[2].split(',')))
    imp = sys.argv[3].split(',')
    op = sys.argv[4]

    topsis(ip, wt, imp, op)
