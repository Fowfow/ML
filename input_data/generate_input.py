#!/usr/bin/python3


def generate_input_csv(filename="input_data/ccs.csv", nrows=100, ncols=4):
    import numpy as np

    with open(filename, mode="w") as ccs_csv:
        header = "Id"
        for i in range(ncols):
            header += ",C" + str(i+1) 
        ccs_csv.write(header+",S\n")
        for i in range(nrows):
            s = str(i+1)
            rnd = []
            for j in range(ncols):
                rnd.append(np.random.uniform(-0.5, 0.5, 1)[0])
                s += "," + str(rnd[j])
            tmp = "1" if (rnd[0] > 0.) else "0"
            s += "," + tmp
            s += "\n"
            ccs_csv.write(s)
