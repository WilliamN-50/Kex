import numpy as np


def reshape_data(in_file, out_file, delimiter=" "):
    """
    Build a new data structure by combining data points from in_file.
    ____________________________
    Structure of in_file:
    1,2
    3,4
    5,6
    7,8
    ____________________________
    Structure of out_file:

    1,3,2,4
    1,5,2,6
    1,7,2,8
    3,5,4,6
    3,7,4,8
    5,7,6,8
    ____________________________
    """
    in_data = np.loadtxt(in_file, delimiter=delimiter)
    rows = in_data.shape[0]
    out_data = np.zeros((rows*(rows-1)//2, 4))
    n = 0
    for i in range(rows):
        for j in range(i+1, rows):
            out_data[n][0] = in_data[i][0]
            out_data[n][1] = in_data[j][0]
            out_data[n][2] = in_data[i][1]
            out_data[n][3] = in_data[j][1]
            n += 1

    np.savetxt(out_file, out_data)


def main():
    file_name = "data_outdata.csv"
    output_file = "test_output.csv"
    reshape_data(file_name, output_file)


if __name__ == '__main__':
    main()


