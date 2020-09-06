

import numpy as np

def proj(v, w):
    #Projects vector w onto vector v
    return np.dot(v, w) / np.dot(v, v) * v


def unit_move(v, w):
    nw = np.linalg.norm(w)
    if nw == 0:
        return 0
    nv = np.linalg.norm(v)
    if nv == 0:
        return 0
    return np.dot(v, w)

def apply_move(array, component):
    l = []
    for row in array:
        l.append(unit_move(row, component))
    return np.array(l)


def size_band(input_vector, lower_bound, upper_bound, output_vector):
    assert len(input_vector) == len(output_vector)


    n = len(input_vector)
    i = 0
    l = []

    while i < n:
        if lower_bound < input_vector[i] < upper_bound:
            l.append(output_vector[i])
        i += 1

    return np.array(l)


