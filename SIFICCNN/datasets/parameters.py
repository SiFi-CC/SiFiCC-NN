import numpy as np
def get_parameters(mode):
    print("Parameters for mode: ", mode)
    if mode == "CC":
        graph_attribute_slice_edge = 2
    elif mode == "CM" or mode == "CMbeamtime":
        graph_attribute_slice_edge = 1
    else:
        raise ValueError("Invalid mode: ", mode)
    return graph_attribute_slice_edge
