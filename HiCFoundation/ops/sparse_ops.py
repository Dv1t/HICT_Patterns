from scipy.sparse import triu,coo_matrix
import numpy as np
def sym_mat(input_array):
    down_array = triu(input_array,1,format='coo').T
    new_row = np.concatenate([input_array.row,down_array.row])
    new_col = np.concatenate([input_array.col,down_array.col])
    new_data = np.concatenate([input_array.data,down_array.data])
    shape=input_array.shape
    final_array = coo_matrix((new_data,(new_row,new_col)),shape=shape)
    return final_array


def array_to_coo(array):
    """
    Convert a regular 2D NumPy array to a scipy.sparse.coo_matrix.

    Parameters:
    - array (numpy.ndarray): The input 2D array.

    Returns:
    - scipy.sparse.coo_matrix: The converted COO matrix.
    """
    # Find the non-zero elements in the array
    row, col = np.nonzero(array)

    # Get the values of the non-zero elements
    data = array[row, col]

    # Create the COO matrix
    coo_mat = coo_matrix((data, (row, col)), shape=array.shape)

    return coo_mat

def filter_sparse_region(input_row,input_col,input_data,
                         start_index,end_index):
    """
    input_row: the row index of the sparse matrix
    input_col: the column index of the sparse matrix
    input_data: the data of the sparse matrix
    start_index: the start index of the region to filter
    end_index: the end index of the region to filter
    """
    select_index1 = (input_row>=start_index) & (input_row<end_index)
    select_index2 = (input_col>=start_index) & (input_col<end_index)
    select_index = select_index1 & select_index2
    new_row = input_row[select_index]-start_index
    new_col = input_col[select_index]-start_index
    new_data = input_data[select_index]
    new_shape = end_index-start_index
    final_array = coo_matrix((new_data,(new_row,new_col)),shape=(new_shape,new_shape))
    return final_array

def filter_sparse_rectangle(input_row,input_col,input_data,
                            start_row,end_row,start_col,end_col):
    """
    input_row: the row index of the sparse matrix
    input_col: the column index of the sparse matrix
    input_data: the data of the sparse matrix
    start_row: the start row index of the rectangle to filter
    end_row: the end row index of the rectangle to filter
    start_col: the start column index of the rectangle to filter
    end_col: the end column index of the rectangle to filter
    """
    select_index1 = (input_row>=start_row) & (input_row<end_row)
    select_index2 = (input_col>=start_col) & (input_col<end_col)
    select_index = select_index1 & select_index2
    new_row = input_row[select_index]-start_row
    new_col = input_col[select_index]-start_col
    new_data = input_data[select_index]
    new_shape = (end_row-start_row,end_col-start_col)
    final_array = coo_matrix((new_data,(new_row,new_col)),shape=new_shape)
    return final_array
