
from ops.sparse_ops import sym_mat
from scipy.sparse import triu
import os
import numpy as np
import scipy.sparse as sp
import math

def trimDiags(a: sp.coo_matrix, iDiagMax: int, bKeepMain: bool):
    """Remove diagonal elements whose diagonal index is >= iDiagMax
    or is == 0

    Args:
        a: Input scipy coo_matrix
        iDiagMax: Diagonal offset cutoff
        bKeepMain: If true, keep the elements in the main diagonal;
        otherwise remove them

    Returns:
        coo_matrix with the specified diagonals removed
    """
    gDist = np.abs(a.row - a.col)
    idx = np.where((gDist < iDiagMax) & (bKeepMain | (gDist != 0)))
    return sp.coo_matrix((a.data[idx], (a.row[idx], a.col[idx])),
                         shape=a.shape, dtype=a.dtype)
def meanFilterSparse(a: sp.coo_matrix, h: int):
    """Apply a mean filter to an input sparse matrix. This convolves
    the input with a kernel of size 2*h + 1 with constant entries and
    subsequently reshape the output to be of the same shape as input

    Args:
        a: `sp.coo_matrix`, Input matrix to be filtered
        h: `int` half-size of the filter

    Returns:
        `sp.coo_matrix` filterd matrix
    """
    assert h > 0, "meanFilterSparse half-size must be greater than 0"
    assert sp.issparse(a) and a.getformat() == 'coo',\
        "meanFilterSparse input matrix is not scipy.sparse.coo_matrix"
    assert a.shape[0] == a.shape[1],\
        "meanFilterSparse cannot handle non-square matrix"
    fSize = 2 * h + 1
    # filter is a square matrix of constant 1 of shape (fSize, fSize)
    shapeOut = np.array(a.shape) + fSize - 1
    mToeplitz = sp.diags(np.ones(fSize),
                         np.arange(-fSize+1, 1),
                         shape=(shapeOut[1], a.shape[1]),
                         format='csr')
    ans = sp.coo_matrix((mToeplitz @ a) @ mToeplitz.T)
    # remove the edges since we don't care about them if we are smoothing
    # the matrix itself
    ansNoEdge = ans.tocsr()[h:(h+a.shape[0]), h:(h+a.shape[1])].tocoo()
    # Assign different number of neighbors to the edge to better
    # match what the original R implementation of HiCRep does
    rowDist2Edge = np.minimum(ansNoEdge.row, ansNoEdge.shape[0] - 1 - ansNoEdge.row)
    nDim1 = h + 1 + np.minimum(rowDist2Edge, h)
    colDist2Edge = np.minimum(ansNoEdge.col, ansNoEdge.shape[1] - 1 - ansNoEdge.col)
    nDim2 = h + 1 + np.minimum(colDist2Edge, h)
    nNeighbors = nDim1 * nDim2
    ansNoEdge.data /= nNeighbors
    return ansNoEdge


def smooth_matrix(input_dict,dMax=200,hsize=11,max_value=1000):
    
    new_dict={}
    size_limit=224
    for key in input_dict:
        current_mat = input_dict[key]
        current_mat.data = np.minimum(max_value,current_mat.data)
        current_mat = sym_mat(current_mat)
        if current_mat.shape[0]<=size_limit:
            continue
        nDiags = current_mat.shape[0] if dMax < 0 else min(dMax, current_mat.shape[0])
        m1 = trimDiags(current_mat, nDiags, False)

        if hsize>0:
            # apply smoothing
            #m1.data = np.log10(m1.data+1)
            m1 = meanFilterSparse(m1, hsize)
            #m1.data = np.power(10,m1.data)-1
        new_dict[key]=triu(m1,0,format='coo')
    return new_dict
from ops.io_utils import load_pickle, write_pickle
def smooth_pkl(input_pkl,output_pkl,
               dMax=200,hsize=11,max_value=1000):
    input_dict = load_pickle(input_pkl)
    new_dict = smooth_matrix(input_dict,dMax,hsize,max_value)
    write_pickle(new_dict,output_pkl)
    return output_pkl