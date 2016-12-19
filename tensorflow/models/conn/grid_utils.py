'''

Collection of functions to manipulate a grid board and compute its connectivity


The coordinates of a cell are represented with a pair (row_index,column_index)

The key function to create a labelled dataset is
def generate_boards(
        number_boards=8,
        board_size=10,
        stone_probability=0.45):

This function is called by
def make_dataset_file(
        full_file_name = 'board_dataset.hdf5',  # where the (b, 0, 1, c) tensor should be saved
        number_boards=8,
        board_size=4,
        stone_probability=0.45):

        
        

See board_utils_demo.py

@author: f.maire@qut.edu.au
created on
2016/01/23

Revised on 2016/02/29
Addded histogram analysis

Revised on 2016/03/21
Addded function segment_intersection(a,b, rtol=1e-5):

todo:
create a function to create a really hard dataset.


'''

# for compatibility with python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys  # for sys.maxsize
import os  # for directory manipulation

import numpy as np
import matplotlib.pyplot as plt  # to plot the histograms

# import random # for randint

import h5py
import pickle  # , gzip


# ----------------------------------------------------------------


def get_connection_length(board, start_set=None, target_set=None):
    '''
        Determine the connectivity between the cell sets 'start_set' and
        'target_set'. That is, check for the existence of a path of connected stones
        from start_set to target_set.

        If start_set is None, it is initialized with the set of stones of row 0 (top row).
        if target_set is None, it is initialized with the set of stones of the bottom row.


        Return  connection_length
             connection_length is None  is the cell sets are not connected
             otherwise, it is the length of the connecting path
             
        Params
            board : uint8 np array,
                    zero -> background
                    non-zero -> stone
            start_set, target_set : cell coords  of the cell sets.sys.maxsize
                                    Lists of (row,col) pairs
                                    
    '''
    num_rows, num_cols = board.shape[0], board.shape[1]

    # .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
    
    def getNeighbours(A, T):
        '''
            Return the set 'N' of stony neighbors of 'A', that are not in 'T' (taboo set)
            The neighbors are represented with the pairs of their (row, column) coordinates.
            Note that this is a sub-function.
        '''
        N = set()
        for r, c in A:
            for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                rn, cn = r + dr, c + dc
                if ((0 <= rn < num_rows) and (0 <= cn < num_cols) and board[rn, cn] != 0 and (rn, cn) not in T):
                    N.add((rn, cn))
        return N
    
    # .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .

    if start_set is None:
        start_set = [(0, c) for c in range(num_cols) if board[0, c] != 0]

    if target_set is None:
        target_set = [(num_rows - 1, c) for c in range(num_cols) if board[num_cols - 1, c] != 0]

    g = 0  # generation index
    T = set()  # taboo cells
    F = set(start_set)  # frontier

    while not F.intersection(target_set):  # reached target?
        g += 1
        T.update(F)
        F = getNeighbours(F, T)
        if not F:  # cul de sac!
            return None
    return g


# ----------------------------------------------------------------


def generate_boards(
        number_boards=8,
        board_size=10,
        stone_probability=0.45):
    '''
    Generate a "connectivity" dataset X, y
    
    Return boards, connectivity, number_connected
    
    where boards is a 3D tensor of ubytes
        board[i] is a 2D grid. The cell board[i, r, c] == 1 if there is a stone 
        in this cell, otherwise if the cell is empty, then board[i, r, c]  is set to 0.
        connectivity[i] is the length of the connection between the top and bottom rows.
        If no path exists, then connectivity[i] is a huge number, namely  sys.maxsize
    
    '''
    
    # generate board, values random from 0-1
    boards = np.random.rand(number_boards, board_size, board_size)
    
    # boards.dtype is 'float64'
    stone_locations = boards <= stone_probability
    empty_locations = np.logical_not(stone_locations)  # boards > stone_probability
        
    boards[stone_locations] = 1
    boards[empty_locations] = 0
    
    connectivity = np.empty((number_boards,), dtype=int)

    number_connected = 0
    for i in range(number_boards):
        
        # generate label matrix where white (non zero) is considered stone, and 4 connectivity
        board = boards[i]
        
        # debug print 'i = ',i,' board shape = ', board.shape

        conn_len = get_connection_length(board)
        if conn_len is not None:
            connectivity[i] = conn_len
            number_connected += 1
        else:
            connectivity[i] = sys.maxsize

    return boards, connectivity, number_connected  # X, y


# ----------------------------------------------------------------


def make_board_generator(
        min_conn_len, 
        max_conn_len, 
        board_size=10, 
        stone_probability=0.45):
    fail_count = 0
    while True:
        # generate board, values random from 0-1
        board = np.random.rand(board_size, board_size)

        # boards.dtype is 'float64'
        stone_locations = board <= stone_probability
        empty_locations = np.logical_not(stone_locations)  # boards > stone_probability
            
        board[stone_locations] = 1
        board[empty_locations] = 0
        
        conn_len = get_connection_length(board)
        
        if conn_len is None:
            connectivity = sys.maxsize
        else:
            connectivity = conn_len
        
        if (min_conn_len <= connectivity <= max_conn_len):
            # print "success after ", fail_count," failures"
            fail_count = 0  # reset counter
            yield board, connectivity  # X, y
        else:
            fail_count += 1  # print "failed"


# ----------------------------------------------------------------
 

def make_constrained_dataset(
        full_file_name,  # where the (b,0,1,c) tensor should be saved
        min_conn_len,
        max_conn_len,
        board_side_len,
        stone_probability,
        num_positive_examples,  # connected boards
        num_negative_examples):
    '''
    Build a labelled dataset of connected and non connected boards and save it to a file
    '''
    n = board_side_len
    theo_max_len = int(n * n / 2 + n / 2)  # theoretical max length

    X = np.zeros((num_positive_examples + num_negative_examples, n, n), np.int)  # boards
    y = np.zeros((num_positive_examples + num_negative_examples, ), np.int)  # connection length
    
    npe = 0  # num of positive examples generated so far
    nne = 0  # num of negative examples generated so far
    i = 0  # index of the next board to generate

    print('making pos examples')
    # make a hard connected board generator
    pos_board_gen = make_board_generator(min_conn_len, theo_max_len, n, stone_probability=stone_probability)
    while npe < num_positive_examples:
        board, conn_len = next(pos_board_gen)
        X[i, :, :] = board
        y[i] = conn_len
        i += 1
        npe += 1
        if npe % 100 == 0:
            print (npe, ' ')
        if npe % 1000 == 0:
            print ('\n - - -')
    print('making neg examples')
    # make a non-connected board generator
    neg_board_gen = make_board_generator(sys.maxsize, sys.maxsize, n, stone_probability=stone_probability)
    while nne < num_negative_examples:
        board, conn_len = next(neg_board_gen)
        X[i, :, :] = board
        y[i] = conn_len
        i += 1
        nne += 1
        if nne % 100 == 0:
            print (nne, ' ',)
        if npe % 1000 == 0:
            print('\n - - -')

    # if not os.path.exists(filename):
    my_dir = os.path.dirname(full_file_name)
    if not os.path.exists(my_dir):
        os.makedirs(my_dir)
           
    with h5py.File(full_file_name, "w") as f:
        dset_X = f.create_dataset("X", data=X, compression="gzip")
        dset_X.attrs['description'] = 'tensor of shape (n_boards,n_rows,n_columns) of b boards'
        dset_y = f.create_dataset("y", data=y, dtype='i', compression="gzip")
        dset_y.attrs['description'] = 'connection length, vector of shape (b,)'
        
       
# ----------------------------------------------------------------

        
def make_dataset_file(
        full_file_name='board_dataset.hdf5',  # where the (b,0,1,c) tensor should be saved
        number_boards=8,
        board_size=4,
        stone_probability=0.45):
    '''
    Build a labeled dataset of connected and non connected boards and save it to a file
    '''
        
    X, y, n = generate_boards(number_boards, board_size, stone_probability)
    
    print('Generated boards = {0}, Connected board = {1}, Proportion = {2}'.format(number_boards, n, float(n) / number_boards))
    
    # save the tensor
    fileExt = full_file_name[full_file_name.rfind('.'):]
    if fileExt == '.hdf5':
        with h5py.File(full_file_name, "w") as f:
            dset_X = f.create_dataset("X", data=X, compression="gzip")
            dset_X.attrs['description'] = 'tensor of shape (n_boards,n_rows,n_columns) of b boards'
            dset_y = f.create_dataset("y", data=y, dtype='i', compression="gzip")
            dset_y.attrs['description'] = 'connection length, vector of shape (b,)'
    elif fileExt == '.pkl':
        with open(full_file_name, 'wb') as f:
            pickle.dump(f, X, -1)
            pickle.dump(f, y, -1)
    else:
        raise Exception('Unexpected file name extension!')

                
# ----------------------------------------------------------------


def load_dataset(
        full_file_name='board_dataset.hdf5',  #
        start_idx=0,  # slice index
        stop_idx=None):  # slice index
    '''
    Load an existing subset of a dataset, and return it as a pair X,y

    Note; to find out the number of elements in the dataset use
    with h5py.File(full_file_name, "r") as f:
        dset_X = f['X']
        print dset_X.shape
    '''
    with h5py.File(full_file_name, "r") as f:
        dset_X = f['X']
        dset_y = f['y']

        X_shape = list(dset_X.shape)
        if stop_idx is None:
            stop_idx = X_shape[0]
        n = stop_idx - start_idx

        assert 0 <= start_idx < stop_idx <= dset_X.shape[0]

        X_shape[0] = n
        y_shape = list(dset_y.shape)
        y_shape[0] = n
        
        X = np.empty(X_shape)
        dset_X.read_direct(X,  # write from dset_X to X
                           np.s_[start_idx:stop_idx],  # source: slice in dset_X
                           np.s_[:]  # destination: slice in X is [0:n]
                           )
        y = np.empty(y_shape)
        dset_y.read_direct(y,
                           np.s_[start_idx:stop_idx],  # source: slice in dset_y
                           np.s_[:]  # destination: slice in y is [0:n]
                           )
    print('- ' * 20)
    print('Loaded dataset ', full_file_name)
    print('X.dtype = ', X.dtype, ' X.shape = ', X.shape)
    print('y.dtype = ', y.dtype, ' y.shape = ', y.shape)
    print('- ' * 20)
    return X, y
    
        
# ----------------------------------------------------------------
    

def show_conn_len_hist(y, max_conn_len):
    '''
    Show the histogram of a vector of connection lengths
    Parameters:
      y : array of integers (connection lengths)
      max_conn_len : only  y[y<max_conn_len]  is plotted
    
    '''
    
    y = y[y < max_conn_len]
    print('upper threshod = ', max_conn_len)
    print('filtered y = ', y)
    my_bins = range(0, max_conn_len + 1)
    hist_y = np.histogram(y, bins=my_bins)
    print (hist_y)
    plt.hist(y, bins=my_bins)
    plt.show()

if __name__ == "__main__":
    pass
