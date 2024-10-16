#!/usr/bin.env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from future import standard_library

from pyHSICLasso import HSICLasso

import scipy.io as sio

standard_library.install_aliases()


def main():

    #Numpy array input example
    hsic_lasso = HSICLasso()
    data = sio.loadmat("../tests/test_data/matlab_data.mat")
    X = data['X'].transpose()
    Y = data['Y'][0]
    print("X:",type(X),X.shape)
    print("Y:",type(Y),Y.shape)
    featname = [ 'Feat%d' % x for x in range(1, X.shape[1] + 1) ]
    print("featname:",len(featname))

    hsic_lasso.input(X,Y,featname=featname)
    hsic_lasso.regression(1)
    hsic_lasso.dump()
    hsic_lasso.plot_path()

    #Save parameters
    hsic_lasso.save_param()

if __name__ == "__main__":
    main()
