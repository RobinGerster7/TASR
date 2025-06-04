import random
import numpy as np
import torch

from scripts.detectors.__base__ import Detector
import multiprocessing

class ECEM(Detector):
    """
    Ensemble-Based Cascaded Constrained Energy Minimization (E-CEM) algorithm.

    Reference:
        @article{zhao2019ensemble,
        title={Ensemble-Based Cascaded Constrained Energy Minimization for Hyperspectral Target Detection},
        author={Zhao, Rui and Shi, Zhenwei and Zou, Zhengxia and Zhang, Zhou},
        journal={Remote Sensing},
        volume={11},
        number={11},
        pages={1310},
        year={2019},
        publisher={Multidisciplinary Digital Publishing Institute}}
    """

    def __init__(self):
        Detector.__init__(self)
        self.imgt = None
        self.tgt = None
        self.img = None
        self.data = None
        self.windowsize = [1/4, 2/4, 3/4, 4/4]  # window size
        self.num_layer = 10  # the number of detection layers
        self.num_cem = 6  # the number of CEMs per layer
        self.Lambda = 1e-6  # the regularization coefficient
        self.show_proc = True  # show the process or not


    def parmset(self, **parm):
        self.windowsize = parm['windowsize']  # parameters
        self.num_layer = parm['num_layer']
        self.num_cem = parm['num_cem']
        self.Lambda = parm['Lambda']
        self.show_proc = parm['show_proc']

    def setlambda(self):
        switch = {
            'san': 1e-6,
            'san_noise': 6e-2,
            'syn_noise': 5e-3,
            'cup': 1e-1
        }
        if self.name in switch:
            return switch[self.name]
        else:
            return 1e-10

    def cem(self, img, tgt):
        size = img.shape   # get the size of image matrix
        lamda = random.uniform(self.Lambda/(1+self.Lambda), self.Lambda)  # random regularization coefficient
        R = np.dot(img, img.T/size[1])   # R = X*X'/size(X,2);
        w = np.dot(np.linalg.inv((R+lamda*np.identity(size[0]))), tgt)  # w = (R+lamda*eye(size(X,1)))\d ;
        result = np.dot(w.T, img)  # y=w'* X;
        return result

    def ms_scanning_unit(self, winowsize):
        d = self.img.shape[0]
        winlen = int(d*winowsize**2)
        size = self.imgt.shape  # get the size of image matrix
        result = np.zeros(shape=(int((size[0]-winlen+1)/2+1), size[1]))
        pos = 0
        for i in range(0, size[0]-winlen+1, 2):  # multi-scale scanning
            imgt_tmp = self.imgt[i:i+winlen-1, :]
            result[pos, :] = self.cem(imgt_tmp, imgt_tmp[:, -1])
            pos += 1
        return result

    def cascade_detection(self, mssimg):   # defult parameter configuration
        size = mssimg.shape
        result_forest = np.zeros(shape=(self.num_cem, size[1]))
        for i_layer in range(self.num_layer):
            for i_num in range(self.num_cem):
                result_forest[i_num,:] = self.cem(mssimg, mssimg[:, -1])
            weights = self.dual_sigmoid(np.mean(result_forest, axis=0))  # sigmoid nonlinear function
            mssimg = mssimg*weights
        result = result_forest[:, 0:-1]
        return result

    def dual_sigmoid(self, x):
        x = np.array(x)
        weights = 1.0 / (1.0 + np.exp(-x))
        return weights

    def forward_old(self, pool_num=4):
        self.imgt = np.hstack((self.img, self.tgt))
        p = multiprocessing.Pool(pool_num)  # multiprocessing
        results = p.map(self.ms_scanning_unit, self.windowsize)  # Multi_Scale Scanning
        p.close()
        p.join()
        mssimg = np.concatenate(results, axis=0)
        cadeimg = self.cascade_detection(mssimg)  # Cascaded Detection
        result = np.mean(cadeimg, axis=0)[:self.imgt.shape[1]].reshape(-1, 1) #has shape (H*W,1) should be (1,1,H,W)
        return result

    def forward(self, img: np.ndarray, target: np.ndarray):
        """
        Forward pass of the ECEM detector using only NumPy operations.

        Args:
            img (np.ndarray): Input hyperspectral image of shape (B, C, H, W) with B=1.
            target (np.ndarray): Target vector of shape (B, C, 1, 1) with B=1.

        Returns:
            np.ndarray: Detection result of shape (1, 1, H, W).
        """
        B, C, H, W = img.shape
        assert B == 1, "Batch size must be 1."

        # Drop batch dimension (B=1) -> (C, H, W)
        img = img[0]
        target = target[0, :, 0, 0]  # (C, 1, 1) -> (C,)

        # Reshape for processing
        self.img = img.reshape(C, H * W)  # Shape (C, H*W)
        self.tgt = target.reshape(C, 1)  # Shape (C, 1)

        # Perform computation in NumPy
        result = self.forward_old()  # Expected shape (H*W, 1)

        # Reshape result to match output shape (1, 1, H, W)
        result = result.astype(np.float32).reshape(1, 1, H, W)

        return result
