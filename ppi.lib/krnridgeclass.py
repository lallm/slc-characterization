## Implementation of a Kernel ridge classifier along the lines of
## sklearns implementation of ridge classification.
##
## P.S. 2025 <peter@sykacek.net>
import copy
## prepare the demo by overloading KRR to allow classification
## fit a ridge regression classifier with RBF kernels
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
## note that this is done via ridge regression and mapping
## y to 0.5 and -0.5
import numpy as np
def softmax(vals):
    ## softmax transformation converts [N x k] matrix
    ## to N normalized 1-of-K probabilities.
    probs=np.zeros_likt(vals)
    K=vals.shape[1]
    for kid in range(K):
        probs[:, k]=1./np.sum(np.exp(vals-vals[:,[kid]*K]), axis=1)
    return probs
from sklearn.kernel_ridge import KernelRidge as KRR
## the following class fakes classification by maping labels to
## distinct real numbers (0-> -trgval 1-> trgval) and making use of
## KRR predict etc just revert that mapping.  This is how the
## RidgeClassifier in sklearn is handeled in linear cases and
## ***definitely a hack which lacks theoretical rigor***, to unlock
## kernel ridge regression for classification.
##
## The main motivation for this code is to demonstrate how the dependency
## between modeling choice (here the nonlinearity) and aspects of how the data 
## is distributed affect predictive performance. 
class KRC(BaseEstimator, ClassifierMixin):
    def __init__(self, krr, kernel="rbf", alpha=1.0, gamma=1.0, trgval=4.0):
        ## krr is an instance of kernel ridge regression. Note that the implementation is at
        ## present restricted to RBF kernels, although the __init__ parameters suggest otherwise.
        self.krr=krr
        self.krr.set_params(kernel=kernel, alpha=alpha, gamma=gamma)
        self.kernel=kernel
        self.alpha=alpha
        self.gamma=gamma
        assert trgval>0, "trgval must not be negative"
        self.trgval=trgval
    def fit(self, X, y):
        ## fit function converts 0 1 labels to -self.trgval self.trgval 
        ## and calls self.krr.fit()
        ## note the extension to 1-of-K multi class classification challenges
        ## which depends on fitting K KRR objects. (despite that it is
        ## not required, we fit two KRRs in case of a two class problem, because it leads
        ## to easier to manage code.)
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.unqy=np.array(list(set(y.ravel().tolist())))
        self.unqy.sort()
        ## note that there is a slight inefficiency here: we fit two
        ## krr models for binary class problems. This has however the
        ## advantage that we can use the same code for binary and
        ## multi class problems.
        self.nrclssmdls=len(self.unqy)
        self.allkrr=[]
        for cid, cy in enumerate(self.unqy):
            y4t=np.array([0.0]*y.shape[0])
            y4t[y==self.unqy[cid]]=self.trgval
            y4t[y!=self.unqy[cid]]=-self.trgval
            self.allkrr.append(copy.deepcopy(self.krr))
            self.allkrr[cid].fit(X, y4t)
        return self
    def predict(self, X):
        # predict labels as inferred by fit.
        # Input validation
        X = check_array(X)
        ally=np.zeros((X.shape[0], self.nrclssmdls))
        for cid in range(self.nrclssmdls):
            ally[:, cid]=self.allkrr[cid].predict(X)
        ## map the predicted values to binary lclass labels
        yp=self.unqy[np.argmax(ally, axis=1)]
        return yp
    def score(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # calculate accuracy as score
        return sum(self.predict(X)==y)/len(y)
    def predict_proba(self, X):
        X = check_array(X)
        yp=self.krr.predict(X)
        ## while such predictions are definitely no probabilities, 
        ## we can still limit the range to 0-1. The transformation 
        ## is controlled by specifying trgval which can be adjusted
        ## to get a smooth transition from small to large probabilities.
        ## here we just take the predictions and apply a logistic cds
        ## to map the values to probabilities.
        ally=np.zeros((X.shape[0], self.nrclssmdls))
        for cid in range(self.nrclssmdls):
            ally[:, cid]=self.allkrr[cid].predict(X)
        return softmax(ally)
    def set_params(self, **params):
        ## set_params is used by GSCV to modify the
        ## parameteters of the KRC and the embedded KRR object.
        ## check whether self.trgval is provided (this is the only 
        ## parameter belonging to KRC)
        if "trgval" in params.keys():
            ## if so we modify self.trgval and 
            ## remove "trgval" from the dictionary, 
            ## before forwarding the other
            ## parameters to KRR.
            trgval=params["trgval"]
            assert trgval>0, "trgval must not be negative"
            self.trgval=trgval
            del params["trgval"]
        ## forward to krr setparams
        self.krr.set_params(**params)
        try:
            for cid in range(len(self.allkrr)):
                self.allkrr[cid].set_param(**param)
        except:
            pass
        ## print(self.krr.get_params())
        return self
