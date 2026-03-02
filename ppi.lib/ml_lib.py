# the subsequent file contains code which provides usefull additions
# for sklearn to aid applications of machine learning and pattern
# recognition in computational biology.
#
# (C) P. Sykacek 2017 - 2022 <peter@sykacek.net>

import pandas as pd
import numpy as np
import copy
#import statsmodels.api as sm
#import statsmodels.formula.api as mdls
import scipy as sp
from sklearn import metrics
## new approach to parallelise
from joblib import parallel_backend
## used for function crossvalprobs
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer

def cumsum(vals):
    ## converts vals to a cumulative sum over vals entries.
    ## ravels vals before calculation and does not sort
    ## IN -> OUT: vals -> cumulative sum(vals)
    vals=vals.ravel().copy()
    csum=vals[0]
    for i in range(1, len(vals)):
        csum=csum+vals[i]
        vals[i]=csum
    return vals

def cumprob(probs, dosort=True):
    ## converts a 1-of-C probability to a cumulative probability
    ## sorts in dependence of dosort (default is to sort)
    ## IN-> OUT: probs -> cumprobs(probs), sortdx
    if dosort:
        srtdx=np.argsort(probs)
        cumprobs=probs(srtdx)
    else:
        srtdx=None
        cumprobs=probs
    return cumsum(cumprobs), srtdx

import os
def creapath(fnam):
    fpath=os.path.dirname(fnam)
    try:
        os.makedirs(fpath)
    except:
        pass

def smartopen(fnam, creapathflag=True, fmode="tw"):
    ## smart open takes a path of a filename specification and makes sure
    ## that all directories exist such that the file can be written.
    ## after that the function opens the file and returns a handle.
    if creapathflag:
        creapath(fnam)
    ## open the file
    return open(fnam, mode=fmode)

## read and write kernel allocation probabilities to a file
def writeallocs(fnambase, QIn, labels=None, totext=False):
    ## writeallocs(fnambase, QIn, labels=None, totext=False): writes
    ## kernel allocation probabilities to a file.  fnambase is a file
    ## name base with direactories. The function makes sure that
    ## missing directories are created. In dependence of labels we
    ## either write only QIn, or QIn and sample labels. If
    ## totext==False the file for storing QIn is named
    ## fnambase+"_QIn.npy". In case totext==True the file is named
    ## fnambase+"_QIn.csv". If labels are provided we check for
    ## compatible dimensions and in dependence of totext store the
    ## contents in files named fnambase+"_labs.npy" or fnambase+"_labs.csv"
    ##
    ## IN
    ##
    ## fnambase: string with directory specification which represents
    ##           a base for all result files to be written.
    ##
    ## QIn: a [nsamples x nkernels] matrix of kernel allocation
    ##           probabilities.
    ##
    ## labels: an optional [nsamples x] vector of row labels.
    ##      These labels are written to a separate file.
    ##
    ## totext: A boolean flag which indicates whether we write QIn and
    ##      labels as text or as binary file. Defaults to False such
    ##      that files are written in native numpy format.
    ##
    ## OUT - none.
    ##
    ## (C) P. Sykacek 2023 <peter@sykacek.net>
    havelabs=labels is not None
    if havelabs:
        nlabs=labels.shape[0]
        [nsmp, nK]=QIn.shape
        if nsmp != nlabs:
            raise Exception("File size mismatch in writeallocs.")
    if totext:
        qinam=fnambase+"_QIn.csv"
        labnam=fnambase+"_labs.csv"
    else:
        qinam=fnambase+"_QIn"   ## .npy is attached automatically
        labnam=fnambase+"_labs"
    ## make sure that fnambase exists
    creapath(qinam)
    ## we may now write the files:
    if totext:
        np.savetxt(qinam, QIn, delimiter=",")
        if havelabs:
            np.savetxt(labnam, labels, delimiter=",")
    else:
        np.save(qinam, QIn)
        if havelabs:
            np.save(labnam, labels)
            
def readallocs(fnambase):
    ## readallocs is compatible with writeallocs and reads allocation
    ## probabilities from files. fnambase is expanded to all possible
    ## file names and the function determines automatically whether to
    ## read binary or text and whether labels exist or not.
    ##
    ## IN
    ##
    ## fnambase: string with directory specification which represents
    ##           a base for all result files to be written.
    ##
    ## OUT - a tuple 
    ## (
    ## QIn:  a [nsamples x nkernels] matrix of kernel allocation
    ##           probabilities.
    ##
    ## labels: an optional [nsamples x] vector of row labels.  These
    ##       labels are read from a separate file. If no compatible
    ##       label file is found this parameter is None.
    ## )
    ## (C) P. Sykacek 2023 <peter@sykacek.net>

    qfnam=fnambase+"_QIn.csv"
    fromtxt=False
    try:
        QIn=np.loadtxt(qfnam, delimiter=",")
        fromtxt=True
    except:
        qfnam=fnambase+"_QIn.npy"
        QIn=np.load(qfnam) ## no exception to signal wrong file name.
    if fromtxt:
        lbfnam=fnambase+"_labs.csv"
        try:
            labels=np.loadtext(lbfnam, delimiter=",")
        except:
            labels=None
    else:
        lbfnam=fnambase+"_labs.npy"
        try:
            labels=np.load(lbfnam)
        except:
            labels=None
    return (QIn, labels)


def uniquerndsub(dat, nr):
    # provide a unique random subsample of dat with nr entries
    #
    # (C) P. Sykacek 2017

    # nr of rows in dat
    mxsz=dat.shape[0]
    # we may subsample only if nr is smaller than
    # the number of samples in dat.
    if nr < mxsz:
        # choice can be used to subsample.
        # the boolean variable replace controls whether
        # we replace a drawn sample with itself.
        # with replace=false we get unique values.
        rwid=np.random.choice(mxsz, size=(nr,), replace=False)
        dat=dat[rwid,:] 
    return dat.copy()

def thinplate(inpt, K):
    # def thinplate(inpt, K) calculates np.shape(K)[1] output activations 
    # (columns in out) from each input (inpt) with each kernel (K). out has
    # 1+shape(K)[1]+shape(in)[1] columns and shape(in)[0] rows. 
    # the number of columns in in and K must be identical.
    # (C) P. Sykacek, 2017 <peter@sykacek.net>
    #print(inpt.shape)
    #print(K.shape)
    nrin=inpt.shape[0]
    out=np.zeros((nrin, K.shape[0]))
    # loop over columns
    for i in range(np.shape(K)[0]):
        # all square distances from i-th kernel (row) in K simultanously
        #print(inpt-K[np.repeat(i, nrin)])
        
        dist=inpt-K[np.repeat(i, nrin)]
        if len(dist.shape)>1:
            dist=np.sum(dist*dist, 1)
            #print("A")
            #dist=dist[:,0]
        else:
            #print("B")
            dist=dist*dist
        #print(dist.shape)
        out[:,i]=dist+np.finfo(float).eps # prevent log of zeros
        lgout=np.log(out[:,i])
        out[:,i]=out[:,i]*lgout;
    #print(out.shape)
    #print(inpt.shape)
    if len(inpt.shape)==1:
        inpt.shape=(nrin, 1)
    return np.concatenate((np.ones((nrin, 1)), inpt, out), axis=1)

def fastgauss(inpt, K, l):
    # function [out]=fastgauss(inpt, K, l) calculates K.shape[1] output activations 
    # (columns in out) from each input (inpt) with each kernel (K). out has
    # 1+inpt.shape[1]+K.shape[1] columns and in.shape[0] rows. the number of 
    # columns in in and K must be identical. l is a row vector with the 
    # square routs of the precisions (inverse std. dev) in each dimension. 
    # (C) P. Sykacek, 2017 <peter@sykacek.net>
    nrin=inpt.shape[0]
    out=np.zeros((nrin, K.shape[0]))
    # loop over columns
    #print(inpt.shape)
    #print(K.shape)
    #print(l)
    for i in range(np.shape(K)[0]):
        # all square distances from i-th kernel (row) in K simultanously
        # print(l[np.repeat(i, nrin)])
        dist=(inpt-K[np.repeat(i, nrin)])*l[np.repeat(0, nrin)]
        dist=np.sum(dist*dist, 1)
        dist=np.exp(-0.5*dist)
        #print(dist.shape)
        out[:,i]=dist
    return np.concatenate((np.ones((nrin, 1)), inpt, out), axis=1)

def evids2mp(evids):
    # evids2mp converts log marginal likelihoods to model
    # probabilities.  The function is in general usefull to convert
    # unnormalised log probabilities to probabilities.
    #
    # IN
    #
    # evids: [nsample x nprobs] array like datastructure with
    #        log evidence compatible infomration.
    #
    # OUT
    #
    # probs: [nsample x nprobs] array like datastructure with
    #        normalised probabilities.
    #
    # (C) P. Sykacek 2019 <peter@sykacke.net>
    
    if type(evids)==type([]):
        # convert to numpy array
        evids=np.array(evids)
    if len(evids.shape)< 2:
        nprob=evids.shape[0]
    else:
        nprob=evids.shape[1]
    # initialise probs
    probs=np.zeros_like(evids, dtype=np.double)
    for pdx in range(nprob):
        if len(evids.shape)< 2:
            probs[pdx]=1/np.sum(np.exp(evids-evids[pdx]))
        else:
            ## we operate on column pdx
            probs[:, pdx]=1/np.sum(np.exp(evids-evids[:, [pdx]*nprob]), axis=1)
    return probs


def logit(pvals, myeps=10**-100):
    ## logit transform of p-values to "unfold" the underlying metric
    ## 
    ## convert to numpy array
    if type(pvals) != type(np.array([])):
        if type(pvals) == type([]):
            pvals=np.array(pvals)
        else:
            pvals=np.array([pvals])
    ## make sure the value is > 0
    onemp=1-pvals
    pvals[pvals<myeps]=myeps
    onemp[onemp<myeps]=myeps
    ## return logit transformed p-values.
    return np.log(pvals)-np.log(onemp)

def kldisc(P1, P2, whichlog="2"):
    # function kldisc(P1, P2, whichlog) calculates the Kullback
    # Leibler divergences between discrete Probability measures P1 and
    # P2. The KL measure calculates d=sum_k P1(k) log(P1(k)/P2(k)).
    # Potential Warning messages should be ignored, they are taken
    # care of by the algorithm.
    #
    # IN
    #
    # P1 [nr samples x nr events] : each row specifies a distribution over 
    #                               a discrete event set.
    # P2 [nr samples x nr events] : each row specifies a distribution over 
    #                               a discrete event set.
    # whichlog: '2' - log 2 based (bit) 'e' - log e based (nat). 
    #
    # OUT
    #
    # d [nr. samples x 1]: distances between P1 and P2 calculated as described above.
    #
    # (C) P. Sykacek 2018 <peter@sykacek.net>
    if whichlog=='2':
        Plg=np.log2(P1)-np.log2(P2)
    else:
        Plg=np.log(P1)-np.log(P2)
    # if there are any nans in Plg we set them 0 by del'Hospital
    Plg[np.isnan(Plg)]=0
    # if Plg is -infinity we can set it to 0 since lim x-> 0 x*log(x) is 0.
    Plg[np.logical_and(np.isinf(Plg), np.sign(Plg)==-1)]=0
    return np.sum(P1*Plg, axis=1)

# we define now a function for AIC, BIC calculation
def calc_aic_bic(llhs, npars, ndata):
    # calculate AIC and BIC model metrics.
    # IN
    # llhs:   numpy array of N log likelihood values
    # npars:  numpy array of N model sizes (number of model parameters)
    # ndata:  number of datapoints in training set (a scalar)
    # OUT
    # {aics:np.array([N aic values], bics: np.array([N bic values]}
    #
    # (C) P.Sykacek 2017 <peter@sykacek.net>
    return {'aics':llhs-npars, 'bics':llhs-0.5*npars*np.log(ndata)}

def best_models(aicsbics, mdlpars):
    # extract the best model parameters based on aics and bics
    # IN
    # aicsbics: a dict with 'aics' key referring to a numpy array of AIC values
    #           and the 'bics' key referring to a numpy array of BIC values
    # mdlpars:  a list of list of model parameters.
    #           the first index selects a parameter type and the
    #           second index selects the parameters of that type
    #           for different 
    # OUT
    # {'aicpars':list_of_bestaicpars, 'bicpars':list_of_bestbicpars}
    # 
    # (C) P. Sykacek 2017 <peter@sykacek.net>

    # sort aics in decreasing order and thake the first as best
    best_aic=np.argsort(-aicsbics['aics'])[0]
    # sort aics in decreasing order and thake the first as best
    best_bic=np.argsort(-aicsbics['bics'])[0]
    # we may now loop through mdlpars and extract the best parameters
    # according to aic and bic
    aicpars=[]
    bicpars=[]
    for partyp in mdlpars:
        aicpars.append(partyp[best_aic])
        bicpars.append(partyp[best_bic])
    return {'aicpars':aicpars, 'bicpars':bicpars}

import sklearn.cluster as clust

def lbl2oneofc(targets):
    # converts labels to a 1-of-c target coding
    # IN
    # targets: zero or one based label vector
    # OUT
    # oneofc: one of c coded representation (
    #         column k of row n is 1 if targets[n] is k)
    # (C) P. Sykacek 2017 <peter@sykacek.net>
    targets=targets-np.min(targets)
    nrsamples=targets.shape[0]
    maxclass = np.max(targets)+1
    targs=np.zeros((nrsamples, maxclass))
    for i in range(maxclass):
        # take the row indices from nonzero 
        idone=np.array([list(np.nonzero(targets==i)[0])])
        # idone=idone+nrsamples*i;
        #print(i, idone)
        targs[idone, i]=np.ones((idone.shape[0], 1))
    return targs

def wrap_kmeans(data, mink, maxk, nrep=20, init='random', maxit=100, verbose=False):
    # wrap_kmeans(data, mink, maxk, nrep=20, maxit=100):
    # wrapper around sklearn.cluster.KMeans algorithm that provides a 
    # quick and dirty way of infering the optimal number 
    # of cluster centers.
    # The sum of squares distances to each cluster center are
    # regarded as a negative log likelihood (implying a multivariate 
    # isotropic Gaussian on each kernel and P(k|x)=1 for that 
    # kernel, hence the dirty...).
    # The corresponding deviance is penalized by AIC and BIC.
    # IN
    # data: [nsamples x ndim] data matrix to be clustered.
    # mink, maxk: range of kernel numbers to be searched through.
    # nrep: number of repetitions of KMeans fit.
    # maxit: largest number of iterations in a single k-means fit.
    # OUT (a dict) { 
    # 'aics':   np.array of AIC values for all k
    # 'bics':   np.array of BIC values for all k
    # 'llhs':   np.array of llh values for all k
    # 'shlts':  np.array of silhouette coefficients for all k (to be maximised)
    # 'chinds': np.array of Calinski-Harabaz Index values for all k (to be maximized)
    #           https://stats.stackexchange.com/questions/52838/what-is-an-acceptable-value-of-the-calinski-harabasz-ch-criterion
    # 'allk':   np.array of all k's tested
    # 'aicpars':kernels from best AIC model, and 
    # 'bicpars':kernels from best BIC model
    # 'aicopt': optimal fitted AIC model
    # 'bicopt': optimal fitted BIC model}
    #
    # (C) P. Sykacek 2017, <peter@sykacek.net>
    
    # shlt = metrics.silhouette_score(X, cluster_labels)
    # chind = metrics.calinski_harabaz_score(X, labels)
    if verbose:
        print('mink:', mink, ' maxk:', maxk, ' nrep:', nrep, ' njobs:', njobs, ' init:', init, ' maxit:', maxit)
    datadim=data.shape[1]
    nsamples=data.shape[0]
    llhs=[]
    shlts=[]
    chinds=[]
    npars=[]
    allpars=[]
    kmeval=[]
    sumdists=[]
    allmdls=[]
    for k in range(mink, maxk+1):  # outer loop : sweep over different kernel numbers
        if verbose:
            print('Doing:', k, 'kernels.')
        # get clustering results for nrep reinitialisations running
        # njobs in parallel res contains the best fit.
        # Note: 'threding' does not help as parallel execution may be blopcked by GIL.
        #       With Intel Python the imptrovement is in low level numeric libraries which
        #       use their efficient implementations like BLAS, LAPACK, DAAL etc.
        # 'loky' is python process based and allows for parallel calls of optimizations
        # grid search etc.
        #with parallel_backend('loky', n_jobs=njobs):
        res=clust.KMeans(n_clusters=k, init=init, n_init=nrep,  max_iter=maxit, tol=0.000001).fit(data)
        centers=res.labels_
        # calculate silhouette score and append it to the vector
        try:
            slsc=metrics.silhouette_score(data, centers)
        except:
            slsc=np.nan
        shlts.append(slsc)
        # calculate Calinski-Harabaz Index value and append it to the vector
        try:
            chsc=metrics.calinski_harabasz_score(data, centers)
        except:
            chsc=np.nan
        chinds.append(chsc)
        # tranform the data to distance space and select the distance
        # to the assigned label
        all_dist=res.transform(data)
        # select the distance to the best label 
        min_dist=all_dist[range(all_dist.shape[0]), res.labels_]
        # interpret the negative inertia (sum of distances of 
        # datapoints to nearest kernel center) as log likelihood
        llh=-np.sum(min_dist**2)
        # add the log prior likelihood via a 1 of c target coding:
        nink=np.sum(lbl2oneofc(res.labels_), axis=0)
        llh=llh+np.sum(nink*np.log(nink/nsamples))
        # collect log likelihood, number of model parameters and model parameters
        llhs.append(llh)
        npars.append(np.prod(res.cluster_centers_.shape))
        allpars.append(res)
        kmeval.append(res.inertia_)
        sumdists.append(np.sum(min_dist*min_dist))
        #allmdls.append(res)
    # model fitting is done and we may calculate aic and bic and the best parameters
    # according to both.
    mdl_metrics=calc_aic_bic(np.array(llhs), np.array(npars), nsamples)
    # best_models expects a list of different model parameters each
    # entry containing a list coefficients per model order.
    best_pars=best_models(mdl_metrics, [allpars])
    # concatenate the dictionaries and return the result.
    mdl_metrics.update(best_pars)
    mdl_metrics['allk']=np.array(list(range(mink, maxk+1)))
    mdl_metrics['llhs']=np.array(llhs)
    mdl_metrics['shlts']=np.array(shlts)
    mdl_metrics['chinds']=np.array(chinds)
    mdl_metrics['kmeval']=np.array(kmeval)
    mdl_metrics['sumdists']=np.array(sumdists)
    return mdl_metrics


def sklearngmmallocP(gmm, X):
    ## Function sklearngmmallocP calculates the kernel allocation
    ## probabilities of a fitted sklearn gmm object for samples in
    ## X. The purpose of this function is making sure that we
    ## calculate propoer probabilies as the behaviour of
    ## gmm.predict_proba is unclear. Help says that the dunction
    ## calculates the component density for every sample in X and that
    ## is NOT what we want. It may howeve just be a verbal inaccuracy.
    ##
    ## IN
    ##
    ## gmm: a fitteg sklearn GaussianMIxture object
    ##
    ## X: a compatible input data matrix (drawn from the training data
    ##    distribution).
    ##
    ## OUT
    ##
    ## Palloc: [N x K] matrix with kernel allocation probabilities
    ##    (soft clusteruing).

    ## shortcut to avoid potential problems with one kernel GMMs
    if gmm.n_components==1:
        return np.ones((X.shape[0], 1))
    mns=gmm.means_ ## [n_components x n_features]
    n_features=mns.shape[1]
    n_smpls=X.shape[0]
    if n_features != X.shape[1]:
        raise Exception("Data in X is not compatible with the Gaussian mixture model.")
    ##
    ## dimension of L depends on gmm.covariance_type
    ## "sperical": [n_components x]
    ## "tied": [n_features x n_features]
    ## "diag": [n_components x n_features]
    ## "full": [n_components x n_features x n_features]
    L=gmm.precisions_ ## depends on gmm.covariance_type (see above)
    P=gmm.weights_    ## kernel prior: [n_components x]
    ## calculate the log probabilities
    lgP=np.zeros((n_smpls, gmm.n_components))
    ## initialise Lk with L (covers the "tied" case)
    Lk=L
    for k in range(gmm.n_components):
        ## set the precision for kernel k
        if gmm.covariance_type=="spherical":
            Lk=np.ones((n_features, n_features))*L[k]
        elif  gmm.covariance_type=="diag":
            Lk=np.diag(L[k,:])
        elif gmm.covariance_type=="full":
            Lk=L[k,:,:]
        ## calculate lgP for component k
        ## initialise with sample independent value
        lgP[:,k]=np.log(P[k])-0.5*n_features*np.log(2*np.pi)+0.5*np.log(np.linalg.det(Lk))
        ## add the sample dependent value under kernel k
        mnk=mns[k,:]
        mnk.shape=(1, n_features)
        lgP[:,k]=lgP[:,k]-0.5*np.sum(np.dot(X-mnk[[0]*n_smpls,:], Lk)*(X-mnk[[0]*n_smpls,:]), axis=1)
    ## convert lgP to probabilities:
    return evids2mp(lgP)
# for plotting: a function which establishes a linear map of input
# values to a specified range.
def linmap(vals, vmin=0.0, vmax=5.0):
    # linmap establishes a linear map of input values to a specified
    # range.
    # IN
    # vals: a numpy.array of float values
    # vmin: lower bound of target range
    # vmax: upper bound of target range
    #
    # OUT
    #
    # vals: a numpy.array of float values linearly mapped to the
    #       target range.
    #
    # (C) P. Sykacek 2018 <peter@sykacek.net>
    vals=vals-np.min(vals)
    vals=vals/(np.max(vals)-np.min(vals))
    vals=vals*(vmax-vmin)+vmin
    return vals


# some definitions:
# mcnemar is from Github:
# https://gist.github.com/kylebgorman/c8b3fb31c1552ecbaafb
from scipy.stats import binom

def mcnemar(b, c):
    """
    Compute McNemar's test using the "mid-p" variant suggested by:
    
    M.W. Fagerland, S. Lydersen, P. Laake. 2013. The McNemar test for 
    binary matched-pairs data: Mid-p and asymptotic are better than exact 
    conditional. BMC Medical Research Methodology 13: 91.
    
    `b` is the number of observations correctly labeled by the first---but 
    not the second---system; `c` is the number of observations correctly 
    labeled by the second---but not the first---system.
    """

    n = b + c
    x = min(b, c)
    dist = binom(n, .5)
    p = 2. * dist.cdf(x)
    midp = p - dist.pmf(x)
    return midp

def lab2cnt(y_1, y_2, t):
    # lab2cnt converts two sets of predicted labels and known truth
    # to McNemars counts
    return (sum(np.logical_and((y_1==t), (y_2 !=t))), sum(np.logical_and((y_1!=t), (y_2 ==t))))

##pred=lab2defpred(t)

# helper datatypes and functions for integrating a new datatype into
# sklearn.
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# define default classifier which is derived from
# sklearn base classes which ensure
# compatibility with other sklearn features.
class DefClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    def fit(self, X, y):
        # fit function for default predictions
        # calculates class prior and majority label
        # ignores X!
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.unqlab_=np.unique(y)
        self.P_=np.zeros(self.unqlab_.shape)
        for index, lab in np.ndenumerate(self.unqlab_):
            self.P_[index]=sum(y==lab)
        self.P_=self.P_/sum(self.P_)
        self.predlab_=self.unqlab_[np.argmax(self.P_)]
        # Return the classifier
        return self
    def predict(self, X):
        # predict default label as inferred by fit.
        # no rows = no samples

        # Check whether fit had been called
        check_is_fitted(self, ['P_', 'unqlab_', 'predlab_'])

        # Input validation
        X = check_array(X)
        nsampl=X.shape[0]
        predy=np.repeat(self.predlab_, nsampl)
        return predy
    def score(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # calculate accuracy as score
        return sum(self.predict(X)==y)/len(y)


## define an exception class
class PyBioExcept(Exception):
    pass
    
## a class for representing esets in Python
class PyEset:
    ## data type for representing bioconductor esets in Python.
    ## 
    ## (C) P. Sykacek 2019 <peter@sykacek.net>

    def compatindex(value, rdx, cdx):
        ## check whether value is a PyEset() and whether
        ## all index values in value agree with both rdx and cdx.
        if type(value) != type(PyEset()):
            raise PyBioExcept("value must be a PyEset() object!")
        brdx=value.exprs.index
        bcdx=value.pdata.index
        ok=len(rdx)==len(brdx) and len(cdx)==len(bcdx)
        if ok:
            brds=pd.Series(brdx)
            brds.sort()
            rds=pd.Series(rdx)
            rds.sort()
            rcds=pd.Series(bcdx)
            rcds.sort()
            cds=pd.Series(cdx)
            cds.sort()
            ok=ok and all(brds==rds) and all(rcds==cds)
        return ok
    def __init__(self, exprs=None, fdata=None, pdata=None):

        if exprs is not None:
            if type(exprs) == type(np.array([])):
                ## we make sure exprs is a dataframe
                exprs=pd.DataFrame(exprs)
        self.exprs=exprs
        self.fdata=fdata
        self.pdata=pdata
        if self.exprs is not None:
            if type(self.exprs) != type(pd.DataFrame()):
                raise PyBioExcept("Wrong data type!")
            if self.fdata is not None:
                if type(self.fdata) != type(pd.DataFrame()):
                    raise PyBioExcept("Wrong data type!")
                ## here we know that fdata is a dataframe and should check the
                ## agreement between row indices in fdata and exprs
                if len(self.fdata.index) != len(self.exprs.index) or not all(self.exprs.index == self.fdata.index):
                    ## we have a mismatch in feature data and expression data.
                    raise PyBioExcept("Missmatch in feature ids!")
                ## final step: we make sure that the order bewteen
                ## exprs and fdata is fine
                self.fdata=self.fdata.loc[self.exprs.index]
            if self.pdata is not None:
                if type(self.pdata) != type(pd.DataFrame()):
                    raise PyBioExcept("Wrong data type!")
                ## here we know that fdata is a dataframe and should check the
                ## agreement between row indices in fdata and exprs
                if len(self.pdata.index) != len(list(self.exprs)) or not all(pd.Series(list(exprs))==self.pdata.index):
                    ## we have a mismatch in feature data and expression data.
                    raise PyBioExcept("Missmatch in sample ids!")
                ## final step: we make sure that the order bewteen exprs and pdata is fine.
                self.pdata=self.pdata.loc[list(self.exprs)]
        
    def loadfromfile(self, fnambase, dataext="_AMP_data.csv",
                 ftrext="_features.csv", phenoext="_pheno.csv",
                 sep="\t", whichlog=np.log2, map2log=True):
        ## initialise a Python eset from files.
        datfnam=fnambase+dataext
        ftrfnam=fnambase+ftrext
        phenofnam=fnambase+phenoext

        ## load data:
        exprs=pd.read_csv(datfnam, sep=sep, index_col=0)
        fdata=pd.read_csv(ftrfnam, sep=sep, index_col=0)
        pdata=pd.read_csv(phenofnam, sep=sep, index_col=0)
        ## convert expressions to log if indicated
        if map2log:
            print("map2log")
            rownams=exprs.index
            colnams=list(exprs)
            exprs=np.array(exprs)
            exprs=whichlog(exprs)
            ## back to dataframe
            exprs=pd.DataFrame(exprs, columns=colnams, index=rownams)
        self.__init__(exprs=exprs, pdata=pdata, fdata=fdata)
        
    def savetofile(self, fnambase, dataext="_AMP_data.csv",
                   ftrext="_features.csv", phenoext="_pheno.csv",
                   sep="\t", whichexp=np.exp2, map2exp=True):
        ## save current pyeset object to csv files.
        datfnam=fnambase+dataext
        ftrfnam=fnambase+ftrext
        phenofnam=fnambase+phenoext
        ## covert expressions to exponential scale:
        if map2exp:
            rownams=self.exprs.index
            colnams=list(self.exprs)
            exprs=np.array(self.exprs)
            exprs=whichexp(exprs)
            ## back to dataframe
            self.exprs=pd.DataFrame(exprs, columns=colnams, index=rownams)
        ## save data to csv files:
        self.exprs.to_csv(datfnam, sep=sep, index_label=False)
        self.pdata.to_csv(phenofnam, sep=sep, index_label=False)
        self.fdata.to_csv(ftrfnam, sep=sep, index_label=False)
       
        
    def __getitem__(self, selector):
        ## selector is a tuple with row and column index entries or values.
        if len(selector) !=2:
            raise PyBioExcept("Subseting requires a row and column selector!")
        rowsel=selector[0]
        ## remove the Series type
        if type(rowsel)==type(pd.Series()):
            rowsel=rowsel.values
        colsel=selector[1]
        ## remove the Series type
        if type(colsel)==type(pd.Series()):
            colsel=colsel.values
        if type(rowsel)==type(list()) and type(rowsel[0]) in [type(True), type(self.exprs.index[0])]:
            ## we have a row selector for loc
            exprs=self.exprs.loc[rowsel,:]
            fdata=self.fdata.loc[rowsel,:]
        else:
            ## we try iloc which also covers slices
            exprs=self.exprs.iloc[rowsel,:]
            fdata=self.fdata.iloc[rowsel,:]
        if type(colsel)==type(list()) and type(colsel[0]) in [type(True), type(self.exprs.index[0])]:
            ## we have a column selector for loc
            exprs=exprs.loc[:,colsel]
            pdata=self.pdata.loc[colsel,:]
        else:
            ## we try iloc which also covers slices
            exprs=exprs.iloc[:,colsel]
            pdata=self.pdata.iloc[colsel,:]
        ## we have now all selected and return a deep copy of a new PyEset
        return copy.deepcopy(PyEset(exprs, fdata, pdata))
    def __setitem__(self, selector, value):
        ## selector is a tuple with row and column index entries or values.
        if len(selector) !=2:
            raise PyBioExcept("Subseting requires a row and column selector!")
        if type(value) != type(PyEset()):
            raise PyBioExcept("Right hand value must be a PyEset() object!")
        rowsel=selector[0]
        colsel=selector[1]
        if type(rowsel)==type(list()) and type(rowsel[0]) in [type(True), type(self.exprs.index[0])]:
            ## we have a row selector for loc
            exprs=self.exprs.loc[rowsel,:]
        else:
            ## we try iloc which also covers slices
            exprs=self.exprs.iloc[rowsel,:]
        ## from exprs we get a loc compatible row index:
        rdx=exprs.index
        if type(colsel)==type(list()) and type(colsel[0]) in [type(True), type(self.exprs.index[0])]:
            ## we have a column selector for loc
            pdata=self.pdata.loc[colsel,:]
        else:
            ## we try iloc which also covers slices
            pdata=self.pdata.iloc[colsel,:]
        ## from pdata we get a loc compatible index
        cdx=pdata.index
        ## we do now check whether the indices agree with the roight hand side PyEset()
        if not PyEset.compatindex(value, rdx, cdx):
            PyBioExcept("Right hand value indices must agree with left hand side selector!")
        ## we can now do the seting of values:
        self.exprs.loc[rdx, cdx]=value.exprs.loc[rdx, cdx]
        self.fdata.loc[rdx,:]=value.fdata.loc[rdx,:]
        self.pdata.loc[cdx,:]=value.pdata.loc[cdx,:]
    def tolabeleddata(self, labelcols):
        ## tolabeleddata converts a PyEset to a dict of inputs (X),
        ## targets (Y), rownams, Xcolnams and Ycolnams. Samples are
        ## rows of X and Y (Note Y is a pd.Series if labelcols
        ## contains only one entry).
        ##
        ## OUT:
        ##
        ## dict(X=... inputs [nsample x nfeatures] matrix
        ##      Y=... targets [nsampe x ntargs] matrix or
        ##                    [nsampe x 0] vector (only one target selected)
        ##      Xcolnams ... feature names
        ##      Xrownams ... sample names
        ##      Ycolnams ... target names
        ## )
        ## (C) P. Sykacek 2019 <peter@sykacek.net>
        X=np.transpose(np.array(self.exprs))
        Xcolnams=self.exprs.index
        Xrownams=list(self.exprs)
        Y=np.array(self.pdata.loc[:, labelcols])
        if len(Y.shape)>1:
            Y=np.transpose(Y)
        Ycolnams=labelcols
        return {"X":X, "Y":Y, "Xcolnams":Xcolnams, "Xrownams":Xrownams, "Ycolnams":Ycolnams}

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
    probs=np.zeros_like(vals)
    K=vals.shape[1]
    for kid in range(K):
        probs[:, kid]=1./np.sum(np.exp(vals-vals[:,[kid]*K]), axis=1)
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
        self.classes_=self.unqy
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
        ## yp=self.krr.predict(X)
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

new_aggregated="../aggregation_tabular/agg_final.csv"
new_transformed="../aggregation_tabular/agg_trans.csv"
### data loading
### 1) column names for different processing steps
col4tmlab="Transmembrane"                      ## column name in classic_raw holding the transmembran labels                                          
col4eratelab="GMM_Label_Combined"              ## column name in classic_raw holding the evolutionary rate labels  

## input label columns to be used in function extract4ml as parameters Xcols with discardXcols=True
rem_mns_fnam="../aggregation_tabular/colnams_remove_means.txt"      ## column names containing mean values
rem_sums_fnam="../aggregation_tabular/colnams_remove_sums.txt"      ## column names containing sum values
## input label columns to be used in function extract4ml as parameters Xcols with discardXcols=True
keep_but_small_counts="../aggregation_tabular/colnamesdropsmallcounts.txt" ## retains all but small counts 
keep_mean_props="../aggregation_tabular/collnamesselmeanprops.txt"         ## retains mean values and proportions
keep_sum_props="../aggregation_tabular/colnamesselsumprops.txt"            ## retains sum values and proportions (including all counts)

## column names in classic_aggregated which appeared in addition to
## the data in classic_raw interesting for prediction and unsupervised analysis
cols_from_aggregated_fnam="../aggregation_tabular/extract_labels_from_raw.txt"
def getcolnams(fnam):
    ## read column names from files (for merging data and feature extraction)
    ## IN
    ##
    ## fnam: (one of the above csv files with column names)
    ##
    ## OUT
    ##
    ## colnams: a list with white space stripped column names.
    with open(fnam,"r") as f:
        string = f.read()
    colnams=[part.strip() for part in string.split(",")]
    return colnams

def extract4ml(fname=new_transformed, iscsvfile=True, idcolnam="ExpID", Xcols=[], discardXcols=True, ycol=col4tmlab, classlabs=[col4tmlab, col4eratelab]):
    ## extracts data from the "classically" aggregasted features for
    ## machine learning. Returns a tuple (X, y, rowids, Xcolids)
    ## which describes and provides the input data and labels.
    ##
    ## IN
    ##
    ## fname: csv file name with features. Defaults to new_transformed
    ##
    ## iscsvfile: boolean indicstion whether we read a csv file or a
    ##        generic table. Defaults to True.
    ##
    ## idcolnam: name or column with row ids (entries returned as rowids)
    ##
    ## Xcols: definition of column names which end up in the retruned
    ##        input matrix X.  For discardXcols==True the entries in
    ##        Xcols are discarded. For discardXcols==False and Xcols
    ##        != [], the entries in Xcols are retained.
    ##
    ## discardXcols: Boolean which controls the handling of
    ##        Xcols. Defaults to True.
    ##
    ## ycol: name of label column which ends up in y. Defaults to
    ##        col4tmlab. If not found or empty string, y=np.array([])
    ##
    ## classlabs: label columns which together with idcolnam are
    ##        discarded from the data when creating X. Defaults to
    ##        [col4tmlab, col4eratelab]
    ##
    ## OUT
    ##
    ## tuple with
    ##
    ## (X, input values as stored in fname and extraced according to
    ##        the definition of column names which end up in X.
    ##
    ## y, target values according to the above defuinition.
    ##
    ## rowids, row ids as found in column name idcolnam
    ##
    ## Xcolids) names of the columns in X.

    if iscsvfile:
        pdfrm=pd.read_csv(fname)
    else:
        pdfrm=pd.read_table(fname)
    ## initialise Xcolids with all column names
    Xcolids=list(pdfrm)
    ## extract the rowids and the label
    if idcolnam != "" and idcolnam in Xcolids:
        rowids=pdfrm[idcolnam].to_list()
        Xcolids.remove(idcolnam)
    else:
        rowids=[]
    if ycol != "" and ycol in Xcolids:
        y=pdfrm[ycol].to_numpy()
    else:
        y=np.array([])
    ## we prepare Xcolds such that it contains only column names which should end up in X
    if discardXcols:
        ## we remove the label columns and the entries in Xcols from Xcolids
        disccols=Xcols+classlabs
        Xcolids=[colnam for colnam in Xcolids if not colnam in disccols]
    elif Xcols!=[]:
        Xcolids=Xcols
    else:
        ## drop the label collumns
        Xcolids=[colnam for colnam in Xcolids if not colnam in classlabs]
    ## finally we are ready to extract the inputs
    X=pdfrm[Xcolids].to_numpy()
    return (X, y, Xcolids, rowids)

embed_fnam_mn="../aggregation_tabular/GNN_mean_ALL_FOLDS.csv"
embed_fnam_mx="../aggregation_tabular/GNN_max_ALL_FOLDS.csv"
embed_fnam_sm="../aggregation_tabular/GNN_sum_ALL_FOLDS.csv"
embed_fnam_s2s="../aggregation_tabular/GNN_set2set_ALL_FOLDS.csv"

labcol="y_true"
idcolnam="ExpID"
probcol="y_prob"
maxembeddingdim=512
Xcolids=["emb_{0}".format(cid) for cid in range(maxembeddingdim)]

def ext_gnn_preds(fnam=embed_fnam_s2s, idcolnam=idcolnam,
                  labcol=labcol, getembed=True, Xcolids=Xcolids,
                  probcol=probcol):
    ## Function ext_gnn_embed extracts different calculated quantities
    ## and the associated sample information from trained GNNs which
    ## are provided in the csv file fnam. The informatio which we
    ## extract are 1) the sample ids (column idcol); 2) the sample
    ## label (column labcol) and in dependence of the flag "getembed"
    ## 3) the GNN embedding after the final pooling and flattening
    ## (the default for getembed=True) or unbiased label probabilities
    ## (probcol) in case getembed==False.
    ##
    ## IN
    ##
    ## fnam: csv file name with GNN embedding, predicted probabilities
    ##       and sample annotation information. Defaults to
    ##       embed_fnam_s2s="../aggregation_tabular/GNN_set2set_ALL_FOLDS.csv"
    ##
    ## idcolnam: column name of sample ids (defaults to idcolnam="ExpID")
    ##
    ## labcol: column name with sample labels (defaults to labcol="y_true")
    ##
    ## getembed: controls data extraction (default = True which leads
    ##       to extracting the embedded values).
    ##
    ## Xcolids: column names for the embedding dimensions. Defaults to
    ##       Xcolids=["emb_0", ..., "emb_512"] which gets reduced to
    ##       the actually available embedding dimensions (will in
    ##       general be fewer)
    ##
    ## probcol: column name with predicted probabilities. Defaults to
    ##       probcol="y_prob".
    ##
    ## OUT a tupple with
    ##
    ##(X,      : in dependence of getembed either the embedding or the predicted probabilities
    ##
    ## y,      : known target labels
    ##
    ## Xcolids,: Column ids which are extracted
    ##
    ## rowids) : row ids (sample ids).

    dfrm=pd.read_csv(fnam)
    allcolnams=list(dfrm)
    rowids=dfrm.loc[:,idcolnam].to_list()
    y=dfrm.loc[:, labcol].to_numpy()
    if getembed:
        Xcolids=[colid for colid in Xcolids if colid in allcolnams]
        X=dfrm.loc[:, Xcolids].to_numpy()
    else:
        Xcolids=[probcol]
        X=dfrm.loc[:, Xcolids].to_numpy()
    ## done and return all values#
    return (X, y, Xcolids, rowids)

def roc_auc_shannon_acc_mcnemar(Pcoloc, ycoloc):
    ## ROC_AUC_Shannon calculates ROC curve coordinates (y=sens over
    ## x=1-spec), or (y=tpr, x=fpr) the area under the ROC curve (AUC)
    ## for a vector of binary probabilities for predicting ycoloc and
    ## an estimate of the Shannon information which is contained in
    ## Pcoloc **about correctly predicted ycoloc**. We set the Shannon
    ## infomration for all wrongly predicted samples to 0. The
    ## function also calculates the prediction accuracy assuming that
    ## Pcoloc is an unbiased prediction of ycoloc this is a
    ## generalisation accuracy and finally calculates the McNemar
    ## p-value testing the null hypothesis that Pcoloc based
    ## predictions and majority vote predictions provide the same
    ## accuracy agauinst a two sided alternative. The majoruity vote
    ## corresponds to the class which has the lardest number of
    ## samples in ycoloc and is estimated directly. This might be
    ## biased but this will not matter because the unbiased majority
    ## and the biased majority are in most situations identical.
    ##
    ## The only prerequisite for this assessment is that we are
    ## assessing a binary classification problem.
    ##
    ## IN
    ##
    ## Pcoloc: unbiased posterior for predicting the binary labels in
    ##         ycoloc. The code assumes that Pcoloc holds the
    ##         probabilities for ycoloc==1 It is thus [Nsample x]
    ##
    ## ycoloc: binary (colocation) indictor which is 1 for all sample
    ##         pairs which share a geographical location. A [Nsample x] vector
    ##         
    ## OUT:  (a tuple)
    ##
    ## (ROCx, ROCy,   ROC curve coordinates
    ##
    ##  AUC,  area under the ROC curve
    ##
    ##  shannon, modified Shannon channel capacity (KL zero for
    ##        misclassified samples)
    ##
    ##  acc,  genrealisation accuracy predicting ycoloc
    ##
    ## pval) pval of McNemars test when comparing against the majority
    ##        vote The latter assesses information content in X about
    ##        y with a significance test.
    
    Pcoloc=Pcoloc.ravel()
    ycoloc=ycoloc.ravel()
    ## first we calculate the ROC curve coordinates
    Psrtd=np.sort(Pcoloc)
    Pc=Pcoloc.copy()
    Pc.shape=(Pc.shape[0], 1)
    Psrtd.shape=(1, Psrtd.shape[0])
    isabove=Pc[:, [0]*Pc.shape[0]]>=Psrtd[[0]*Pc.shape[0],:]
    tpcnt=np.sum(isabove[ycoloc==1,:], axis=0)
    fpcnt=np.sum(isabove[ycoloc==0,:], axis=0)
    ROCy=tpcnt/np.sum(ycoloc==1)
    ROCx=fpcnt/np.sum(ycoloc==0)
                 
    nsmpl=Pcoloc.shape[0]
    Pcoloc.shape=(nsmpl,)
    Ppcl=np.mean(ycoloc)
    ## 1) get accuracy and McNemar p-value
    acc=100.0*np.sum((Pcoloc>0.5).astype(int)==ycoloc)/Pcoloc.shape[0]
    ## 2) create the default prerictions
    if Ppcl < 0.5:
        ymajor=np.zeros_like(ycoloc)
    else:
        ymajor=np.ones_like(ycoloc)
    ## 3) get McNemar counts and the p-value
    na,nb=lab2cnt((Pcoloc>0.5).astype(int), ymajor, ycoloc)
    pval=mcnemar(na, nb)
    ## prepare probabilties for Shannon calculation
    yPclpred=Pcoloc>0.5
    Pcl4shannon=Pcoloc.copy()
    Pcoloc.shape=(Pcoloc.shape[0], 1)
    ## in case it predicts a whron colocation state we set the
    ## colocation probability to the prior. This effectively sets the
    ## corresponding KL divergence to 0.
    #Pcl4shannon[yPclpred!=ycoloc]=Ppcl
    #Pcl4shannon.shape=(Pcoloc.shape[0], 1)
    Pc4s=np.concatenate((Pcoloc, 1-Pcoloc), axis=1)
    Ppcl=np.array([[Ppcl, 1-Ppcl]])
    ## see e.g. Eqn (7) from The impact of quantitative optimization
    ## of hybridization conditions on gene expression analysis
    KLS=kldisc(Pc4s, Ppcl[[0]*nsmpl,:])
    #print(yPclpred.shape)
    #print(ycoloc.shape)
    wronglabels=yPclpred!=ycoloc
    #print(wronglabels.shape)
    KLS[wronglabels.ravel()]=-KLS[wronglabels.ravel()]
    shannon=np.mean(KLS) 
    ## We may express the AUC as the fraction of sample pairs where
    ## one sample has y==1 and the second y==0 for which we
    ## find Pcoloc of a member from the second set larger than
    ## Pcoloc from the first set.
    P1=Pcoloc[ycoloc==1,0]
    n1=P1.shape[0]
    P1.shape=(n1,1)
    P2=Pcoloc[ycoloc==0,0]
    n2=P2.shape[0]
    P2.shape=(n2,1)
    P1=P1[:,[0]*n2]
    P2=P2[:,[0]*n1].T
    allone=np.ones(P1.shape)
    allone[P1<=P2]=0
    #print(allone.shape)
    AUC=np.mean(allone) ## fraction of evets where P1 is larger P2.
    return (ROCx, ROCy, AUC, shannon, acc, pval)

def pval2ind(pval):
    ## map pvalues to significance indicator:
    ## pval < 0.001 -> ***
    ## pval < 0.01  -> **
    ## pval < 0.05  -> *
    ## pval < 0.1   ->.
    ## else -> not sig.
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    elif pval < 0.1:
        return "."
    else:
        return "not sig."

def crossvalprobs(estimator, X, y,
                  cv=StratifiedKFold(n_splits=5, shuffle=True),
                  verbosity=True):
    ## crossvalprobs can only be applied to classification type
    ## estimators which provide a predioct_proba function. It uses a
    ## StratifiedKFold object to split data such that the label
    ## distibution is maintained constant over all splits and
    ## aggregates the predicted probabilities in a manner which is
    ## similar top cross_val_predict. THhe function is however
    ## inherently serial because parallelisation should only happen at
    ## the level of the estimator (GSCV is usually mandatory).
    ##
    ## IN
    ##
    ## estimator: a parameterised classification model (or a grid
    ##            search cv instaance) must provide methods fit() and
    ##            predict_proba()
    ##
    ## X, y: Inputs and classification targets.
    ##
    ## cv: a StratifedKfold instance. Defaults to StratifiedKfold with
    ##            5 folds and shuffle=True.
    ##
    ## verbosity: A boolean to control the output provided on the
    ##            console.  Defaults to True which cases the function
    ##            to write the fold iterations to the console.
    ##
    ## OUT
    ##
    ## Probs: a [Nsample x K] matrix of unbiased probabilities for
    ##        class which are predicted for the test samples.

    ## prepare the output probabilities as one-hot encoded labels
    ## this is only the initialisation and will be exchaged with
    ## unbiased probabilities.
    if len(y.shape)>1:
        ## we have a one hot coding and initialise Pros with it.
        Probs=y.copy()
    else:
        ## we have a label vector which we map to a one hot coding
        Probs=np.asarray(OneHotEncoder(sparse_output=False).fit_transform(y.reshape((y.shape[0],1))))
    mxrnd=cv.get_n_splits()
    ## loop over the iterator
    for cntr, (trix, tsix) in enumerate(cv.split(X, y)):
        if verbosity:
            print("round {0} of {1}".format(cntr+1, mxrnd))
        ## get training data
        Xtr=X[trix,:]
        try:
            ## try wheter y is one hot coded
            ytr=y[trix,:]
        except:
            ## oterwise we have a simple vector
            ytr=y[trix]
        ## train the estimator
        estimator.fit(Xtr, ytr)
        ## get the test inputs
        Xts=X[tsix,:]
        Probs[tsix,:]=estimator.predict_proba(Xts)
    return Probs

def prob_scoring(y_true, Probs):
    ## prob_scoriing is a GSCV compatible scoring function, which
    ## evaluates the log likelihood of y_true under the predicted
    ## probabilities Probs. The motivation of this scorer is to make
    ## sure that we prefer predictors which are certain about class as
    ## this translates to large mutual information.
    ##
    ## IN
    ##
    ## y_true: given labels
    ##
    ## Probs: probabilities prediced for all labels under the trained
    ##        model.
    ##
    ## OUT
    ##
    ## llh: log likelihood (maximal 0, minimal -infinity) larger is better is true!

    ## adapt Probs.
    if len(Probs.shape)==1:
        Probs.shape=(Probs.shape[0],1)
        Probs=np.concatenate((1-Probs, Probs), axis=1)
    elif Probs.shape[1]==1:
        Probs=np.concatenate((1-Probs, Probs), axis=1)
    if len(y_true.shape)==1:
        ## we move y_true to one hot coded labels
        y_true=np.asarray(OneHotEncoder(sparse_output=False).fit_transform(y_true.reshape((y_true.shape[0],1))))
    ## we have now y_true as one hot coded matrix which has the same
    ## shape as Probs by taking the sum of logarithm of the element
    ## wise power of Probs to y_true we get the desired metric
    return np.sum(np.log(Probs**y_true))

prob_scorer=make_scorer(prob_scoring, greater_is_better=True, needs_proba=True)

def ftrnam2accronym(fnam="../aggregation_tabular/feature_names2_accronyms.csv", ftrcol="features", accrocol="features_abbreviation"):
    ## maps the mlDIAMANT feature names to unique shorter accronyms
    dfrm=pd.read_csv(fnam)
    return dict(zip(dfrm.loc[:, ftrcol].to_list(), dfrm.loc[:,accrocol].to_list()))
    
MINVAL=10.0**-101
CNTPRIOR=1

def saflog(x, minval=MINVAL):
    ## take safe natural logarithm
    if type(x) != type(np.array([])):
        x=np.array(x)
    x[x<minval]=minval   
    return np.log(x)

    
def maptreeimp(vals):
    ## maps tree importance values by taking logs and
    ## subtracting the smallest value
    vals=saflog(vals)
    return vals-np.min(vals)

mapcols={"names":lambda x:x,
         "RFC_gini_importance":lambda x:x,        ## importances are between 0 and 1 -> larger is better
         "RFC_entropy_importance":lambda x:x,     ## importances are between 0 and 1 -> larger is better
         "RFC_logloss_importance":lambda x:x,     ## importances are between 0 and 1 -> larger is better
         "Bayes_logodds":lambda x:x-np.min(x),    ## Bayes factors are on a nominal scale and moved to positive values -> larger is better
         "SVC_acc":lambda x:x,                    ## accuracy are in percent, moved to 0 and 1, logged and transformed to positive values -> larger is better
         "SVC_auc":lambda x:x,                    ## AUC values are between 0 and 1 -> larger is better
         "SVC_Shannon": lambda x:x,               ## Shannon channel capacities are on a positive nominal scale and kept -> larger is better
         "SVC_McNemar": lambda x: -saflog(x)      ## p-values are on a 0-1 scale, taken log and negated to convert  smaller is better -> larger is better
         }

valcols=["RFC_gini_importance", "RFC_entropy_importance", "RFC_logloss_importance",
         "Bayes_logodds", "SVC_acc", "SVC_auc", "SVC_Shannon", "SVC_McNemar"]



def vals2sharedrank(resdf, mapcols=mapcols, valcols=valcols):
    ## vals2sharedrank takes all columns from valcols, maps the values
    ## and aggregates them according to the averaged relative
    ## sizes. The function returns a reordered version of resdf in
    ## decreasing order of the aggregated performance. To allow for a
    ## meaningful visualisation all numerical valuies are replaced by
    ## relative sizes.
    ##
    ## IN
    ##
    ## resdf: result dataframe as created by the Jupyter notebook
    ##        supervised.ipynb and stored in file
    ##        ./physicoftrs4transstate_eval.csv
    ##
    ## mapcols: dictionary with mapping functions for all feature
    ##        importance values as calculated by supervised.ipynb
    ##
    ## valcols: feature importance values as calculated in
    ##        supervised.ipynb
    ##
    ## OUT
    ##
    ## ndf: A dataframe with all feature importance values represented
    ##         as (transformed) feature specific proportions such that
    ##         larger corresponds to more important. Rows are ordered
    ##         by decreased average proportion value when aggregating
    ##         accross features.

    ## copy to preserve the data
    ndf=resdf.copy()
    ## map individual column values
    for col in valcols:
        vals=mapcols[col](ndf.loc[:, col].to_numpy())
        ndf.loc[:, col]=vals/np.sum(vals)
    ## aggregate all columns by averaging and negating to get the values for ranking
    vals4rank=-np.mean(ndf.loc[:,valcols], axis=1)
    id4rank=np.argsort(vals4rank)
    return ndf.iloc[id4rank,:]


def vals2rankprob(resdf, namcol="names", mapcols=mapcols,
                   valcols=valcols, npos=5, mapnams={}):
    ## val2rankprob ranks resdf by all columns in valcols and counts
    ## the counts of the different resdf[:, namcol] entries at position
    ## 0:npos-1. The aggregated feature counts for the top npos rank positions
    ## is returned in a dataframe.
    ##
    ## IN
    ##
    ## resdf: result dataframe as created by the Jupyter notebook
    ##        supervised.ipynb and stored in file
    ##        ./physicoftrs4transstate_eval.csv
    ##
    ## mapcols: dictionary with mapping functions for all feature
    ##        importance values as calculated by supervised.ipynb
    ##
    ## valcols: feature importance values as calculated in
    ##        supervised.ipynb
    ##
    ## npos: number of leading ranks considered in the transformation.
    ##
    ## mapnams: dictionary to map the namcol entries in resdf to
    ##        names which are easier to read.
    ##
    ## OUT
    ##
    ## topscore: A dataframe with npos+1 columns indicating in the names column
    ##        all features which occur after ranking in the top npos position.

    ## copy to preserve the data
    ndf=resdf.copy()
    ## map individual column values
    for col in valcols:
        ## we map the values
        vals=-1.0*mapcols[col](ndf.loc[:, col].to_numpy())
        ## and replace the column with rank positions
        ndf.loc[:, col]=np.argsort(vals)
    ## to continue we move the dataframe to a numpy array
    rankinf=ndf.loc[:, valcols].to_numpy()
    ## print(rankinf)
    rankvals={}
    for pos in range(npos):
        rankvals["pos{0}".format(pos+1)]=np.zeros((rankinf.shape[0],)).tolist()
    #rankvals={"pos1":np.zeros((rankinf.shape[0],)).tolist(),
    #          "pos2":np.zeros((rankinf.shape[0],)).tolist(),
    #          "pos3":np.zeros((rankinf.shape[0],)).tolist(),
    #          "pos4":np.zeros((rankinf.shape[0],)).tolist(),
    #          "pos5":np.zeros((rankinf.shape[0],)).tolist()}
    rankdf=pd.DataFrame(rankvals ,index=ndf.loc[:,namcol])
    for pos in range(npos):
        ## aggregate all pos positions in rankinf
        posagg=np.sum(rankinf==pos, axis=1)
        ## print(posagg)
        rowdxvals=ndf.loc[posagg>0, namcol]
        poscolnam="pos{0}".format(pos+1)
        rankdf.loc[rowdxvals, poscolnam]=posagg[posagg>0]
    ## we can finally drop all features fromrankdf which have zeros in all columns
    rankdf.drop(index=ndf.loc[np.sum(rankdf.to_numpy(), axis=1)==0, namcol], inplace=True)
    return rankdf


greyscale =[str((1.0*x)/255) for x in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]] # grayscale values can be used as colour in plt.pie()


import matplotlib.pyplot as plt

def createpies(rankdf, labelcol="index", valcols=[], colors=greyscale, explode=0.025, subfigsz=[4.8, 4.8], pie_title="Rank position {0}"):
    ## crates a series of subplots with pie charts
    ##
    ## IN
    ##
    ## rankdf: a dataframe with labels in labelcol and rank counts in columns "pos1".. "posn".
    ##
    ## labelcol: column in rankdf with label information. Defaults to
    ##         "index" which indicates that the dataframe index
    ##         contains the label information.
    ##
    ## valcols: column mames which give rise to the different rank
    ##          positions and pie charts.  if [] we assume the above
    ##          namind convention and generate for every such column a
    ##          pie chart.
    ##
    ## colors: color information for the pie segments. Defaults to grey scale.
    ##
    ## explode: a float value which identifies the gap size between
    ##          individual pie slices.
    ##

    if len(valcols)==0:
        ## safety initialisation
        valcols=[]
        labelcols=[]
        ## we create the value column names under the default assumption
        dfcolnams=list(rankdf)
        maxcols=len(dfcolnams)
        for colit in range(maxcols):
            colnam="pos{0}".format(colit+1)
            if colnam in dfcolnams:
                valcols.append(colnam)
                labelcols.append(pie_title.format(colit+1))
    ## create a figure and subplots
    fig, axtupple=plt.subplots(1, len(valcols), figsize=[subfigsz[0]*len(valcols), subfigsz[1]])
    allnams=rankdf.index.to_numpy()
    for posid, ax in enumerate(list(axtupple)):
        ## generate the labels and values such that zeros get omitted.
        vals=rankdf.loc[allnams, valcols[posid]].to_numpy()
        havevals=vals!=0
        labels=allnams[havevals]
        vals=vals[havevals]
        ax.pie(vals, explode=[explode]*vals.shape[0], labels=labels, colors=colors, textprops={'fontsize': 20})
        ax.set_title(labelcols[posid], fontsize=22, fontweight='bold')
    return fig

emptystore={"mthd":[], "rocx":[], "rocy":[], "auc":[], "acc":[], "shif":[], "pval":[]} 

def clssmetrics_2_store(mthd, rocx, rocy, auc, acc, shif, pval, store=emptystore):
    ## adds the classification metrics as estimated by function roc_auc_shannon_acc_mcnemar
    ## to store.
    ##
    ## IN
    ##
    ## mthd: short method string which describes the classifier which
    ##       is characterised by the subsequent metrics.
    ##
    ## rocx: [Nsamples x] vector of x coordinates of roc curve (array like)
    ##
    ## rocy: [Nsamples x] vector of y coordinates of roc curve (array like)
    ##
    ## auc:  float (0.5 <= auc <=1) AUC (area under ROC curve)
    ##
    ## acc:  float (P (majority class prior) <= acc <= 1.0) Generalisation accuracy
    ##
    ## shif: float (0 < shif) modified Shannon channel capacity
    ##
    ## pval: float 0 < pval <1.0 p-value of McNemar test when
    ##       comparing predictions against the majority vote
    ##
    ## store: a dictionary with keys: "mthd", "rocx", "rocy", "auc",
    ##       "acc", "pval", each representing a Python list which
    ##       contains as elements the respective parameter values of
    ##       previous calls of clssmetrics_2_store. Defaults to a
    ##       dictionary with all keys refering to empty lists.
    ##
    ## OUT
    ##
    ## store: the input store dictionary with all entries augmented by the
    ##       parameter values of the current calls.
    ##print(store)
    store["mthd"].append(mthd)
    store["rocx"].append(rocx)
    store["rocy"].append(rocy)
    store["auc"].append(auc)
    store["acc"].append(acc)
    store["shif"].append(shif)
    store["pval"].append(pval)
    return store

import pickle
defaultstore="../results/classmetrics.pkl"
def dumpstore(store, fname=defaultstore):
    ## Function dumpstore uses pickle to write "store" to a file
    ## named according to parameter fname.
    ##
    ## IN
    ##
    ## store: a dictionary with keys: "mthd", "rocx", "rocy", "auc",
    ##       "acc", "pval", each representing a Python list which
    ##       contains as elements the respective parameter values of
    ##       previous calls of clssmetrics_2_store.
    ##
    ## fname: name of pickle file; defaults to
    ##       defaultstore="../results/classmetrics.pkl"
    ##
    ## OUT - no output generated
    with open(fname, "wb") as ofile:
        pickle.dump(store, ofile)
    
def readstore(fname=defaultstore):
    ## Function readstore uses pickle to read "store" from a file
    ## named according to parameter fname.
    ##
    ## IN
    ##
    ## fname: name of pickle file; defaults to
    ##       defaultstore="../results/classmetrics.pkl"
    ##
    ## OUT - no output generated
    ##
    ## store: a dictionary with keys: "mthd", "rocx", "rocy", "auc",
    ##       "acc", "pval", each representing a Python list which
    ##       contains as elements the respective parameter values which
    ##       were generated with clssmetrics_2_store and pickled with function
    ##       dumpstore.
    
    with open(fname, "rb") as ifile:
        store=pickle.load(ifile)
    return store


## function to visualise several ROC curves which were collected with
## clssmetrics_2_store
import matplotlib.pyplot as plt
deflnw=3
defmrksz=3
defsym=["b-", "r--", "g-.", "c:", "m:", "b:", "r:", "g:", "c--", "m--"]

def store2roc(store, lnw=deflnw, mrksz=defmrksz, allsym=defsym,
              legprop={'family': 'DejaVu Sans Mono'},
              xlab="1-specificity (false positive rate)",
              ylab="sensitivity (true positive rate)",
              ttlstr="Transmembran status from physicochemical properties",
              legpatt="{0:6} AUC:{1:0.3f} Acc:{2:2.2f} Shannon:{3:1.3f} McNemar: {4:3}"):
    ## creates annotated ROC curves with annotation provided as part
    ## of the figure legend store is meant to be created by function
    ## clssmetrics_2_store. 
    
    p2s=pval2ind
    nplots=len(store["mthd"])
    mthds=store["mthd"]
    rocxs=store["rocx"]
    rocys=store["rocy"]
    aucs=store["auc"]
    accs=store["acc"]
    shif=store["shif"]
    pvals=store["pval"]
    nplots=len(mthds)
    for cplt in range(nplots):
        legstr=legpatt.format(mthds[cplt], aucs[cplt], accs[cplt], shif[cplt], p2s(pvals[cplt]))
        plt.plot(rocxs[cplt], rocys[cplt], allsym[cplt], linewidth=lnw,  markersize=mrksz,  label=legstr) 
    plt.legend(prop=legprop)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(ttlstr)
    plt.tight_layout()

def gscvscores2modeleval(gscv):
    ## Function gscvscores2modeleval extracts the parameters and the
    ## average scores pover folds from a fitted gscv object. The
    ## function creates a dictionary with all hyperparameter values
    ## which occur as keys in gscv.cv_resuts_.params[0], arranges all
    ## hyperparameter values approproately and pairs the parameter
    ## setting with the average score of the respective position in the array one obtains for all
    ## ccross all gscv.cv_resuts_.split0_test_score to
    ## gscv.cv_resuts_.split<nfold-1>_test_score entries. Note that
    ## the average is already provided as gscv.mean_test_score
    ##
    ## IN
    ##
    ## gscv: a fitted instance of sklearns GSCV meta learner.
    ##
    ## OUT
    ##
    ## resdict: a dictionary with a key score and one key per
    ##          hyperparameter in the paramgrid which got provided
    ##          when instantiating GSCV. All keys reference vectors
    ##          (lists) which contain the gridded parameter values and
    ##          the resulting score from averaging accross all folds
    ##          properly paired.

    ## get all parameters which occurd in the paramdict:
    reskeys=["meanscore", "sumscore"]+list(gscv.cv_results_["params"][0].keys())
    resdict={}
    for key in reskeys:
        resdict[key]=[]
    scorearray=[]
    fldctr=0
    scorekey="split{0}_test_score".format(fldctr)
    while scorekey in gscv.cv_results_.keys():
        scorearray.append(gscv.cv_results_[scorekey].tolist())
        fldctr+=1
        scorekey="split{0}_test_score".format(fldctr)
    sumscore=np.sum(scorearray, axis=0)
    for entry in gscv.cv_results_["params"]:
        for key in entry.keys():
            resdict[key].append(entry[key])
    resdict["meanscore"]=gscv.cv_results_["mean_test_score"]
    resdict["sumscore"]=sumscore.tolist()
    return resdict
