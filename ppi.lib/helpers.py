## Defines helpers for the machine learning course. The origin of
## some of the algorithms dates back to MatLab versions which were
## developed more than 20 years ago.
##
## (C) P. Sykacek 2003-2025 <peter@sykacek.net>

## test functions for the last python programming exercise
## (quicksort)
import sys
# this is a pointer to the module object instance itself.
this = sys.modules[__name__]
this.basedir=None
import random as rnd
import operator as op
def setbase(basedir="../"):
    this.basedir=basedir
    
def getwords(fname="course_data/sowpods.txt"):
    ## sowpods.txt is by Peter Norvig and provided at
    ## http://norvig.com/ngrams/sowpods.txt for download.
    with open(this.basedir+fname) as ifl:
        allwords=[instr.strip() for instr in ifl]
    return allwords

def selwords(allwords, selct, dupct):
    ## randomly selects selct words from allwords and
    ## duplicates randomly selected dupct words.
    ## the final operation reshuffles after selection and
    ## duplication and returns the list of words.
    selwrds=rnd.choices(allwords, k=selct)
    ## duplicate the first dupct. This is unbiased as selwrds is a
    ## random selection of allwords.
    selwrds=selwrds+selwrds[0:dupct]
    ## return after reshuffling
    rnd.shuffle(selwrds)
    return selwrds

def isordered(elemlist, ordop=op.le):
    ## check whether elemlist is ordered which means that
    ## ordop(elemlist[i], elemlist[i+j]) is True for all j > 0
    ## ordop defaults to op.le which tests for increasing order of elements.
    return all(ordop(elemlist[idx], elemlist[idx+1]) for idx in range(0,len(elemlist)-2)) 

## preprocessing functions for the UCI repository datasets which we
## use for the extended ML practical exercises.
## We start with a kNN based imputation of missing values.
## The nature of the algorithm is mainly suitable for numerical data.
## Preprocessing will thus remove non numerical features with missing
## values. A key difficulty with processing UCI data is the fact that
## there are occasional misspelled entities (e.g. more than one way of
## indicating missing information) in at least some of the datasets.


from sklearn.preprocessing import OneHotEncoder as OHE
import pandas as pd
import numpy as np
import ucimlrepo as uci


## CONV: a data type for assessing convegence independent of the scale
## of a numerical sequence The motivation behind this implementation
## is that a fixed numerical delta value like for example 10**-10 or
## any other value can not decide for appropriate convergence as
## numerical accuracy depends on the mantissa and not on the
## exponend. We hence use here a datatype which analyses the behavious
## of the digits in the mantissa and declare a numerical sequence as
## converged, if the mtol most significant digits do not change within
## nintol successive values.
##
## Python
##
## v0.1 - 2022
##
## v0.2 - 2025 added a protection against very small values which
##             would otherwise overload the log10 calculations in the
##             mapping to identical mantissa values.
##
## (C) P. Sykacek 2003-2025 (based on a previous implementation in MatLab)
class CONV:
    def __init__(self, domax=True, mtol=13, nintol=5, valtyp=np.float64, verbose=False):
        ## convergence assessment via least significant digit evaluation.
        ##
        ## domax: flag which specifies whether we maximise (defaults to True)
        ##
        ## mtol: least significant digit position which we monitor for convergence.
        ##
        ## nintol: number of calls of isconv(cscore) we need to observe without 
        ##         change at position mtol in the mantissa of the float before we
        ##         assign "convergence".
        ##
        ## valtyp: value type used in convergence assesment. Defaults to float64 (double precision) 
        dtinfo=np.finfo(valtyp)
        self.mtol=mtol
        self.nintol=nintol
        self.domax=domax
        #if domax:
        #    self.oldscore=dtinfo.min
        #else:
        #    self.oldscore=dtinfo.max
        self.oldscore=None
        self.resolution=dtinfo.resolution
        self.tiny=dtinfo.tiny
        self.cintol=0
        self.verbose=verbose
    def reset(self):
        ## reset operation of convergence assessment
        ## should be called, if the data type which uses
        ## class CONV uses an operation which leads to
        ## a non monotonic behaviour of the optimisation metric.
        self.oldscore=None
    def isconverged(self, cscore):
        ## we assess whether cscore allows to assess convergence.
        ## this happens if the difference between cscore and its
        ## previous value (which we record as self.oldscore) do not
        ## differ in the first mtol digits of the mantissa (after the
        ## comma).  Working on a specific number of mantissa digits as
        ## convergence criterion has the advantage that the assessment
        ## is independent of the scale of the value differences.
        if self.oldscore is not None:
            oldscore=np.maximum(np.absolute(self.oldscore), self.tiny)
            safcscore=cscore
            ## take positive values and do not allow smaller values than tiny.
            cscore=np.maximum(np.absolute(cscore), self.tiny)
            ecs=np.fix(np.log10(cscore))
            eos=np.fix(np.log10(oldscore))
            esd=ecs-eos
            expsc=(ecs+eos)/2
            ## the next two instructions remove the shared exponend (if identical!!)
            ## This will for identical exponends provide the mantissa.
            nsc=cscore/(10.0**expsc)
            osc=oldscore/(10.0**expsc)
            ## if exponends are the same diffsc has exponend 0.
            diffsc=nsc-osc
            if not np.isfinite(diffsc) or self.verbose:
                print("diffsc {0} oldscr {1} newscr {2} expsc {3} nsc {4} osc {5}".format(diffsc, oldscore, cscore, expsc, nsc, osc))
            ## The following boolean expression is True if the difference
            ## between both scores is smaller or equal to the number of tolerance
            ## digits of the mantissa (mtol) AND both exponends are identical
            convonce=np.absolute(diffsc) <= 10.0**-self.mtol 
            if convonce: ## we count the number of iterations where we assess convergence in mantissa digits.
                self.cintol=self.cintol+1
            else:
                self.cintol=0
            ## we finally check whether we have a problem concerning
            ## optimisation if we expect monotonic behaviour.
            if self.domax:
                haveproblem=(self.oldscore-safcscore)>0 and not convonce
            else:
                haveproblem=(self.oldscore-safcscore)<0 and not convonce
            if haveproblem:
                print("diffsc {0} cmpval {1} score difference {2}".format(diffsc, 10.0**-self.mtol, self.oldscore-safcscore)) 
            converged=self.cintol >= self.nintol
            if converged:
                self.cintol=0
            self.oldscore=safcscore
        else:
            self.oldscore=cscore
            converged=False
            haveproblem=False
        return (converged, haveproblem)

def updateXmiss(Xcmplt, Xmiss, missflg, k, dltdiv):
    ## updates all entries in Xmiss which are missing. The code
    ## assumes that Xmiss is fully initialised. We can thus use all
    ## values in Xmiss for the distance calculations. All individual
    ## Xmiss rows (samples) are subsequently approximated as weighted
    ## average of the k closest samples in Xcmplt. The weights are
    ## proportional to the inverse distances where inversion is
    ## safeguarded for overflow. After calculating the approximate
    ## values we update Xmiss at all missing value positions. The
    ## latter are indicated by missflg. Function updateXmiss performs
    ## one such update and is called repeatedly from kNNimpute to
    ## perform a kNN based imputation of missing values.
    ##
    ## IN
    ##
    ## Xcmplt: [ncmpl x nftr] array with complete data.
    ##
    ## Xmiss: [nmiss x nftr] array with data that contained missing
    ##        values.  ALl missing values in Xmiss are assumed to be
    ##        initialised.  Function kNNimpute sets them to the
    ##        average values from the samples which do not contain
    ##        missing values.
    ##        
    ## missflg: [nmiss x nftr] boolean array indicating which values in Xmiss are missing (True==missing value).
    ##
    ## k: number of neighbors
    ##
    ## dltdiv: divisor for the average distances between the
    ##    imputation candidate and the complete neighbours. The divided
    ##    average distance is added to each individual distance before
    ##    calculating the imputation weights. The latter are
    ##    proportional to the inverse distance between neighbours and
    ##    the imputation candidate.
    ##
    ## OUT
    ##
    ## Xmiss: [nmiss x nftr] Data with missing values updated
    ##
    ## (C) P. Sykacek 2025

    ## prepare calculationg distances by reshaping Xcmplt and Xmiss:

    nftrs=Xcmplt.shape[1]
    ncmplt=Xcmplt.shape[0]
    nmiss=Xmiss.shape[0]
    ## Xmiss gets converted such that the matrix with missing samples
    ## gets replicated in the third dimension as often as we have
    ## complete samples.
    Xm2=Xmiss.copy()
    Xm2.shape=(Xm2.shape[0], Xm2.shape[1], 1)
    Xm2=Xm2[:,:,[0]*ncmplt]
    ## complete gets reshaped such that the samples are in the third dimension
    Xc2=Xcmplt.T.reshape((1,nftrs,ncmplt))
    ## and replicated in dimension 1 such that we have as many replications of every
    ## complete sample as we have missing ones.
    Xc2=Xc2[[0]*nmiss,:,:]
    ## subtracting Xc2 from Xm2 gives us matrices which subract every
    ## complete sample from every missing sample.  Xdiff[n,:,:] holds
    ## in the third dimension vectors which are the differences
    ## between the n-th missing sample and all complete samples.
    ## Xdiff[n,:,m] is hence a vector containing the difference
    ## between all feature values of the n-th missing and the m-th
    ## complete sample.
    Xdiff=Xm2-Xc2
    ## alldist contains in row n the Euclidean distances beween the
    ## n-th missing sample and all complete samples.
    alldist=np.sum(Xdiff**2, axis=1)**0.5
    ## calculate a smoothness contribution which has a dual purpose in
    ## avoiding numerical problems when calculating weights.
    mndistfrac=np.mean(alldist, axis=1)/dltdiv
    mndistfrac.shape=(mndistfrac.shape[0], 1)
    weights=(alldist+mndistfrac[:, [0]*ncmplt])**(-1)
    sumweights=np.sum(weights, axis=1)
    sumweights.shape=(sumweights.shape[0], 1)
    weights=weights/sumweights[:, [0]*ncmplt]
    ## weights is a [nmiss x ncomplt] matrix of weights.  the imputed
    ## values are thus given by the matrix product weights.dot(Xcmplt)
    ## or weights @ Xcmplt. (@ is the numpy operator for matrix
    ## products)
    ## Before we calculate the product, we set all weights to zero which are
    ## larget than the k-th closest weight.
    srtw=np.sort(weights, axis=1)
    wghtthrs=srtw[:,k]
    wghtthrs.shape=(wghtthrs.shape[0], 1)
    ## set the weights which are smaller than the k-th largest weight to zero
    ## to update the missing values from the k closest points in the complete data.
    weights[weights<wghtthrs[:, [0]*ncmplt]]=0.0
    Xupd=weights@Xcmplt
    ## we update only the missing values
    #print(Xupd.shape)
    #print(Xmiss.shape)
    #print(missflg.shape)
    Xmiss[missflg]=Xupd[missflg]
    return Xmiss

def kNNimpute(X, dtctmiss=lambda X:np.isnan(X), imputerows=True, k=10,
              maxit=50, dltdiv=10.0**10, conv=CONV(domax=False, mtol=10)):
    ## kNNimpute X operates on a numpy array X and imputes rows or
    ## columns. The function can be parameterized to arbitrary missing
    ## value symbols. It allows imputing rows or columns and to
    ## control smoothness and convergence.  Imputations with kNN were
    ## to my best knowledge introduced by O. Troanskaya in a
    ## Bioinformatice publication.  This implementation imputes
    ## repreatedly until the sum of quared differences between all
    ## imputation units (rows or columns) of two successive
    ## imputations are declared "converged" by conv. The
    ## implementation differs from an earlier implementation in FSPMA
    ## by using imputed values during distance calculations. This
    ## modification allows for a simultaneous calculation of all
    ## distances by matrix operations thus avouding one loop and hence
    ## considerably speeding up calculation. The new calculation has
    ## the added advantage of improved compatibility with GPUs
    ## allthough it is at present CPU based.
    ##
    ## IN:
    ##
    ## X: A numpy array to be imputed
    ##
    ## dtctmiss: a function which operates on X and retuirns an
    ##    equally shaped numpy array of booleans where True indicates
    ##    the missing entries of X which we wish to impute.
    ##
    ## imputerows: A flag which controls the row or column wise
    ##    operation. Deaults to True, hence we assume that rows are
    ##    samples.
    ##
    ## k: number of nearest neighbours to be considered in the
    ##    calculation of the imputed values.
    ##
    ## maxit: maximal number of iterations for imputing.
    ##
    ## dltdiv: divisor for the average distances between the
    ##    imputation candidate and the k neighbours.  the divided
    ##    average distance is added to each individual distance before
    ##    calculating the imputation weights. The latter are
    ##    proportional to the inverse distance between neighbours and
    ##    the imputation candidate.
    ##    
    ## conv: an instance of Class CONV to assess convergence of the
    ##    imputation procedure. Convergence assessments are based on
    ##    the sum of distances beween the imputed values. The distance
    ##    calculation considers only imputation candidates and the
    ##    feature subspace which needs imputation.
    ##    
    ## OUT
    ##
    ## Ximputed: A numpy array which contains the values of X except
    ##    for all missing data which get set to best guesses based on
    ##    the implemented imputation scheme.
    ##
    ## (C) P. Sykacek 2005-2025 (based on R code in FSPMA from 2005)

    ## in case imputation should be column wise, we transpose
    if not imputerows:
        X=X.T
    ## we can now do a row wise imputation
    impcands=dtctmiss(X)
    ##print(impcands.shape)
    ##sumimp=np.sum(impcands, axis=0)
    ##print(sumimp.shape)
    ##sumimp=np.sum(impcands, axis=1)
    ##print(sumimp.shape)
    impcols=np.arange(0, impcands.shape[1])[np.sum(impcands, axis=0)>0]
    improws=np.arange(0, impcands.shape[0])[np.sum(impcands, axis=1)>0]
    cmpltrws=np.arange(0, impcands.shape[0])[np.sum(impcands, axis=1)==0]
    Xcmplt=X[cmpltrws,:]
    Ximpute=X[improws,:]
    print("samples: {0} complete: {1} missing: {2}".format(X.shape[0], len(cmpltrws), len(improws)))
    ## obtain boolean indices for all values in Ximpute which are missing.
    missflg=dtctmiss(Ximpute)
    ## and initialise Ximpute by the respective averages in the complete data
    avobs=np.mean(Xcmplt, axis=0)
    avobs.shape=(1, avobs.shape[0])
    avobs=avobs[[0]*Ximpute.shape[0],:]
    Ximpute[missflg]=avobs[missflg]
    converged=False
    icntr=0
    while not converged and maxit>icntr:
        icntr=icntr+1
        ## update Ximpute by Xcmplt according to a kNN based approach
        Ximpute_new=updateXmiss(Xcmplt, Ximpute, missflg, k, dltdiv)
        ## calculate sum of square distances beween Ximpute and Ximpute_new:
        Xdiff=Ximpute-Ximpute_new
        dist=np.sum(Xdiff[missflg])
        (converged, haveproblem)=conv.isconverged(dist)
        Ximpute=Ximpute_new
    ## write the imputed values back:
    X[impcands]=Ximpute[missflg]
    ## we have finally consider transposing if we were operating on columns
    if not imputerows:
        X=X.T
    return X

def delmissrows(X, y=None, dtctmiss=lambda X:np.isnan(X)):
    ## deletw rows from X and y if X contains missing values.
    ##
    ## IN
    ##
    ## X: [N x M] array of values
    ##
    ## y: [N x] vector of targets (None if not present)
    ##
    ## dtctmiss: function for detecting missing values.
    ##     defaults to numpy.isnan(X)
    ##
    ## OUT
    ##
    ## X:[N_1 <N x M]: Copy of X with rows which contain missing
    ##     values removed.
    ##
    ## y:[N_1 x ]: vector of targets with entries that correspond to
    ##     X's missing value rows removed (or None)
    ##
    ## (C) P. Sykacek 2025
    missvals=dtctmiss(X)
    cmpltrows=np.sum(missvals, axis=1)==0
    X=X[cmpltrows,:]
    if y is not None:
        y=y[cmpltrows]
        return (X,y)
    else:
        return X

def delmisscols(X, y=None, dtctmiss=lambda X:np.isnan(X)):
    ## deletw rows from X and y if X contains missing values.
    ##
    ## IN
    ##
    ## X: [N x M] array of values
    ##
    ## y: [N x] vector of targets (None if not present)
    ##
    ## dtctmiss: function for detecting missing values.
    ##     defaults to numpy.isnan(X)
    ##
    ## OUT
    ##
    ## X:[N_1 <N x M]: Copy of X with rows which contain missing
    ##     values removed.
    ##
    ## y:[N_1 x ]: vector of targets with entries that correspond to
    ##     X's missing value rows removed (or None)
    ##
    ## (C) P. Sykacek 2025
    missvals=dtctmiss(X)
    cmpltcols=np.sum(missvals, axis=0)==0
    X=X[:, cmpltcols]
    if y is not None:
        return (X,y)
    else:
        return X
    
def treatmiss(X, y=None, trtmiss=lambda X, y: (kNNimpute(X), y)):
    ## function treatmiss treats missing values which might be
    ## contained in the input data. Options for treasting missing
    ## values are provided by the function parameter trtmiss, which
    ## defaults to kNNimpute(X) with default parameters. Other options
    ## are the above defined functions delmissrows which also removes
    ## entries from y and delmisscols which like kNNimnpute operates
    ## only on X.
    ##
    ## IN
    ##
    ## X: [N xM] dim array with input values that might contain
    ##      missing values.
    ##
    ## y: [N x]: vector with corresponding target values. Defaults to
    ##      None in case there are none, or the missing value
    ##      treatment does not affect y.
    ##
    ## trtmiss: function which processes the missing values. Defaults to
    ##      kNNimpute with defautl values.
    ##
    ## (C) P. Sykacek 2025

    X,y=trtmiss(X,y)
    if y is not None:
        return (X,y)
    else:
        return X
    
def map2double(inval):
    ## casts inval to a double float. in case this does not work due
    ## to a mismatched format, the function returns np.nan
    ##
    ## IN
    ##
    ## inval: input value which should be mapped to a doule float.
    ##
    ## OUT
    ##
    ## outval: either the input sequence interpreted as double float
    ##     or np.nan.
    try:
        return float(inval)
    except:
        return np.nan

import matplotlib.pyplot as plt
def vismiss(X, dtctmiss=lambda X:np.isnan(X)):
    ## visualizes missing values

    miss=np.flip(dtctmiss(X), axis=0)
    nrows, ncols=miss.shape
    allx=np.arange(ncols)
    ally=np.flip(np.arange(nrows))
    X, Y=np.meshgrid(allx, ally)
    plt.scatter(X.ravel(), Y.ravel(), c=miss.ravel(), marker='.')
    plt.show()
    
def condense(dframe, isok=lambda val: np.isfinite(val), minavailfrac=0.5):
    ## condenses the dataframe by filtering out columns which have
    ## less than minavailfrac samples pass the isok mapping.
    ##
    ## IN
    ##
    ## dframe: a dataframe
    ##
    ## isok: a function which assesses the status of values (returning
    ##      True for values passing the test). defaults to
    ##      numpy.finite() hence only finite values are fine.
    ##
    ## minavailfrac: minimum fraction of entries in a dataframe column
    ##      which have to pass the test. The function retains colu,mns
    ##      only if they pass the test. Default value is 0.5 (50%).
    ##
    ## OUT
    ##
    ## dframe: a filtered version of the dataframe.
    for colnam in list(dframe):
        fracavail=1.0*np.sum(isok(dframe[colnam]))/dframe.shape[0]
        if fracavail < minavailfrac:
            dframe.drop(colnam, axis=1, inplace=True)
    return dframe
    
def uciinptcdr(repoinfo, recode=True, delmisscat=True,  minavailfrac=0.5, trtmisscont=lambda X, y: (kNNimpute(X), y), dtctmiss=lambda X:np.isnan(X), verbose=False):
    ## function uciinptcdr recodes input values from the uci
    ## repository.  The function accepts the return object of the
    ## function ucimlrepo.fetch_ucirepo and processes the input data
    ## to allow for analysis with NN type numerical ML methods.  The
    ## default operation (recode=True) is to recode binary inputs to
    ## -0.5, 0.5 and map categorical inputs to a 1 hot coding which
    ## depends on importin the sklearn encoder as OHE. Integer and
    ## continuous variables are left as is. In case we set
    ## recode=False, the function operates in "del" mode where all non
    ## integer and continuous variables are removed. To avoid
    ## inconsistencies with the meta information the function extracts
    ## and returns only the data component.
    ##
    ## Some extra code is required to process data to react to
    ## wrong vlaue categories (binary is sometimes categorical).
    ##
    ## IN
    ##
    ## repoinfo: an uci ml repo object as returned from function
    ##     ucimlrepo.fetch_ucirepo()
    ##
    ## recode: A flag which controls whether non numerical data are
    ##     recoded (recode=True) or removed (recode=False).
    ##
    ## delmisscat: A flag which controls whether categorical and
    ##     binary featurs with missing values get removed
    ##     (delmisscat=True). The default value is True and there is at
    ##     present no sensible other approach available to deal with
    ##     missing values in non numerical features.
    ##
    ## minavailfrac: minimum fraction of available samples in a feature to retain it.
    ##     Defaults to 0.5 (50%). Features with fewer samples get discarded.
    ##
    ## trtmisscont: a function which treats missing values in
    ##     numerical inputs. The default initialisation calls
    ##     kNNimpute with its default initialisation but that can be
    ##     changed. The default assumes that np.nan represents missing
    ##     values.
    ##
    ## dtctmiss: function to detect missing values in features.
    ##
    ## OUT
    ##
    ## ucidata: a processed ucimlrepo.dotdict object which mirrors the
    ##    structure of repoinfo.data
    ##
    ## (C) P. Sykacek 2025 <peter@sykacek.net>

    ## extract the dataframe which describes the input variables. 
    vardef=repoinfo.variables
    ftrs=repoinfo.data.features
    ## mapped features
    trgftr={}
    ## store for 1-hot coded inputs (only required for treating missing values in numerical data in case recode==False)
    catcoded=[]
    ## we iterate over all feature definitions in vardef
    for idx, row in vardef.iterrows():
        if row['role']=="Feature":
            ## we process the data
            if row['type']=='Continuous':
                ## we copy the numerical values
                tmp=ftrs[row['name']]
                if len(tmp.shape)>1:
                    for col in range(tmp.shape[1]):
                        trgftr["{0}_{1}".format(row['name'], col)]=tmp.iloc[:,col]
                else:
                    trgftr[row['name']]=ftrs[row['name']]
            elif row['type']=='Integer':
                ## we copy the numerical values
                tmp=ftrs[row['name']]
                if len(tmp.shape)>1:
                    for col in range(tmp.shape[1]):
                         trgftr["{0}_{1}".format(row['name'], col)]=tmp.iloc[:,col]
                else:
                    trgftr[row['name']]=ftrs[row['name']]
                ## trgftr[row['name']]=ftrs[row['name']]
            elif (row['type']=='Categorical' or row['type']=='Binary') and not(delmisscat and row['missing_values']=="yes"):
                ## extract and reformat data
                idata=ftrs[row['name']].to_numpy().ravel()
                idata.shape=(idata.shape[0],1)
                ## transform to 1 hot coding
                ohe=OHE()
                cdt=np.asarray(ohe.fit_transform(idata).todense())
                if len(cdt.shape)>1:
                    if cdt.shape[1]==2:
                        ## one hot coding of binary categories leats to 2 columne which are redundant.
                        ## take the first column to temove redundancy in 1-hot endoding of binary and subtract 0.5
                        vals=cdt[:,0]-0.5
                        ##print("bin vals shape:{0}".format(vals.shape))
                        if recode:
                            ## store the recoded data
                            trgftr[row['name']]=vals.squeeze().tolist()
                        else:
                            ## add it to catcoded to aid imputation of numerical inputs.
                            catcoded.append(vals.squeeze().tolist())
                    else:
                        ## the annotation is wrong and we have indeed a Categorical input:
                        for col in range(cdt.shape[1]):
                            tmp=cdt[:,col]
                            tmp.shape=(len(tmp),)
                            if recode:
                                ## store the recoded data
                                trgftr["{0}_{1}".format(row['name'], col)]=tmp.tolist()
                            else:
                                ## add it to catcoded to aid imputation of numerical inputs.
                                catcoded.append(tmp.tolist())
                else:
                    ## we get the one hot coded data in one column (which may not happen at all...)
                    vals=cdt-0.5
                    vals.shape=(len(vals),)
                    if recode:
                        ## store the recoded data
                        trgftr[row['name']]=vals.tolist()
                    else:
                        ## add it to catcoded to aid imputation of numerical inputs.
                        catcoded.append(vals.squeeze().tolist())
                if not recode:
                    ## no recoding. We store the binary data directly in trgftr.
                    trgftr[row['name']]=idata.tolist()
            elif False and row['type']=='Categorical' and not(delmisscat and row['missing_values']=="yes"):
                ## This is now obsolete code but left here in case something changes.
                ## 
                ## extract and reformat data
                idata=ftrs[row['name']].to_numpy().ravel()
                idata.shape=(idata.shape[0],1)
                ## and 1-hot code them
                ohe=OHE()
                cdt=np.asarray(ohe.fit_transform(idata).todense())
                ## we add the 1-hot coded features to the output or to catcoded
                for col in range(cdt.shape[1]):
                    tmp=cdt[:,col]
                    tmp.shape=(len(tmp),)
                    if recode:
                        trgftr["{0}_{1}".format(row['name'], col)]=tmp.tolist()
                    else:
                        ## no recoding - store the data in catcoded
                        catcoded.append(tmp.tolist())
                if not recode:
                    ## no recoding -> We store the original categorical feature
                    trgftr[row['name']]=idata.tolist()
    ## we are now done processing the input features and continue with
    ## imputing the numerical data.
    Xd=pd.DataFrame(trgftr)
    try:
        Xd=Xd.map(map2double)
    except:
        Xd=Xd.applymap(map2double)
    Xd=condense(Xd, minavailfrac=minavailfrac)
    X=Xd.to_numpy()
    if verbose:
        vismiss(X)
    if dtctmiss(X).any():
        print("Treating missing values...")
        ## we hace missing values and depending on recode, there are two modes of operation
        if recode:
            ## X contains also the recoded categorical and binary features and we just have to call kNNimpute:
            X, yd =trtmisscont(X, None)
        else:
            ## We hacve to concatenate X by catcoded and use the augmented infromation to impute.
            ncols=X.shape[1]
            Xtmp, yd=trtmisscont(np.concatenate(X, np.array(catcoded), axis=1), None)
            ## after treating missing values we remove the coded data again
            X=Xtmp[:,0:ncols]
        ## with missing values gone, we replace the entries in trgftr with
        ## the columns in X.
        trgftr={}
        for colid, key in enumerate(list(Xd)):
            trgftr[key]=X[:, colid].tolist()
    
    ## we are now ready to create the ucimlrepo.dotdict object which
    ## mirrors the structure of repoinfo.data and is returned 
    ucidata={"features":pd.DataFrame(trgftr), "targets":repoinfo.data.targets}
    ## which we return as ucimlrepo.dotdict object
    return uci.dotdict(ucidata)


getfoldmax=lambda y:min(np.sum(y.reshape(y.shape[0], 1)[:,[0]*len(set(y.tolist()))]==np.array(list(set(y.tolist()))).reshape(1, len(set(y.tolist())))[[0]*y.shape[0],:], axis=0))

def relabel(ordtarg, Ppri, P_on_inc=True):
    ## relabel accepts as ordtarg any order inducing target
    ## information such that a discretization of ordtarg leads to
    ## ordered class labels. An order relation among the ordtarg
    ## values is a prerequisite for converting the targets to ordered
    ## labels.
    ##
    ## IN
    ##
    ## ordtarg: a [nsample x ] vector of target values which exhibit an
    ##    order relation.
    ##
    ## Ppri: a [nclass x] vector with Prior probabilities of the
    ##    ordered class labels. The size of the vector determines the
    ##    number of target labels and the Probabilities determine the
    ##    fractions of orgtarg samples which we assign with the
    ##    different class labels.
    ##
    ## P_on_inc: boolean flag with "True" indicating that the order
    ##    among the indices in Ppri relate to increasing values of
    ##    ordtarg. The latter implies that when n2>n1 Ppri[n1] denotes
    ##    the prior probability of a subset of samples which come with
    ##    smaller ordtarg values than the sample set which has the
    ##    prior probability P[n2].
    ##    
    ## OUT
    ##
    ## yt: a [nsample x] vector with 0.. nclass-1 labels which reflect
    ##    the order among the 1-of-nclass class labels.
    ##
    ## (C) P. Sykacek 2025.
 
    ## if P_on_inc is False we revert the order od Ppri
    if not P_on_inc:
        Ppri=np.flip(Ppri)
    ## to determine the threshods, we first sort ordtarg
    thrsvals=np.sort(ordtarg)
    ## and get the upper thresholds for the bins which correspond to
    ## the different class labels.
    cumP=np.array([Ppri[0]]*len(Ppri))
    for cid in range(1, len(Ppri)):
        cumP[cid:]+=Ppri[cid]
    ##print("cumP: {0}".format(cumP))
    ## create the threshod values in decreasing order
    thrsvals=np.array([max(ordtarg)]+[thrsvals[round(thrsvals.shape[0]*cumP[i])] for i in range(cumP.shape[0]-2, -1, -1)])
    thrsvals.shape=(1, thrsvals.shape[0])
    ##print("thrsvals: {0}".format(thrsvals))
    ordtarg.shape=(ordtarg.shape[0], 1)
    ## the column sum of the boolean assessment results in 1-based
    ## labels which reflect the prior Ppri and the order in ordtrg.
    yt=np.sum(ordtarg[:,[0]*thrsvals.shape[1]]<= thrsvals[[0]*ordtarg.shape[0],:], axis=1)
    ## we return zero based labels
    return yt-1

import pandas as pd
def getuciwisc(fnam="../course_data/uci/wdbc.data"):
    hdrnam=["radius", "texture", "perimeter", "area", "smoothness",  "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"]
    hdrnam=hdrnam+["radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",  "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se"]
    hdrnam=hdrnam+["radius_wst", "texture_wst", "perimeter_wst", "area_wst", "smoothness_wst",  "compactness_wst", "concavity_wst", "concave_points_wst", "symmetry_wst", "fractal_dimension_wst"]
    print(hdrnam)
    data=pd.read_csv(fnam, header=None, names=["ID", "target"]+hdrnam)
    print(list(data))
    y=data[["target"]].to_numpy().ravel()
    X=data[hdrnam].to_numpy()
    return X, y, data
