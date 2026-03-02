## generic functions which are required for different purposes.
import numpy as np

def scaler_safexp(value):
    ## safe exponentiation for a scaler which sets underflow to zero and overflow to max float.
    ## note that the code requires to activate numpy exceptions
    try:
        res=np.exp(value)
    except:
        if value>0:
            res=np.finfo(np.double).max
        else:
            res=0.0
    return res

def safexp(vals):
    ## safe exponentiation which we use to avoid runtime problems.
    ## IN:
    ## vals: a scaler or an array type
    ##
    ## OUT
    ## res: a scaler or array type with same shape as vals and
    ##      all elements exponentiated safely (we set underflow to
    ##      0 and overflow to max float).
    ##
    ## (C) P. Sykacek 2022 <peter@sykacek.net>
    
    try:
        islist=len(vals)>0
        vals=np.array(vals)
        alldims=vals.shape
        nelem=np.prod(alldims)
        res=np.reshape(vals.copy(), (np.prod(alldims),))
        res=np.array([scaler_safexp(xi) for xi in res])
        res=np.reshape(res, alldims)
    except:
        res=scaler_safexp(vals)
    return res
        
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
    # latest version adds overflow protection. 
    # (C) P. Sykacek 2019 - 2022 <peter@sykacke.net>
    np.seterr(all='raise')
    tolist=False
    simpvec=False
    ## make sure that evids is a numpy array
    try:
        dims=evids.shape
    except:
        tolist=True
        evids=np.array(evids)
        dims=evids.shape
    if len(dims)==1:
        simpvec=True
        ncols=dims[0]
        nrows=1
        evids.shape=(nrows, ncols)
    else:
        ## evids2mp does not support more than two dimensions and its
        ## ok if the following expression chockes...
        nrows, ncols=dims
    ## check whether we have any None or non finite values and set
    ## them to the minimum of all existing and finite values.
    idnone=np.isin(evids, None)
    if np.sum(idnone)>0:
        evids[idnone]=np.nan
        evids=evids.astype(np.float64)
    isfin=np.isfinite(evids)
    isnf=np.logical_not(isfin)
    if isnf.any():
        minval=np.min(evids[isfin])
        evids[isnf]=minval
    # initialise probs
    probs=np.zeros_like(evids, dtype=np.double)
    try:
        ## fast column wise code which may raise an exception if we get an over- or underflow in exp
        for ccol in range(ncols):
            probs[:,ccol]=1/np.sum(np.exp(evids-evids[:,[ccol]*ncols]), axis=1)
    except:
        ## slower row wise calculation which we do in case the above calculation fails
        for crow in range(nrows):
            ## one calculation per row
            for ccol in range(ncols):
                ## calculate the probability in the current row as we
                ## did above note however that we use the safe
                ## exponentiation which we define above!
                try:
                    allexp=safexp(evids[crow, :]-evids[crow,ccol])
                except:
                    print("rw:{0} col:{1} all:{2}".format(crow, ccol, evids[crow, :]))
                try:
                    probs[crow, ccol]=1/np.sum(allexp)
                except:
                    probs[crow, ccol]=0
    np.seterr(all='ignore')
    ## renormalise probabilities as they may not add up...
    sumP=np.sum(probs, axis=1)
    idz=sumP==0
    sumP.shape=(nrows, 1)
    probs=probs/sumP[:,[0]*ncols]
    probs[idz,:]=1.0/ncols
    if tolist:
        ## we were provided with a list of evid values and return probs as list
        probs=probs.ravel().tolist()
    elif simpvec:
        ## we were provided with a vector of evid values and return probs as a vector
        probs=probs.ravel()
    return probs

def saf_evids2mp(evids, doprotect=False):
    # evids2mp converts log marginal likelihoods to model
    # probabilities.  The function is in general usefull to convert
    # unnormalised log probabilities to probabilities.
    #
    # IN
    #
    # evids: [nsample x nprobs] array like datastructure with
    #        log evidence compatible infomration.
    #
    # doprotect: Flag which activates overflow protection
    #        defaults to False
    #
    # OUT
    #
    # probs: [nsample x nprobs] array like datastructure with
    #        normalised probabilities.
    #
    # latest version adds overflow protection. 
    # (C) P. Sykacek 2019 - 2022 <peter@sykacke.net>
    
    if type(evids)==type([]):
        # convert to numpy array
        evids=np.array(evids)
    if len(evids.shape)< 2:
        nprob=evids.shape[0]
    else:
        nprob=evids.shape[1]
    ## values for overflow protection
    lgmx=np.log(np.finfo(np.double).max)
    ## trial and error gave the following as smallest value which allows exponentiation.
    lgmn=10**-300 
    # initialise probs
    probs=np.zeros_like(evids, dtype=np.double)
    for pdx in range(nprob):
        if len(evids.shape)< 2:
            if doprotect:
                l2e=np.minimum(np.maximum(evids-evids[pdx], lgmn), lgmx)
            else:
                l2e=evids-evids[pdx]
            probs[pdx]=1/np.sum(np.exp(evids-evids[pdx]))
        else:
            ## we operate on column pdx
            if doprotect:
                l2e=np.minimum(np.maximum(evids-evids[:, [pdx]*nprob], lgmn), lgmx)
            else:
                l2e=evids-evids[:, [pdx]*nprob]
            probs[:, pdx]=1/np.sum(np.exp(l2e), axis=1)
    ## in case wqe have doprotect=True we renormalise to make sure the probabilities add up to 1.
    if doprotect:
        if len(evids.shape)< 2:
            probs=probs/np.sum(probs)
        else:
            sumP=np.sum(probs, 1)
            for pdx in range(nprob):
               probs[:, pdx]=probs[:, pdx]/sumP
    return probs

def logexpsum(evids):
    ## function which calculates the log of the exponentiated sum for
    ## evids.  This function is applied to np.array like one dim and
    ## two dim vectors or matrices.  In case evids is a [nsmpl x
    ## ninstance] matrix the sum is applied row wise and leads to a
    ## one dim vector [nsmpl x] of log sums. The purpose of logexpsum
    ## is to avoid overflows and underflows. We perform to this end a
    ## row wise calculation which subtracts the largest value.
    ## Assuming the entries in a row denoted as xk we express:
    ## log(sum_k exp(xk))=xK+log(sum_k exp(xk-xK)). By chosing
    ## K=argmax_k(xk) (the index of the largest value) we reduce the
    ## risk of overflow. What remainins is a problem with very large
    ## dynamic range in xk. This can lead to underflow and will be
    ## taken care of by using safexp(xk_xK).
    ##
    ## (C) P. Sykacek 2022 <peter@sykacek.net>
    np.seterr(all='ignore')
    try:
        nrows, ncols=evids.shape
    except:
        evids=np.array(evids)
        ## this should choke in case we provide a misspecified data type
        nrows, ncols=evids.shape
    ## determine the largest value in each row
    maxevid=np.max(evids, axis=1)
    maxevid.shape=(nrows,1)
    ## subtract maxevids
    evids=evids-maxevid[:, [0]*ncols]
    ## exponentiate evids - this is done by mapping a safe calculation
    ## over all elements in the matrix
    np.seterr(all='raise')
    evids=safexp(evids)
    maxevid.shape=(nrows,)
    np.seterr(all='ignore')
    return maxevid+np.log(np.sum(evids, axis=1))



def saflogit(allP, tol=0.000001, whichlog=np.log):
    ## numerically stable version of logit. We make sure that the
    ## probabilities are constrained in a range such as to allow for a
    ## numerically stable conversion to logits.
    ##
    ## IN
    ##
    ## allP: a numpy compatible datatype (which can be converted to
    ##        np.array)
    ##
    ## tol: tolerance to 0 which we enforce befpre taking logs.  can
    ##        be set to np.finfo(float).tiny or np.finfo(float).eps,
    ##        defaults however to 0.001.
    ##
    ## whichlog: one of np.log functions (defaults to np.log).
    ##
    ## OUT
    ##
    ## lgitP: logit(allP) logit transformed probabilities.
    ##
    ## (C) P. Sykacek 2020 <peter@sykacek.net>
    
    allP=np.array(allP)
    ## calculate 1-P
    omP=1-allP
    ## and test both for 
    i2sml=allP<tol
    if np.sum(i2sml)>0:
        allP[i2sml]=tol
    i2lg=allP>1
    if np.sum(i2lg)>0:
        allP[i2lg]=1.0
    i2sml=omP<tol
    if np.sum(i2sml)>0:
        omP[i2sml]=tol
    i2lg=omP>1
    if np.sum(i2lg)>0:
        omP[i2lg]=1.0
    ## we can now calculate the logit:
    return np.log(allP)-np.log(omP)

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
            oldscore=np.absolute(self.oldscore)
            safcscore=cscore
            cscore=np.absolute(cscore)
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

def listargsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def pargridparse(pargrid):
    ## pargridparse inspects all dict entries in pargrid and converts singleton entries to lists.
    for key in pargrid.keys():
        try:
            dummy=pargrid[key][0]
        except:
            ## in case of problems with element access we convert the value to a singleton list.
            pargrid[key]=[pargrid[key]]
    return pargrid

def Ztrans(values, logfunc=np.log):
    ## Fishers Z-transform
    values=np.array(values)
    return (0.5*(logfunc(1+values)-logfunc(1-values))).tolist()
