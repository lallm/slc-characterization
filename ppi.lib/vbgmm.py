## sklearn compatible implementation of Bayesian Gaussian mixture models
## with hints on implementation details.
##
## the model is implemented according to the DAG in Richardson and
## Green to reduce the tendency that inference allocates meaningless
## modes (e.g. allmost empty clusters).
##
## We use two important design considerations:
##
## a) diagonal vs. full covariance matrices.
## b) kernel specific or shared covarianve matrices.
##
## (C) P. Sykacek <peter@sykacek.net> 2022
##
## Important hint for efficient implementations.
##
## np.stack can be used to stack a list of matrices or vectors.
## stacking the rate martices of the Wishart or the precision matrices
## of Gaussians as [d x d x nK] dim tensor TST allows us immediately to
## calculate TST * scvec where scvec is a one dim vector [nK x .].
## np.linalg.det works on the other hand alongside axis 0!
## This is also the dimension which we iterate over when looping
## for ct in TST:
##
## Since we want to apply det to or iterate over the correct [d x d]
## rate or precision matrices we need to switch dimensions:
##
## for c_prec in np.moveaxis(TST, 2, 0):
##
## does the trick.
##
## To move between both tensor representations we use function:
##
## swp4loop(tensor_in_[dxdxnK]_order) to a tensor_in_[nKxdxd]_order
##
## and function
##
## swpback(tensor_in_[nKxdxd]_order) to obtain the equivalent tensor_in_[dxdxnK]_order.

## some code documentation
##
##def test(cf=lambda **arg: print(arg), **kwarg):
##    cf(**kwarg)

# ## generating samples from multinomial-1 (categiorical) distribution
# ## and from kernel specific multivariate nomral
# ## densities. Unfortunately there is no simple approach to sample
# ## without a loop.
# import numpy as np
# nsample=100
# nK=4
# mn=np.array([1,1])
# mns=np.stack([mn]*nK, axis=1)*np.arange(1,5)
# mns=mns.T
# covs=np.stack([np.diag([1,1])]*nK, axis=2)
# Pk=np.array([0.2,0.3,0.1,0.4])
# ## alldx represents randomly chosen kernel indices.
# alldx=np.argmax(np.random.multinomial(1, Pk, size=nsample), axis=1)
# ## allx is a matrix of observations (here two dimensional with nsample rows) 
# allx=np.stack([np.random.multivariate_normal(mns[idx], covs[:,:,idx]) for idx in alldx], axis=0)

# ## code fragment to elucidate the meaning of alpha and Beta in scipy.stats.invwishart
# import scipy.stats as sps
# iW=sps.invwishart.rvs
# Beta=np.diag([1,4])*500 ## in the context of the expectation below, Beta is an inverse scale (rate or precision like formulation)
# alpha=1000.0            ## ths is the degrees of freedom
# Ecov=(1/alpha)*Beta     ## The expected precision of such a Wishart prior is alpha*inv(Beta) the covariance (inverse precision) is thus 1/alpha*Beta
# allcovs=iW(alpha, Beta, 1000)  ## if precisions are Wishart distributed, inverse precisions (covariances) follow an inverse Wishart distribution.
# ## since np.mean(allcovs, axis=0) approximates Ecov, we established that scipy inverse Wishart is parameterised by rate matrices (inverse scale).
#import warnings
#warnings.filterwarnings("ignore")

import ray
import numpy as np
import pandas as pd
import scipy.stats as sps
## some functions which are important for the derivation
#from scipy.special import psi       ## digamma (logarithmic derivative of gamma function)
#from scipy.special import gamma     ## gamma
#from scipy.special import gammaln   ## log gamma function
## equivalent ray compatible imports:
import scipy.special
psi=lambda val: scipy.special.psi(val)
gamma=lambda val:scipy.special.gamma(val)
gammaln=lambda val:scipy.special.gammaln(val)
import scipy.linalg as spl
def pinv(X, **pinvpar):
    try:
        Xi=spl.pinv(X, **pinvpar)
    except:
        nonfin=np.logical_not(np.isfinite(X))
        print("non finite values in pinv:{0}".format(np.sum(nonfin)))
        X[np.isnan(X)]=0
        nonfin=np.logical_not(np.isfinite(X))
        isneg=X<0
        ispos=X>0
        X[np.logical_and(nonfin,isneg)]=-10^100
        X[np.logical_and(nonfin,ispos)]=10^100
        Xi=pinv(X, **pinvpar)
    return Xi

import copy
ddup=copy.deepcopy
def lggammad(x, d):
    ## log of multivariate gamma function with dimension d.
    try:
        ## allows calling by vector
        return d*(d-1)/4*np.log(np.pi)+np.array([np.sum(gammaln(xi+0.5*np.arange(-(d-1),1,1))) for xi in x])
    except:
        return d*(d-1)/4*np.log(np.pi)+np.array(np.sum(gammaln(x+0.5*np.arange(-(d-1),1,1) ) ))

def gammad(x,d):
    ## multivariate gamma function with dimension d.
    return np.exp(lggammad(x,d))

def psid(x, d):
    ## multivariate digamma function (logarithmic derivative of multivariate gamma function) with dimension d
    try:
        return np.array([np.sum(psi(xi+0.5*np.arange(-(d-1),1,1))) for xi in x]) ## allows calling by vector
    except:
        return np.sum(psi(x+0.5*np.arange(-(d-1),1,1)))

logdet=lambda arrays:np.linalg.slogdet(arrays)[1] ## we take the log determinant and ignore the sign!! (see np.linalg.slogdet)
## tensor swap functions:
swp4loop=lambda mytensor:np.moveaxis(mytensor, 2, 0)
## move the loop order of the [d x d x nK] tensor to the multiplication compatible order
swpback= lambda mytensor:np.moveaxis(mytensor, 0, 2)


# ## test code of convergence assessment.
# def logistic(x, fact=1.0, const=0.0):
#      return fact/(1+np.exp(-x))+const

# x=np.linspace(0,30,num=5000)
# y=logistic(x, fact=10**50, const=-10)
# convtest=CONV(verbose=True)
# for cy in y:
#     (converged, haveproblem)=convtest.isconverged(cy)
#     if converged:
#         break
#     if haveproblem:
#         print("Problem at {0}".format(cy))
# if converged:
#     print("converged at {0}".format(cy))
# else:
#     print("not converged")
import sklearn.model_selection as skms
import genericfuncs as gfs
import importlib
importlib.reload(gfs)
## we shorten definitions which were moved to genericfuncs
def evids2mp(vals):
    ## print("call gfs.evids2mp")
    return gfs.evids2mp(vals)
    
logexpsum=gfs.logexpsum
CONV=gfs.CONV
listargsort=gfs.listargsort
pargridparse=gfs.pargridparse
from sklearn.cluster import KMeans    
import sklearn.base as sbs
## REMARK: a recent modification in sklearn leads to incompatibilities
## when deriving ray actor classes from sklearn base classes.
## Hence the comment of all sklearn base classes we used to derive from.
##
## Comment by P. Sykacek 01 2024.

class VBGMM:    #(sbs.BaseEstimator, sbs.ClusterMixin, sbs.DensityMixin):
    ## VB GMM
    def __init__(self, nK=1, PIndPriCnt=10, g=0.05, H=2.0, alpha=1.5,
                 xi=None, kappa=1.0, covtyp="full", covmode="individual",
                 mdlinit="kmeans", prtfrac=2.0, kmnsinit=5, maxit=300, mtol=9,
                 nintol=5, verbose=False, maxrestart=5):
        ## nK: number of components
        ##
        ## PIndPriCnt: prior counts in Dirichlet over kernel prior
        ##             probability. This is either a positive real
        ##             number or an arraytype with nK counts.
        ##
        ## g, H: shape and rate type parameters of prior precision (matrix) of a Gamma
        ##       (or Wishart) prior over inverse kernel covariance matrix.
        ##       H depends on covtyp. If "full" H is a symmetric positive matrix.
        ##       if "diag" H is a rate parameter of a Gamma distribution.
        ##
        ## alpha: shape (degrees of freedom) parameter of a Gamma or Wishart density.
        ##
        ## To follow R&G 1997 g, H and alpha should be linked. In case
        ## we use diagonal covarianvce matrices these parameters are
        ## the parameters of Gamma densities. Using L[d,d] to denote
        ## the precision in dimension d we have Beta[d,d]~Gamma(g,
        ## H[d,d]) and L[d,d]~Gamma(alpha, Beta[d,d]). In this case we
        ## can apply the R&G suggestion on the diagonal and use alpha
        ## > 1 > g > 0 and to set H[d,d] to a small multiple of
        ## 1/R[d]^2.  To obtain this parameterisation we use scalars
        ## for alpha, g and H and redefine all H[d,d] as
        ## H*1/R[d]^2. Otherwise if H is a [d x d] matrix we use it
        ## directly as H. If we use full covariance Gaussians the same
        ## applies for H. We have however maintain the constraints of
        ## [d x d] Wishart densities and use degrees of freedom which
        ## are larger than d-1. In situations where H is a scaler, we
        ## assume that the R&G prior should be tailored to a
        ## multivariate setting. We will then add d-1 to g and alpha.
        ## The specification of both degrees of freedom (g and alpha)
        ## can thus always follow the recommendations given in R&G
        ## 1997 for univariate GMMs. Furthermore we adjust the given
        ## g, H and alpha in response to the type of covariance
        ## matrix: covtyp "full" uses the parameters after adjusting
        ## them to the dimension of the input data. For covtyp "diag"
        ## we use moment matched parameters of Gamma densities such
        ## that the expectation of a Wishart with diagonal inverse
        ## scale matrix corresponds to the diagonalised expected
        ## precisions which result from having independent Gamma
        ## densities on the main diagonal.
        ##
        ## xi, kappa: mean and precision matrix of Gaussian prior over
        ##       kernel means.  Both parameters need to be compatible
        ##       with the data (dimension!).  To obtain the Richardson
        ##       & Green JRSSB 1997 setting, leave xi at the default
        ##       value "None" and specify a scalar precision factor as
        ##       kappa. In this case we set xi to the midpoint of the
        ##       range of the data and in correspondence to R&G use a
        ##       diagonal precision matrix kappa*1/R[d]^2 as precision
        ##       matrix. R&G state that "kappa is a small factor" and
        ##       propose 1.0 (page 742), which is our default value.
        ##
        ## covtyp: type of covarinace matrix - "full" or
        ##        "diag". Defaults to "full" where we use Wishart
        ##        diostributions to represent the inverse kernel
        ##        covariance matrices.  In case we use "diag",
        ##        inference works with Gamma distributions on
        ##        individual dimensions of the diagonal kernel
        ##        precision matrix.
        ##
        ## covmode: Defines the graph structure of the DAG. In case we
        ##        use "individual" (the default), all kernel densities
        ##        have their own precision matrices. In case we use
        ##        "shared" all kernel densities have one shared
        ##        precision matrix.
        ##
        ## mdlinit: Defines how the Q distributions will be
        ##        initialised.  The modes are "random" and "kmeans"
        ##        which is the default. Note that even kmeans adds
        ##        some perturbation (determined by prtfrac) to the
        ##        kernel means such that repreated inference will lead
        ##        to different solutions. Mode "random" initialises
        ##        from kmeans as well but replaces the kernel means
        ##        with randomly chosen samples.
        ##        
        ## prtfrac: perturbation fraction parameter which is used for
        ##        randomisation of the kernel means which we obtain
        ##        from kmeans. The parameter is multiplicative to the
        ##        number of kernels and has default value 2.
        ##
        ## See the code of function initQs for details of how mdlinit
        ## and prtfrac influence both initialisation options.
        ##
        ## kmnsinit: ninit for kmeans which is only relevant if we
        ##        chose mdlinit as "kmeans".
        ##
        ## maxit: maximum number of iteration before inference is stopped anyway.
        ##
        ## mtol, nintol: mantissa based convergence assessment. mtol
        ##        specifies the position of the least significant
        ##        digit which we consider during convergence
        ##        assessment while nintol is the number of successive
        ##        iterations where we must not observe variation in
        ##        the negative free energy at any more important digit
        ##        of the mantissa.
        ##
        ## verbose: boolean to control the output generated. If True
        ##        additional sanity checks and outputs are generated.
        ##        Defaults to False.
        ## maxrestart: maximum number of restarts before giving up.
        ##        A restart is initiated if a numerical operation
        ##        leads to overflow. Defaults to 5.

        self.maxrestart=maxrestart
        self.restarts=0
        ## self.convass is used to assess convergence.
        self.convass=CONV(domax=True, mtol=mtol, nintol=nintol, valtyp=np.float64)
        self.maxit=maxit
        ## GMM parameters and priors
        self.nK=nK
        ## we make sure that the prior counts are a
        ## vector with shape (nK,)
        try:
            if len(PIndPriCnt)!=self.nK:
                self.dlt=np.array([PIndPriCnt[0]]*self.nK)
            else:
                self.dlt=np.array(PIndPriCnt)
        except:
            ## PIndPriCnt is a scaler
            self.dlt=np.array([PIndPriCnt]*self.nK)
        if any(self.dlt <= 0):
            raise Exception('Wrong specification of prior counts (PIndPriCnt must contain positive values).')
        ## type of model and prior parameters for the covariance
        ## structure and the mean parameters.
        self.iscovfull=covtyp=="full"
        if not (covtyp == "full" or covtyp == "diag"):
            raise Exception('covtyp has to be "full" or "diag"')
        self.issharedcov=covmode=="shared"
        if not (covmode == "individual" or covmode == "shared"):
            raise Exception('covmode has to be "individual" or "shared"')
        ## set up prior for inverse kernel covariances 
        try:
            d=H.shape[0]
            if np.any(H!=H.T):
                raise Exception("H needs to be symmetric.")
            self.indim=d
            self.H=H
            self.updateH=False
        except:
            self.H=H
            self.updateH=True
            self.indim=None
        ## in dependence of self.updateH and self.iscovfull we set the parameters g and alpha
        if self.updateH: ## or not self.iscovfull:
            ## if we do not know the data dimension or we use diagonal
            ## covariances, where R&G can be applied directly, we use
            ## the provided parameters without modification.
            self.g=g
            self.alpha=alpha
        else:
            ## we know the data dimension and use Wishart
            ## distributions over self.indim dimensional full size
            ## inverse covarinance matrices. The same hyperparameters
            ## are also used in the diagonal case. That is why we comment the above or... 
            self.g=self.g+self.indim-1
            self.alpha=self.alpha+self.indim-1
        ## specify the parameters of the Gaussian prior over all kernel means.
        self.xi=xi
        self.kappa=kappa
        try:
            ## this makes only sense if we have multivariate H and kappa.
            Hshp=H.shape
            kpshp=kappa.shape
            if Hshp!=kpshp:
                raise Exception("H and kappa need identical dimensions.")
            if np.any(kappa!=kappa.T):
                raise Exception("The inverse covariance matrix kappa needs to be symmetric.")
        except:
            pass

        self.mdlinit=mdlinit
        self.prtfrac=prtfrac
        self.kmnsinit=kmnsinit
        
        ## the following goes to the wrapping class
        #self.ninit=ninit
        #self.cpuno=cpuno
        self.verbose=verbose
        dtinfo=np.finfo(np.float64)
        self.nFE=dtinfo.min
        self.isfitted=False
    def mdlstatus(self):
        ## function mdlstatus prints the status of the VBGMM instance on screen or to a file.
        try:
            QLalphshp=self.QLalpha.shape
        except:
            QLalphshp="double"
        try:
            QLbetashp=self.QLbeta.shape
        except:
            QLbetashp="double"
        try:
            QBgshp=self.QBg.shape
        except:
            QBgshp="double"
        try:
            QBHshp=self.QBH.shape
        except:
            QBHshp="double"
        
        loginfo="Nk:{0} dim:{1} Nsmpl: {2} cov:{3}, {4}\n".format(self.nK, self.indim, self.Nsmpl, ["diag","full"][self.iscovfull], ["individual", "shared"][self.issharedcov]) 
        loginfo=loginfo+"nFE:{0} Qmm:{1} QmL:{2}\n".format(self.nFE, self.Qmm.shape, self.QmL.shape)
        loginfo=loginfo+"QLalph:{0} QLBeta:{1} QBg:{2} QBH:{3}\n".format(QLalphshp, QLbetashp, QBgshp, QBHshp)
        return loginfo
    def modelstats(self, X):
        ## Function modelstats provides information about the model.
        ## The purpose of the function is to provide information which
        ## allows judging aspects of the solution which could hint
        ## convergence to a meaningless solution.  The following
        ## characteristics are provided:
        ##
        ## 1) peak density of all kernels.
        ## 
        ## 2) KL between p(Lk) and Q(Lk) that is the distance btween
        ## prior and posterior over all kernel precision matrices.
        ##
        ## 3) The generalized variance of every kernel according to <Lk>^(-1) (Anderson p. 264).
        ##
        ## 4) The rank of all Lk^(-1)
        ##
        ## 5) The condition number of all Lk^(-1)
        ##
        ## 6) KL between p(mk) and Q(mk) that is the distance btween
        ## prior and posterior over all kernel means.
        ##
        ## 7) Number of samples per kernel which by the approximate
        ##    posterior Q(Ik) get assigned to the respective kernel.
        ##
        ## 8) The generalized variance of every kernel according to a
        ##    kernel specific sample covariance matrix.
        ##
        ## 9) Rank of kernel specific sample covariance matrix.
        ##
        ## IN
        ##
        ## X: input data that was used to fit the model.
        ##
        ## OUT
        ##
        ## res: dictionary with keys each representing a list of
        ##      values
        ##
        ## "pkd": [pmx1, ..., pmxK] - maximum density values
        ##
        ## "KLL": [KLL1, ..., KLLK] - KL(p(Lk)||Q(Lk))
        ##
        ## "GVL": [|<L1>^-1|, ..., |<LK>^-1|] - generalised variances
        ##        according to parameters
        ##
        ## "Lr" : [Lr1, ..., LrK] - rank of all <Lk>^-1
        ##
        ## "Lcnd": [Lc1, ..., LcK] - condition number of all <Lk>^-1
        ##
        ## "KLm": [KLm1, ..., KLmK] - KL(p(mk)||Q(mk))
        ##
        ## "nk" : [n1, ..., nK] - number of samples assigned to kernel
        ##        k (kn = argmax_k(P(In=1,..K|xn))) with nk = sum of
        ##        kn=k over all n samples
        ##
        ## "GVSC": [|S1|, ..., |SK|] - general variance of sample cov matrix.
        ##
        ## "SCr": [Sr1, ..., SrK] - rank of sample covariance matrix.
        ##
        ## "SCcnd": [Sc1, ..., ScK] - condition numbers of sample covariance matrix.

        mvn=sps.multivariate_normal
        ## initialise kernel specific stores for debug info
        pkd=[]
        KLL=self.KLL()
        if self.issharedcov:
            KLL=np.array([KLL]*self.nK) ## we have only one value and augment it.
            ## expected covariance mtrices are calculated
            ## as 3d tensor the expression is the same, irrespective
            ## of whether we have full or diagonal covarinace
            ## matrices.
            allELk=np.stack([self.QLalpha*self.QLbeta_i]*self.nK, axis=2)   ## sample and kernel independent
            allCovk=np.stack([(1/self.QLalpha)*self.QLbeta]*self.nK, axis=2)
        else:
            ## we get the tensor by multiplying alpha and Beta_i
            ## (multiplication is properly expanded).  The calculation
            ## is independent of the structure of the kernel
            ## covariance matrices.
            allELk=self.QLalpha*self.QLbeta_i
            allCovk=(1/self.QLalpha)*self.QLbeta
        GVL=[]
        Lr=[]
        Lcnd=[]
        KLm=self.KLm()
        ## get the kernel counts
        krnind=np.zeros_like(self.QIn)
        mxcol=np.argmax(self.QIn, axis=1)
        krnind[list(range(krnind.shape[0])), mxcol]=1
        nk=np.sum(krnind, axis=0)
        GVSC=[]
        SCr=[]
        SCcnd=[]
        for k in range(self.nK):
            ## get maximal log kernel densities:
            pkd.append(mvn.logpdf(x=self.Qmm[k,:], mean=self.Qmm[k,:],  cov=allCovk[:,:,k]))
            ## get the generalised covariance
            ## use log det:
            sgn, val=np.linalg.slogdet(allCovk[:,:,k])
            GVL.append(sgn*val)
            ## matrix rank of kernel covariance matrices
            Lr.append(np.linalg.matrix_rank(allCovk[:,:,k]))
            ## condition number of kernel covariance matrices
            Lcnd.append(np.linalg.cond(allCovk[:,:,k]))
            Xk=X[mxcol==k,:]
            ## sample covariance matrix of samples allocated to kernel k (argmax...)
            if sum(mxcol==k)> 0:
                try:
                    scm=np.cov(Xk, rowvar=False)
                    ## generlaised variance of sample covariance matrix
                    sgn, val=np.linalg.slogdet(scm)
                    GVSC.append(sgn*val)
                    ## matrix rank of sample covariance matrix
                    SCr.append(np.linalg.matrix_rank(scm))
                    ## condition number of sample covariance matrix
                    SCcnd.append(np.linalg.cond(scm))
                except:
                    GVSC.append(-1)
                    SCr.append(-1)
                    SCcnd.append(-1)
            else:
                GVSC.append(-1)
                SCr.append(-1)
                SCcnd.append(-1)
        return {"pkd":pkd, "KLL":KLL, "GVL":GVL, "Lr":Lr, "Lcnd":Lcnd, "Klm":KLm,
                "Klm":KLm, "nk":nk, "GVSC":GVSC, "SCr":SCr, "SCcnd":SCcnd, "nFE":self.nFE}
                
    def finalise_priors(self, X):
        ## Function finalise_priors uses data X to finalise the
        ## prior specification.  This finalisation step is needed
        ## if we specify the prior parameters according to R&G
        ## 1997, who propose adapting the priors over inverse
        ## kernel covariances and the prior over kernel means to
        ## the data.
        ##
        ## get sample size and data dimension.
        self.Nsmpl, self.indim=X.shape
        ## get maximum, minimum and range
        Xmx=np.amax(X, axis=0)
        Xmn=np.amin(X,axis=0)
        ## range
        R=Xmx-Xmn
        ## finalise the prior over kernel means
        if self.xi is None:
            ## we update xi and kappa
            ## self.xi as midpoint
            self.xi= (Xmn+Xmx)/2
            ## and self.kappa as diagonal matrix R&G: "small
            ## multiple of inverse squared range" where the
            ## provided self.kappa value is the small multiple
            self.kappa=np.diag(self.kappa/(R**2))
        ## and now we take care of the prior over inverse kernel covariance matrices:
        if self.updateH:
            ## self.H is a scaler
            self.H=np.diag(self.H/(R**2))
            ## in this case we did not know the dimension of the
            ## data during construction and need to adjust
            ## self.alpha and self.g if we infer a full covariance
            ## structure.
            if self.iscovfull or True:
                self.g=self.g+self.indim-1
                self.alpha=self.alpha+self.indim-1
            ## in case we use Gamma parameterised diagonal covariance
            ## matrices in the kernels, we adjust g, H and alpha to
            ## match the moments:
            if not self.iscovfull:
                self.g=0.5*self.g
                self.alpha=0.5*self.alpha
                self.H=0.5*self.H ## works because H is diagonal!! 
            self.updateH=False
            
        #print("updating for nk:{0}".format(self.nK))
        #print(self.H)

    def Qbetamx(self):
        ## Update of Q(Beta) which requires no external
        ## information. The update depends on the type and the
        ## mode of the covarianve metric and operates on
        ## the parent sufficient statistics (the prior)
        ## self.H, the posterior rate type parameters of p(beta)
        ## self.g, the posterior deg. of freedom parameters of p(beta)
        ## and the expected sufficient statistics of all child nodes which are 
        ## self.QLbeta, the posterior rate type parameters of Q(L) or Q(Lk)
        ## self.QLbeta_i (the inverse of the above)
        ## self.QLalpha, the posterior deg. of freedom parameters of Q(L) or Q(Lk)
        ##
        ## the update creates current values for self.QBg which is
        ## a degrees of freedom like parameter and self.QBH which
        ## is a [d x d] rate type matrix. If we use diagonal
        ## covariance matrices self.QBH is diagonal. If we use
        ## full cov matrices self.QBH is also full. self.QBg is
        ## always a scalar quantity.
        #print(self.QLalpha)
        #print(self.QLbeta_i)
        if self.issharedcov:
            if self.iscovfull:
                ## we have only one covariance matrix and update Q(Beta) from there.
                self.QBg=self.g+self.alpha
                self.QBH=self.H+self.QLalpha*self.QLbeta_i
                ## enforce symmetry
                self.QBH=0.5*(self.QBH+self.QBH.T)
            else:
                ## we have self.indim gamma densities on the
                ## diagonal entries of L
                self.QBg=self.g+self.alpha
                self.QBH=self.H+self.QLalpha*self.QLbeta_i
        else:
            if self.iscovfull:
                ## we have self.nK full covariance matrices and update Q(Beta) from there.
                self.QBg=self.g+self.alpha*self.nK
                self.QBH=self.H+np.sum(self.QLalpha*self.QLbeta_i, axis=2)
                ## enforce symmetry
                self.QBH=0.5*(self.QBH+self.QBH.T)
            else:
                self.QBg=self.g+self.alpha*self.nK
                self.QBH=self.H+np.sum(self.QLalpha*self.QLbeta_i, axis=2)
    def QInmx(self, X):
        ## maximise the neg free energy w.r.t. Q(In)
        ##
        ## This function provides self.QI which is a [no sample x
        ## no kernels] matrix and represents the probabilities of
        ## a multinomial one distribution over the kernel
        ## indicators. The function also calculates self.nFEdata
        ## which summarizes the data contribution to the negative
        ## free energy (the approximate log marginal likelihood).
        ## The expression of self.nFEdata is however only correct
        ## if we evaluate the neg. free energy just after
        ## maximising w.r.t Q(In). This implies that self.QInmx(X)
        ## should be the last update before calling self.getnFE().
        d=self.indim
        #Nsmpl=X.shape[0]
        ## some preparations
        if self.issharedcov:
            if self.iscovfull:                    
                explogdetLk=psid(0.5*self.QLalpha, d)+ d*np.log(2)-logdet(self.QLbeta) ## sample and kernel independent
                ## expand it to identical values (row vector with nK entries)
                explogdetLk=np.ones((1,self.nK))*explogdetLk
                allELk=np.stack([self.QLalpha*self.QLbeta_i]*self.nK, axis=2)   ## sample and kernel independent
                ## allELk is a tensor and we iterate over individual precision matrices
                ## we have again got to swap axis here:
                xT_ELk_x=np.array([np.sum(np.dot(X, ELk)*X, axis=1) for ELk in swp4loop(allELk)]).T    ## sample and kernel dependent
                mT_ELk_x=np.zeros((self.Nsmpl, self.nK))
                ## we calculate the kernel specific projections in a loop, sample and kernel dependent
                for k in range(self.nK):
                    mT_ELk_x[:,k]=np.dot(np.dot(self.Qmm[k], allELk[:,:,k]), X.T)
                ## we want mT_Elk_m as row vector with self.nK dimensions sample independent
                mT_ELk_m=np.array([[np.dot(np.dot(self.Qmm[k,:],allELk[:,:,k]),self.Qmm[k,:])+np.trace(np.dot(allELk[:,:,k],self.QmLi[:,:,k])) for k in range(self.nK)]])
            else:
                explogdetLk=self.indim*psi(self.QLalpha)-np.sum(np.log(np.diag(self.QLbeta)))  ## sample and kernel independent
                ## expand it to identical values (row vector with nK entries)
                explogdetLk=np.ones((1,self.nK))*explogdetLk
                #ELk=self.QLalpha*self.QLbeta_i  ## sample and kernel independent
                allELk=np.stack([self.QLalpha*self.QLbeta_i]*self.nK, axis=2)
                xT_ELk_x=np.array([np.sum(np.dot(X, ELk)*X, axis=1) for ELk in swp4loop(allELk)]).T
                #xT_ELk_x=np.sum(np.dot(X, ELk)*X, axis=1) 
                #xT_ELk_x.shape=(len(xT_ELk_x),1)
                #xT_ELk_x=xT_ELk_x[:,[0]*self.nK] # sample depednent kernel independent
                mT_ELk_x=np.zeros((self.Nsmpl, self.nK))
                ## we calculate the kernel specific projections in a loop
                for k in range(self.nK):
                    #mT_ELk_x[:,k]=np.sum(np.dot(self.Qmm[[k]*self.Nsmpl,:],ELk)*X, axis=1)
                    mT_ELk_x[:,k]=np.sum(np.dot(self.Qmm[[k]*self.Nsmpl,:], allELk[:,:,k])*X, axis=1)
                ## we want mT_ELk_m as row vector with self.nK dimensions sample independent
                #mT_ELk_m=np.array([[np.dot(np.dot(self.Qmm[k,:],ELk),self.Qmm[k,:])+np.sum(np.diag(ELk)*np.diag(self.QmLi[:,:,k])) for k in range(self.nK)]])
                mT_ELk_m=np.array([[np.dot(np.dot(self.Qmm[k,:],allELk[:,:,k]),self.Qmm[k,:])+np.sum(np.diag(allELk[:,:,k])*np.diag(self.QmLi[:,:,k])) for k in range(self.nK)]])
        else:
            if self.iscovfull:
                ## self.QLbeta_i.shape=(d,d,nK) -> for det to work we have to
                ## apply swp4loop to get then nK log determinants
                ## of the nK inverted Beta_k matrices.
                explogdetLk=psid(0.5*self.QLalpha, d)+ d*np.log(2)-logdet(swp4loop(self.QLbeta)) ## sample indepenent kernel dependent
                explogdetLk.shape=(1, self.nK)
                allELk=self.QLalpha*self.QLbeta_i ## this works on the original [d x d x nK] tensor  ## sample indepenent kernel dependent
                ## allELk is a tensor and we iterate over individual precision matrices
                ## we have again got to swap axis here:
                xT_ELk_x=np.array([np.sum(np.dot(X, ELk)*X, axis=1) for ELk in swp4loop(allELk)]).T    ## sample and kernel dependent
                mT_ELk_x=np.zeros((self.Nsmpl, self.nK))
                ## we calculate the kernel specific projections in a loop, sample and kernel dependent
                for k in range(self.nK):
                    mT_ELk_x[:,k]=np.dot(np.dot(self.Qmm[k], allELk[:,:,k]), X.T)
                ## we want mT_Elk_m as row vector with self.nK dimensions sample independent
                mT_ELk_m=np.array([[np.dot(np.dot(self.Qmm[k,:],allELk[:,:,k]),self.Qmm[k,:])+np.trace(np.dot(allELk[:,:,k],self.QmLi[:,:,k])) for k in range(self.nK)]])
            else:
                ## need to apply swp4loop to get the axis right for the loop:
                explogdetLk=self.indim*psi(self.QLalpha)-np.array([np.sum(np.log(np.diag(Beta_k))) for Beta_k in swp4loop(self.QLbeta)]) ## sample indepenent kernel dependent
                explogdetLk.shape=(1, self.nK)
                allELk=self.QLalpha*self.QLbeta_i
                ## allELk is a tensor and we iterate over individual precision matrices
                ## and again swp4loop:
                xT_ELk_x=np.array([np.sum(np.dot(X, ELk)*X, axis=1) for ELk in swp4loop(allELk)]).T
                mT_ELk_x=np.zeros((self.Nsmpl, self.nK))
                ## we calculate the kernel specific projections in a loop
                for k in range(self.nK):
                    mT_ELk_x[:,k]=np.sum(np.dot(self.Qmm[[k]*self.Nsmpl,:], allELk[:,:,k])*X, axis=1)
                ## we want mT_Elk_m as row vector with self.nK dimensions sample independent
                mT_ELk_m=np.array([[np.dot(np.dot(self.Qmm[k,:],allELk[:,:,k]),self.Qmm[k,:])+np.sum(np.diag(allELk[:,:,k])*np.diag(self.QmLi[:,:,k])) for k in range(self.nK)]])
        ## we have now all model specific parts defined and can express Fkn:
        ## start with <log(Pk)> and some constant values 
        ElP=psi(self.QP)-psi(np.sum(self.QP))-0.5*self.indim*np.log(2*np.pi)
        ElP.shape=(1, self.nK)
        ## all row vectors get expanded to be identical for all
        ## samples ([[0]*self.Nsmpl,:] as row indices). Sample and
        ## kernel specific contorubtions get summed.
        Fkn=ElP[[0]*self.Nsmpl,:]+0.5*explogdetLk[[0]*self.Nsmpl,:]
        #print(Fkn.shape)
        #print(xT_ELk_x.shape)
        #print(mT_ELk_x.shape)
        #print(mT_ELk_m[[0]*self.Nsmpl,:].shape)
        Fkn=Fkn-0.5*(xT_ELk_x - 2 * mT_ELk_x + mT_ELk_m[[0]*self.Nsmpl,:])
        ## Fkn can now be converted to self.QIn:
        try:
            self.QIn=evids2mp(Fkn)
            ## print(np.sum(self.QIn))
        except:
            self.Fkn=Fkn
            raise Exception("self.QIn=evids2mp(Fkn) in VBGMM.QInmx(X) failed due to overflow!")
        if self.verbose:
            nFnans=np.sum(np.isnan(Fkn))
            nQnans=np.sum(np.isnan(self.QIn))
            if nFnans > 0 or nQnans > 0:
                print("Nan in Fkn:{0} Nan in Q(In):{1}".format(nFnans, nQnans))
        ## final calculation: negative free enery gontribution if
        ## we evaluate F(Q) immediately after calling QInmx. To be
        ## of dual use for evaluating F(Q) and for calculating log
        ## densities we store all sample contributions as
        ## vector. (self.mxFQIn are the sample specific
        ## approximate log data densities and as a sum the
        ## contribution to the neg. free energy).
        ##self.Fkn=Fkn
        ##self.mxFQIn=np.log(np.sum(np.exp(Fkn), axis=1))
        ## safe calculation
        self.mxFQIn=logexpsum(Fkn)
    def QPmx(self):
        ## maximise F(Q) w.r.t Q(P) (the kernel prior)
        self.QP=self.dlt+np.sum(self.QIn, axis=0)  ## count parameters of a Dirichlet density

    def Qmmx(self, X):
        ## maximise F(Q) w.r.t all Q(m).
        ## express <Lk> (4 different modes):
        prictr=np.stack([self.kappa]*self.nK, axis=2)
        if self.issharedcov:
            ELk=self.QLalpha*self.QLbeta_i   ## kernel independent
            allELk=np.stack([ELk]*self.nK, axis=2)  ## match with kernel specific situation
        else:
            allELk=self.QLalpha*self.QLbeta_i
        ## get the sum of kernel soecuific weights from Q(In)
        kw=np.sum(self.QIn, axis=0)
        ## we can now use the tensor trick of numpy (see comment
        ## at the beginning of this file) and express the
        ## precision matrices of the kernel means in one
        ## expression:
        self.QmL=prictr+allELk*kw
        ## QmLi exists already and is only recalculated.
        for k in range(self.nK):
            ## enforce symmetry
            ##docalc=True
            ##while docalc:
            try:
                self.QmL[:,:,k]=0.5*(self.QmL[:,:,k]+self.QmL[:,:,k].T)
                self.QmLi[:,:,k]=pinv(self.QmL[:,:,k])
            except:
                print("Qmmx problem:{0}".format(np.sum(np.logical_not(np.isfinite(self.QmL[:,:,k])))))
            ## enforce symmetry
            self.QmLi[:,:,k]=0.5*(self.QmLi[:,:,k]+self.QmLi[:,:,k].T)
            ## expression for modes of Q(m) which we do inside the inversion loop
            #print(self.QmLi[:,:,k])
            #print("k:{0} P:{1}".format(k, self.QIn[:,k]))
            #print(np.sum(self.QIn[:,[k]*self.indim]*X, axis=0))
            #print(allELk[:,:,k])
            #print(np.dot(self.kappa, self.xi))
            self.Qmm[k,:]=np.dot(self.QmLi[:,:,k], np.dot(self.kappa, self.xi)+np.dot(np.sum(self.QIn[:,[k]*self.indim]*X, axis=0),allELk[:,:,k]))

    def QLmx(self, X):
        ## maximise F(Q) w.r.t all Q(L).
        ## calculate expectation of beta:
        EBt=self.QBg*pinv(self.QBH)
        ## assure symmetry
        EBt=0.5*(EBt+EBt.T)
        ## prepare the expectation in the exponend which
        ## contributes to QLbeta. We calculate this here as we need
        ## the result in all different cases
        E_exp=np.zeros((self.indim, self.indim, self.nK))
        E_exp_diag=np.zeros((self.indim, self.indim, self.nK))
        for k in range(self.nK):
            ## we need a loop over kernels
            cX=X*self.QIn[:,[k]*self.indim]
            smPx=np.sum(cX, axis=0)  ## sum over all n of P(In=k)*xn
            cX=cX.T
            smPxxT=np.dot(cX, X)    
            ## reduce numerical problems and force symmetry
            smPxxT=0.5*(smPxxT+smPxxT.T)
            #ctr1=cX.dot(X)
            #print(ctr1.shape)
            #ctr2=np.dot(self.Qmm[k,:].T, self.Qmm[k,:])+ self.QmLi[:,:,k]
            #print(ctr2.shape)
            #ctr3=np.outer(self.Qmm[k,:], np.sum(cX.T, axis=0))
            #print(ctr3.shape)
            mk_out_smPx=np.outer(self.Qmm[k,:], smPx)  # outer produict of mk and the weighted sum of all observations
            ## should be symmetric
            E_exp[:,:,k]=smPxxT+(np.outer(self.Qmm[k,:], self.Qmm[k,:])+self.QmLi[:,:,k])*np.sum(self.QIn[:,k])-mk_out_smPx-mk_out_smPx.T
            ## store the diagonal of E_exp in E_exp_diag.
            E_exp_diag[:,:,k]=np.diag(np.diag(E_exp[:,:,k]))
        ## the calculation of the Q distribution parameters depends on the mode
        if self.issharedcov:
            if self.iscovfull:
                ## shared full which is case 2:
                self.QLalpha=self.alpha+self.Nsmpl
                self.QLbeta=EBt+np.sum(E_exp, axis=2) ## shared kernel covariance require summation of the expected value over kernels and thus dimension 3 (and index 2)
                self.QLbeta_i=pinv(self.QLbeta)
            else:
                ## shared diagonal (Gamm densities on individual precisions) which is case 4: 
                self.QLalpha=self.alpha+0.5*self.Nsmpl
                self.QLbeta=np.diag(np.diag(EBt+0.5*np.sum(E_exp, axis=2)))
                self.QLbeta_i=np.diag(1/np.diag(self.QLbeta))
        else:
            if self.iscovfull:
                ## kernel specific full precision matrices which is case 1:
                self.QLalpha=self.alpha+np.sum(self.QIn, axis=0)  # a (nK,) vector
                self.QLbeta=np.stack([EBt]*self.nK, axis=2)+E_exp # a (indim, indim, nK) tensor
                for k in range(self.nK):
                    ## assure symmetry of self.QLbeta[:,:,k]
                    try:
                        self.QLbeta[:,:,k]=0.5*(self.QLbeta[:,:,k]+self.QLbeta[:,:,k].T)
                        self.QLbeta_i[:,:,k]=pinv(self.QLbeta[:,:,k])
                    except:
                        print("QLmx problem:{0}".format(np.sum(np.logical_not(np.isfinite(self.QLbeta[:,:,k])))))
                    ## assure symmetry of inverse
                    self.QLbeta_i[:,:,k]=0.5*(self.QLbeta_i[:,:,k]+self.QLbeta_i[:,:,k].T)
            else:
                ## kernel specific diagonal precision matrices (Gamma on dimension and kernel specific precisions) which is case 3:
                self.QLalpha=self.alpha+0.5*np.sum(self.QIn, axis=0)  # a (nK,) vector
                self.QLbeta=np.stack([np.diag(np.diag(EBt))]*self.nK, axis=2)+0.5*E_exp_diag # a (indim, indim, nK) tensor (nK diagonal indim x indim matrices).
                for k in range(self.nK):
                    self.QLbeta_i[:,:,k]=np.diag(1/np.diag(self.QLbeta[:,:,k]))
    def KLL(self):
        ## calculate for all different cases the KL divergence between
        ## p(Lk) and Q(Lk) where Lk denotes the kernel specific
        ## inverse covariance matrices.
        ##
        ## OUT
        ##
        ## nKLL: if self.issharedcov is True one Kulback Leibler divergence for the shared covariance matrix
        ##       if self.issharedcov is False: a vector with self.nK Kulback Leibler divergences (one per kernel).
                ## some shorthand notations
        ## functions
        log=np.log
        det=np.linalg.det
        ## pinv=np.linalg.pinv
        tr=np.trace
        diag=np.diag
        ## constants and expressions
        d=self.indim
        lg2=np.log(2)
        g=self.g
        H=self.H
        Hi=pinv(H)
        g_h=self.QBg
        H_h=self.QBH
        nprob=np.sum(np.logical_not(np.isfinite(H_h)))
        if nprob!=0:
            print("problems with H_h:{0}".format(nprob))
        H_hi=pinv(H_h)
        alpha=self.alpha
        ## posterior Q(Lambda) (inv cov of Gaussian kernel densities)
        alph_ht=self.QLalpha ## either self.nK scalars or one sclalar (depending on self.isshared)
        bt_ht=self.QLbeta    ## either self.nK inv. covs or one inv. cov (depending on self.isshared) QLbeta is diagonal for iscovfull==False.
        bt_hti=self.QLbeta_i ## inverse of previous
        if self.iscovfull:
            ## KL divrgence for full covariance matrices
            if self.issharedcov:
                ## code for -KL(p(L)||Q(L) (shared full cov)
                nKLL=0.5*alpha*(psid(0.5*g_h,d)+d*np.log(2)-logdet(H_h))-0.5*alph_ht*logdet(bt_ht)
                nKLL=nKLL+0.5*(alpha-alph_ht)*(psid(0.5*alph_ht, d)-logdet(bt_ht))+lggammad(0.5*alph_ht, d)-lggammad(0.5*alpha, d)
                nKLL=nKLL+0.5*alph_ht*d-0.5*alph_ht*np.trace(np.dot(g_h*H_hi, bt_hti))
            else:
                ## code for -KL(p(L_k)||Q(L_k) (individual full cov and diagonal cov)
                ## implementation uses martfun(swo4loop(tensor)) which reorders the tensors such that
                ## matrix functions like determinant or trace operate on the correct [d x d] dim matrices
                ELk=alph_ht*bt_hti
                nKLLk=0.5*alpha*(psid(0.5*g_h,d)+d*np.log(2)-logdet(H_h))-0.5*alph_ht*logdet(swp4loop(bt_ht))
                nKLLk=nKLLk+0.5*(alpha-alph_ht)*(psid(0.5*alph_ht, d)-logdet(swp4loop(bt_ht)))
                nKLLk=nKLLk+lggammad(0.5*alph_ht, d)-lggammad(0.5*alpha, d)
                nKLL=nKLLk+0.5*np.array([np.trace(np.dot(bt_ht[:,:,k]-g_h*H_hi, ELk[:,:,k])) for k in range(self.nK)])
        else:
            ## KL divergence for diagonal covariance matrices.
            if self.issharedcov:
                ## code for -KL(p(L)||Q(L) (shared diagonal cov)
                betahat=np.diag(bt_ht)
                allELk=alph_ht/betahat
                Ebt=g_h/np.diag(H_h)
                nKLLd=alpha*(psi(g_h)-np.log(np.diag(H_h)))-alph_ht*np.log(betahat)+gammaln(alph_ht)-gammaln(alpha)
                nKLL=nKLLd+(alpha-alph_ht)*(psi(alph_ht)-np.log(betahat))+(betahat-Ebt)*allELk
                nKLL=np.sum(nKLL)
            else:
                ## code for -KL(p(L)||Q(L) (kernel specific diagonal cov)
                ## to simplify calculation we convert the tensor bt_ht to a [nK x d] matrix
                betahat=np.stack([np.diag(bt_ht[:,:,k]) for k in range(self.nK)], axis=0)
                aht_col=alph_ht.copy()
                aht_col.shape=(self.nK, 1)
                allELk=aht_col[:, [0]*self.indim]/betahat    ## the expectations of kernel precision matrix k is in row k.   
                Ebt=g_h/np.diag(H_h)
                Ebt.shape=(1,self.indim)                ## the expectation of the dioagonal beta as row vector
                ## we can now use sums over columns of betahat to sum over the different dimensions
                ElgdtH=psi(g_h)-np.log(diag(H_h))
                ElgdtH.shape=(1,self.indim)
                ElgdtH=ElgdtH[[0]*self.nK,:]
                ## calculation are done in parallel for all diumensions and kernels.
                ## The contributions are either [nk x dim] or scalar.
                nKLLk=alpha*ElgdtH-aht_col[:,[0]*self.indim]*np.log(betahat)+gammaln(aht_col[:,[0]*self.indim])-gammaln(alpha)
                #print(nKLLk.shape)
                nKLLk=nKLLk+(alpha-aht_col[:,[0]*self.indim])*(psi(aht_col[:,[0]*self.indim])-np.log(betahat))
                #print(nKLLk.shape)
                nKLL=nKLLk+(betahat-Ebt[[0]*self.nK,:])*allELk
                nKLL=np.sum(nKLL, axis=1)
        return nKLL
    def KLm(self):
        ## calculate for every kernel the KL divergence between p(mk) and Q(mk):
        nKLpmQmk=0.5*(logdet(self.kappa)-np.array([logdet(swp4loop(self.QmL))]))
        mm=self.Qmm
        xi=self.xi
        kappa=self.kappa
        L_hti=self.QmLi
        nKLpmQmk=nKLpmQmk+0.5*np.array([self.indim-np.dot(np.dot((self.xi-mm[k,:]),kappa),(self.xi-mm[k,:]))-np.trace(np.dot(kappa,L_hti[:,:,k])) for k in range(self.nK)])
        return nKLpmQmk
    def FQ(self):
        ## evaluation of the negative free energy. The expressions
        ## assume that we evaluate the negative free energy just
        ## after maximising w.r.t Q(In) in this case we have just
        ## the sum of the expected negative KL divergences between
        ## all prior distributions and the corresponding Q
        ## distributions plus self.mxFQIn which represents the
        ## expected data contribution after the maximisation step
        ## w.r.t. Q(In)

        ## the expressions of some negative KL divergences depend
        ## on the chosen covariance structure of the Gaussians

        ## some shorthand notations
        ## functions
        log=np.log
        det=np.linalg.det
        ## pinv=np.linalg.pinv
        tr=np.trace
        diag=np.diag
        ## constants and expressions
        d=self.indim
        lg2=np.log(2)
        g=self.g
        H=self.H
        Hi=pinv(H)
        g_h=self.QBg
        H_h=self.QBH
        nprob=np.sum(np.logical_not(np.isfinite(H_h)))
        if nprob!=0:
            print("problems with H_h:{0}".format(nprob))
        H_hi=pinv(H_h)
        alpha=self.alpha
        ## posterior Q(Lambda) (inv cov of Gaussian kernel densities)
        alph_ht=self.QLalpha ## either self.nK scalars or one sclalar (depending on self.isshared)
        bt_ht=self.QLbeta    ## either self.nK inv. covs or one inv. cov (depending on self.isshared) QLbeta is diagonal for iscovfull==False.
        bt_hti=self.QLbeta_i ## inverse of previous
        if self.iscovfull:
            ## We have a Wishart density over the beta matrix
            nKL_beta=0.5*(g*logdet(H)-g_h*logdet(H_h))+lggammad(0.5*g_h,d)-lggammad(0.5*g,d)+0.5*(g-g_h)*(psid(0.5*g_h,d)-logdet(H_h))+0.5*d*g_h-0.5*g_h*tr(np.dot(H,H_hi))
        else:
            ## We have a product of Gamma densities over the
            ## diagonal entries of the beta matrix (off diagonal
            ## elements are all zero)
            nKL_beta=np.sum(g*log(diag(H))-g_h*log(diag(H_h))+gammaln(g_h)-gammaln(g)+(g-g_h)*(psi(g_h)-log(diag(H_h)))+(diag(H_h)-diag(H))*(g_h/diag(H_h)))
        nKLL=self.KLL()
        if not self.issharedcov:
            ## self.KLL returns a vector with self.nK entries which we have to add together.
            nKLL=np.sum(nKLL)
        ## neg. KL btw. p(P) and Q(P)
        ## self.dlt is the prior counts in the Dirichlet p(P)
        ## self.QP is the posterior counts in the Dirichlet Q(P).  
        dlt=np.sum(self.dlt)
        dlt_hat=np.sum(self.QP)
        nKLP=gammaln(dlt)-gammaln(dlt_hat)+np.sum(gammaln(self.QP)-gammaln(self.dlt)+(self.dlt-self.QP)*(psi(self.QP)-psi(dlt_hat)))
        ## sum over all kernel specific neg KLs
        ## self.xi and self.kappa specify the prior p(m_k)
        ## self.QmL[:,:,k] and self.QmLi[:,:,k] are the inverse cov and the cov of Q(m_k) and self.Qmm[k,:] its mode.
        ## self.indim is the dimension of the input data.
        #ctr1=np.array([logdet(swp4loop(self.QmL))])
        #print(ctr1.shape)
        nKLpmQmk=self.KLm()
        nKLpmQm=np.sum(nKLpmQmk)
        ## self.mxFQIn is the likelihood contribution if the previous maximization before evaluating F(Q) was w.r.t all Q(In) (the unknown kernel indicators).
        #print(nKL_beta)
        #print(nKLP)
        #print(nKLpmQm)
        #print(nKLL)
        #print(np.sum(np.isnan(self.mxFQIn)))
        self.nKL_beta=nKL_beta
        self.nKLP=nKLP
        self.nKLpmQm=nKLpmQm
        self.nKLL=nKLL
        self.mxFQ=np.sum(self.mxFQIn)
        if self.verbose:
            try:
                nKL_beta_dim=nKL_beta.shape
            except:
                nKL_beta_dim="scl"
            try:
                nKLP_dim=nKLP.shape
            except:
                nKLP_dim="scl"
            try:
                nKL_pmQm_dim=nKL_pmQm.shape
            except:
                nKL_pmQm_dim="scl"
            try:
                nKLL_dim=nKLL.shape
            except:
                nKLL_dim="scl"
            try:
                mxFQ_dim=self.mxFQ.shape
            except:
                mxFQ_dim="scl"
            print("KLBeta:{0} nKLP:{1} nKLpmQm:{2} nKLL:{3} mxFQ:{4}".format(nKL_beta_dim, nKLP_dim, nKL_pmQm_dim, nKLL_dim, mxFQ_dim))
        FQ=nKL_beta+nKLP+nKLpmQm+nKLL+self.mxFQ
        return FQ
    def initQs(self, X):
        ## we do now initialise all Q distributions in dependence of
        ## the specification during model construction
        ## if self.mdlinit=="kmeans":
        ## we run sklearn k-means to obtain an initialisation for the clusters.
        kmnsmdl=KMeans(n_clusters=self.nK, n_init=self.kmnsinit)
        clustno=kmnsmdl.fit_predict(X)
        ## counts for initialisation of Dirichlet over P
        allcnts=[]
        ## modes of all Qmn distributions
        allmnmds=[]
        ## precisions of all Qmn distributions - These are
        ## always full symmetric matrices, even if we use
        ## diagonal covariance matrices as kernel cov.
        allmnprc=[]
        kmptrbcov=np.diag([1/(self.prtfrac*self.nK)]*self.indim)
        kmptrbmn=np.zeros((self.indim,))
        ## rate parameter of Wishart or Gamma if we opt for a
        ## diagonal covarinace structure.
        allprcprc=[]
        allprcdegf=[]
        ## next step: convert the clusters to mean values and
        ## covariance matrices.
        aug_mn=np.mean(X, axis=0)
        aug_sdev=np.std(X, axis=0)
        for ccval in list(set(clustno.tolist())):
            ## get samples of cluster ccval:
            Xc=X[clustno==ccval,:]
            if Xc.shape[0] < Xc.shape[1]:
                ## we augment Xc with rows which are N(mn; sdev) 
                Xcn=np.zeros((Xc.shape[1],  Xc.shape[1]))
                Xcn[:Xc.shape[0],:]=Xc
                simdat=[(np.random.normal(size=(Xc.shape[1],))*aug_sdev+aug_mn).tolist() for _ in range(Xc.shape[0], Xc.shape[1])]
                Xcn[Xc.shape[0]:,:]=np.array(simdat)
                Xc=Xcn
            ## get cluster mean
            cm=np.mean(Xc, axis=0)
            ## and store a perturbed version
            allmnmds.append(cm+np.random.multivariate_normal(kmptrbmn, kmptrbcov, 1))
            ## and the inverse covariance matrix
            ccov=np.cov(Xc.T)
            cprc=pinv(ccov)
            ## degrees of freedom
            cdf=Xc.shape[0]+self.dlt[0] ## we add the prior counts
            cprc=cdf*cprc
            #cprc.shape=cprc.shape+(1,) ## we convert cprc to a 3d tensor to allow concatenation
            allcnts.append(cdf)
            allmnprc.append(cprc)
            ## we may now construct the initialisation of this kernels distributions
            allprcdegf.append(cdf) ## this is a duplicate of allcnts and just done to improve the documentation
            if self.iscovfull:
                ## we use a rate parameter in the Q distribution which matches this kernels
                ## precision matrix. cprc=cdf*Betainv -> cprc^(-1)=cdf^(-1)*Beta -> Beta = cdf*cprc^(-1)=cdf*ccov
                ## to store Betak in a 3dim tensor we have to modify its shape
                cbta=cdf*ccov
                allprcprc.append(cbta)
            else:
                ## we use a diagonal covarianve matrix, which we code as self.indim Gamma densities.
                ## cprc=cdf/beta -> beta=cdf*ccov -> we just extract the diagonal matrix of ccov!
                cbta=cdf*np.diag(np.diag(ccov))
                allprcprc.append(cbta)
        ## We can now convert the statistics which we collect
        ## in the previous loop from the kmeans run to
        ## parameters of Q distributions. In addition to the
        ## above distinction between full and diagonal
        ## covariance matrices we have to distinguish between
        ## kernel specific and shared covariance
        ## matrices. We start by aggregating the statistics
        #for mtr in allprcprc:
        #    print(mtr.shape)
        #for mtr in allmnprc:
        #    print(mtr.shape)
        allprcprc=np.stack(allprcprc, axis=2) ## tensor with Betak in 3rd dimension (axis=2)
        if self.issharedcov:
            ## we augment the covariance statistics.
            self.QLbeta=np.sum(allprcprc, axis=len(allprcprc.shape)-1) ## we sum over kernels which works for both types
            ##print("shared cov - dim allprcprc:{0} dim QLbeta: {1}".format(allprcprc.shape, self.QLbeta.shape))
            if not self.iscovfull:
                self.QLbeta=np.diag(np.diag(self.QLbeta))
                self.QLbeta_i=np.diag(1/np.diag(self.QLbeta))
            else:
                self.QLbeta_i=pinv(self.QLbeta) ## invert upon calculation as we need the inverse at several places.
            self.QLalpha=np.sum(allprcdegf)
        else:
            ## we have nK Q distributions which characterize the inverse kernel covariance matrices.
            self.QLbeta=allprcprc     ## QLbeta is thus a 3 dim tensor
            self.QLbeta_i=self.QLbeta.copy()
            if self.iscovfull:
                for k in range(self.QLbeta_i.shape[2]):
                    self.QLbeta_i[:,:,k]=pinv(self.QLbeta_i[:,:,k])
            else: ## diagonal inversion is simpler...
                for k in range(self.QLbeta_i.shape[2]):
                    self.QLbeta_i[:,:,k]=np.diag(1/np.diag(self.QLbeta_i[:,:,k]))
            self.QLalpha=np.array(allprcdegf)   ## and QLalpha a 1 dim vector which works well for diagonal settings. 
            ### which we convert to a 3 dim tensor for full covariance matrices (efficient calculations of expectations)
            #self.QLalpha.shape=(1,1,len(self.QLalpha))   ## and QLalpha a 3 dim tensor
        ## initialise the nK Q distributions over the kernel means.
        #for k,mk in enumerate(allmnmds):
            # print("allmnmds[{0}].shape:{1}".format(k, mk.shape))
            ## allmnmds[k].shape=(len(mk,)
        self.Qmm=np.concatenate(allmnmds, axis=0) # we use a matrix with d columns and nK rows to store the modes of the Gaussians over the nK kernel means
        self.QmL=np.stack(allmnprc, axis=2) # we use a 3 dim tensor with nK entries in dim 2 to store the precision matrices of the Gaussians over the nK kernel means
        self.QmLi=self.QmL.copy()           # we also invert
        for k in range(self.nK):
            self.QmLi[:,:,k]=pinv(self.QmLi[:,:,k])
        if self.mdlinit=="random": ## initialise from random overrides the calculated kernel means. 
            ## random initilaisation sets the modes of all Qmk to randomly chosen data points.
            Xrws=np.random.permutation(X.shape[0])[0:self.nK]
            self.Qmm=X[Xrws,:]
        ## Intialisation of Q(m) and Q(L) is followed by updating Q(Beta)
        self.Qbetamx()
        if self.verbose:
            statstrg=self.mdlstatus()
            print(statstrg)
        ## we initialise Q(P) from allcnts
        self.QP=self.nK*self.dlt+allcnts
        ## and are finally in the position to update Q(In) for
        ## all samples.
        self.QInmx(X)
        ## After that we use a round of updates:
        ## Q(P), Q(m), Q(L) Q(Beta) and a final update of
        ## Q(In) which is followed by calculating the initial
        ## value of the neg. free energy.
        self.QPmx()    ## maximise w.r.t. Q(P)
        self.Qmmx(X)   ## maximise w.r.t  Q(m) (Gaussians over kernel mean values)
        self.QLmx(X)   ## maximise w.r.t  Q(L) (Wishart or Gamma over kernel precision matrices)
        self.Qbetamx() ## maximise w.r.t  Q(Beta) (Wishart or Gamma over kernel precision matrices)
        self.QInmx(X)  ## maximise w.r.t  Q(In)
        self.nFE=self.FQ()  ## initialise the negative free energy
        dummy=self.convass.isconverged(self.nFE)  ## initialise convergence assessment
        ## With this step the initialisation completed we can move to inference.
    def restart(self, X):
        ## restart VB inference from beginning
        self.convass.reset()   ## reset convergence assessment (otherwise we get a monotonicity warning)
        ## self.initQs(X)      ## reinitialisation is not required as part of fit() (see below)
        self.fit(X)            ## fit the model.
    def fit(self, X):
        ## fit implements VB inference as loop over updates.  This
        ## is done in a loop and preserves the update order of the
        ## initialisation.
        ##
        ## we jave to run the entire set of functions to make fit()
        ## compatible with sklearn (and parallel inference)
        ## print("############################  fit")
        self.convass.reset()
        self.finalise_priors(X)
        self.initQs(X)
        ## now we can start the actual fit().
        cit=0
        doit=True
        self.isfitted=True
        nFEs=[self.nFE]
        self.nKL_betas=[self.nKL_beta]
        self.nKLPs=[self.nKLP]
        self.nKLpmQms=[self.nKLpmQm]
        self.nKLLs=[self.nKLL]
        self.mxFQs=[self.mxFQ]
        
        try:
            while doit:
                ## print("it:{0} nFE:{1}".format(cit, self.nFE))
                if self.verbose and (cit % 100)==0:
                    print("it:{0} nFE:{1}".format(cit, self.nFE))
                self.QPmx()    ## maximise w.r.t. Q(P)
                self.Qmmx(X)   ## maximise w.r.t  Q(m) (Gaussians over kernel mean values)
                self.QLmx(X)   ## maximise w.r.t  Q(L) (Wishart or Gamma over kernel precision matrices)
                self.Qbetamx() ## maximise w.r.t  Q(Beta) (Wishart or Gamma over kernel precision matrices)
                self.QInmx(X)  ## maximise w.r.t  Q(In)
                ##print(self.expparams())
                nFE=self.FQ()  ## calculate the resulting free energy
                ## after all updates we check for convergence
                (converged, haveproblem)=self.convass.isconverged(nFE)
                if haveproblem:
                    print("Warning at iteration:{0} oFE:{1} nFE:{2}".format(cit, self.nFE, nFE))
                self.nFE=nFE
                nFEs.append(nFE)
                self.nKL_betas.append(self.nKL_beta)
                self.nKLPs.append(self.nKLP)
                self.nKLpmQms.append(self.nKLpmQm)
                self.nKLLs.append(self.nKLL)
                self.mxFQs.append(self.mxFQ)
                cit=cit+1
                doit=(cit < self.maxit) and not converged
        except:
            print("################################################### refitting ##############################################")
            if self.restarts < self.maxrestart:
                self.restarts=self.restarts+1
                self.restart(X)
        self.allnFEs=np.array(nFEs)
        self.nKL_betas=np.array(self.nKL_betas)
        self.nKLPs=np.array(self.nKLPs)
        self.nKLpmQms=np.array(self.nKLpmQms)
        self.nKLLs=np.array(self.nKLLs)
        self.mxFQs=np.array(self.mxFQs)
        ## print("#######################################  sum(self.QIn):{0}".format(np.sum(self.QIn)))
    def expparams(self):
        ## returns the expected GMM parameters as string for diagnosis options
        Pk=self.QP/sum(self.QP)
        mk=self.Qmm
        if self.issharedcov:
            ELki=(1/self.QLalpha)*self.QLbeta
            allELki=np.stack([ELki]*self.nK, axis=2)
        else:
            allELki=(1/self.QLalpha)*self.QLbeta
        EBeta=self.QBg*pinv(self.QBH)
        retstr="Pk:{0} \n\nBeta:{1}\n\n".format(Pk, EBeta)
        for k in range(self.nK):
            retstr=retstr+"m:{0} \n\ncov:{1}\n\n".format(mk[k], allELki[:,:,k])
        return retstr
    def remapkrn(self, truekernels):
        ## remapkrn remaps the internal kernel order as to maximize
        ## agreement with truekernels. The function hence resolves the
        ## unknown identification of the predicted kernels and adjusts
        ## the internal representation correspongingly. This affects
        ## self.QIn self.QP, self.Qmm, self.QmL, self.QmLi and
        ## self.QLalpha and self.QLbeta, if we use kernel specific
        ## covariance matrices.
        ##
        ## In addition to indentifying the internal representation with
        ## the provided kernel indices, the function returns
        ## the predicted kernels after remapping. 
        
        ## The provided truekernels should contain the kernel labels
        ## which are mapped to a zero based coding.
        truekernels=np.array(truekernels)
        mink=np.min(truekernels)
        truekernels=truekernels-mink
        nsmp=truekernels.shape[0]
        if nsmp != self.Nsmpl:
            print("Mismatch in size true kernels do not match data.")
            return -1
        ## identify kernels
        kmap={}
        for k in list(set(truekernels.tolist())):
            ## find the maximum column index in all rows of the latent indicator probability which match kernel k
            argmxk=np.argmax(self.QIn[truekernels==k,:], axis=1)
            ## the most frequent kernel number which we find in argmxk is the kernel which corresponds to the external indicator k
            m2k=0
            mxkcnt=-1
            for tryk in range(self.QIn.shape[1]):
                nk=np.sum(argmxk==tryk)
                if nk>mxkcnt:
                    mxkcnt=nk
                    m2k=tryk
            kmap[k]=m2k
        ## The mapping between external and internal kernel indices
        ## can now be used to remap the internal variables.
        kout=list(kmap.keys())
        kin=list(kmap.values())
        self.QIn[:,kout]=self.QIn[:,kin]
        self.QP[kout]=self.QP[kin]
        self.Qmm[kout]=self.Qmm[kin]
        self.QmL[:,:,kout]=self.QmL[:,:,kin]
        self.QmLi[:,:,kout]=self.QmLi[:,:,kin]
        if not self.issharedcov:
            self.QLalpha[kout]=self.QLalpha[kin]
            self.QLbeta[:,:,kout]=self.QLbeta[:,:,kin]
        ## the final task is converting the remapped self.QIn to most
        ## probable kernel predictions which match the order of the
        ## provided truekernels and adjust the indices to the provided
        ## labels.
        return np.argmax(self.QIn, axis=1)+mink
    def score(self, X=None, screval=lambda obj, X=None, **screvalkwarg: obj.nFE, **scorekwarg):
        ## evaluate the score of the GMM using data X and self.
        ## This is a generic implementation which allows
        ## overloading the scoring function which is used for
        ## evaluation.  The default operation is without data (X)
        ## and with a lambda function which returns the GMMs
        ## negative free energy.  note that improvement
        ## transtlates to larger score values. For model
        ## comparisons score should be maximised and that should
        ## be considered when providing custom scoring functions
        ## and when using score(). To allow overriging the default
        ## score calculation on can proviude a function screval
        ## which accepts an instancs of VBGMM as first parameter,
        ## A data vector X as second parameter and keyword
        ## arguments. Upon calling screval score passes self, X
        ## and all provided keyword arguments (**scorekwarg) on to
        ## screval and returns the score value. The default
        ## implementation of screval returns the objects nFE value
        ## (the negative free enregy) which can be used as model
        ## selection yardstick. When calling vbobj.score() this
        ## functionality is obtained.
        if not self.isfitted:
            raise Exception("Model not fitted!")
        return screval(self, X=X, **scorekwarg)
    def predict_proba(self, X=None):
        ## predicts the component probabilities of a sample matrix
        ## X. In case X is None we just return the probabilities
        ## which were obtained for all training samples.
        ## Calculation may use self.QInmx(X), but has to preserve
        ## self.QIn and self.mxFQIn for coherence with the updates
        ## during model fitting.
        if not self.isfitted:
            raise Exception("Model not fitted!")
        if X is not None:
            Nsmpl=self.Nsmpl
            self.Nsmpl=X.shape[0]
            QInsaf=self.QIn.copy()
            mxFQInsaf=self.mxFQIn.copy()
            self.QInmx(X)
            Pk=self.QIn.copy()
            self.QIn=QInsaf.copy()
            self.mxFQIn=mxFQInsaf.copy()
            self.Nsmpl=Nsmpl
        else:
            Pk=self.QIn.copy()
        return Pk
    def predict(self, X=None):
        ## predicts the >>one based<< (most probable) component
        ## indices predicts the component probabilities of a sample matrix
        ## X. If X==None we predict the labels of the samples that were
        ## used for inference.
        if not self.isfitted:
            raise Exception("Model not fitted!")
        Pk=self.predict_proba(X=X)
        return 1+np.argmax(Pk, axis=1)

    def fit_predict(self, X):
        ## fits a VB GMM and predicts the one based (most
        ## probable) component indices of a sample matrix X
        self.fit(X)
        return self.predict()
    def predict_logdens(self, X=None):
        ## predicts the log densities of samples under the fitted
        ## GMM (a vector with X.shape[0] entries). As the
        ## derivation of the VBGMM shows, the approximate log
        ## densities arrise within self.QInmx(X) after
        ## maximisation w.r.t Q(In) as data contribution to the
        ## approximate marginal log likelihood. After preserving
        ## the individual sample contributions from model fitting,
        ## we can thus use self.QInmx(X) to obtain approximate log
        ## density values for all samples.
        if not self.isfitted:
            raise Exception("Model not fitted!")
        if X is not None:
            ## we save the internal values, similarly to
            ## predict_proba call self.QInmx(X) and restopre the
            ## internal state.
            Nsmpl=self.Nsmpl
            self.Nsmpl=X.shape[0]
            QInsaf=self.QIn.copy()
            mxFQInsaf=self.mxFQIn.copy()
            self.QInmx(X)
            lgpdf=self.mxFQIn.copy() ## contains the approximate log data densities
            self.QIn=QInsaf.copy()
            self.mxFQIn=mxFQInsaf.copy()
            self.Nsmpl=Nsmpl
        else:
            ## we wish to obtain the training sample log density values
            lgpdf=self.mxFQIn.copy()
        return lgpdf
    def sample(self, nsample=1, takeexp=True):
        ## generates nsample samples of fitted GMM. When setting
        ## takeexp=True, samples are drawn from a density model
        ## which is parameterised by the expectations of all model
        ## parameters. When setting takeexp to False every sample
        ## is drawn by using a random sample
        if not self.isfitted:
            raise Exception("Model not fitted!")
        ## for an efficient implementaion of sampling, we use a
        ## tensor notation which may be provided to numpy
        ## multivariate normal random number generation to
        ## generate all n samples with one function call.
        if takeexp:
            ## step 1 - obtain the expected model parameters which we use for sampling.
            Pk=self.QP/sum(self.QP) ## kernel prior
            mk=self.Qmm  ## mk[k,:] is the expectation of the k-th kernel mean
            ## express <Lk> (4 different modes): Note that the kernel index should be provided in third dimension.
            ## we use allELki[:,:,k] as index for the covariance matrix of kernel k
            if self.issharedcov:
                ELki=(1/self.QLalpha)*self.QLbeta   ## kernel independent NOTE THIS IS THE COVARIANCE
                allELki=np.stack([Elki]*self.nK, axis=2)  ## match with kernel specific situation
            else:
                allELki=(1/self.QLalpha)*self.QLbeta  ## again an expression for the covariance 
            ## we can now generate samples according to the expectations of the GMM parameters
            alldx=np.argmax(np.random.multinomial(1, Pk, size=nsample), axis=1)
            ## and generate samples by a loop of n calls of np.random.multivariate_normal
            allx=np.stack([np.random.multivariate_normal(k[idx], allELki[:,:,idx]) for idx in alldx], axis=0)
        else:
            ## sample the allocation probabilities,
            ## and the resulting kernel indices
            alldx=np.argmax(np.stack([np.random.multinomial(1, np.random.dirichlet(self.QP), size=1) for cdx in range(nsample)], axis=0), axis=1)
            ## next we sample all mean and covarianvce
            ## matrices and finally the samples.
            ## shortcuts
            mvn=np.random.multivariate_normal
            iW=sps.invwishart.rvs ## the inverse parameters are degrees of freedom and rate (precision like) we use thus alpha and beta as parameters
            ## adjust the parameters such that they agree with the case we work on
            if self.issharedcov:
                alpha=np.array([self.QLalpha]*self.nK)
                Beta=np.stack([self.QLbeta]*self.nK, axis=2)
            else:
                alpha=self.QLalpha
                Beta=self.QLBeta
            ## We nest drawing random variates: the outer mvn
            ## draws one observation from kernel kdx which is
            ## based on randomly drawn kernel parameters.  The
            ## kernel mean is mvn with mean Qmm[kdx] and
            ## covariance QmLi[:,:,kdx] The kernel covariance
            ## matrix is an inverse Wishart with alpha[kdx] degf
            ## and inverse scale matrix Beta[:,:,kdx] Note that
            ## this expression is in line with the VB derivation
            ## and the equations which describe
            ## scipy.stats.invwishart. Also note that the
            ## terminology in the scipy help is imprecise as they
            ## speak about scale matrices when they actually use
            ## an inverse scale in the equation. A code fragment
            ## at the top of the document shows that iW is
            ## parameterised by rate (inverse scale) matrices!
            allx=np.stack([mvn(mvn(self.Qmm[kdx], self.QmLi[:,:,kdx]), iW(alpha[kdx], Beta[:,:,kdx])) for kdx in alldx], axis=0)
        return allx
    
   
import ray
import psutil
import logging
def rayinit(nocpus=None, doshutdown=False, resourcedict={"ParLrnActrs":None}, resourcefact=1, **rayarg):
    try:
        ## set number of parallel tasks
        NoResources=psutil.cpu_count(True)*resoucefact
        resourcedict["ParLrnActrs"]=NoResources
        ##print( resourcedict)
        ray.init(num_cpus=nocpus, logging_level=logging.CRITICAL, resources={"ParLrnActrs":NoResources}, **rayarg)
    except:
        if doshutdown:
            ray.shutdown()
            ## set number of parallel tasks
            NoResources=psutil.cpu_count(True)*resoucefact
            resourcedict["ParLrnActrs"]=NoResources
            ray.init(num_cpus = nocpus, logging_level=logging.CRITICAL, resources={"ParLrnActrs":NoResources}, **rayarg)
            
## we disable all warning logs (which tell us not to use as many actors as we do)
import logging
logging.disable(logging.WARNING)
## alterantive to the ray server objects whcih do not work.

@ray.remote
def parfit(X, vbgo):
    ## wrapper arround vbgo.fit(X) which fits a VBGMM object to
    ## datamatrix X such as to allow submissions of parallel vbgmm
    ## inference runs usind the ray infrastructure.
    ##
    ##
    ## IN
    ##
    ## X: [N x M] matrix with rows assumed to be drawn from a M-variate Gaussian.
    ##
    ## vbgo: initialised vbgmm object which is to be fitted to X.
    ##
    ## OUT
    ##
    ## vbgo: fitted VBGMM instance.
    ##
    vbgo.fit(X)
    return vbgo

## obsolete as it does not work as intended and leads to overloading
## the machine with far too many tasks. For ease of correcting the
## problem, we modify ProbEns and change it for function based
## parallelisation.
##
## @ray.remote
## class PARLEARNER(VBGMM):
##     def __init__(self, parid=None, **otherargs):
##         ## initialise the LEARNER object with **otherargs (containing modified hyperparameters)
##         VBGMM.__init__(self, **otherargs)
##         ## parid is identical for all objects which were
##         ## generated with the same hyperparameters self._parid
##         ## This allows for an efficient identification of remote learning actors which use the same hyperparameters (select the best
##         ## result from several reinitialisations)
##         self._parid=parid
##     def getparid(self):
##         return self._parid 
##     def getobj(self):
##         ## return the object itself.
##         return self
        
class SERIALLEARNER(VBGMM):
    def __init__(self, parid=None, **otherargs):
        ## initialise the LEARNER object with **otherargs (containing modified hyperparameters)
        VBGMM.__init__(self, **otherargs)
        ## parid is identical for all objects which were
        ## generated with the same hyperparameters self._parid
        ## This allows for an efficient identification of remote learning actors which use the same hyperparameters (select the best
        ## result from several reinitialisations)
        self._parid=parid
    def getparid(self):
        return self._parid 
    def getobj(self):
        ## return the object itself.
        return self
    def parallel2serial(self, obj):
        ## parallel2serial is a method which robustly
        ## initialises a SERIALLEARNER from a PARLEARNER. We
        ## create to this end a copy of all attributes which
        ## exist in both objects.
        for mykey in obj.__dict__.keys():
            self.__dict__[mykey]=ddup(obj.__dict__[mykey])            
            #print("{0} in target".format(mykey))
            #if mykey in obj.__dict__.keys():
            #    #print("{0} in source.".format(mykey))
            #    self.__dict__[mykey]=ddup(obj.__dict__[mykey])
            #    #if mykey=="isfitted":
            #    #    print("{0} in taret is {1}".format(mykey, self.__dict__[mykey]))
        ## print("ID: {0} fitted:{1}".format(self._parid, self.isfitted))
        return self

class ProbEns:
    def __init__(self, LEARNER, paramgrid, ninit, ncpus=-1, useHT=False, taskmul=2, rayargs={}, 
                 score4ensemblecomb=lambda obj, X=None, **screvalkwarg:obj.nFE,
                 score2weight=lambda scores:evids2mp(scores),
                 score4mdlsel=lambda obj, X=None, **screvalkwarg:obj.nFE,
                 nomdls2sel=None, **scorekwargs):
        ## A wrapper for VBGMM comnpatible data types which will be
        ## used to parallelise inference and to obtain probabilistic
        ## ensemble predictions.
        ##
        ## LEARNER: Type which is compatible with VBGMM. The
        ##          most important properties of LEARNER is the need
        ##          of a VBGMM compatible score function and the fact
        ##          that objects of type LEARNER must provide an
        ##          attrivbute nFE which contains the score value (in
        ##          VBGMM this attribute contains the approximate log
        ##          marginal likelihood of the model).
        ##
        ## paramgrid: A dictionary with LEARNER.__init__(self, ...)
        ##          compatible parameter names. ProbEns uses
        ##          sklearn.model_selection.ParameterGrid(paramgrid)
        ##          to generate a list of dictionaries which enumerate
        ##          all parameter combinations. For efficient
        ##          dispatching it is advisable to start the
        ##          hyperparameterlists with values which lead to more
        ##          involved calculations (e.g. kernel numbers in
        ##          decreasing order).
        ##
        ## ninit:   integer value of reinitialisations of every parameterised learner.
        ##
        ## ncpus: number of cores used for parallelising model fiting
        ##          and model evaluation. -1 uses all available cores.
        ## 
        ## useHT: Boolean flag which defines whether hyperthreads
        ##          count as cpus. This flag only matters in
        ##          combination with cpuno=-1 and determines whether
        ##          hyperthreding is considered in obtaining the
        ##          number of processes which we execute in parallel.
        ##
        ## taskmul: integer factor to control task dispatching.
        ##
        ## dofuncpar: Boolean which if True uses ray function parallelisation for inference
        ##          This is only considered in case 
        ##          
        ## score4ensemblecomb: a function with prototype funv(obj,
        ##          X=None, **screvalkwarg) which returns a double
        ##          score The function defaults to returning obj.nFE
        ##          which is the approximate log marginal likelihood
        ##          of a VB fitted learner. 
        ##
        ## score2weight: A function which is used to combine the
        ##          result of all learner.score(X, score4ensemblecomb,
        ##          **scorekwargs) return values to ensemble
        ##          weights. The defailt is evids2mp which converts
        ##          log marginal likelihoods to probabilities.  The
        ##          weights are subsequently used in the prediction
        ##          functions to mix iundividual model predictions.
        ##
        ## score4mdlsel: a function with prototype funv(obj, X=None,
        ##          **screvalkwarg) which returns a double score
        ##          value. Function score4mdlsel is also used as
        ##          parameter to learner.score calls, serves however
        ##          the purpose of ranking all models for selection.
        ##          The default implementation is a dupolication of
        ##          score4ensemblecomb.
        ##
        ## nomdls2sel: Number of top performing models after ranking
        ##          by score4mdlsel which are considered for ensemble
        ##          predictions.
        ##
        ## scorekwargs: key word aeguiments which are provided to
        ##          learner.score evaluations. Compatibility requires that
        ##          these parameters are forwarded from LEARNER::score
        ##          to the passed on scoring functions. It is thus advisable that
        ##          the implementations of score4ensemblecomb and score4mdlsel
        ##          can deal with keyword arguments they are not aware of.
        ##
        ## (C) P. Sykacek 2022-2025 <peter@sykacek.net>
        print("initi self.rayargs")
        self.rayargs=rayargs
        ## local datatype derived from LERNER which adds a parameter
        ## id to identify the important object characteristics
        ## (hyperparameters) when the code returns from dispatched
        ## inference.
        class PARAGG:
            ## data type for aggergating results with identical hyper
            ## parameter values
            def __init__(self, paramid):
                self.paramid=paramid
                self.optval=None
                self.obj=None
            def aggres(self, obj, score):
                try:
                    if not np.isnan(score):
                        if self.optval is None: 
                            self.optval=score
                            self.obj=obj
                        elif self.optval < score:
                            self.optval=score
                            self.obj=obj
                except:
                    print(score)
        ## store the types
        self.SERIALLEARNER=LEARNER
        ## handle parallel processing
        ## store information for parallel processing
        self.ncpus=ncpus
        self.useHT=useHT
        self.parinit=False
        if self.ncpus==-1:
            self.ncpus=psutil.cpu_count(self.useHT)
        ## maximum running number of tasks=taskmul times the number of threads.
        self.maxtasks=psutil.cpu_count(True)*taskmul
        ## conditional start of parallel processing. This happens only
        ## if self.nocpus >1 and will in this case set the flag
        ## self.parinit True.
        self.startparproc()
        ## generate all parametrisations which we wish to test
        self.allpars=list(skms.ParameterGrid(pargridparse(paramgrid)))
        ## prepare all objects for dispatching
        self.alldispatchremote=[]
        self.alldispatchlocal=[]
        self.aggdict={}    ## parameter ID based aggregatiuon of results 
        self.objdict={}    ## mapping from object ids to parameter ids
        ## mapping from parameter ids to hyperparameter values.  Note:
        ## dict.fromkeys must not be initialised with an empty list as
        ## this will lead to all keys refering to the same list!!
        self.hpardict=dict.fromkeys(["paramid"]+list(self.allpars[0].keys()), None)
        for key in self.hpardict.keys():
            self.hpardict[key]=ddup([])
        #print(self.hpardict)
        for paramid, hyperdict in enumerate(self.allpars):
            ## initialsie aggregation of results (keep track of the
            ## optimum oer hyper parameter combination.)
            self.hpardict["paramid"].append(paramid)
            for knam in hyperdict.keys():
                self.hpardict[knam].append(hyperdict[knam])
            self.aggdict[paramid]=PARAGG(paramid)
            ## To lead to good optima, every hyperparameter combination
            ## should be inferred several times. We achieve this in
            ## parallel by creating multiple object instances for
            ## dispatching.
            for nrestart in range(ninit):
                if self.parinit:
                    ## parallelise via function calls
                    clearner=SERIALLEARNER(paramid, **hyperdict)
                    self.alldispatchremote.append(clearner)
                    self.objdict[clearner]=paramid
                ## sequential processing uses a LEARNER type as well
                self.alldispatchlocal.append(LEARNER(paramid, **hyperdict))
        ## finally we initialise the functions which we will use for
        ## score evaluation and ensemble averaging.
        ##
        ## self.score4ensemblecomb is used for scoring model averaging.
        self.score4ensemblecomb=score4ensemblecomb
        ## self.score2weight converts score values to mixing weights
        self.score2weight=score2weight
        ## self.score4mdlsel is used to score model selection.
        self.score4mdlsel=score4mdlsel
        ## number of models which we select for the ensemvle predicitons from all inferred models
        self.nomdls2sel=nomdls2sel
        ## self.scorekwargs contains the keyword arguments for the score functions and is
        ## used as arguments when calling "score".
        self.scorekwargs=ddup(scorekwargs)
        ## lists for references to fit.remote()
        self.allfitrefs=[]
        self.allfitdone=[]
        ## lists for references to score.remote()
        self.allscorerefs=[]
        self.allscoredone=[]
    def startparproc(self):
        ## start the scheduler.
        if self.ncpus>1:
            self.parinit=True
            ## start ray
            rayinit(nocpus=self.ncpus, **self.rayargs)
    def stopparproc(self):
        if self.parinit:
            ## stop ray.
            ray.shutdown()
    def fit(self, X):
        ## fit optimises all LEARNER objects in parallel and tracks the
        ## best solution for the same hypeprarameter combination.
        self.allfitrefs=[] ## reinitialise to allow sequential calls of fit (in case this is for example done on different samples).
        self.allfitdone=[]
        self.havescores=False
        if self.parinit:
            ## ray based parallel inference.
            ## put X to the ray opbject store.
            Xref=ray.put(X)
            ## we use a pattern which avouds that we call to many tasks in parallel.
            for callno, learner in enumerate(self.alldispatchremote):
                ##if callno>self.maxtasks:
                    ## after we started self.maxtasks we make sure that we have callno-self.maxtasks
                    ## tasks completed.
                ##    ray.wait(self.allfitrefs, num_returns=callno-self.maxtasks)
                ## this is now modified to use parfit which returns an
                ## inferred bvbgmm object.
                self.allfitrefs.append(parfit.remote(Xref, learner))
            ## we do not block here! We just make sure that all references are kept.
            ## ray.get(self.allfitrefs+self.allfitdone)
        else:
            ## single core implementation without overhead
            for learner in self.alldispatchlocal:
                learner.fit(X)
    def getscores(self, X=None):
        ## getscores is the only blocking function! we need to call
        ## getscores also when aggregating individual model
        ## predictions.
        if self.parinit and not self.havescores:
            ##self.allscorerefs=[]
            ## we need to collect the results of the parallel
            ## implementation by calling the getobj function from
            ## alldispatchremote and copy the results as SERIALLEARNER
            ## objects to self.alldispatchlocal
            ##
            ## we use a pattern which avouds that we call to many tasks in parallel.
            ##for callno, learner in enumerate(self.alldispatchremote):
            ##    if callno>self.maxtasks:
            ##        ## after we started self.maxtasks we make sure that we have callno-self.maxtasks
            ##        ## tasks completed.
            ##        ray.wait(self.allscorerefs, num_returns=callno-self.maxtasks)
            ##    self.allscorerefs.append(learner)
            ## we do not have to wait for fitting to be done as from
            ## the ray documentation: "Methods called on different
            ## actors can execute in parallel, and methods called on
            ## the same actor are executed serially in the order that
            ## they are called."
            allparobjs=ray.get(self.allfitrefs) ## this is the blocking wait after inference.
            print(len(allparobjs))
            print(len(self.alldispatchremote))
            for idx, parobj in enumerate(allparobjs):
                self.alldispatchlocal[idx].parallel2serial(parobj)
            self.havescores=True
        ## parallel and serial processing has now the results in
        ## self.alldispatchlocal. We can thus aggregate the scores for
        ## both implementations
        for learner in self.alldispatchlocal:
            #print("learner:{0} fitted:{1} nFE:{2}".format(learner.getparid(), learner.isfitted, learner.nFE)) 
            cscore=learner.score(X, screval=self.score4mdlsel, **self.scorekwargs)
            ## aggregate scores for identical paramids which
            ## represents the same hyperparameter combinations.
            ## self.aggdict is prepared bay the constructuor for
            ## every hyperparameter combination.
            self.aggdict[learner.getparid()].aggres(learner.getobj(), cscore)
        ## we have now either from parallel processing or from the
        ## sequential solution in self.aggdict the optimal solution
        ## for every tested hyperparameter combination.

        ## If no
        ## calculation for a particular hyperparameter combination was
        ## successfull, we remove the respective entry from
        ## self.aggdict. This may happen in case we fit a complex model
        ## on a small number of samples.
        ## delkeys=[]
        ## for key in self.aggdict.keys():
        ##    if self.aggdict[key].obj is None:
        ##        delkeys.append(key)
        ## for key in delkeys:
        ##    del self.aggdict[key]
        ## we extract the parids and scores from self.aggdict
        allparids=list(self.aggdict.keys())
        self.allscores=[]
        for parid in allparids:
            self.allscores.append(self.aggdict[parid].optval)
        ## if the number of models in self.aggdict is larger than
        ## self.nomdls2sel we chose the top scored self.nomdls2sel
        ## instances.
        self.parids4preds=ddup(allparids)
        if self.nomdls2sel is not None:
            if len(allscores)>self.nomdls2sel:
                ## we subselect the self.nomdls2sel inferred models
                scoreorder=listargsort(allscores)
                scoreorder=scoreorder[0:self.nomdls2sel]
                ## self.parids4preds are indices to self.aggdict which contain the
                ## top ranked
                self.parids4preds=allparids[scoreorder]
        ## the final step in fit is calculating the ensemble weights.
        self.ensscores=[]
        for parid in self.parids4preds:
            if self.aggdict[parid].obj is not None:
                self.ensscores.append(self.aggdict[parid].obj.score(X, self.score4ensemblecomb, **self.scorekwargs))
            else:
                self.ensscores.append(None)
        ## self.ensweigts are the mixing weights of the ensemble predicitions
        try:
            ensscores=self.ensscores.copy()
            #minval=np.min(ensscores[np.isfinite(ensscores)])
            #ensscores[np.logical_not(np.isfinite(ensscores))]=minval
            self.ensweights=self.score2weight(ensscores)
        except:
            self.ensweights=self.ensscores.copy()
    def getscorestats(self):
        ## function returns model parameters and their scores.  the
        ## return value is based on self.hpardict which contains all
        ## parameter ids as list under the key name "paramid" and the
        ## model hyperparameter values as list under the respective
        ## key name.
        ##
        ## make sure we have the scores
        self.getscores()
        scoredict=ddup(self.hpardict)
        selscores=[]
        ## we should collect the selection scores from
        ## self.aggdict[parid].optval
        for paramid in scoredict["paramid"]:
            if paramid in self.aggdict.keys():
                selscores.append(self.aggdict[paramid].optval)
            else:
                selscores.append(None)
        scoredict["selscores"]=selscores
        ## self.ensscores and self.ensweights are already in the correct order.
        scoredict["ensscores"]=self.ensscores
        scoredict["ensweights"]=self.ensweights
        ## return the scores as dataframe.
        return scoredict
    def optmdlparid(self):
        ## return the parameter id of the optimal model.
        ## this is done by traversing self.aggdict.
        first=True
        for cpid in self.aggdict.keys():
            if first:
                optparid=cpid
                optval=self.aggdict[cpid].optval
                first=False
            elif self.aggdict[cpid].optval > optval:
                optparid=cpid
                optval=self.aggdict[cpid].optval
        return optparid
    def predict(self, X=None):
        ## we use getscores to block and obtain all scores and
        ## subsequently predict the cluster labels with the optimal
        ## model. Note that another useful implementation would
        ## aggregate. This is however in general non trivial and thus
        ## not yet implemented.
        self.getscores(X=X) ## this is blocking
        optparamid=self.optmdlparid()
        return self.aggdict[optparamid].obj.predict(X=X)
    def fit_predict(self, X):
        ## combination of fit and predict.
        self.fit(X)
        return self.predict(X=None) ## note that we do not need X here and this speeds up.
    def predict_proba(self, X=None):
        ## like predict, however calling the optimal objects
        ## predict_proba function.
        self.getscores(X=X) ## this is blocking
        optparamid=self.optmdlparid()
        return self.aggdict[optparamid].obj.predict_proba(X=X)
    def predict_logdens(self, X=None, domax=False):
        ## In case the ensemble weights sum up to 1, we predict the
        ## log value of the ensemble average of the density function
        ## of all considered models.
        ##
        ## If calculation of the ensemble weights failed, or
        ## domax=True, we use the model which got the highest
        ## selection score.
        ## IN
        ## X: a [nsmpl x ndim] matrix of inpit values
        ## OUT
        ## logdens: a [nsmpl x ] vector of logarithmic data densities
        ##          which correspond to the rows of X.
        
        myeps=np.finfo(np.double).resolution
        domax=domax or np.abs(1.0-np.sum(self.ensweights))>myeps
        if domax:
            optparamid=self.optmdlparid()
            logdens=self.aggdict[optparamid].obj.predict_logdens(X=X)
        else:
            ## we set up a list of log density values which we
            ## estimate from all individual models.
            allpx=[]
            for key, entry in self.aggdict.items():
                allpx.append(np.exp(entry.obj.predict_logdens(X=X)))
            alldens=np.stack(allpx, axis=1)
            logdens=np.log(np.dot(alldens, self.ensweights))
        return logdens

## test parallel inference.
gendata=False
if gendata:
    ## gerenate 2 dim test data and infer a vbgmm
    nsample=200
    lc=1.75
    mk=np.array([[-lc,-lc],[0,0],[lc,lc]])
    nK=mk.shape[0]
    PIndPriCnt=7.0
    sdvl=1
    corr=-0.5
    cov=np.array([[sdvl**2, corr*sdvl**2],[corr*sdvl**2,sdvl**2]])
    Pk=[0.2,0.3,0.5]
    kdx=np.argmax(np.random.multinomial(1, Pk, size=nsample), axis=1)
    X=np.stack([np.random.multivariate_normal(mk[idx], cov) for idx in kdx], axis=0)


dosim=False
if dosim:
    np.seterr(all='raise')
    paramgrid={"nK":list(range(1,8)), "PIndPriCnt":[7], "covtyp":["full", "diag"], "covmode":["individual", "shared"]}
    #paramgrid={"nK":list(range(1,8)), "PIndPriCnt":[7], "covtyp":["diag"], "covmode":["shared"]}
    vbgmm=ProbEns(SERIALLEARNER, paramgrid, ninit=10, ncpus=12)
    vbgmm.fit(X)
    scorestats=vbgmm.getscorestats()
    print(scorestats)
    #plabels=vbgmm.predict_proba()
    #print(plabels)
    vbgmm.stopparproc()
