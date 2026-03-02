## code and certified solutions for some NIST non linear regression problems
## https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml
##
## (C) P. Sykacek 2020 <peter@sykacek.net> for use in teaching
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
## available nonlinear NIST datasets, legends and predictors
nistnlrg_names=["Misra1a", "DanWood", "Hahn1", "Roszman1", "Nelson", "Gauss1"]
## information about data from NIST homepage
nistnlrg_legend={
    "Misra1a":"""NIST study involving dental research in monomolecular
    adsorption. The response variable is volume, and the predictor
    variable is pressure.""",
    "DanWood":"""E.S.Keeping, Introduction to Statistical Inference, Van Nostrand
    Company, Princeton, NJ, 1962, p. 354. The response variable is
    energy radiated from a carbon filament lamp per cm**2 per second,
    and the predictor variable is the absolute temperature of the
    filament in 1000 degrees Kelvin.""",
    "Hahn1":"""These data are the result of a NIST study involving the thermal
    expansion of copper. The response variable is the coefficient of
    thermal expansion, and the predictor variable is temperature in
    degrees kelvin. """,
    "Roszman1":"""These data are the result of a NIST study involving quantum defects
    in iodine atoms. The response variable is the number of quantum
    defects, and the predictor variable is the excited energy
    state. The argument to the ARCTAN function is in radians.""",
    "Nelson":"""These data are the result of a study involving the analysis of
    performance degradation data from accelerated tests, published in
    IEEE Transactions on Reliability. The response variable is
    dialectric breakdown strength in kilo-volts, and the predictor
    variables are time in weeks and temperature in degrees Celcius.""" ,
    "Gauss1":"""The data are two well-separated Gaussians on a decaying 
    exponential baseline plus normally distributed zero-mean noise with 
    variance = 6.25"""
}

## specify the default names to data map
nistnlrg_data={
    "Misra1a":"../course_data/misra1a.csv",
    "DanWood":"../course_data/danwood.csv",
    "Hahn1":"../course_data/hahn1.csv",
    "Roszman1":"../course_data/roszman1.csv",
    "Nelson":"../course_data/nelson.csv",
    "Gauss1":"../course_data/gauss1.csv"
}

## setbasedir allows overloading the default basedir 
def setbasedir(basedir="../"):
    ## sets a new basedir in the module global mapping from NIST data names
    ## to data files. (required for compatibity with google colab)
    global nistnlrg_data
    nistnlrg_data={
        "Misra1a":basedir+"course_data/misra1a.csv",
        "DanWood":basedir+"course_data/danwood.csv",
        "Hahn1":basedir+"course_data/hahn1.csv",
        "Roszman1":basedir+"course_data/roszman1.csv",
        "Nelson":basedir+"course_data/nelson.csv",
        "Gauss1":basedir+"course_data/gauss1.csv"
    }

## NIST certified predictors for the data. The functions and certified
## parameters are from the NIST homepage.
def misra1apred(x, bt1=2.3894212918e+02, bt2=5.5015643181e-04):
    if type(x) != type(np.array([])):
        x=np.array(x)
    return bt1*(1-np.exp(-bt2*x))

def danwoodpred(x, bt1=7.6886226176e-01, bt2=3.8604055871):
    if type(x) !=  type(np.array([])):
        x=np.array(x)
    return bt1*np.exp(bt2*np.log(x))

def hahn1pred(x, b1=1.0776351733, b2=-1.2269296921e-01, b3=4.0863750610e-03,
              b4=-1.4262662514e-06, b5=-5.7609940901e-03, b6=2.4053735503e-04,
              b7=-1.2314450199e-07):
    if type(x) != type(np.array([])):
        x=np.array(x)
    return (b1+b2*x+b3*x**2+b4*x**3)/(1+b5*x+b6*x**2+b7*x**3)

def roszman1pred(x, b1=2.0196866396e-01, b2=-6.1953516256E-06,
                 b3=1.2044556708E+03, b4=-1.8134269537E+02):
    if type(x) !=  type(np.array([])):
        x=np.array(x)
    return b1-b2*x - np.arctan(b3/(x-b4))/np.pi

def nelsonpred(x, b1=2.5906836021, b2=5.6177717026e-09, b3=-5.7701013174e-02):
    if type(x) != type(np.array([])):
        x=np.array(x)
    return np.exp(b1-b2*x[:,0]*np.exp(-b3*x[:,1]))

def gauss1pred(x, b1=9.8778210871e+01, b2=1.0497276517e-02,
               b3=1.0048990633e+02, b4=6.7481111276e+01,
               b5=2.3129773360E+01, b6=7.1994503004E+01,
               b7=1.7899805021E+02, b8=1.8389389025E+01):
    if type(x) != type(np.array([])):
        x=np.array(x)
    return b1*np.exp(-b2*x)+b3*np.exp(-(x-b4)**2/b5**2)+b6*np.exp(-(x-b7)**2/b8**2)

# dict from name to certified NIST predictor
nistnlrg_bestpred={
    "Misra1a": lambda x: misra1apred(x),
    "DanWood": lambda x:danwoodpred(x),
    "Hahn1": lambda x: hahn1pred(x),
    "Roszman1": lambda x: roszman1pred(x),
    "Nelson":lambda x: nelsonpred(x),
    "Gauss1":lambda x:gauss1pred(x)
}

# prepare data loading (regressor column names in appropriate order)
nistnlrg_regnams={
    "Misra1a":["x"],
    "DanWood":["x"],
    "Hahn1":["x"],
    "Roszman1":["x"],
    "Nelson":["x1","x2"],
    "Gauss1":["x"]
}

def to_numpy(pdt):
    ## convert a pandas dataframe to a numpy array
    try:
        return pdt.to_numpy()
    except:
        return pdt.values

## exception 
class NISTError(Exception):
    pass

class NISTnlReg:
    ## class for handling NIST nonlinear regression data and certified predictors
    def __init__(self, names=nistnlrg_names):
        self.nistnlrg_names=copy.deepcopy(names)
    def getdata(self, name, sep=","):
        ## load the nist data an provide the tuple (X, y) data is
        ## appropriately reordered to be compatible with the
        ## corresponding predictor.
        ##
        ## IN
        ## name: name of NIST data
        ## sep:  column separar in csv file
        ##
        ## OUT
        ## (
        ## X: regressors
        ## y: response variables
        ## )
        ## (C) P. Sykacek 2019 <peter@sykacek.net>     
        if name not in self.nistnlrg_names:
            raise NISTError("Dataset {0} unavailable. Load any of {1}".format(name, ", ".join(self.nistnlrg_names)))
        ## get filename and load as pandas dataframe
        pdat=pd.read_csv(nistnlrg_data[name], sep=sep)
        ## extract response variable
        y=pdat[["y"]]
        y=to_numpy(y)
        ## extract regressors
        X=pdat[nistnlrg_regnams[name]]
        X=to_numpy(X)
        return(X, y)
    def getlegend(self, name):
        ## provide an explanation for a NIST nonlinear regression data
        ## set
        ##
        ## IN
        ##
        ## name: name of NIST data
        ##
        ## OUT
        ##
        ## legend: explanation form the NIST nonlinear regression data homepage
        ##
        ## (C) P. Sykacek 2019 <peter@sykacek.net>
        if name not in self.nistnlrg_names:
            raise NISTError("Dataset {0} unavailable. Load any of {1}".format(name, ", ".join(self.nistnlrg_names)))
        return nistnlrg_legend[name]
    def getpredictor(self, name):
        ## provide the certified predictor which NIST makes available
        ## for the data at the NIST nonlinear regression data
        ## homepage.
        ##
        ## IN
        ##
        ## name: name of NIST data
        ##
        ## OUT
        ##
        ## preditor: a function of type f(x) which is defined and
        ##       parameterised according to NISTs certified
        ##       predictor. The function f(x) can be applied to the
        ##       regressor matrix X as loaded by getdata(name) and
        ##       used to obtain the NIST certified predictions for
        ##       that data.
        ##       
        ## (C) P. Sykacek 2019 <peter@sykacek.net>
        if name not in self.nistnlrg_names:
            raise NISTError("Dataset {0} unavailable. Load any of {1}".format(name, ", ".join(self.nistnlrg_names)))
        return nistnlrg_bestpred[name]
    def illustrate(self, name, fig=None, subplotno=None, nsamples=100, predictor=None):
        ## provide a visualisation of data and NIST certified function.
        ##
        ## IN
        ##
        ## name: name of NIST data
        ##
        ## fig: a matplotlib figure handle or None
        ##
        ## subplotno: a number for the subplot (211, 212 would be
        ##      both plots for two rows and one column) or None
        ##
        ## nsamples: nr of samples (default 100) created evely spaced
        ##      per data dimension to feed predictor.
        ##
        ## predictor: a f(X) compatible preditor which reoresents any
        ##      predictor for the NIST datum. Defaults to None in
        ##      which case we use the NIST certified predictor.
        ##
        ## OUT
        ##
        ## fig: a matplotlib figure handle
        ##
        ## (C) P. Sykacek 2019 <peter@sykacek.net>        
        if name not in self.nistnlrg_names:
            raise NISTError("Dataset {0} unavailable. Load any of {1}".format(name, ", ".join(self.nistnlrg_names)))
        if fig is None:
            fig=plt.figure()
        ## get data
        (Xd, yd)=self.getdata(name)
        ## and predictor
        if predictor is None:
            predictor=self.getpredictor(name)
        ## we will now apply predictor to a generated dataset
        dim=len(Xd.shape)
        if dim==2:
            if Xd.shape[1] == 1:
                dim=1
        if dim >1:
            ## two dim dataset.
            xmin=np.min(Xd, axis=0)
            xmax=np.max(Xd, axis=0)
            x1=np.linspace(xmin[0], xmax[0], nsamples)
            x2=np.linspace(xmin[1], xmax[1], nsamples)
            [X, Y]=np.meshgrid(x1,x2)
            allx=X.copy()
            allx.shape=(np.prod(allx.shape),1)
            ally=Y.copy()
            ally.shape=(np.prod(ally.shape),1)
            Xgrid=np.hstack((allx, ally))
            print(Xgrid.shape)
            allz=predictor(Xgrid)
            Z=allz
            Z.shape=X.shape
            ## we may now plot X Y and Z as wireplot
            if subplotno is None:
                subplotno=111
            ax = fig.add_subplot(subplotno, projection='3d')
            ax.plot_wireframe(X, Y, Z, color="green", linewidth=3, rstride=10, cstride=10)
            ## add the samples
            ax.scatter(Xd[:,0], Xd[:,1], yd, 'b.')
            ax.set_title("NIST {0}".format(name))
        else:
            ## 1 dim input as for most data
            xmin=min(Xd)
            xmax=max(Xd)
            allx=np.linspace(xmin, xmax, nsamples)
            ally=predictor(allx)
            if subplotno is None:
                subplotno=111
            plt.subplot(subplotno)
            plt.plot(allx, ally, "g-", linewidth=3)
            plt.plot(Xd, yd, 'b.')
            plt.title("NIST {0}".format(name))
        return fig
            
def testall():
    ## test function which loads all data and displays it together with
    ## allsubplots=[321, 322, 323, 324, 325, 326]
    fig=plt.figure(figsize=(15, 10))
    nistnldt=NISTnlReg()
    for idx, name in enumerate(nistnlrg_names):
        subplotno=321+idx
        dummy=nistnldt.illustrate(name, fig=fig, subplotno=subplotno)
    plt.show()
        
        
#testall()
#fig=plt.figure(figsize=(15, 15))
#nistnldt=NISTnlReg()
#fig=nistnldt.illustrate(name="Nelson")
#fig.show()
