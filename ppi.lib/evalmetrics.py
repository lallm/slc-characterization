"""
Minimal subset of evalmetrics.py — only what's needed for dendrograms.
"""
import numpy as np    
def colocP(Pk):
    nsmp, ninst = Pk.shape
    clP = -np.ones((int(0.5*(nsmp**2-nsmp)),))
    sdx = 0
    for frw in range(nsmp-1):
        edx = sdx + nsmp - frw - 1
        clP[sdx:edx] = np.sum(Pk[[frw]*(nsmp-frw-1),:] * Pk[frw+1:nsmp,:], axis=1)
        sdx = edx
    return clP

def kldisc(P1, P2, whichlog=np.log2, tol=10**(-323)):
    nsmp, ninst = P1.shape
    P1 = P1.copy(); P2 = P2.copy()
    P1[P1<tol] = tol; P2[P2<tol] = tol
    P1 /= P1.sum(axis=1, keepdims=True)
    P2 /= P2.sum(axis=1, keepdims=True)
    Plg = whichlog(P1) - whichlog(P2)
    Plg[np.isnan(Plg)] = 0
    Plg[np.logical_and(np.isinf(Plg), np.sign(Plg)==-1)] = 0
    return np.sum(P1 * Plg, axis=1)

def symkldisc(P1, P2, whichlog=np.log2):
    return kldisc(P1, P2, whichlog) + kldisc(P2, P1, whichlog)


def clPlmetric(Pk, labels, metric=symkldisc):
    nsmp = Pk.shape[0]
    Pcl = colocP(Pk)
    lab1 = np.array([-1]*len(Pcl)); lab2 = np.array([-1]*len(Pcl))
    allmetrics = np.zeros(len(Pcl))
    sdx = 0
    for frw in range(nsmp-1):
        edx = sdx + nsmp - frw - 1
        Pcl[sdx:edx] = np.sum(Pk[[frw]*(nsmp-frw-1),:] * Pk[frw+1:nsmp,:], axis=1)
        allmetrics[sdx:edx] = metric(Pk[[frw]*(nsmp-frw-1),:], Pk[frw+1:nsmp,:])
        lab1[sdx:edx] = labels[frw]
        lab2[sdx:edx] = labels[frw+1:nsmp].ravel()
        sdx = edx
    return Pcl, lab1, lab2, allmetrics

def QIn2symkl(QIn, y):
    _, lab1, lab2, allmetrics = clPlmetric(QIn, y)
    return allmetrics, lab1, lab2

def davgquant(vals, qantval=0.5, isqntlower=True):
    vals = np.sort(vals)
    if isqntlower:
        sdx, edx = 0, int(vals.shape[0]*qantval)
    else:
        sdx, edx = int(vals.shape[0]*qantval), vals.shape[0]
    return np.mean(vals[sdx:edx])

def dopair(set1, set2):
    n1, n2 = set1.shape[0], set2.shape[0]
    idx1 = list(range(n1)) * n2
    idx2 = sorted(list(range(n2)) * n1)
    return (set1[idx1,:] if len(set1.shape)>1 else set1[idx1],
            set2[idx2,:] if len(set2.shape)>1 else set2[idx2])

def labcombs(labels):
    unqlab = sorted(list(set(labels.ravel().tolist())))
    lab1, lab2 = [], []
    for i in range(len(unqlab)-1):
        lab1 += [unqlab[i]] * (len(unqlab)-i-1)
        lab2 += unqlab[i+1:]
    return np.array(lab1, dtype=int), np.array(lab2, dtype=int)

def QIn2coloc(QIn, y, myeps=1e-300, whichlog=np.log):
    nsmp = QIn.shape[0]
    clP = colocP(QIn)
    clP[clP < myeps] = myeps
    lab1 = np.array([-1]*len(clP)); lab2 = np.array([-1]*len(clP))
    sdx = 0
    for frw in range(nsmp-1):
        edx = sdx + nsmp - frw - 1
        lab1[sdx:edx] = y[frw]
        lab2[sdx:edx] = y[frw+1:nsmp].ravel()
        sdx = edx
    return -whichlog(clP), lab1, lab2

def agglomerate_multi_allocprobs(allQIn, ally,
                                  QI_metric=QIn2coloc,
                                  aggsamples=lambda v: davgquant(v, qantval=0.25, isqntlower=True),
                                  aggruns=lambda v: np.mean(v, axis=1)):
    yproc = ally[0].copy().ravel()
    unqy = np.array(sorted(set(yproc.tolist())), dtype=int)
    Zmatr = np.zeros((len(unqy)-1, 4))
    clustlab = np.max(unqy) + 1
    lb1, lb2 = labcombs(unqy)

    alb1, alb2, allmetrics = [], [], []
    for n in range(len(allQIn)):
        cmetric, l1, l2 = QI_metric(allQIn[n], ally[n])
        alb1.append(l1.astype(int)); alb2.append(l2.astype(int))
        allmetrics.append(cmetric)

    unqlabmap = np.zeros((len(unqy), len(unqy)))
    unqlabmap[:,0] = unqy.copy()
    clv = 1

    while clv < len(unqy):
        alldist = -np.ones((len(lb1), len(allQIn)))
        for n in range(len(allQIn)):
            for lct in range(len(lb1)):
                cdx = np.logical_or(
                    np.logical_and(alb1[n]==lb1[lct], alb2[n]==lb2[lct]),
                    np.logical_and(alb1[n]==lb2[lct], alb2[n]==lb1[lct]))
                alldist[lct, n] = aggsamples(allmetrics[n][cdx])
        combdist = aggruns(alldist)
        idagg = np.argmin(combdist)
        agg1, agg2 = lb1[idagg], lb2[idagg]
        unqy[unqy==agg1] = clustlab; unqy[unqy==agg2] = clustlab
        for n in range(len(allQIn)):
            alb1[n][alb1[n]==agg1] = clustlab; alb1[n][alb1[n]==agg2] = clustlab
            alb2[n][alb2[n]==agg1] = clustlab; alb2[n][alb2[n]==agg2] = clustlab
        unqlabmap[:,clv] = unqy.copy()
        nunq = np.array(list(set(unqy.tolist())), dtype=int)
        lb1, lb2 = labcombs(nunq.copy())
        Zmatr[clv-1, 0] = agg1; Zmatr[clv-1, 1] = agg2
        Zmatr[clv-1, 2] = combdist[idagg]
        Zmatr[clv-1, 3] = np.sum(nunq==clustlab)
        clustlab += 1; clv += 1

    for clv in range(1, unqlabmap.shape[1]):
        unqlab = sorted(set(unqlabmap[:,clv].tolist()))
        ymap = dict(zip(unqlab, range(len(unqlab))))
        for src, dst in ymap.items():
            unqlabmap[unqlabmap[:,clv]==src, clv] = dst
    return unqlabmap, Zmatr