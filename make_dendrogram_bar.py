"""
Created on Sat Apr 3 18:39:34 2021
Last modified on Thurs March 31 09:16:00 2022

@author: 
Original matlab code for graph filtration: Hye Kyung Lee
Translation to python and addition of visualization code: Sungwoo Bae with the help of matlab2python

"""
import numpy as np
    
def make_dendrogram_bar(history,duration,cvertical_x=None,cvertical_y=None,chorizontal_x=None,chorizontal_y=None,cdots=None): 
    if (cvertical_x is None) and (cvertical_y is None) and (chorizontal_x is None) \
            and (chorizontal_y is None) and (cdots is None):
        is_new = 1
    else:
        is_new = 0
    
    ncc = duration.shape[0]

    # Estimate the depth of dendrogram
    nlayer = []
    # Find CCs with no parent
    length_history = np.array(list(map(lambda x: len(x), history)))
    ind_notempty = np.where(np.sum(duration, axis=1) != 0)[0]
    ind_empty = np.setdiff1d([range(len(history))], ind_notempty)
    ind_past = np.setdiff1d(np.where(length_history == 0)[0], ind_empty)
    nlayer.append(ind_past.tolist())

    while len(ind_past) < len(ind_notempty):
        tind = np.array(list(map(lambda x: len(np.intersect1d(x, ind_past)) == len(x), history)))
        ttind = np.setdiff1d(np.where(tind)[0], np.concatenate((ind_past,ind_empty)))
        if len(ttind) != 0:
            nlayer.append(ttind.tolist())
            ind_past = np.concatenate((ind_past, ttind))
    
    if is_new == 1:
        # Estimate bars in a dendrogram
        nvertical_x = np.zeros((ncc,2))
        nvertical_y = np.zeros((ncc,2))
        nhorizontal_x = np.zeros((ncc,2))
        nhorizontal_y = np.zeros((ncc,2))
        ndots = np.zeros((ncc,2))
        
        sval_ind = dict(sorted(zip(range(len(duration[nlayer[0],1])), duration[nlayer[0],1]), key=lambda x: x[1], reverse=True))
        sval = np.array(list(sval_ind.values()))
        sind = np.array(list(sval_ind.keys())).astype(int)
        sind = np.array(nlayer[0])[sind]

        for i in range(len(sind)):
            ii = sind[i]
            nvertical_x[ii,:] = np.array([i,i])
            nvertical_y[ii,:] = np.array([duration[ii,0], duration[ii,1]])
            ndots[ii,:] = np.array([i, duration[ii,0]])

        for i in range(1,len(nlayer)):
            for j in range(len(nlayer[i])):
                tx = nvertical_x[history[nlayer[i][j]], 0]
                if len(tx)>0: 
                    nvertical_x[nlayer[i][j],:] = np.mean(tx) * np.ones((1,2))
                    nhorizontal_x[nlayer[i][j],:] = [np.min(tx), np.max(tx)]
                    ndots[nlayer[i][j],0] = np.mean(tx)
                ndots[nlayer[i][j],1] = duration[nlayer[i][j],0]
                nvertical_y[nlayer[i][j],:] = duration[nlayer[i][j],:]
                nhorizontal_y[nlayer[i][j],:] = duration[nlayer[i][j],0] * np.ones((1,2))

    else:
        ncc = duration.shape[0]

        nvertical_x = cvertical_x
        nvertical_y = cvertical_y
        nhorizontal_x = chorizontal_x
        nhorizontal_y = chorizontal_y
        ndots = cdots

        nvertical_x[ind_empty,:] = 0
        nvertical_y[ind_empty,:] = 0
        nhorizontal_x[ind_empty,:] = 0
        nhorizontal_y[ind_empty,:] = 0
        ndots[ind_empty,:] = 0

        for j in range(len(nlayer[0])):
            ii = nlayer[0][j]
            nvertical_y[ii,:] = np.sort(duration[ii,:])
            nhorizontal_x[ii,:] = 0
            nhorizontal_y[ii,:] = 0
            ndots[ii,:] = np.array([nvertical_x[ii,0], nvertical_y[ii,1]])
        
        for i in range(1,len(nlayer)):
            for j in range(len(nlayer[i])):
                ii = nlayer[i][j]
                tx = nvertical_x[history[ii],0]
                if len(tx)>0:
                    nvertical_x[ii,:] = np.mean(tx) * np.ones((1,2))
                    nhorizontal_x[ii,:] = [np.min(tx), np.max(tx)]
                    ndots[ii,0] = np.mean(tx)
                ndots[ii,1] = duration[ii,0]
                nvertical_y[ii,:] = duration[ii,:]
                nhorizontal_y[ii,:] = duration[ii,0] * np.ones((1,2))
    
    return nvertical_x,nvertical_y,nhorizontal_x,nhorizontal_y,ndots,nlayer
