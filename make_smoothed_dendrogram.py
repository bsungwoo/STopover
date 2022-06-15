import numpy as np
from scipy import sparse
    
def make_smoothed_dendrogram(cCC,cE,cduration,chistory,lim_size=np.array([0,np.inf])):
    
    max_size = lim_size[1]
    min_size = lim_size[0]
    
    p = max(map(lambda x: max(x), cCC))
    ncc = len(cCC)
    length_duration = cduration[:,0].T - cduration[:,1].T
    length_cc = np.array(list(map(lambda x: len(x), cCC)))

    # Layer of dendrogram
    layer = []
    # find CCs with no parent
    length_history = np.array(list(map(lambda x: len(x), chistory)))
    # Leaf CCs
    ind_past = np.where(length_history == 0)[0]
    layer.append(ind_past.tolist())

    while len(ind_past) < ncc:
        tind = np.array(list(map(lambda x: len(np.intersect1d(np.array(x),ind_past)) == len(x), chistory)))
        ttind = np.setdiff1d(np.where(tind == 1)[0], ind_past)
        if len(ttind) != 0:
            layer.append(np.unique(ttind).tolist())
            ind_past = np.concatenate((ind_past,ttind))
    
    # initialization
    nCC = cCC
    nduration = cduration
    nchildren = chistory
    nE = cE
    nparent = -np.ones(ncc, dtype=int)
    ilayer = []

    for i in range(ncc):
        if len(nchildren[i]) != 0:
            for j in range(len(nchildren[i])):
                nparent[nchildren[i][j]] = i
        # Add the index of elements in 'layer' which includes more than 0 number of i
        # Presumed to have only one nonzero element for each i (**)
        ilayer.append(np.nonzero(list(map(lambda x: sum(map(lambda y: y==i, x)), layer)))[0][0])
    
    # Delete CCs of which size is smaller than min_size
    ck_delete = np.zeros(ncc, dtype=int)
    for i in range(len(layer)):
        for j in range(len(layer[i])):
            ii = layer[i][j]
            if ii != 0:
                if (length_cc[ii] <  min_size) & (ck_delete[ii] == 0):
                    if nparent[ii] != -1:
                        # find sisters and brothers
                        jj = np.array(nchildren[nparent[ii]])
                        ck = np.array([1 if (e >= min_size) else 0 for e in length_cc[jj]])
                    else:
                        ck = np.array([ncc + 1], dtype=int)
                    
                    if sum(ck) <= 1:
                        # All the children come back into the parent's belly
                        ii = nparent[ii]
                        if sum(ck) == 1:
                            tind = jj[np.where(ck == 1)[0]][0]
                            nchildren[ii] = nchildren[tind]
                            nparent[nchildren[ii]] = ii
                            nduration[ii,:] = np.array([np.max(nduration[np.append(ii,tind),0]), 
                                                        np.min(nduration[np.append(ii,tind),1])])
                        else:
                            nduration[ii,:] = np.array([np.max(nduration[np.append(ii,jj),0]), 
                                                        np.min(nduration[np.append(ii,jj),1])])
                            nchildren[ii] = []
                        nE[ii,:] = 0
                        nE[:,ii] = 0
                        nE[ii,ii] = nduration[ii,0]
                        length_duration[ii] = nduration[ii,0] - nduration[i,1]

                        # delete all children of my parent
                        delete_list = np.array(jj)
                        for k in range(len(jj)):
                            if ck[k] == 0:
                                ind_notdelete = np.where(ck_delete == 0)[0]
                                ind_children = ind_notdelete[np.where(list(map(lambda x: len(np.setdiff1d(nCC[x],nCC[jj[k]]))==0, ind_notdelete)))[0]]
                                delete_list = np.append(delete_list, ind_children)
                        jj = np.unique(delete_list)

                        ck_delete[jj] = 1
                        for k in range(len(jj)):
                            nCC[jj[k]] = []
                            nchildren[jj[k]] = []
                            nparent[jj[k]] = 0
                            nE[jj[k],:] = 0
                            nE[:,jj[k]] = 0
                            nduration[jj[k],:] = 0
                            length_cc[jj[k]] = 0
                            length_duration[jj[k]] = 0
                            layer[ilayer[jj[k]]] = [0 if e==jj[k] else e for e in layer[ilayer[jj[k]]]]
                    else:
                        ck_delete[ii] = 1
                        if sum(ck) <= ncc:
                            nchildren[nparent[ii]] = np.setdiff1d(nchildren[nparent[ii]],ii).tolist()
                        nCC[ii] = []
                        nchildren[ii] = []
                        nparent[ii] = 0
                        nE[ii,:] = 0
                        nE[:,ii] = 0
                        nduration[ii,:] = 0
                        length_cc[ii] = 0
                        length_duration[ii] = 0
                        layer[ilayer[ii]] = [0 if e==ii else e for e in layer[ilayer[ii]]]
    
# Layer update
# Estimate the depth of dendrogram
    layer = []
    length_history = list(map(lambda x: len(x), nchildren))
    
    ind_notempty = np.where(np.sum(nduration, axis=1) != 0)[0]
    ind_empty = np.setdiff1d(range(len(nchildren)), ind_notempty)
    ind_past = np.setdiff1d(np.where(length_history == 0)[0], ind_empty)
    layer.append(ind_past.tolist())

    while len(ind_past) < len(ind_notempty):
        tind = np.array(list(map(lambda x: 1 if len(np.intersect1d(x, ind_past)) == len(x) else 0, nchildren)))
        ttind = np.setdiff1d(np.where(tind == 1)[0], np.concatenate((ind_past,ind_empty)))
        if len(ttind) != 0:
            layer.append(ttind.tolist())
            ind_past = np.concatenate((ind_past, ttind))
    
    length_duration = nduration[:,0].T - nduration[:,1].T
    length_cc = list(map(lambda x: len(x), nCC))

    sval_ind = dict(sorted(zip(range(len(length_duration)), length_duration), key=lambda x: x[1], reverse=True))
    sval = np.array(list(sval_ind.values()))
    sind = np.array(list(sval_ind.keys()))

    tind = np.where(sval > 0)[0]
    sval = sval[tind]
    sind = sind[tind]
    tval = np.max(length_cc)
    tind = np.argmax(length_cc)
    sind = np.append(np.setdiff1d(sind,tind), tind)
    
    # Select CCs with the longest duration
    while len(sind) != 0:
        ii = sind[0]
        jj = ind_notempty[np.where([True if len(np.setdiff1d(nCC[e],nCC[ii])) == 0 else False for e in ind_notempty])[0]]
        jj = np.setdiff1d(jj,ii)
        iparent = ind_notempty[np.where([True if len(np.setdiff1d(nCC[e],nCC[ii])) < len(nCC[e]) else False for e in ind_notempty])[0]]
        iparent = np.setdiff1d(iparent, np.append(jj,ii))
        
        # Select ii
        nduration[ii,:] = np.array([np.max(nduration[np.append(ii,jj),0]), np.min(nduration[np.append(ii,jj),1])])
        nchildren[ii] = []
        nE[ii,:] = 0
        nE[:,ii] = 0
        nE[ii,ii] = nduration[ii,0]
        # delete all children of my parent
        for k in range(len(jj)):
            nCC[jj[k]] = []
            nchildren[jj[k]] = []
            nparent[jj[k]] = 0
            nE[jj[k],:] = 0
            nE[:,jj[k]] = 0
            nduration[jj[k],:] = 0
        sind = np.array([e for e in sind if e not in np.concatenate((iparent,np.append(jj,ii)))])
        sind = np.unique(sind)
        ind_notempty = np.setdiff1d(ind_notempty,jj)

    
    return nCC,nE,nduration,nchildren
