"""
Created on Sat Apr 3 18:39:34 2021
Last modified on Thurs March 31 09:16:00 2022

@author: 
Original matlab code for graph filtration: Hye Kyung Lee
Translation to python and addition of visualization code: Sungwoo Bae with the help of matlab2python

"""
import numpy as np
from scipy import sparse
import graph_tool.all as gt

def make_original_dendrogram_cc(U, A, threshold): 
# CC: connected components, ncc-list, each element is a set of voxels
# E: ncc-by-ncc sparse matrix connectivity matrix between CCs
# duration: ncc-by-2 array, the birth and death of CCs
# history: ncc-list, each element has the index from which the CC come
    
    p = len(U)
    CC = [[]]*p
    E = sparse.dok_matrix((p,p))
    ncc = -1
    ck_cc = -np.ones(p, dtype=int)
    duration = np.zeros((p,2))
    history = [[]]*p

    for i in range(len(threshold)):
        # Choose current voxels that satisfy threshold interval   
        if i == 0:
            cvoxels = np.where(U >= threshold[i])[0]
        else:
            cvoxels = np.where((U >= threshold[i]) & (U < threshold[i-1]))[0]
        # Define pairwise index arrays
        index_x, index_y = np.meshgrid(cvoxels, cvoxels, indexing='ij')

        # Create directed graph and extract strongly connected components
        G = gt.Graph(directed=True)
        G.add_vertex(len(cvoxels))
        G.add_edge_list(np.transpose(A[index_x,index_y].nonzero()))
        comp, hist = gt.label_components(G, directed=True)
        CC_profiles = comp.a
        S = max(CC_profiles) + 1
        
        nCC = [[]]*S
        nA = sparse.dok_matrix((p,S))
        neighbor_cc = np.array([])
        for j in range(S):
            nCC[j] = cvoxels[np.where(CC_profiles==j)[0]]
            # neighbors of current voxels 
            neighbor_voxels = np.where(np.sum(A[nCC[j],:].toarray(), axis=0) > 0)[0]
            # CC index to which neighbors belong (to differentiate null value with )
            tcc = np.setdiff1d(np.unique(ck_cc[neighbor_voxels]), -1)
            nA[tcc,j] = 1
            neighbor_cc = np.concatenate((neighbor_cc, tcc))
        neighbor_cc = np.unique(neighbor_cc).astype(int)
        
        if len(neighbor_cc) == 0:
            for j in range(S):
                ncc += 1
                CC[ncc] = np.sort(nCC[j]).tolist()
                ck_cc[CC[ncc]] = ncc
                duration[ncc,0] = threshold[i]
                E[ncc,ncc] = threshold[i]
                history[ncc] = []
        else:
            nA_tmp = sparse.dok_matrix((S+len(neighbor_cc), S+len(neighbor_cc)))
            nA_tmp.setdiag(1)
            nA_tmp[:len(neighbor_cc),len(neighbor_cc):] = nA[neighbor_cc,:]
            nA_tmp[len(neighbor_cc):,:len(neighbor_cc)] = nA[neighbor_cc,:].T
            nA = nA_tmp.copy()

            # Estimate connected components of clusters
            G = gt.Graph(directed=True)
            G.add_vertex(S+len(neighbor_cc))
            G.add_edge_list(np.transpose(nA.nonzero()))
            comp, hist = gt.label_components(G, directed=True)
            CC_profiles = comp.a
            S = max(CC_profiles) + 1
            
            for j in range(S):
                tind = np.where(CC_profiles==j)[0]
                tind1 = neighbor_cc[tind[np.where(tind < len(neighbor_cc))]]
                tind2 = tind[np.where(tind >= len(neighbor_cc))] - len(neighbor_cc)
                
                if len(tind1) == 1:
                    nCC_tind2 = [ee for e in tind2 for ee in nCC[e]]
                    CC[tind1[0]] = list(set(CC[tind1[0]]+nCC_tind2))
                    ck_cc[CC[tind1[0]]] = tind1[0]
                else:
                    ncc += 1
                    CC_tind1 = [ee for e in tind1 for ee in CC[e]]
                    nCC_tind2 = [ee for e in tind2 for ee in nCC[e]]
                    CC[ncc] = list(set(CC_tind1+nCC_tind2))
                    ck_cc[CC[ncc]] = ncc
                    duration[ncc,0] = threshold[i]
                    duration[tind1,1] = threshold[i]

                    E_mod = np.eye(len(tind1))*E[tind1,:][:,tind1].toarray() + (1 - np.eye(len(tind1))) * threshold[i]
                    for ind, e in enumerate(tind1):
                        E[tind1,e] = E_mod[:,ind]
                    
                    E[ncc,tind1] = threshold[i]
                    E[tind1,ncc] = threshold[i]
                    E[ncc,ncc] = threshold[i]
                    history[ncc] = tind1.tolist()
    
    # Remove the empty list from the end
    for index, cc in enumerate(reversed(CC)):
        if len(cc) == 0: continue
        else:
            rev_count = -index
            break
    
    CC = CC[:rev_count]
    history = history[:rev_count]
    E = E[:rev_count,:rev_count].copy()
    duration = duration[:rev_count,:]

    return CC,E,duration,history
