"""
Created on Sat Apr 3 18:39:34 2021
Last modified on Thurs March 31 09:16:00 2022

@author: 
Original matlab code for graph filtration: Hyekyoung Lee
Translation to python and addition of visualization code: Sungwoo Bae with the help of matlab2python

"""
import numpy as np
from scipy import sparse


# The algorithm is identical to networkx python module
def extract_connected_nodes(edge_list, sel_node_idx):
    '''
    ## Extract node indices of a connected component which contains a selected node
    (Algorith is identical to python networkx _plain_bfs function)
    ### Input
    edge_list: list containing array of all nodes connected with each node
    idx: index of the selected node (from 0 to node number)

    ### Output
    set of nodes constituting the connected components containing the selected node
    '''
    cc_set = set()
    next_neighbor = {sel_node_idx}
    while next_neighbor:
        curr_neighbor = next_neighbor
        next_neighbor = set()
        # For each vertex among current neighbors of the selected node (n-hop), find the next neighbor node and add (n+1-hop)
        # Iterate until all nodes in a connected component are found and all the vertices in the next_neighbor is empty
        for vertex in curr_neighbor:
            if vertex not in cc_set:
                cc_set.add(vertex)
                next_neighbor.update(edge_list[vertex])
    return cc_set


def connected_components_generator(A):
    '''
    ## Generator for returning connected components for the given adjacency matrix  
    (Algorith is identical to python networkx connected_components function)
    ### Input
    A: sparse matrix for spatial adjacency matrix across spots/grids (0 and 1)

    ### Output
    set of nodes constituting the connected components 
    '''
    all_cc_set = set()
    edge_list = [A[vertex].tocoo().col for vertex in range(A.shape[0])]
    # Generate the new connected component only if the vertex provided is not included in the previously generated connected components
    for vertex in range(A.shape[0]):
        if vertex not in all_cc_set:
            cc_set = extract_connected_nodes(edge_list, vertex)
            all_cc_set.update(cc_set)
            yield cc_set


def make_original_dendrogram_cc(U, A, threshold): 
    '''
    ## Compute original dendrogram with connected components
    ### Input
    U: gene expression profiles of a feature across p spots (p * 1 array)
    A: sparse matrix for spatial adjacency matrix across spots/grids (0 and 1)
    threshold: threshold value for U
    
    ### Output
    CC: connected components, ncc-list, each element is a index of spatial spots/grids
    E: ncc-by-ncc sparse matrix connectivity matrix between CCs
    duration: ncc-by-2 array, the birth and death of CCs
    history: ncc-list, each element has the index from which the CC come
    '''
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

        # Extract connected components for the adjacency matrix (containing voxels between the two threshold values)
        CC_profiles = [cc for cc in connected_components_generator(A[index_x,index_y])]
        S = len(CC_profiles)
        
        nCC = [[]]*S
        nA = sparse.dok_matrix((p,S))
        neighbor_cc = np.array([])
        for j in range(S):
            nCC[j] = cvoxels[list(CC_profiles[j])]
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
            CC_profiles = [cc for cc in connected_components_generator(nA)]
            S = len(CC_profiles)
            
            for j in range(S):
                tind = np.array(list(CC_profiles[j]))
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
