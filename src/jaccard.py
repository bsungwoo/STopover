def calculate_jaccard_matrix(cc_list, jaccard_type="weighted", num_workers=1, progress_callback=None):
    """
    Calculate Jaccard indices between all pairs of connected components.
    
    Parameters:
    -----------
    cc_list : list of numpy.ndarray
        List of connected components arrays.
    jaccard_type : str, optional
        Type of Jaccard index to calculate. Default is "weighted".
    num_workers : int, optional
        Number of worker threads. Default is 1.
    progress_callback : callable, optional
        Callback function to report progress.
        
    Returns:
    --------
    numpy.ndarray
        Matrix of Jaccard indices.
    """
    # Debug information
    print(f"Python: Starting calculate_jaccard_matrix with {len(cc_list)} components")
    if len(cc_list) > 0:
        print(f"Python: First component type: {type(cc_list[0])}, shape: {cc_list[0].shape}, dtype: {cc_list[0].dtype}")
        print(f"Python: First component sample values: {cc_list[0].flatten()[:10]}")
    
    # Check if cc_list is empty
    if len(cc_list) == 0:
        print("Python: Empty cc_list, returning empty matrix")
        return np.zeros((0, 0))
    
    # Ensure all arrays are numpy arrays with int dtype
    for i, cc in enumerate(cc_list):
        if not isinstance(cc, np.ndarray):
            print(f"Python: Converting component {i} to numpy array")
            cc_list[i] = np.array(cc, dtype=np.int32)
        elif cc.dtype != np.int32:
            print(f"Python: Converting component {i} dtype from {cc.dtype} to int32")
            cc_list[i] = cc.astype(np.int32)
    
    try:
        # Call the C++ function
        print("Python: Calling parallel_jaccard_composite")
        jaccard_indices = parallelize.parallel_jaccard_composite(
            cc_list, cc_list, jaccard_type, num_workers, progress_callback
        )
        print(f"Python: Received {len(jaccard_indices)} results from parallel_jaccard_composite")
        
        # Reshape the results into a matrix
        n = len(cc_list)
        jaccard_matrix = np.zeros((n, n))
        
        # Fill the matrix
        idx = 0
        for i in range(n):
            for j in range(n):
                if idx < len(jaccard_indices):
                    jaccard_matrix[i, j] = jaccard_indices[idx]
                    idx += 1
                else:
                    print(f"Python: Warning - not enough results to fill matrix at ({i},{j})")
        
        return jaccard_matrix
    except Exception as e:
        print(f"Python: Exception in calculate_jaccard_matrix: {str(e)}")
        import traceback
        traceback.print_exc()
        return np.zeros((len(cc_list), len(cc_list))) 