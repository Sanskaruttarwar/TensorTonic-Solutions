def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """

    if len(set_a) == 0 and len(set_b) == 0:
        return 0.0
    
    set1 = set(set_a)
    set2 = set(set_b)

    inst = set1.intersection(set2)
    uni = set1.union(set2)

    return len(inst)/len(uni)

    