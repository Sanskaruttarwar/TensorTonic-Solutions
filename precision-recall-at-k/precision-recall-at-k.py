def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    prec = 0.
    rec = 0.
    
    top_k = recommended[:k]
    count = sum(item in relevant for item in top_k)

    if count == 0:
        return [prec,rec]

    prec = count/k
    rec = count/len(relevant)

    return [prec,rec]

    