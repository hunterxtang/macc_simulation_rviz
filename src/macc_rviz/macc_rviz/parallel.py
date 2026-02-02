import numpy as np

def find_parallel_groups(substructures, dependencies):
    """
    Given substructures and their dependencies (edges a->b),
    return groups of substructures that can be built in parallel.
    """
    num = len(substructures)
    indeg = {i:0 for i in range(num)}
    for a,b in dependencies:
        indeg[b]+=1

    ready = [i for i in indeg if indeg[i]==0]
    groups = []

    while ready:
        groups.append(ready)
        next_ready = []
        for r in ready:
            # remove r from graph
            for a,b in dependencies:
                if a==r:
                    indeg[b]-=1
                    if indeg[b]==0:
                        next_ready.append(b)
        ready = next_ready
    return groups
