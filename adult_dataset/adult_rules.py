

def basic_rules(features, point_from, point_to):
    """

    Args:
        features: list of feature name
        point_from: numpy array
        point_to: numpy array

    Returns: Binary 1 is transition if permissible, 0 if not

    """
    for i, feature in enumerate(features):
        if feature == 'age':
            return 0 if point_to[i] < point_from[i] else 1
        elif feature == 'education':
            return 0 if point_to[i] < point_from[i] or point_to[i] - point_from[i] > 1 else 1
        elif feature == 'sex':
            return 0 if point_from[i] != point_to[i] else 1
        elif feature == 'weekly-hours':
            return 0 if point_to[i] - point_from[i] > 5 else 1
        elif 'race' in feature:
            return 0 if point_from[i] != point_to[i] else 1
        else:
            return 1


def tom_test_rules(features, point_from, point_to):
    """

    Args:
        features: list of feature name
        point_from: numpy array
        point_to: numpy array

    Returns: Binary 1 is transition if permissible, 0 if not

    """
    # Variables: ["age","employment-type","education-num","occupation","race","sex","weekly-hours"]

    if (point_from != point_to).sum() > 2: return 0 # Can only change a limited number of features at a time.

    if point_to[0] < point_from[0]: return 0 # Age
    if point_to[0] - point_from[0] > 1: return 0 

    # if point_from[1] != point_to[1]: return 0 # Employment type

    if point_to[2] < point_from[2]: return 0 # Education.
    # if point_to[2] >= 9 and point_from[2] < 9: return 0 
    if point_to[2] - point_from[2] > 1: return 0 
    
    # if point_from[3] != point_to[3]: return 0 # Occupation
    # if point_from[3] != 2 and point_to[3] == 2: return 0 # Can't move to managerial
    # if point_from[3] != 8 and point_to[3] == 8: return 0 # Can't move to professional specialty.
    # if point_from[3] in {0,2,8,10,11} and point_to[3] in {1,3,4,5}: return 0 # Can't move from office work to manual labour...
    # if point_from[3] in {1,3,4,5} and point_to[3] in {0,2,8,10,11}: return 0 # ...and vice versa.

    # if point_from[2] < 8 and point_from[3] != point_to[3]: return 0 # Can't change occupation if not completed high school.

    if point_from[4] != point_to[4]: return 0 # Race

    if point_from[5] != point_to[5]: return 0 # Sex

    if abs(point_to[6] - point_from[6]) > 1: return 0 # Weekly hours

    return 1