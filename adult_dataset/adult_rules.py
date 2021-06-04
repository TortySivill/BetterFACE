

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
    for i, feature in enumerate(features):
        if feature == 'age':
            return 0 if point_to[i] < point_from[i] else 1
        elif feature == 'education':
            return 0 if point_to[i] < point_from[i] or point_to[i] - point_from[i] > 1 else 1
        elif feature in {'sex', 'race'}:
            return 0 if point_from[i] != point_to[i] else 1
        elif feature == 'weekly-hours':
            return 0 if point_to[i] - point_from[i] > 1 else 1
        else:
            return 1
