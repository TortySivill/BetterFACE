

def basic_rules(feature, value_from, value_to):
    """

    Args:
        feature: feature name
        value_from: feature value of start point
        value_to: feature value of end point

    Returns: Binary 1 is transition if permissible, 0 if not

    """
    if feature == 'age':
        return 0 if value_to < value_from else 1
    elif feature == 'education':
        return 0 if value_to < value_from or value_to - value_from > 1 else 1
    elif feature == 'sex':
        return 0 if value_from != value_to else 1
    elif feature == 'weekly-hours':
        return 0 if value_to - value_from > 5 else 1
    elif 'race' in feature:
        return 0 if value_from != value_to else 1
    else:
        return 1

def tom_test_rules(feature, value_from, value_to):
    """

    Args:
        feature: feature name
        value_from: feature value of start point
        value_to: feature value of end point

    Returns: Binary 1 is transition if permissible, 0 if not

    """
    if feature == 'age':
        return 0 if value_to < value_from else 1
    elif feature == 'education':
        return 0 if value_to < value_from or value_to - value_from > 1 else 1
    elif feature in {'sex', 'race'}:
        return 0 if value_from != value_to else 1
    elif feature == 'weekly-hours':
        return 0 if value_to - value_from > 1 else 1
    else:
        return 1
