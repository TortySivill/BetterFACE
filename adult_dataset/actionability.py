"""

Rules are written as lambda functions over transitions, with x0 = start point and x1 = end point.
NOTE: Rules should return True if the transition is ***not*** actionable.
NOTE: Add new rule at the end, ensuring dictionary key is higher than previous one.

Variable order: ["age","employment-type","education-num","occupation","race","sex","weekly-hours"]

See adult_categories.json and adult_udm_bin_edges.json for category order and bin definitions.

"""
rule_base = {

    # Age cannot decrease.
    0: lambda x0, x1: x1[0] < x0[0],

    # Age can only increase by one category.
    1: lambda x0, x1: x1[0] - x0[0] > 1, 

    # Employment type cannot change.
    2: lambda x0, x1: x1[1] != x0[1],

    # Education cannot decrease.
    3: lambda x0, x1: x1[2] < x0[2],

    # Cannot go to college if haven't done already.
    4: lambda x0, x1: x1[2] >= 9 and x0[2] < 9,

    # Cannot increase education by more than one class.
    5: lambda x0, x1: x1[2] - x0[2] > 1,

    # Occupation cannot change.
    6: lambda x0, x1: x1[3] != x0[3],

    # Cannot move between office work and manual labour.
    7: lambda x0, x1: (x0[3] in {0,2,8,10,11} and x1[3] in {1,3,4,5}) or (x1[3] in {0,2,8,10,11} and x0[3] in {1,3,4,5}),

    # Cannot move to a managerial job.
    8: lambda x0, x1: x0[3] != 2 and x1[3] == 2,

    # Cannot move to a professional specialty.
    9: lambda x0, x1: x0[3] != 8 and x1[3] == 8,
    
    # Cannot change occupation if above a certain age.
    10: lambda x0, x1: x0[0] > 6 and x1[3] != x0[3],

    # Cannot change occupation if not completed high school.
    11: lambda x0, x1: x0[2] < 8 and x1[3] != x0[3],

    # Fixed race.
    12: lambda x0, x1: x1[4] != x0[4],

    # Fixed sex.
    13: lambda x0, x1: x1[5] != x0[5],

    # Cannot change weekly hours by more than one category.
    14: lambda x0, x1: abs(x1[6] - x0[6]) > 1,

    # Cannot change more than two features.
    15: lambda x0, x1: (x0 != x1).sum() > 2,

}