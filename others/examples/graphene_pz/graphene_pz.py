# graphene_pz.py
"""
input file for graphene (pz orbital).
"""

graphene_pz = {
    "model": "graphene_pz",  # name of model.
    "group": 191,  # No. of space group.
    "cell": {"a": 2.435, "b": 2.435, "c": 1000},  # set large enough interlayer distance.
    #
    "site": {"C": ("[2/3,1/3,0]", "pz")},  # position of C site and its orbital.
    # C-C bonds up to 20th neighbors.
    "bond": [("C", "C", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])],
    #
    "spinful": False,  # spinless.
}
