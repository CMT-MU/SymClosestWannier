# graphene.py
"""
input file for graphene.
"""
graphene = {
    "model": "graphene",  #  name of model.
    "group": 191,  # No. of space group.
    "cell": {"a": 2.435, "b": 2.435, "c": 1000},  # set large enough interlayer distance.
    #
    "site": {"C": ("[2/3,1/3,0]", "pz")},  # positions of C site and its orbital.
    "bond": [("C", "C", [1, 2, 3, 4, 5, 6])],  # C-C bonds up to 6th neighbors.
    #
    "spinful": False,  # spinless.
    #
    "k_point": {"Γ": "[0,0,0]", "K": "[1/3,1/3,0]", "M": "[1/2,0,0]"},
    "k_path": "Γ-M-K-Γ",
}
