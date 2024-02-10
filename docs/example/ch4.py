# ch4.py
"""
input file for ch4 molecule.
"""
ch4 = {
    "model": "ch4",  # name of model.
    "group": "Td",  # name of point group.
    "cell": {"c": 10},  # set large enough interlayer distance.
    #
    "site": {"C": ("[0,0,0]", "s p"), "H": ("[1/3,1/3,1/3]", "s")},  # positions of C and H sites and their orbitals.
    "bond": [("C", "H", 1), ("H", "H", 1)],  # nearest-neighbor C-H and H-H bonds.
    #
    "spinful": True,  # spinful.
}
