# srvo3_d.py
"""
input file for SrVO3 (V t2g orbitals).
"""

srvo3_d = {
    "model": "srvo3_d",  # name of model.
    "group": 221,  # No. of space group.
    "cell": {"a": 3.8409, "b": 3.8409, "c": 3.8409},
    #
    # V site at the origin; it is at [1/2,1/2,1/2] in srvo3_d.win, which is equivalent.
    "site": {"V": ("[0,0,0]", ["dyz", "dxz", "dxy"])},  # position of V site and its t2g orbitals.
    # V-V bonds up to 30th neighbors.
    "bond": [
        (
            "V",
            "V",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        ),
    ],
    #
    "spinful": False,  # spinless.
}
