# srvo3_dp.py
"""
input file for SrVO3 (V t2g and O p orbitals).
"""

srvo3_dp = {
    "model": "srvo3_dp",  # name of model.
    "group": 221,  # No. of space group.
    "cell": {"a": 3.8409, "b": 3.8409, "c": 3.8409},
    #
    # V and O sites at the positions in srvo3_dp.win.
    "site": {"V": ("[1/2,1/2,1/2]", ["dyz", "dxz", "dxy"]), "O": ("[0,1/2,1/2]", "p")},
    # V-V, O-O and V-O bonds up to 30th neighbors.
    # fmt: off
    "bond": [
        ("V", "V", [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30,
        ]),
        ("O", "O", [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30,
        ]),
        ("V", "O", [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30,
        ]),
    ],
    # fmt: on
    #
    "spinful": False,  # spinless.
}
