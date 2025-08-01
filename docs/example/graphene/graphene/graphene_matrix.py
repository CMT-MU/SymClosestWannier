"""
== SAMB in full matrix form in real space (* only for crystal) ===
- model : model name.
- molecule : molecule or crystal ?
- group : (tag, detailed str)
- dimension : dimension of full matrix
- ket : ket basis list, orbital@site
- version : MultiPie version
- k_point* : representative k points
- k_path* : high-symmetry line in k space
- cell_site : { name_idx(pset): (position, SOs) }
- A* : transform matrix, [a1,a2,a3]
- matrix : { "z_#": "matrix"}
"""

graphene = {
    "model": "graphene",
    "molecule": False,
    "group": ("D6h^1", "space group No. 191 : D6h^1 / P6/mmm : PG D6h"),
    "dimension": 2,
    "ket": ["pz@C_1", "pz@C_2"],
    "cell_site": {
        "C_1": ("[2/3, 1/3, 0]", "[1,6,7,8,9,10,14,15,16,17,23,24]"),
        "C_2": ("[1/3, 2/3, 0]", "[2,3,4,5,11,12,13,18,19,20,21,22]"),
    },
    "version": "1.3.3",
    "k_point": {"Γ": "[0,0,0]", "K": "[1/3,1/3,0]", "M": "[1/2,0,0]"},
    "k_path": "Γ-M-K-Γ",
    "A": "[[2.435, -1.2175, 6e-14], [0.0, 2.10877185821511, 1.1e-13], [0.0, 0.0, 1000.0000000000001]]",
    "matrix": {
        "z_001": {(0, 0, 0, 0, 0): "sqrt(2)/2", (0, 0, 0, 1, 1): "sqrt(2)/2"},
        "z_002": {
            (-1, 0, 0, 1, 0): "sqrt(6)/6",
            (1, 0, 0, 0, 1): "sqrt(6)/6",
            (0, 0, 0, 1, 0): "sqrt(6)/6",
            (0, 0, 0, 0, 1): "sqrt(6)/6",
            (0, 1, 0, 1, 0): "sqrt(6)/6",
            (0, -1, 0, 0, 1): "sqrt(6)/6",
        },
        "z_003": {
            (-1, 0, 0, 0, 0): "sqrt(3)/6",
            (1, 0, 0, 0, 0): "sqrt(3)/6",
            (-1, 0, 0, 1, 1): "sqrt(3)/6",
            (1, 0, 0, 1, 1): "sqrt(3)/6",
            (-1, -1, 0, 1, 1): "sqrt(3)/6",
            (1, 1, 0, 1, 1): "sqrt(3)/6",
            (0, -1, 0, 1, 1): "sqrt(3)/6",
            (0, 1, 0, 1, 1): "sqrt(3)/6",
            (-1, -1, 0, 0, 0): "sqrt(3)/6",
            (1, 1, 0, 0, 0): "sqrt(3)/6",
            (0, -1, 0, 0, 0): "sqrt(3)/6",
            (0, 1, 0, 0, 0): "sqrt(3)/6",
        },
        "z_004": {
            (-1, -1, 0, 1, 0): "sqrt(6)/6",
            (1, 1, 0, 0, 1): "sqrt(6)/6",
            (-1, 1, 0, 1, 0): "sqrt(6)/6",
            (1, -1, 0, 0, 1): "sqrt(6)/6",
            (1, 1, 0, 1, 0): "sqrt(6)/6",
            (-1, -1, 0, 0, 1): "sqrt(6)/6",
        },
        "z_005": {
            (-2, -1, 0, 1, 0): "sqrt(3)/6",
            (2, 1, 0, 0, 1): "sqrt(3)/6",
            (0, -1, 0, 1, 0): "sqrt(3)/6",
            (0, 1, 0, 0, 1): "sqrt(3)/6",
            (-2, 0, 0, 1, 0): "sqrt(3)/6",
            (2, 0, 0, 0, 1): "sqrt(3)/6",
            (1, 2, 0, 1, 0): "sqrt(3)/6",
            (-1, -2, 0, 0, 1): "sqrt(3)/6",
            (1, 0, 0, 1, 0): "sqrt(3)/6",
            (-1, 0, 0, 0, 1): "sqrt(3)/6",
            (0, 2, 0, 1, 0): "sqrt(3)/6",
            (0, -2, 0, 0, 1): "sqrt(3)/6",
        },
        "z_006": {
            (-1, -2, 0, 0, 0): "sqrt(3)/6",
            (1, 2, 0, 0, 0): "sqrt(3)/6",
            (1, 2, 0, 1, 1): "sqrt(3)/6",
            (-1, -2, 0, 1, 1): "sqrt(3)/6",
            (1, -1, 0, 1, 1): "sqrt(3)/6",
            (-1, 1, 0, 1, 1): "sqrt(3)/6",
            (-2, -1, 0, 1, 1): "sqrt(3)/6",
            (2, 1, 0, 1, 1): "sqrt(3)/6",
            (-1, 1, 0, 0, 0): "sqrt(3)/6",
            (1, -1, 0, 0, 0): "sqrt(3)/6",
            (2, 1, 0, 0, 0): "sqrt(3)/6",
            (-2, -1, 0, 0, 0): "sqrt(3)/6",
        },
        "z_007": {
            (-2, -2, 0, 0, 0): "sqrt(3)/6",
            (2, 2, 0, 0, 0): "sqrt(3)/6",
            (-2, -2, 0, 1, 1): "sqrt(3)/6",
            (2, 2, 0, 1, 1): "sqrt(3)/6",
            (0, -2, 0, 1, 1): "sqrt(3)/6",
            (0, 2, 0, 1, 1): "sqrt(3)/6",
            (-2, 0, 0, 1, 1): "sqrt(3)/6",
            (2, 0, 0, 1, 1): "sqrt(3)/6",
            (0, -2, 0, 0, 0): "sqrt(3)/6",
            (0, 2, 0, 0, 0): "sqrt(3)/6",
            (-2, 0, 0, 0, 0): "sqrt(3)/6",
            (2, 0, 0, 0, 0): "sqrt(3)/6",
        },
    },
}
