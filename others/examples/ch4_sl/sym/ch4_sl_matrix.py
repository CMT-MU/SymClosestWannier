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
ch4_sl = {'model': 'ch4_sl', 'molecule': True, 'group': ('Td', 'point group No. 31 : Td / -43m'), 'dimension': 8, 'ket': ['s@C_1', 'px@C_1', 'py@C_1', 'pz@C_1', 's@H_1', 's@H_2', 's@H_3', 's@H_4'], 'cell_site': {'C_1': ('[0, 0, 0]', '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]'), 'H_1': ('[1/3, 1/3, 1/3]', '[1,5,9,16,17,18]'), 'H_2': ('[-1/3, -1/3, 1/3]', '[2,6,11,13,21,23]'), 'H_3': ('[1/3, -1/3, -1/3]', '[3,7,12,15,19,24]'), 'H_4': ('[-1/3, 1/3, -1/3]', '[4,8,10,14,20,22]')}, 'version': '1.2.11', 'matrix': {'z_001': {(0, 0, 0, 0, 0): '1'}, 'z_002': {(0, 0, 0, 1, 1): 'sqrt(3)/3', (0, 0, 0, 2, 2): 'sqrt(3)/3', (0, 0, 0, 3, 3): 'sqrt(3)/3'}, 'z_003': {(0, 0, 0, 4, 4): '1/2', (0, 0, 0, 5, 5): '1/2', (0, 0, 0, 6, 6): '1/2', (0, 0, 0, 7, 7): '1/2'}, 'z_004': {(0, 0, 0, 4, 0): 'sqrt(2)/4', (0, 0, 0, 0, 4): 'sqrt(2)/4', (0, 0, 0, 5, 0): 'sqrt(2)/4', (0, 0, 0, 0, 5): 'sqrt(2)/4', (0, 0, 0, 6, 0): 'sqrt(2)/4', (0, 0, 0, 0, 6): 'sqrt(2)/4', (0, 0, 0, 7, 0): 'sqrt(2)/4', (0, 0, 0, 0, 7): 'sqrt(2)/4'}, 'z_005': {(0, 0, 0, 4, 1): 'sqrt(6)/12', (0, 0, 0, 1, 4): 'sqrt(6)/12', (0, 0, 0, 5, 1): '-sqrt(6)/12', (0, 0, 0, 1, 5): '-sqrt(6)/12', (0, 0, 0, 6, 1): 'sqrt(6)/12', (0, 0, 0, 1, 6): 'sqrt(6)/12', (0, 0, 0, 7, 1): '-sqrt(6)/12', (0, 0, 0, 1, 7): '-sqrt(6)/12', (0, 0, 0, 4, 2): 'sqrt(6)/12', (0, 0, 0, 2, 4): 'sqrt(6)/12', (0, 0, 0, 5, 2): '-sqrt(6)/12', (0, 0, 0, 2, 5): '-sqrt(6)/12', (0, 0, 0, 6, 2): '-sqrt(6)/12', (0, 0, 0, 2, 6): '-sqrt(6)/12', (0, 0, 0, 7, 2): 'sqrt(6)/12', (0, 0, 0, 2, 7): 'sqrt(6)/12', (0, 0, 0, 4, 3): 'sqrt(6)/12', (0, 0, 0, 3, 4): 'sqrt(6)/12', (0, 0, 0, 5, 3): 'sqrt(6)/12', (0, 0, 0, 3, 5): 'sqrt(6)/12', (0, 0, 0, 6, 3): '-sqrt(6)/12', (0, 0, 0, 3, 6): '-sqrt(6)/12', (0, 0, 0, 7, 3): '-sqrt(6)/12', (0, 0, 0, 3, 7): '-sqrt(6)/12'}, 'z_006': {(0, 0, 0, 5, 4): 'sqrt(3)/6', (0, 0, 0, 4, 5): 'sqrt(3)/6', (0, 0, 0, 7, 6): 'sqrt(3)/6', (0, 0, 0, 6, 7): 'sqrt(3)/6', (0, 0, 0, 6, 4): 'sqrt(3)/6', (0, 0, 0, 4, 6): 'sqrt(3)/6', (0, 0, 0, 7, 5): 'sqrt(3)/6', (0, 0, 0, 5, 7): 'sqrt(3)/6', (0, 0, 0, 7, 4): 'sqrt(3)/6', (0, 0, 0, 4, 7): 'sqrt(3)/6', (0, 0, 0, 6, 5): 'sqrt(3)/6', (0, 0, 0, 5, 6): 'sqrt(3)/6'}, 'z_007': {(0, 0, 0, 1, 1): '-sqrt(6)/6', (0, 0, 0, 2, 2): '-sqrt(6)/6', (0, 0, 0, 3, 3): 'sqrt(6)/3'}, 'z_008': {(0, 0, 0, 1, 1): 'sqrt(2)/2', (0, 0, 0, 2, 2): '-sqrt(2)/2'}, 'z_009': {(0, 0, 0, 4, 1): '-sqrt(3)/12', (0, 0, 0, 1, 4): '-sqrt(3)/12', (0, 0, 0, 5, 1): 'sqrt(3)/12', (0, 0, 0, 1, 5): 'sqrt(3)/12', (0, 0, 0, 6, 1): '-sqrt(3)/12', (0, 0, 0, 1, 6): '-sqrt(3)/12', (0, 0, 0, 7, 1): 'sqrt(3)/12', (0, 0, 0, 1, 7): 'sqrt(3)/12', (0, 0, 0, 4, 2): '-sqrt(3)/12', (0, 0, 0, 2, 4): '-sqrt(3)/12', (0, 0, 0, 5, 2): 'sqrt(3)/12', (0, 0, 0, 2, 5): 'sqrt(3)/12', (0, 0, 0, 6, 2): 'sqrt(3)/12', (0, 0, 0, 2, 6): 'sqrt(3)/12', (0, 0, 0, 7, 2): '-sqrt(3)/12', (0, 0, 0, 2, 7): '-sqrt(3)/12', (0, 0, 0, 4, 3): 'sqrt(3)/6', (0, 0, 0, 3, 4): 'sqrt(3)/6', (0, 0, 0, 5, 3): 'sqrt(3)/6', (0, 0, 0, 3, 5): 'sqrt(3)/6', (0, 0, 0, 6, 3): '-sqrt(3)/6', (0, 0, 0, 3, 6): '-sqrt(3)/6', (0, 0, 0, 7, 3): '-sqrt(3)/6', (0, 0, 0, 3, 7): '-sqrt(3)/6'}, 'z_010': {(0, 0, 0, 4, 1): '1/4', (0, 0, 0, 1, 4): '1/4', (0, 0, 0, 5, 1): '-1/4', (0, 0, 0, 1, 5): '-1/4', (0, 0, 0, 6, 1): '1/4', (0, 0, 0, 1, 6): '1/4', (0, 0, 0, 7, 1): '-1/4', (0, 0, 0, 1, 7): '-1/4', (0, 0, 0, 4, 2): '-1/4', (0, 0, 0, 2, 4): '-1/4', (0, 0, 0, 5, 2): '1/4', (0, 0, 0, 2, 5): '1/4', (0, 0, 0, 6, 2): '1/4', (0, 0, 0, 2, 6): '1/4', (0, 0, 0, 7, 2): '-1/4', (0, 0, 0, 2, 7): '-1/4'}, 'z_011': {(0, 0, 0, 5, 4): '-sqrt(6)/6', (0, 0, 0, 4, 5): '-sqrt(6)/6', (0, 0, 0, 7, 6): '-sqrt(6)/6', (0, 0, 0, 6, 7): '-sqrt(6)/6', (0, 0, 0, 6, 4): 'sqrt(6)/12', (0, 0, 0, 4, 6): 'sqrt(6)/12', (0, 0, 0, 7, 5): 'sqrt(6)/12', (0, 0, 0, 5, 7): 'sqrt(6)/12', (0, 0, 0, 7, 4): 'sqrt(6)/12', (0, 0, 0, 4, 7): 'sqrt(6)/12', (0, 0, 0, 6, 5): 'sqrt(6)/12', (0, 0, 0, 5, 6): 'sqrt(6)/12'}, 'z_012': {(0, 0, 0, 6, 4): '-sqrt(2)/4', (0, 0, 0, 4, 6): '-sqrt(2)/4', (0, 0, 0, 7, 5): '-sqrt(2)/4', (0, 0, 0, 5, 7): '-sqrt(2)/4', (0, 0, 0, 7, 4): 'sqrt(2)/4', (0, 0, 0, 4, 7): 'sqrt(2)/4', (0, 0, 0, 6, 5): 'sqrt(2)/4', (0, 0, 0, 5, 6): 'sqrt(2)/4'}, 'z_013': {(0, 0, 0, 4, 2): '1/4', (0, 0, 0, 2, 4): '1/4', (0, 0, 0, 5, 2): '1/4', (0, 0, 0, 2, 5): '1/4', (0, 0, 0, 6, 2): '-1/4', (0, 0, 0, 2, 6): '-1/4', (0, 0, 0, 7, 2): '-1/4', (0, 0, 0, 2, 7): '-1/4', (0, 0, 0, 4, 3): '-1/4', (0, 0, 0, 3, 4): '-1/4', (0, 0, 0, 5, 3): '1/4', (0, 0, 0, 3, 5): '1/4', (0, 0, 0, 6, 3): '1/4', (0, 0, 0, 3, 6): '1/4', (0, 0, 0, 7, 3): '-1/4', (0, 0, 0, 3, 7): '-1/4'}, 'z_014': {(0, 0, 0, 4, 1): '-1/4', (0, 0, 0, 1, 4): '-1/4', (0, 0, 0, 5, 1): '-1/4', (0, 0, 0, 1, 5): '-1/4', (0, 0, 0, 6, 1): '1/4', (0, 0, 0, 1, 6): '1/4', (0, 0, 0, 7, 1): '1/4', (0, 0, 0, 1, 7): '1/4', (0, 0, 0, 4, 3): '1/4', (0, 0, 0, 3, 4): '1/4', (0, 0, 0, 5, 3): '-1/4', (0, 0, 0, 3, 5): '-1/4', (0, 0, 0, 6, 3): '1/4', (0, 0, 0, 3, 6): '1/4', (0, 0, 0, 7, 3): '-1/4', (0, 0, 0, 3, 7): '-1/4'}, 'z_015': {(0, 0, 0, 4, 1): '1/4', (0, 0, 0, 1, 4): '1/4', (0, 0, 0, 5, 1): '-1/4', (0, 0, 0, 1, 5): '-1/4', (0, 0, 0, 6, 1): '-1/4', (0, 0, 0, 1, 6): '-1/4', (0, 0, 0, 7, 1): '1/4', (0, 0, 0, 1, 7): '1/4', (0, 0, 0, 4, 2): '-1/4', (0, 0, 0, 2, 4): '-1/4', (0, 0, 0, 5, 2): '1/4', (0, 0, 0, 2, 5): '1/4', (0, 0, 0, 6, 2): '-1/4', (0, 0, 0, 2, 6): '-1/4', (0, 0, 0, 7, 2): '1/4', (0, 0, 0, 2, 7): '1/4'}, 'z_016': {(0, 0, 0, 0, 1): 'sqrt(2)/2', (0, 0, 0, 1, 0): 'sqrt(2)/2'}, 'z_017': {(0, 0, 0, 0, 2): 'sqrt(2)/2', (0, 0, 0, 2, 0): 'sqrt(2)/2'}, 'z_018': {(0, 0, 0, 0, 3): 'sqrt(2)/2', (0, 0, 0, 3, 0): 'sqrt(2)/2'}, 'z_019': {(0, 0, 0, 2, 3): 'sqrt(2)/2', (0, 0, 0, 3, 2): 'sqrt(2)/2'}, 'z_020': {(0, 0, 0, 1, 3): 'sqrt(2)/2', (0, 0, 0, 3, 1): 'sqrt(2)/2'}, 'z_021': {(0, 0, 0, 1, 2): 'sqrt(2)/2', (0, 0, 0, 2, 1): 'sqrt(2)/2'}, 'z_022': {(0, 0, 0, 4, 4): '1/2', (0, 0, 0, 5, 5): '-1/2', (0, 0, 0, 6, 6): '1/2', (0, 0, 0, 7, 7): '-1/2'}, 'z_023': {(0, 0, 0, 4, 4): '1/2', (0, 0, 0, 5, 5): '-1/2', (0, 0, 0, 6, 6): '-1/2', (0, 0, 0, 7, 7): '1/2'}, 'z_024': {(0, 0, 0, 4, 4): '1/2', (0, 0, 0, 5, 5): '1/2', (0, 0, 0, 6, 6): '-1/2', (0, 0, 0, 7, 7): '-1/2'}, 'z_025': {(0, 0, 0, 4, 0): 'sqrt(2)/4', (0, 0, 0, 0, 4): 'sqrt(2)/4', (0, 0, 0, 5, 0): '-sqrt(2)/4', (0, 0, 0, 0, 5): '-sqrt(2)/4', (0, 0, 0, 6, 0): 'sqrt(2)/4', (0, 0, 0, 0, 6): 'sqrt(2)/4', (0, 0, 0, 7, 0): '-sqrt(2)/4', (0, 0, 0, 0, 7): '-sqrt(2)/4'}, 'z_026': {(0, 0, 0, 4, 0): 'sqrt(2)/4', (0, 0, 0, 0, 4): 'sqrt(2)/4', (0, 0, 0, 5, 0): '-sqrt(2)/4', (0, 0, 0, 0, 5): '-sqrt(2)/4', (0, 0, 0, 6, 0): '-sqrt(2)/4', (0, 0, 0, 0, 6): '-sqrt(2)/4', (0, 0, 0, 7, 0): 'sqrt(2)/4', (0, 0, 0, 0, 7): 'sqrt(2)/4'}, 'z_027': {(0, 0, 0, 4, 0): 'sqrt(2)/4', (0, 0, 0, 0, 4): 'sqrt(2)/4', (0, 0, 0, 5, 0): 'sqrt(2)/4', (0, 0, 0, 0, 5): 'sqrt(2)/4', (0, 0, 0, 6, 0): '-sqrt(2)/4', (0, 0, 0, 0, 6): '-sqrt(2)/4', (0, 0, 0, 7, 0): '-sqrt(2)/4', (0, 0, 0, 0, 7): '-sqrt(2)/4'}, 'z_028': {(0, 0, 0, 4, 1): 'sqrt(2)/4', (0, 0, 0, 1, 4): 'sqrt(2)/4', (0, 0, 0, 5, 1): 'sqrt(2)/4', (0, 0, 0, 1, 5): 'sqrt(2)/4', (0, 0, 0, 6, 1): 'sqrt(2)/4', (0, 0, 0, 1, 6): 'sqrt(2)/4', (0, 0, 0, 7, 1): 'sqrt(2)/4', (0, 0, 0, 1, 7): 'sqrt(2)/4'}, 'z_029': {(0, 0, 0, 4, 2): 'sqrt(2)/4', (0, 0, 0, 2, 4): 'sqrt(2)/4', (0, 0, 0, 5, 2): 'sqrt(2)/4', (0, 0, 0, 2, 5): 'sqrt(2)/4', (0, 0, 0, 6, 2): 'sqrt(2)/4', (0, 0, 0, 2, 6): 'sqrt(2)/4', (0, 0, 0, 7, 2): 'sqrt(2)/4', (0, 0, 0, 2, 7): 'sqrt(2)/4'}, 'z_030': {(0, 0, 0, 4, 3): 'sqrt(2)/4', (0, 0, 0, 3, 4): 'sqrt(2)/4', (0, 0, 0, 5, 3): 'sqrt(2)/4', (0, 0, 0, 3, 5): 'sqrt(2)/4', (0, 0, 0, 6, 3): 'sqrt(2)/4', (0, 0, 0, 3, 6): 'sqrt(2)/4', (0, 0, 0, 7, 3): 'sqrt(2)/4', (0, 0, 0, 3, 7): 'sqrt(2)/4'}, 'z_031': {(0, 0, 0, 4, 2): '1/4', (0, 0, 0, 2, 4): '1/4', (0, 0, 0, 5, 2): '1/4', (0, 0, 0, 2, 5): '1/4', (0, 0, 0, 6, 2): '-1/4', (0, 0, 0, 2, 6): '-1/4', (0, 0, 0, 7, 2): '-1/4', (0, 0, 0, 2, 7): '-1/4', (0, 0, 0, 4, 3): '1/4', (0, 0, 0, 3, 4): '1/4', (0, 0, 0, 5, 3): '-1/4', (0, 0, 0, 3, 5): '-1/4', (0, 0, 0, 6, 3): '-1/4', (0, 0, 0, 3, 6): '-1/4', (0, 0, 0, 7, 3): '1/4', (0, 0, 0, 3, 7): '1/4'}, 'z_032': {(0, 0, 0, 4, 1): '1/4', (0, 0, 0, 1, 4): '1/4', (0, 0, 0, 5, 1): '1/4', (0, 0, 0, 1, 5): '1/4', (0, 0, 0, 6, 1): '-1/4', (0, 0, 0, 1, 6): '-1/4', (0, 0, 0, 7, 1): '-1/4', (0, 0, 0, 1, 7): '-1/4', (0, 0, 0, 4, 3): '1/4', (0, 0, 0, 3, 4): '1/4', (0, 0, 0, 5, 3): '-1/4', (0, 0, 0, 3, 5): '-1/4', (0, 0, 0, 6, 3): '1/4', (0, 0, 0, 3, 6): '1/4', (0, 0, 0, 7, 3): '-1/4', (0, 0, 0, 3, 7): '-1/4'}, 'z_033': {(0, 0, 0, 4, 1): '1/4', (0, 0, 0, 1, 4): '1/4', (0, 0, 0, 5, 1): '-1/4', (0, 0, 0, 1, 5): '-1/4', (0, 0, 0, 6, 1): '-1/4', (0, 0, 0, 1, 6): '-1/4', (0, 0, 0, 7, 1): '1/4', (0, 0, 0, 1, 7): '1/4', (0, 0, 0, 4, 2): '1/4', (0, 0, 0, 2, 4): '1/4', (0, 0, 0, 5, 2): '-1/4', (0, 0, 0, 2, 5): '-1/4', (0, 0, 0, 6, 2): '1/4', (0, 0, 0, 2, 6): '1/4', (0, 0, 0, 7, 2): '-1/4', (0, 0, 0, 2, 7): '-1/4'}, 'z_034': {(0, 0, 0, 6, 4): '1/2', (0, 0, 0, 4, 6): '1/2', (0, 0, 0, 7, 5): '-1/2', (0, 0, 0, 5, 7): '-1/2'}, 'z_035': {(0, 0, 0, 7, 4): '1/2', (0, 0, 0, 4, 7): '1/2', (0, 0, 0, 6, 5): '-1/2', (0, 0, 0, 5, 6): '-1/2'}, 'z_036': {(0, 0, 0, 5, 4): '1/2', (0, 0, 0, 4, 5): '1/2', (0, 0, 0, 7, 6): '-1/2', (0, 0, 0, 6, 7): '-1/2'}}}