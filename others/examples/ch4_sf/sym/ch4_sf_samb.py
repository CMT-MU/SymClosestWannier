"""
=== SAMB (* only for crystal with fourier transform) ===
- info
    - atomic : { "M_#" : ["amp_#"] }
    - site_cluster : { "S_#" : ["smp_#"] }
    - bond_cluster : { "B_#" : ["bmp_#"] }
    - uniform : { "S_#"/"B_#" : ["ump_#"] }
    - structure* : { "B_#" : ["kmp_#"] }
    - Z : { ("M_#", "S_#"/"B_#") : ["z_#"] }
    - version : MultiPie version
    - harmonics : { head : [TagMultipole] }

- data
    - atomic : { "amp_#" : ( TagMultipole, shape, [(i, j, matrix element)] ) }
    - site_cluster : { "smp_#" : ( TagMultipole, [vector component] ) }
    - bond_cluster : { "bmp_#" : ( TagMultipole, [vector component] ) }
    - uniform* : { "ump_#" : ( TagMultipole, shape, [(i, j, matrix element)] ) }
    - structure* : { "kmp_#" : (TagMultipole, "structure factor") }
    - Z : {"z_#" : ( TagMultipole, [(coeff, "amp_#", "smp_#"/"bmp_#/ump_#")] ) }
    - Zk* : {"z_#" : ( TagMultipole, [(coeff, "amp_#", "ump_#", "kmp_#")] ) }
"""
ch4_sf = {'info': {'atomic': {'M_001': ['amp_001', 'amp_002', 'amp_003', 'amp_004'], 'M_002': ['amp_005', 'amp_006', 'amp_007', 'amp_008', 'amp_009', 'amp_010', 'amp_011', 'amp_012', 'amp_013', 'amp_014', 'amp_015', 'amp_016'], 'M_003': ['amp_017', 'amp_018', 'amp_019', 'amp_020', 'amp_021', 'amp_022', 'amp_023', 'amp_024', 'amp_025', 'amp_026', 'amp_027', 'amp_028', 'amp_029', 'amp_030', 'amp_031']}, 'site_cluster': {'S_001': ['smp_001'], 'S_002': ['smp_002', 'smp_003', 'smp_004', 'smp_005']}, 'bond_cluster': {'B_001': ['bmp_006', 'bmp_007', 'bmp_008', 'bmp_009', 'bmp_010', 'bmp_011', 'bmp_012', 'bmp_013'], 'B_002': ['bmp_014', 'bmp_015', 'bmp_016', 'bmp_017', 'bmp_018', 'bmp_019', 'bmp_020', 'bmp_021', 'bmp_022', 'bmp_023', 'bmp_024', 'bmp_025']}, 'uniform': {'S_001': ['ump_001'], 'S_002': ['ump_002', 'ump_003', 'ump_004', 'ump_005'], 'B_001': ['ump_006', 'ump_007', 'ump_008', 'ump_009', 'ump_010', 'ump_011', 'ump_012', 'ump_013'], 'B_002': ['ump_014', 'ump_015', 'ump_016', 'ump_017', 'ump_018', 'ump_019', 'ump_020', 'ump_021', 'ump_022', 'ump_023', 'ump_024', 'ump_025']}, 'Z': {('A1', 'M_001', 'S_001'): ['z_001'], ('A1', 'M_003', 'S_001'): ['z_002', 'z_003'], ('A1', 'M_001', 'S_002'): ['z_004'], ('A1', 'M_001', 'B_001'): ['z_005'], ('A1', 'M_002', 'B_001'): ['z_006', 'z_007'], ('A1', 'M_001', 'B_002'): ['z_008', 'z_009'], ('A2', 'M_002', 'S_001'): ['z_010'], ('A2', 'M_001', 'B_001'): ['z_011'], ('A2', 'M_002', 'B_001'): ['z_012', 'z_013'], ('A2', 'M_001', 'B_002'): ['z_014'], ('E', 'M_002', 'S_001'): ['z_015', 'z_016'], ('E', 'M_003', 'S_001'): ['z_017', 'z_018', 'z_019', 'z_020'], ('E', 'M_001', 'B_001'): ['z_021', 'z_022'], ('E', 'M_002', 'B_001'): ['z_023', 'z_024', 'z_025', 'z_026', 'z_027', 'z_028', 'z_029', 'z_030'], ('E', 'M_001', 'B_002'): ['z_031', 'z_032', 'z_033', 'z_034', 'z_035', 'z_036'], ('T1', 'M_002', 'S_001'): ['z_037', 'z_038', 'z_039'], ('T1', 'M_003', 'S_001'): ['z_040', 'z_041', 'z_042'], ('T1', 'M_001', 'B_001'): ['z_043', 'z_044', 'z_045', 'z_046', 'z_047', 'z_048'], ('T1', 'M_002', 'B_001'): ['z_049', 'z_050', 'z_051', 'z_052', 'z_053', 'z_054', 'z_055', 'z_056', 'z_057', 'z_058', 'z_059', 'z_060', 'z_061', 'z_062', 'z_063', 'z_064', 'z_065', 'z_066'], ('T1', 'M_001', 'B_002'): ['z_067', 'z_068', 'z_069', 'z_070', 'z_071', 'z_072'], ('T2', 'M_002', 'S_001'): ['z_073', 'z_074', 'z_075', 'z_076', 'z_077', 'z_078'], ('T2', 'M_003', 'S_001'): ['z_079', 'z_080', 'z_081', 'z_082', 'z_083', 'z_084'], ('T2', 'M_001', 'S_002'): ['z_085', 'z_086', 'z_087'], ('T2', 'M_001', 'B_001'): ['z_088', 'z_089', 'z_090', 'z_091', 'z_092', 'z_093'], ('T2', 'M_002', 'B_001'): ['z_094', 'z_095', 'z_096', 'z_097', 'z_098', 'z_099', 'z_100', 'z_101', 'z_102', 'z_103', 'z_104', 'z_105', 'z_106', 'z_107', 'z_108', 'z_109', 'z_110', 'z_111'], ('T2', 'M_001', 'B_002'): ['z_112', 'z_113', 'z_114', 'z_115', 'z_116', 'z_117', 'z_118', 'z_119', 'z_120']}, 'version': '1.2.11', 'harmonics': {'Q': ['Qh(0,A1,,)', 'Qh(1,T2,,0)', 'Qh(1,T2,,1)', 'Qh(1,T2,,2)', 'Qh(2,E,,0)', 'Qh(2,E,,1)', 'Qh(2,T2,,0)', 'Qh(2,T2,,1)', 'Qh(2,T2,,2)', 'Qh(3,T1,,0)', 'Qh(3,T1,,1)', 'Qh(3,T1,,2)'], 'G': ['Gh(0,A2,,)', 'Gh(1,T1,,0)', 'Gh(1,T1,,1)', 'Gh(1,T1,,2)', 'Gh(2,E,,0)', 'Gh(2,E,,1)', 'Gh(2,T1,,0)', 'Gh(2,T1,,1)', 'Gh(2,T1,,2)']}}, 'data': {'atomic': {'amp_001': ('Qa(0,A1,,)', (2, 2), [(0, 0, 'sqrt(2)/2'), (1, 1, 'sqrt(2)/2')]), 'amp_002': ('Ma(1,T1,,0|1,-1)', (2, 2), [(0, 1, 'sqrt(2)/2'), (1, 0, 'sqrt(2)/2')]), 'amp_003': ('Ma(1,T1,,1|1,-1)', (2, 2), [(0, 1, '-sqrt(2)*I/2'), (1, 0, 'sqrt(2)*I/2')]), 'amp_004': ('Ma(1,T1,,2|1,-1)', (2, 2), [(0, 0, 'sqrt(2)/2'), (1, 1, '-sqrt(2)/2')]), 'amp_005': ('Qa(1,T2,,0)', (2, 6), [(0, 0, 'sqrt(2)/2'), (1, 1, 'sqrt(2)/2')]), 'amp_006': ('Qa(1,T2,,1)', (2, 6), [(0, 2, 'sqrt(2)/2'), (1, 3, 'sqrt(2)/2')]), 'amp_007': ('Qa(1,T2,,2)', (2, 6), [(0, 4, 'sqrt(2)/2'), (1, 5, 'sqrt(2)/2')]), 'amp_008': ('Qa(1,T2,,0|1,0)', (2, 6), [(0, 2, '-I/2'), (0, 5, '1/2'), (1, 3, 'I/2'), (1, 4, '-1/2')]), 'amp_009': ('Qa(1,T2,,1|1,0)', (2, 6), [(0, 0, 'I/2'), (0, 5, '-I/2'), (1, 1, '-I/2'), (1, 4, '-I/2')]), 'amp_010': ('Qa(1,T2,,2|1,0)', (2, 6), [(0, 1, '-1/2'), (0, 3, 'I/2'), (1, 0, '1/2'), (1, 2, 'I/2')]), 'amp_011': ('Ga(0,A2,,|1,1)', (2, 6), [(0, 1, 'sqrt(6)*I/6'), (0, 3, 'sqrt(6)/6'), (0, 4, 'sqrt(6)*I/6'), (1, 0, 'sqrt(6)*I/6'), (1, 2, '-sqrt(6)/6'), (1, 5, '-sqrt(6)*I/6')]), 'amp_012': ('Ga(2,E,,0|1,-1)', (2, 6), [(0, 1, '-sqrt(3)*I/6'), (0, 3, '-sqrt(3)/6'), (0, 4, 'sqrt(3)*I/3'), (1, 0, '-sqrt(3)*I/6'), (1, 2, 'sqrt(3)/6'), (1, 5, '-sqrt(3)*I/3')]), 'amp_013': ('Ga(2,E,,1|1,-1)', (2, 6), [(0, 1, 'I/2'), (0, 3, '-1/2'), (1, 0, 'I/2'), (1, 2, '1/2')]), 'amp_014': ('Ga(2,T1,,0|1,-1)', (2, 6), [(0, 2, 'I/2'), (0, 5, '1/2'), (1, 3, '-I/2'), (1, 4, '-1/2')]), 'amp_015': ('Ga(2,T1,,1|1,-1)', (2, 6), [(0, 0, 'I/2'), (0, 5, 'I/2'), (1, 1, '-I/2'), (1, 4, 'I/2')]), 'amp_016': ('Ga(2,T1,,2|1,-1)', (2, 6), [(0, 1, '1/2'), (0, 3, 'I/2'), (1, 0, '-1/2'), (1, 2, 'I/2')]), 'amp_017': ('Qa(0,A1,,)', (6, 6), [(0, 0, 'sqrt(6)/6'), (1, 1, 'sqrt(6)/6'), (2, 2, 'sqrt(6)/6'), (3, 3, 'sqrt(6)/6'), (4, 4, 'sqrt(6)/6'), (5, 5, 'sqrt(6)/6')]), 'amp_018': ('Qa(0,A1,,|1,1)', (6, 6), [(0, 2, '-sqrt(3)*I/6'), (0, 5, 'sqrt(3)/6'), (1, 3, 'sqrt(3)*I/6'), (1, 4, '-sqrt(3)/6'), (2, 0, 'sqrt(3)*I/6'), (2, 5, '-sqrt(3)*I/6'), (3, 1, '-sqrt(3)*I/6'), (3, 4, '-sqrt(3)*I/6'), (4, 1, '-sqrt(3)/6'), (4, 3, 'sqrt(3)*I/6'), (5, 0, 'sqrt(3)/6'), (5, 2, 'sqrt(3)*I/6')]), 'amp_019': ('Qa(2,E,,0)', (6, 6), [(0, 0, '-sqrt(3)/6'), (1, 1, '-sqrt(3)/6'), (2, 2, '-sqrt(3)/6'), (3, 3, '-sqrt(3)/6'), (4, 4, 'sqrt(3)/3'), (5, 5, 'sqrt(3)/3')]), 'amp_020': ('Qa(2,E,,1)', (6, 6), [(0, 0, '1/2'), (1, 1, '1/2'), (2, 2, '-1/2'), (3, 3, '-1/2')]), 'amp_021': ('Qa(2,T2,,0)', (6, 6), [(2, 4, '1/2'), (3, 5, '1/2'), (4, 2, '1/2'), (5, 3, '1/2')]), 'amp_022': ('Qa(2,T2,,1)', (6, 6), [(0, 4, '1/2'), (1, 5, '1/2'), (4, 0, '1/2'), (5, 1, '1/2')]), 'amp_023': ('Qa(2,T2,,2)', (6, 6), [(0, 2, '1/2'), (1, 3, '1/2'), (2, 0, '1/2'), (3, 1, '1/2')]), 'amp_024': ('Qa(2,E,,0|1,-1)', (6, 6), [(0, 2, '-sqrt(6)*I/6'), (0, 5, '-sqrt(6)/12'), (1, 3, 'sqrt(6)*I/6'), (1, 4, 'sqrt(6)/12'), (2, 0, 'sqrt(6)*I/6'), (2, 5, 'sqrt(6)*I/12'), (3, 1, '-sqrt(6)*I/6'), (3, 4, 'sqrt(6)*I/12'), (4, 1, 'sqrt(6)/12'), (4, 3, '-sqrt(6)*I/12'), (5, 0, '-sqrt(6)/12'), (5, 2, '-sqrt(6)*I/12')]), 'amp_025': ('Qa(2,E,,1|1,-1)', (6, 6), [(0, 5, '-sqrt(2)/4'), (1, 4, 'sqrt(2)/4'), (2, 5, '-sqrt(2)*I/4'), (3, 4, '-sqrt(2)*I/4'), (4, 1, 'sqrt(2)/4'), (4, 3, 'sqrt(2)*I/4'), (5, 0, '-sqrt(2)/4'), (5, 2, 'sqrt(2)*I/4')]), 'amp_026': ('Qa(2,T2,,0|1,-1)', (6, 6), [(0, 3, '-sqrt(2)/4'), (0, 4, 'sqrt(2)*I/4'), (1, 2, 'sqrt(2)/4'), (1, 5, '-sqrt(2)*I/4'), (2, 1, 'sqrt(2)/4'), (3, 0, '-sqrt(2)/4'), (4, 0, '-sqrt(2)*I/4'), (5, 1, 'sqrt(2)*I/4')]), 'amp_027': ('Qa(2,T2,,1|1,-1)', (6, 6), [(0, 3, '-sqrt(2)*I/4'), (1, 2, '-sqrt(2)*I/4'), (2, 1, 'sqrt(2)*I/4'), (2, 4, '-sqrt(2)*I/4'), (3, 0, 'sqrt(2)*I/4'), (3, 5, 'sqrt(2)*I/4'), (4, 2, 'sqrt(2)*I/4'), (5, 3, '-sqrt(2)*I/4')]), 'amp_028': ('Qa(2,T2,,2|1,-1)', (6, 6), [(0, 5, 'sqrt(2)*I/4'), (1, 4, 'sqrt(2)*I/4'), (2, 5, '-sqrt(2)/4'), (3, 4, 'sqrt(2)/4'), (4, 1, '-sqrt(2)*I/4'), (4, 3, 'sqrt(2)/4'), (5, 0, '-sqrt(2)*I/4'), (5, 2, '-sqrt(2)/4')]), 'amp_029': ('Ga(1,T1,,0|1,0)', (6, 6), [(0, 3, '-sqrt(2)/4'), (0, 4, '-sqrt(2)*I/4'), (1, 2, 'sqrt(2)/4'), (1, 5, 'sqrt(2)*I/4'), (2, 1, 'sqrt(2)/4'), (3, 0, '-sqrt(2)/4'), (4, 0, 'sqrt(2)*I/4'), (5, 1, '-sqrt(2)*I/4')]), 'amp_030': ('Ga(1,T1,,1|1,0)', (6, 6), [(0, 3, 'sqrt(2)*I/4'), (1, 2, 'sqrt(2)*I/4'), (2, 1, '-sqrt(2)*I/4'), (2, 4, '-sqrt(2)*I/4'), (3, 0, '-sqrt(2)*I/4'), (3, 5, 'sqrt(2)*I/4'), (4, 2, 'sqrt(2)*I/4'), (5, 3, '-sqrt(2)*I/4')]), 'amp_031': ('Ga(1,T1,,2|1,0)', (6, 6), [(0, 5, 'sqrt(2)*I/4'), (1, 4, 'sqrt(2)*I/4'), (2, 5, 'sqrt(2)/4'), (3, 4, '-sqrt(2)/4'), (4, 1, '-sqrt(2)*I/4'), (4, 3, '-sqrt(2)/4'), (5, 0, '-sqrt(2)*I/4'), (5, 2, 'sqrt(2)/4')])}, 'site_cluster': {'smp_001': ('Qs(0,A1,,)', '[1]'), 'smp_002': ('Qs(0,A1,,)', '[1/2, 1/2, 1/2, 1/2]'), 'smp_003': ('Qs(1,T2,,0)', '[1/2, -1/2, 1/2, -1/2]'), 'smp_004': ('Qs(1,T2,,1)', '[1/2, -1/2, -1/2, 1/2]'), 'smp_005': ('Qs(1,T2,,2)', '[1/2, 1/2, -1/2, -1/2]')}, 'bond_cluster': {'bmp_006': ('Qb(0,A1,,)', '[1/2, 1/2, 1/2, 1/2]'), 'bmp_007': ('Qb(1,T2,,0)', '[1/2, -1/2, 1/2, -1/2]'), 'bmp_008': ('Qb(1,T2,,1)', '[1/2, -1/2, -1/2, 1/2]'), 'bmp_009': ('Qb(1,T2,,2)', '[1/2, 1/2, -1/2, -1/2]'), 'bmp_010': ('Tb(0,A1,,)', '[I/2, I/2, I/2, I/2]'), 'bmp_011': ('Tb(1,T2,,0)', '[I/2, -I/2, I/2, -I/2]'), 'bmp_012': ('Tb(1,T2,,1)', '[I/2, -I/2, -I/2, I/2]'), 'bmp_013': ('Tb(1,T2,,2)', '[I/2, I/2, -I/2, -I/2]'), 'bmp_014': ('Qb(0,A1,,)', '[sqrt(6)/6, sqrt(6)/6, sqrt(6)/6, sqrt(6)/6, sqrt(6)/6, sqrt(6)/6]'), 'bmp_015': ('Qb(2,E,,0)', '[-sqrt(3)/3, -sqrt(3)/3, sqrt(3)/6, sqrt(3)/6, sqrt(3)/6, sqrt(3)/6]'), 'bmp_016': ('Qb(2,E,,1)', '[0, 0, -1/2, -1/2, 1/2, 1/2]'), 'bmp_017': ('Qb(2,T2,,0)', '[0, 0, sqrt(2)/2, -sqrt(2)/2, 0, 0]'), 'bmp_018': ('Qb(2,T2,,1)', '[0, 0, 0, 0, sqrt(2)/2, -sqrt(2)/2]'), 'bmp_019': ('Qb(2,T2,,2)', '[sqrt(2)/2, -sqrt(2)/2, 0, 0, 0, 0]'), 'bmp_020': ('Tb(1,T2,,0)', '[I/2, I/2, 0, 0, I/2, -I/2]'), 'bmp_021': ('Tb(1,T2,,1)', '[I/2, -I/2, I/2, -I/2, 0, 0]'), 'bmp_022': ('Tb(1,T2,,2)', '[0, 0, I/2, I/2, I/2, I/2]'), 'bmp_023': ('Tb(3,T1,,0)', '[I/2, I/2, 0, 0, -I/2, I/2]'), 'bmp_024': ('Tb(3,T1,,1)', '[-I/2, I/2, I/2, -I/2, 0, 0]'), 'bmp_025': ('Tb(3,T1,,2)', '[0, 0, -I/2, -I/2, I/2, I/2]')}, 'uniform': {'ump_001': ('Qs(0,A1,,)', (5, 5), [(0, 0, '1')]), 'ump_002': ('Qs(0,A1,,)', (5, 5), [(1, 1, '1/2'), (2, 2, '1/2'), (3, 3, '1/2'), (4, 4, '1/2')]), 'ump_003': ('Qs(1,T2,,0)', (5, 5), [(1, 1, '1/2'), (2, 2, '-1/2'), (3, 3, '1/2'), (4, 4, '-1/2')]), 'ump_004': ('Qs(1,T2,,1)', (5, 5), [(1, 1, '1/2'), (2, 2, '-1/2'), (3, 3, '-1/2'), (4, 4, '1/2')]), 'ump_005': ('Qs(1,T2,,2)', (5, 5), [(1, 1, '1/2'), (2, 2, '1/2'), (3, 3, '-1/2'), (4, 4, '-1/2')]), 'ump_006': ('Qu(0,A1,,)', (5, 5), [(0, 1, 'sqrt(2)/4'), (0, 2, 'sqrt(2)/4'), (0, 3, 'sqrt(2)/4'), (0, 4, 'sqrt(2)/4'), (1, 0, 'sqrt(2)/4'), (2, 0, 'sqrt(2)/4'), (3, 0, 'sqrt(2)/4'), (4, 0, 'sqrt(2)/4')]), 'ump_007': ('Qu(1,T2,,0)', (5, 5), [(0, 1, 'sqrt(2)/4'), (0, 2, '-sqrt(2)/4'), (0, 3, 'sqrt(2)/4'), (0, 4, '-sqrt(2)/4'), (1, 0, 'sqrt(2)/4'), (2, 0, '-sqrt(2)/4'), (3, 0, 'sqrt(2)/4'), (4, 0, '-sqrt(2)/4')]), 'ump_008': ('Qu(1,T2,,1)', (5, 5), [(0, 1, 'sqrt(2)/4'), (0, 2, '-sqrt(2)/4'), (0, 3, '-sqrt(2)/4'), (0, 4, 'sqrt(2)/4'), (1, 0, 'sqrt(2)/4'), (2, 0, '-sqrt(2)/4'), (3, 0, '-sqrt(2)/4'), (4, 0, 'sqrt(2)/4')]), 'ump_009': ('Qu(1,T2,,2)', (5, 5), [(0, 1, 'sqrt(2)/4'), (0, 2, 'sqrt(2)/4'), (0, 3, '-sqrt(2)/4'), (0, 4, '-sqrt(2)/4'), (1, 0, 'sqrt(2)/4'), (2, 0, 'sqrt(2)/4'), (3, 0, '-sqrt(2)/4'), (4, 0, '-sqrt(2)/4')]), 'ump_010': ('Tu(0,A1,,)', (5, 5), [(0, 1, '-sqrt(2)*I/4'), (0, 2, '-sqrt(2)*I/4'), (0, 3, '-sqrt(2)*I/4'), (0, 4, '-sqrt(2)*I/4'), (1, 0, 'sqrt(2)*I/4'), (2, 0, 'sqrt(2)*I/4'), (3, 0, 'sqrt(2)*I/4'), (4, 0, 'sqrt(2)*I/4')]), 'ump_011': ('Tu(1,T2,,0)', (5, 5), [(0, 1, '-sqrt(2)*I/4'), (0, 2, 'sqrt(2)*I/4'), (0, 3, '-sqrt(2)*I/4'), (0, 4, 'sqrt(2)*I/4'), (1, 0, 'sqrt(2)*I/4'), (2, 0, '-sqrt(2)*I/4'), (3, 0, 'sqrt(2)*I/4'), (4, 0, '-sqrt(2)*I/4')]), 'ump_012': ('Tu(1,T2,,1)', (5, 5), [(0, 1, '-sqrt(2)*I/4'), (0, 2, 'sqrt(2)*I/4'), (0, 3, 'sqrt(2)*I/4'), (0, 4, '-sqrt(2)*I/4'), (1, 0, 'sqrt(2)*I/4'), (2, 0, '-sqrt(2)*I/4'), (3, 0, '-sqrt(2)*I/4'), (4, 0, 'sqrt(2)*I/4')]), 'ump_013': ('Tu(1,T2,,2)', (5, 5), [(0, 1, '-sqrt(2)*I/4'), (0, 2, '-sqrt(2)*I/4'), (0, 3, 'sqrt(2)*I/4'), (0, 4, 'sqrt(2)*I/4'), (1, 0, 'sqrt(2)*I/4'), (2, 0, 'sqrt(2)*I/4'), (3, 0, '-sqrt(2)*I/4'), (4, 0, '-sqrt(2)*I/4')]), 'ump_014': ('Qu(0,A1,,)', (5, 5), [(1, 2, 'sqrt(3)/6'), (1, 3, 'sqrt(3)/6'), (1, 4, 'sqrt(3)/6'), (2, 1, 'sqrt(3)/6'), (2, 3, 'sqrt(3)/6'), (2, 4, 'sqrt(3)/6'), (3, 1, 'sqrt(3)/6'), (3, 2, 'sqrt(3)/6'), (3, 4, 'sqrt(3)/6'), (4, 1, 'sqrt(3)/6'), (4, 2, 'sqrt(3)/6'), (4, 3, 'sqrt(3)/6')]), 'ump_015': ('Qu(2,E,,0)', (5, 5), [(1, 2, '-sqrt(6)/6'), (1, 3, 'sqrt(6)/12'), (1, 4, 'sqrt(6)/12'), (2, 1, '-sqrt(6)/6'), (2, 3, 'sqrt(6)/12'), (2, 4, 'sqrt(6)/12'), (3, 1, 'sqrt(6)/12'), (3, 2, 'sqrt(6)/12'), (3, 4, '-sqrt(6)/6'), (4, 1, 'sqrt(6)/12'), (4, 2, 'sqrt(6)/12'), (4, 3, '-sqrt(6)/6')]), 'ump_016': ('Qu(2,E,,1)', (5, 5), [(1, 3, '-sqrt(2)/4'), (1, 4, 'sqrt(2)/4'), (2, 3, 'sqrt(2)/4'), (2, 4, '-sqrt(2)/4'), (3, 1, '-sqrt(2)/4'), (3, 2, 'sqrt(2)/4'), (4, 1, 'sqrt(2)/4'), (4, 2, '-sqrt(2)/4')]), 'ump_017': ('Qu(2,T2,,0)', (5, 5), [(1, 3, '1/2'), (2, 4, '-1/2'), (3, 1, '1/2'), (4, 2, '-1/2')]), 'ump_018': ('Qu(2,T2,,1)', (5, 5), [(1, 4, '1/2'), (2, 3, '-1/2'), (3, 2, '-1/2'), (4, 1, '1/2')]), 'ump_019': ('Qu(2,T2,,2)', (5, 5), [(1, 2, '1/2'), (2, 1, '1/2'), (3, 4, '-1/2'), (4, 3, '-1/2')]), 'ump_020': ('Tu(1,T2,,0)', (5, 5), [(1, 2, '-sqrt(2)*I/4'), (1, 4, '-sqrt(2)*I/4'), (2, 1, 'sqrt(2)*I/4'), (2, 3, 'sqrt(2)*I/4'), (3, 2, '-sqrt(2)*I/4'), (3, 4, '-sqrt(2)*I/4'), (4, 1, 'sqrt(2)*I/4'), (4, 3, 'sqrt(2)*I/4')]), 'ump_021': ('Tu(1,T2,,1)', (5, 5), [(1, 2, '-sqrt(2)*I/4'), (1, 3, '-sqrt(2)*I/4'), (2, 1, 'sqrt(2)*I/4'), (2, 4, 'sqrt(2)*I/4'), (3, 1, 'sqrt(2)*I/4'), (3, 4, 'sqrt(2)*I/4'), (4, 2, '-sqrt(2)*I/4'), (4, 3, '-sqrt(2)*I/4')]), 'ump_022': ('Tu(1,T2,,2)', (5, 5), [(1, 3, '-sqrt(2)*I/4'), (1, 4, '-sqrt(2)*I/4'), (2, 3, '-sqrt(2)*I/4'), (2, 4, '-sqrt(2)*I/4'), (3, 1, 'sqrt(2)*I/4'), (3, 2, 'sqrt(2)*I/4'), (4, 1, 'sqrt(2)*I/4'), (4, 2, 'sqrt(2)*I/4')]), 'ump_023': ('Tu(3,T1,,0)', (5, 5), [(1, 2, '-sqrt(2)*I/4'), (1, 4, 'sqrt(2)*I/4'), (2, 1, 'sqrt(2)*I/4'), (2, 3, '-sqrt(2)*I/4'), (3, 2, 'sqrt(2)*I/4'), (3, 4, '-sqrt(2)*I/4'), (4, 1, '-sqrt(2)*I/4'), (4, 3, 'sqrt(2)*I/4')]), 'ump_024': ('Tu(3,T1,,1)', (5, 5), [(1, 2, 'sqrt(2)*I/4'), (1, 3, '-sqrt(2)*I/4'), (2, 1, '-sqrt(2)*I/4'), (2, 4, 'sqrt(2)*I/4'), (3, 1, 'sqrt(2)*I/4'), (3, 4, '-sqrt(2)*I/4'), (4, 2, '-sqrt(2)*I/4'), (4, 3, 'sqrt(2)*I/4')]), 'ump_025': ('Tu(3,T1,,2)', (5, 5), [(1, 3, 'sqrt(2)*I/4'), (1, 4, '-sqrt(2)*I/4'), (2, 3, '-sqrt(2)*I/4'), (2, 4, 'sqrt(2)*I/4'), (3, 1, '-sqrt(2)*I/4'), (3, 2, 'sqrt(2)*I/4'), (4, 1, 'sqrt(2)*I/4'), (4, 2, '-sqrt(2)*I/4')])}, 'Z': {'z_001': ('Q(0,A1,,)', [('1', 'amp_001', 'ump_001')]), 'z_002': ('Q(0,A1,,)', [('1', 'amp_017', 'ump_001')]), 'z_003': ('Q(0,A1,,|1,1)', [('1', 'amp_018', 'ump_001')]), 'z_004': ('Q(0,A1,,)', [('1', 'amp_001', 'ump_002')]), 'z_005': ('Q(0,A1,,)', [('1', 'amp_001', 'ump_006')]), 'z_006': ('Q(0,A1,,)', [('sqrt(3)/3', 'amp_005', 'ump_007'), ('sqrt(3)/3', 'amp_006', 'ump_008'), ('sqrt(3)/3', 'amp_007', 'ump_009')]), 'z_007': ('Q(0,A1,,|1,0)', [('sqrt(3)/3', 'amp_008', 'ump_007'), ('sqrt(3)/3', 'amp_009', 'ump_008'), ('sqrt(3)/3', 'amp_010', 'ump_009')]), 'z_008': ('Q(0,A1,,)', [('1', 'amp_001', 'ump_014')]), 'z_009': ('Q(3,A1,,|1,-1)', [('sqrt(3)/3', 'amp_002', 'ump_023'), ('sqrt(3)/3', 'amp_003', 'ump_024'), ('sqrt(3)/3', 'amp_004', 'ump_025')]), 'z_010': ('G(0,A2,,|1,1)', [('1', 'amp_011', 'ump_001')]), 'z_011': ('G(0,A2,,|1,-1)', [('sqrt(3)/3', 'amp_002', 'ump_011'), ('sqrt(3)/3', 'amp_003', 'ump_012'), ('sqrt(3)/3', 'amp_004', 'ump_013')]), 'z_012': ('G(0,A2,,|1,1)', [('1', 'amp_011', 'ump_006')]), 'z_013': ('G(3,A2,,|1,-1)', [('sqrt(3)/3', 'amp_014', 'ump_007'), ('sqrt(3)/3', 'amp_015', 'ump_008'), ('sqrt(3)/3', 'amp_016', 'ump_009')]), 'z_014': ('G(0,A2,,|1,-1)', [('sqrt(3)/3', 'amp_002', 'ump_020'), ('sqrt(3)/3', 'amp_003', 'ump_021'), ('sqrt(3)/3', 'amp_004', 'ump_022')]), 'z_015': ('G(2,E,,0|1,-1)', [('1', 'amp_012', 'ump_001')]), 'z_016': ('G(2,E,,1|1,-1)', [('1', 'amp_013', 'ump_001')]), 'z_017': ('Q(2,E,,0)', [('1', 'amp_019', 'ump_001')]), 'z_018': ('Q(2,E,,1)', [('1', 'amp_020', 'ump_001')]), 'z_019': ('Q(2,E,,0|1,-1)', [('1', 'amp_024', 'ump_001')]), 'z_020': ('Q(2,E,,1|1,-1)', [('1', 'amp_025', 'ump_001')]), 'z_021': ('G(2,E,,0|1,-1)', [('-sqrt(6)/6', 'amp_002', 'ump_011'), ('-sqrt(6)/6', 'amp_003', 'ump_012'), ('sqrt(6)/3', 'amp_004', 'ump_013')]), 'z_022': ('G(2,E,,1|1,-1)', [('sqrt(2)/2', 'amp_002', 'ump_011'), ('-sqrt(2)/2', 'amp_003', 'ump_012')]), 'z_023': ('Q(2,E,,0)', [('-sqrt(6)/6', 'amp_005', 'ump_007'), ('-sqrt(6)/6', 'amp_006', 'ump_008'), ('sqrt(6)/3', 'amp_007', 'ump_009')]), 'z_024': ('Q(2,E,,1)', [('sqrt(2)/2', 'amp_005', 'ump_007'), ('-sqrt(2)/2', 'amp_006', 'ump_008')]), 'z_025': ('Q(2,E,,0|1,0)', [('-sqrt(6)/6', 'amp_008', 'ump_007'), ('-sqrt(6)/6', 'amp_009', 'ump_008'), ('sqrt(6)/3', 'amp_010', 'ump_009')]), 'z_026': ('Q(2,E,,1|1,0)', [('sqrt(2)/2', 'amp_008', 'ump_007'), ('-sqrt(2)/2', 'amp_009', 'ump_008')]), 'z_027': ('G(2,E,,0|1,-1)', [('1', 'amp_012', 'ump_006')]), 'z_028': ('G(2,E,,1|1,-1)', [('1', 'amp_013', 'ump_006')]), 'z_029': ('Q(2,E,,0|1,-1)', [('-sqrt(2)/2', 'amp_014', 'ump_007'), ('sqrt(2)/2', 'amp_015', 'ump_008')]), 'z_030': ('Q(2,E,,1|1,-1)', [('-sqrt(6)/6', 'amp_014', 'ump_007'), ('-sqrt(6)/6', 'amp_015', 'ump_008'), ('sqrt(6)/3', 'amp_016', 'ump_009')]), 'z_031': ('Q(2,E,,0)', [('1', 'amp_001', 'ump_015')]), 'z_032': ('Q(2,E,,1)', [('1', 'amp_001', 'ump_016')]), 'z_033': ('G(2,E,,0|1,-1)', [('-sqrt(6)/6', 'amp_002', 'ump_020'), ('-sqrt(6)/6', 'amp_003', 'ump_021'), ('sqrt(6)/3', 'amp_004', 'ump_022')]), 'z_034': ('G(2,E,,1|1,-1)', [('sqrt(2)/2', 'amp_002', 'ump_020'), ('-sqrt(2)/2', 'amp_003', 'ump_021')]), 'z_035': ('G(2,E,,0|1,-1)', [('-sqrt(2)/2', 'amp_002', 'ump_023'), ('sqrt(2)/2', 'amp_003', 'ump_024')]), 'z_036': ('G(2,E,,1|1,-1)', [('-sqrt(6)/6', 'amp_002', 'ump_023'), ('-sqrt(6)/6', 'amp_003', 'ump_024'), ('sqrt(6)/3', 'amp_004', 'ump_025')]), 'z_037': ('G(2,T1,,0|1,-1)', [('1', 'amp_014', 'ump_001')]), 'z_038': ('G(2,T1,,1|1,-1)', [('1', 'amp_015', 'ump_001')]), 'z_039': ('G(2,T1,,2|1,-1)', [('1', 'amp_016', 'ump_001')]), 'z_040': ('G(1,T1,,0|1,0)', [('1', 'amp_029', 'ump_001')]), 'z_041': ('G(1,T1,,1|1,0)', [('1', 'amp_030', 'ump_001')]), 'z_042': ('G(1,T1,,2|1,0)', [('1', 'amp_031', 'ump_001')]), 'z_043': ('G(1,T1,,0|1,-1)', [('1', 'amp_002', 'ump_010')]), 'z_044': ('G(1,T1,,1|1,-1)', [('1', 'amp_003', 'ump_010')]), 'z_045': ('G(1,T1,,2|1,-1)', [('1', 'amp_004', 'ump_010')]), 'z_046': ('G(2,T1,,0|1,-1)', [('sqrt(2)/2', 'amp_003', 'ump_013'), ('sqrt(2)/2', 'amp_004', 'ump_012')]), 'z_047': ('G(2,T1,,1|1,-1)', [('sqrt(2)/2', 'amp_002', 'ump_013'), ('sqrt(2)/2', 'amp_004', 'ump_011')]), 'z_048': ('G(2,T1,,2|1,-1)', [('sqrt(2)/2', 'amp_002', 'ump_012'), ('sqrt(2)/2', 'amp_003', 'ump_011')]), 'z_049': ('G(1,T1,,0)', [('sqrt(2)/2', 'amp_006', 'ump_009'), ('-sqrt(2)/2', 'amp_007', 'ump_008')]), 'z_050': ('G(1,T1,,1)', [('-sqrt(2)/2', 'amp_005', 'ump_009'), ('sqrt(2)/2', 'amp_007', 'ump_007')]), 'z_051': ('G(1,T1,,2)', [('sqrt(2)/2', 'amp_005', 'ump_008'), ('-sqrt(2)/2', 'amp_006', 'ump_007')]), 'z_052': ('G(1,T1,,0|1,0)', [('sqrt(2)/2', 'amp_009', 'ump_009'), ('-sqrt(2)/2', 'amp_010', 'ump_008')]), 'z_053': ('G(1,T1,,1|1,0)', [('-sqrt(2)/2', 'amp_008', 'ump_009'), ('sqrt(2)/2', 'amp_010', 'ump_007')]), 'z_054': ('G(1,T1,,2|1,0)', [('sqrt(2)/2', 'amp_008', 'ump_008'), ('-sqrt(2)/2', 'amp_009', 'ump_007')]), 'z_055': ('G(1,T1,,0|1,1)', [('1', 'amp_011', 'ump_007')]), 'z_056': ('G(1,T1,,1|1,1)', [('1', 'amp_011', 'ump_008')]), 'z_057': ('G(1,T1,,2|1,1)', [('1', 'amp_011', 'ump_009')]), 'z_058': ('G(2,T1,,0|1,-1)', [('1', 'amp_014', 'ump_006')]), 'z_059': ('G(2,T1,,1|1,-1)', [('1', 'amp_015', 'ump_006')]), 'z_060': ('G(2,T1,,2|1,-1)', [('1', 'amp_016', 'ump_006')]), 'z_061': ('G(1,T1,,0|1,-1)', [('-sqrt(10)/10', 'amp_012', 'ump_007'), ('sqrt(30)/10', 'amp_013', 'ump_007'), ('sqrt(30)/10', 'amp_015', 'ump_009'), ('sqrt(30)/10', 'amp_016', 'ump_008')]), 'z_062': ('G(1,T1,,1|1,-1)', [('-sqrt(10)/10', 'amp_012', 'ump_008'), ('-sqrt(30)/10', 'amp_013', 'ump_008'), ('sqrt(30)/10', 'amp_014', 'ump_009'), ('sqrt(30)/10', 'amp_016', 'ump_007')]), 'z_063': ('G(1,T1,,2|1,-1)', [('sqrt(10)/5', 'amp_012', 'ump_009'), ('sqrt(30)/10', 'amp_014', 'ump_008'), ('sqrt(30)/10', 'amp_015', 'ump_007')]), 'z_064': ('G(3,T1,,0|1,-1)', [('-sqrt(15)/10', 'amp_012', 'ump_007'), ('3*sqrt(5)/10', 'amp_013', 'ump_007'), ('-sqrt(5)/5', 'amp_015', 'ump_009'), ('-sqrt(5)/5', 'amp_016', 'ump_008')]), 'z_065': ('G(3,T1,,1|1,-1)', [('-sqrt(15)/10', 'amp_012', 'ump_008'), ('-3*sqrt(5)/10', 'amp_013', 'ump_008'), ('-sqrt(5)/5', 'amp_014', 'ump_009'), ('-sqrt(5)/5', 'amp_016', 'ump_007')]), 'z_066': ('G(3,T1,,2|1,-1)', [('sqrt(15)/5', 'amp_012', 'ump_009'), ('-sqrt(5)/5', 'amp_014', 'ump_008'), ('-sqrt(5)/5', 'amp_015', 'ump_007')]), 'z_067': ('G(2,T1,,0|1,-1)', [('sqrt(2)/2', 'amp_003', 'ump_022'), ('sqrt(2)/2', 'amp_004', 'ump_021')]), 'z_068': ('G(2,T1,,1|1,-1)', [('sqrt(2)/2', 'amp_002', 'ump_022'), ('sqrt(2)/2', 'amp_004', 'ump_020')]), 'z_069': ('G(2,T1,,2|1,-1)', [('sqrt(2)/2', 'amp_002', 'ump_021'), ('sqrt(2)/2', 'amp_003', 'ump_020')]), 'z_070': ('G(2,T1,,0|1,-1)', [('-sqrt(2)/2', 'amp_003', 'ump_025'), ('sqrt(2)/2', 'amp_004', 'ump_024')]), 'z_071': ('G(2,T1,,1|1,-1)', [('sqrt(2)/2', 'amp_002', 'ump_025'), ('-sqrt(2)/2', 'amp_004', 'ump_023')]), 'z_072': ('G(2,T1,,2|1,-1)', [('-sqrt(2)/2', 'amp_002', 'ump_024'), ('sqrt(2)/2', 'amp_003', 'ump_023')]), 'z_073': ('Q(1,T2,,0)', [('1', 'amp_005', 'ump_001')]), 'z_074': ('Q(1,T2,,1)', [('1', 'amp_006', 'ump_001')]), 'z_075': ('Q(1,T2,,2)', [('1', 'amp_007', 'ump_001')]), 'z_076': ('Q(1,T2,,0|1,0)', [('1', 'amp_008', 'ump_001')]), 'z_077': ('Q(1,T2,,1|1,0)', [('1', 'amp_009', 'ump_001')]), 'z_078': ('Q(1,T2,,2|1,0)', [('1', 'amp_010', 'ump_001')]), 'z_079': ('Q(2,T2,,0)', [('1', 'amp_021', 'ump_001')]), 'z_080': ('Q(2,T2,,1)', [('1', 'amp_022', 'ump_001')]), 'z_081': ('Q(2,T2,,2)', [('1', 'amp_023', 'ump_001')]), 'z_082': ('Q(2,T2,,0|1,-1)', [('1', 'amp_026', 'ump_001')]), 'z_083': ('Q(2,T2,,1|1,-1)', [('1', 'amp_027', 'ump_001')]), 'z_084': ('Q(2,T2,,2|1,-1)', [('1', 'amp_028', 'ump_001')]), 'z_085': ('Q(1,T2,,0)', [('1', 'amp_001', 'ump_003')]), 'z_086': ('Q(1,T2,,1)', [('1', 'amp_001', 'ump_004')]), 'z_087': ('Q(1,T2,,2)', [('1', 'amp_001', 'ump_005')]), 'z_088': ('Q(1,T2,,0)', [('1', 'amp_001', 'ump_007')]), 'z_089': ('Q(1,T2,,1)', [('1', 'amp_001', 'ump_008')]), 'z_090': ('Q(1,T2,,2)', [('1', 'amp_001', 'ump_009')]), 'z_091': ('Q(1,T2,,0|1,-1)', [('sqrt(2)/2', 'amp_003', 'ump_013'), ('-sqrt(2)/2', 'amp_004', 'ump_012')]), 'z_092': ('Q(1,T2,,1|1,-1)', [('-sqrt(2)/2', 'amp_002', 'ump_013'), ('sqrt(2)/2', 'amp_004', 'ump_011')]), 'z_093': ('Q(1,T2,,2|1,-1)', [('sqrt(2)/2', 'amp_002', 'ump_012'), ('-sqrt(2)/2', 'amp_003', 'ump_011')]), 'z_094': ('Q(1,T2,,0)', [('1', 'amp_005', 'ump_006')]), 'z_095': ('Q(1,T2,,1)', [('1', 'amp_006', 'ump_006')]), 'z_096': ('Q(1,T2,,2)', [('1', 'amp_007', 'ump_006')]), 'z_097': ('Q(2,T2,,0)', [('sqrt(2)/2', 'amp_006', 'ump_009'), ('sqrt(2)/2', 'amp_007', 'ump_008')]), 'z_098': ('Q(2,T2,,1)', [('sqrt(2)/2', 'amp_005', 'ump_009'), ('sqrt(2)/2', 'amp_007', 'ump_007')]), 'z_099': ('Q(2,T2,,2)', [('sqrt(2)/2', 'amp_005', 'ump_008'), ('sqrt(2)/2', 'amp_006', 'ump_007')]), 'z_100': ('Q(1,T2,,0|1,0)', [('1', 'amp_008', 'ump_006')]), 'z_101': ('Q(1,T2,,1|1,0)', [('1', 'amp_009', 'ump_006')]), 'z_102': ('Q(1,T2,,2|1,0)', [('1', 'amp_010', 'ump_006')]), 'z_103': ('Q(2,T2,,0|1,0)', [('sqrt(2)/2', 'amp_009', 'ump_009'), ('sqrt(2)/2', 'amp_010', 'ump_008')]), 'z_104': ('Q(2,T2,,1|1,0)', [('sqrt(2)/2', 'amp_008', 'ump_009'), ('sqrt(2)/2', 'amp_010', 'ump_007')]), 'z_105': ('Q(2,T2,,2|1,0)', [('sqrt(2)/2', 'amp_008', 'ump_008'), ('sqrt(2)/2', 'amp_009', 'ump_007')]), 'z_106': ('Q(2,T2,,0|1,-1)', [('sqrt(2)/2', 'amp_012', 'ump_007'), ('sqrt(6)/6', 'amp_013', 'ump_007'), ('-sqrt(6)/6', 'amp_015', 'ump_009'), ('sqrt(6)/6', 'amp_016', 'ump_008')]), 'z_107': ('Q(2,T2,,1|1,-1)', [('-sqrt(2)/2', 'amp_012', 'ump_008'), ('sqrt(6)/6', 'amp_013', 'ump_008'), ('sqrt(6)/6', 'amp_014', 'ump_009'), ('-sqrt(6)/6', 'amp_016', 'ump_007')]), 'z_108': ('Q(2,T2,,2|1,-1)', [('-sqrt(6)/3', 'amp_013', 'ump_009'), ('-sqrt(6)/6', 'amp_014', 'ump_008'), ('sqrt(6)/6', 'amp_015', 'ump_007')]), 'z_109': ('G(3,T2,,0|1,-1)', [('-1/2', 'amp_012', 'ump_007'), ('-sqrt(3)/6', 'amp_013', 'ump_007'), ('-sqrt(3)/3', 'amp_015', 'ump_009'), ('sqrt(3)/3', 'amp_016', 'ump_008')]), 'z_110': ('G(3,T2,,1|1,-1)', [('1/2', 'amp_012', 'ump_008'), ('-sqrt(3)/6', 'amp_013', 'ump_008'), ('sqrt(3)/3', 'amp_014', 'ump_009'), ('-sqrt(3)/3', 'amp_016', 'ump_007')]), 'z_111': ('G(3,T2,,2|1,-1)', [('sqrt(3)/3', 'amp_013', 'ump_009'), ('-sqrt(3)/3', 'amp_014', 'ump_008'), ('sqrt(3)/3', 'amp_015', 'ump_007')]), 'z_112': ('Q(2,T2,,0)', [('1', 'amp_001', 'ump_017')]), 'z_113': ('Q(2,T2,,1)', [('1', 'amp_001', 'ump_018')]), 'z_114': ('Q(2,T2,,2)', [('1', 'amp_001', 'ump_019')]), 'z_115': ('Q(1,T2,,0|1,-1)', [('sqrt(2)/2', 'amp_003', 'ump_022'), ('-sqrt(2)/2', 'amp_004', 'ump_021')]), 'z_116': ('Q(1,T2,,1|1,-1)', [('-sqrt(2)/2', 'amp_002', 'ump_022'), ('sqrt(2)/2', 'amp_004', 'ump_020')]), 'z_117': ('Q(1,T2,,2|1,-1)', [('sqrt(2)/2', 'amp_002', 'ump_021'), ('-sqrt(2)/2', 'amp_003', 'ump_020')]), 'z_118': ('Q(3,T2,,0|1,-1)', [('sqrt(2)/2', 'amp_003', 'ump_025'), ('sqrt(2)/2', 'amp_004', 'ump_024')]), 'z_119': ('Q(3,T2,,1|1,-1)', [('sqrt(2)/2', 'amp_002', 'ump_025'), ('sqrt(2)/2', 'amp_004', 'ump_023')]), 'z_120': ('Q(3,T2,,2|1,-1)', [('sqrt(2)/2', 'amp_002', 'ump_024'), ('sqrt(2)/2', 'amp_003', 'ump_023')])}}}
