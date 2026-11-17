---
orphan: true
---

# seedname.cwin [既定値]

英語版は[こちら](cwin.md)です。

- 実行方法：pw2cw seedname
  - SEEDNAME は必須です。入力は seedname.cwin から読みます。
      seedname の代わりに、文字列 seedname.cwin を与えても同じです。

- seedname.cwin の入力パラメータ（seedname.win と同じ形式）[既定値]
  - restart           : 再開の位置 'cw'/'w90' (str), ["cw"].
  - seedname          : seedname (str), ["cwannier"].
  - outdir            : 出力ファイルを書き出すディレクトリ。seedname.cwin のディレクトリからの相対パス (str), ["./"].
  - disentangle       : バンドを disentangle するか (bool), [False].
  - proj_min          : 射影度の最小値, [0.0].
  - cwf_mu_max      : エネルギー窓の上端 (float), [None].
  - cwf_mu_min      : エネルギー窓の下端 (float), [None].
  - cwf_sigma_max : エネルギー窓の上端のぼかしの温度 (float), [1.0].
  - cwf_sigma_min : エネルギー窓の下端のぼかしの温度 (float), [0.0].
  - cwf_delta             : 窓関数に加える基準の重み。エネルギー窓の外の状態はこの重みを持ちます (0 <= cwf_delta < 1)。seedname.cwout が射影の階数落ちを警告したときは大きくしてください（例：1e-3） (float), [0.0].
  - svd               : 特異値分解を使うか。使わないときはレーヴディンの直交化を使います (bool), [False].
  - verbose           : 計算の情報を詳しく表示するか (bool, optional), [False].
  - parallel          : 並列のコードを使うか (bool), [False].
  - formatter         : black で整形するか (bool), [False].
  - calc_spreads      : 広がりを計算するか (bool), [False].
  - write_info_data   : 情報とデータを seedname.hdf5（hdf5 形式）に書き出すか (bool), [False].
  - transl_inv        : バンド対角の位置の行列要素に、Marzari&Vanderbilt PRB 56, 12847 (1997) の式 (31) を使うか (bool), [False].
  - use_degen_pert    : バンドが縮退していてバンドの微分が要るとき、縮退のある摂動論を使うか (bool), [False].
  - degen_thr         : 縮退したバンドを計算から除くしきい値, [0.0001].
  - tb_gauge          : tb ゲージを使うか (bool), [False].
  - tb_position       : seedname_r.dat.cw と seedname_tb.dat.cw の位置演算子を、射影の中心 r_ab(R) = tau_a delta_ab delta_R0 で近似するか（このとき seedname.mmn は不要） (bool), [False].
  - write_hr          : seedname_hr.dat.cw、seedname_hr_R_dep.dat.cw、seedname_nr.dat.cw、seedname_nr_R_dep.dat.cw を書き出すか (bool), [False].
  - write_sr          : seedname_sr.dat.cw と seedname_sr_R_dep.dat.cw を書き出すか (bool), [False].
  - write_u_matrices  : seedname_u.mat.cw と seedname_u_dis.mat.cw を書き出すか（restart = w90 では書き出さない） (bool), [False].
  - write_rmn         : seedname_r.dat.cw を書き出すか（tb_position = true でなければ seedname.mmn が必要） (bool), [False].
  - write_vmn         : seedname_v.dat.cw を書き出すか (bool), [False].
  - write_tb          : seedname_tb.dat.cw（wannier90 の形式）を書き出すか（tb_position = true でなければ seedname.mmn が必要。tb_gauge = true では書き出さない） (bool), [False].
  - write_eig         : seedname.eig.cw を書き出すか (bool), [False].
  - write_amn         : write_windowed_amn と同じ。互換性のために残しています (bool), [False].
  - write_windowed_amn : seedname.amn.cw を書き出すか。窓関数 w と proj_min を適用した射影 w(e_mk) <psi_mk|g_n> で、直交化はしません（disentangle しないときは w = 1。restart = w90 では入力の射影） (bool), [False].
  - write_mmn         : seedname.mmn.cw を書き出すか (bool), [False].
  - write_spn         : スピン演算子を seedname.s.cw に書き出すか (bool), [False].

- 対称化だけで使うもの
  - symmetrization    : 対称化するか (bool), [False].
  - mp_outdir         : multipie の出力ファイルを書き出すディレクトリ。seedname.cwin のディレクトリからの相対パス (str). ["./"].
  - mp_seedname       : MultiPie の模型の名前（mp_create が作るディレクトリと .pkl ファイル） (str), ["default"].
  - ket_amn           : seedname.amn の ket の基底のリスト (list), [None]. 各 ket は [atom, sublattice, rank, orbital] で与えます。例：`[[C,1,1,pz],[C,2,1,pz]]`。スピンを含むときは `[[C,1,1,(pz,u)],[C,1,1,(pz,d)]]`。ket_amn = auto なら、軌道のリストを自動で決めます。
  - irreps            : 考える SAMB の既約表現："all"（すべての既約表現）、"full"（恒等表現だけ）、または既約表現のリスト (str/list), ["all"].

- バンド分散の計算だけで使うもの
  - a                 : 参照のバンドのデータの k 点の単位を直す格子定数（Å） (float, optional), [None]（None のときは |a1|）.
  - N1                : 高対称線の分割数 (int, optional), [50].
  - calc_spin_2d      : *.spn で与えるスピン演算子の期待値を出力ファイルに加えるか (bool). [False].

- 状態密度の計算だけで使うもの
  - calc_dos          : 状態密度を計算するか (bool), [False].
  - dos_kmesh         : 状態密度の計算の k 点の Monkhorst-Pack 格子の大きさ。空白で区切った整数で与えます。例：`dos_kmesh = 10 10 10` (list), [[1, 1, 1]].
  - dos_num_fermi     : フェルミエネルギーの数 (int), [50].
  - dos_smr_en_width  : 状態密度のぼかし関数のエネルギー幅（単位は [eV]） (float), [0.001].
  - dos_emax          : 計算するエネルギーの最大値 (float), [None].
  - dos_emin          : 計算するエネルギーの最小値 (float), [None].

- COHP の計算だけで使うもの
  - calc_cohp             : COHP を計算するか (bool), [False].
  - calc_cohp_samb_decomp : COHP を SAMB で分解するか (bool), [False].
  - cohp_bond_length_max  : 計算するボンドの長さの最大値（Å） (float), [6.0].
  - cohp_bond_length_min  : 計算するボンドの長さの最小値（Å） (float), [0.0].
  - cohp_head_atom        : 計算するボンドの先の原子 (str), [None].
  - cohp_head_atom_idx    : 計算するボンドの先の原子の番号 (int), [None].
  - cohp_tail_atom        : 計算するボンドの元の原子 (str), [None].
  - cohp_tail_atom_idx    : 計算するボンドの元の原子の番号 (int), [None].
  - cohp_kmesh            : COHP の計算の k 点の Monkhorst-Pack 格子の大きさ。空白で区切った整数で与えます。例：`cohp_kmesh = 10 10 10` (list), [[1, 1, 1]].
  - cohp_num_fermi        : フェルミエネルギーの数 (int), [50].
  - cohp_smr_en_width     : COHP のぼかし関数のエネルギー幅（単位は [eV]） (float), [0.001].
  - cohp_emax             : 計算するエネルギーの最大値 (float), [None].
  - cohp_emin             : 計算するエネルギーの最小値 (float), [None].


- ゼーマン相互作用を考えるときだけ使うもの
  - zeeman_interaction   : ゼーマン相互作用を考えるか (bool), [False].
  - magnetic_field       : 磁場の強さ (float), [0.0].
  - magnetic_field_theta : 磁場の z 軸からの角度 (float), [0.0].
  - magnetic_field_phi   : 磁場の x 軸からの角度 (float), [0.0].
  - g_factor             : スピンの g 因子 (float), [2.0].


- フェルミ面の計算だけで使うもの
  - fermi_surface        : フェルミ面を計算するか (bool), [False].
  - fermi_surface_kmesh  : 2 次元の k メッシュ [[k1_min, k1_max, N1], [k2_min, k2_max, N2]]（結晶座標）。空白で区切った 6 つの整数 k1_min k1_max N1 k2_min k2_max N2 で与えます。例：`fermi_surface_kmesh = -1 1 10 -1 1 10` (list), [ [[-1, 1, 10], [-1, 1, 10]] ].
  - fermi_surface_view   : k3 の方向。空白で区切った整数で与えます。例：`fermi_surface_view = 0 0 1` (list), [ [0, 0, 1] ].
  - fermi_surface_const  : k3 軸の一定値 [0.0].

- Lindhard 関数だけで使うもの
  - lindhard                    : Lindhard 関数を計算するか (bool), [False].
  - lindhard_freq               : Lindhard 関数を計算する振動数（単位は [eV]） (float), [0.0].
  - lindhard_smr_fixed_en_width : 大域変数 smr_fixed_en_width を上書きします (float), [0.01].
  - lindhard_kmesh              : Lindhard 関数の k 点の Monkhorst-Pack 格子の大きさ。空白で区切った整数で与えます。例：`lindhard_kmesh = 10 10 10` (list), [[1, 1, 1]].
  - lindhard_surface            : 2 次元の q 平面で Lindhard 関数を計算するか (bool), [False].
  - lindhard_surface_qmesh      : 2 次元の q メッシュ [[q1_min, q1_max, N1], [q2_min, q2_max, N2]]（結晶座標）。空白で区切った 6 つの整数 q1_min q1_max N1 q2_min q2_max N2 で与えます。例：`lindhard_surface_qmesh = -1 1 10 -1 1 10` (list), [ [[-1, 1, 10], [-1, 1, 10]] ].
  - lindhard_surface_view       : q3 の方向。空白で区切った整数で与えます。例：`lindhard_surface_view = 0 0 1` (list), [ [0, 0, 1] ].
  - lindhard_surface_const      : q3 軸の一定値 [0.0].
  - qpoint                      : 代表的な q 点 (dict), [None].
  - qpoint_path                 : ブリルアンゾーンの高対称線に沿った q 点、[[k1, k2, k3]]（結晶座標） (str), [None].
  - Nq1                         : 高対称線の分割数 (int, optional), [30].
  - filling                     : 単位胞あたりの電子数, [None].
  - temperature                 : Lindhard 関数を計算する温度 (float), [0.0].

- postcw の計算だけで使うもの
  - hr_input             : hr.dat ファイルのファイル名（パスを含む） (str), [""].
  - use_tb_approximation : 強束縛近似を使うか (bool), [False].
