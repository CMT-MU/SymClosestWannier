---
orphan: true
---

# 実空間の演算子の位相の規約

英語版は[こちら](convention.md)です。

このページでは、`pw2cw` が書き出す実空間の行列（`seedname_hr.dat.cw`、`seedname_tb.dat.cw` など）と `seedname.hdf5` に保存する行列の規約を定め、`tb_gauge` で何が変わるかを説明します。

## 実空間の行列

- $O_{ab}(\boldsymbol{R}) = \langle \phi_{a}(\boldsymbol{0})|O|\phi_{b}(\boldsymbol{R})\rangle$。$\phi_{b}(\boldsymbol{R})$ は格子ベクトル $\boldsymbol{R}$ の胞にある軌道 $b$ です。
- $\boldsymbol{R} = n_1\boldsymbol{a}_1 + n_2\boldsymbol{a}_2 + n_3\boldsymbol{a}_3$ は、Wigner–Seitz 超格子の整数 `irvec` = (n1, n2, n3) で与えます（`util/utility.py:wigner_seitz`）。`ndegen` は各 $\boldsymbol{R}$ の縮重度です。
- 配列の添字は `O[iR, a, b]` です（ベクトル演算子は `O[i, iR, a, b]`、i = x, y, z）。保存する値は、wannier90 と同じく `ndegen` で割り**ません**。
- 格子ベクトル `unit_cell_cart` は、行が $\boldsymbol{a}_1, \boldsymbol{a}_2, \boldsymbol{a}_3$ で、単位は Å です。
- 軌道の順は `ket_amn` の順（`seedname.win` の射影の順）です。対称化した行列（`*_sym`）は MultiPie の SAMB の ket の順で、`sort_ket_matrix` で両者を変換します。

## 対称化した行列と単位胞の外にある原子

MultiPie は模型のサイトを単位胞 $[0,1)$ に置き、その SAMB の格子ベクトル $\boldsymbol{R}$ は、この位置 $\boldsymbol{\tau}^{\rm MP}_a$ を基準にしています。`seedname.nnkp` の位置 $\boldsymbol{\tau}_a$ はどこにあってもかまいません（例えば MoS$_2$ で Mo の面の下にある S は $z<0$）。各軌道について、`site_offset` は $\boldsymbol{\tau}_a = \boldsymbol{\tau}^{\rm MP}_a + \boldsymbol{t}_a$ となる格子ベクトル $\boldsymbol{t}_a$ です（サイトは格子ベクトルの差を除いて対応づけます。スラブの真空のように模型と長さが違う格子ベクトルの方向では、その方向の距離で対応づけます。`ket_amn` を明示したときは、$\boldsymbol{t}_b-\boldsymbol{t}_a$ しか効かないので、模型全体を平行移動したもの $\boldsymbol{\tau}_a = \boldsymbol{\tau}^{\rm MP}_a + \boldsymbol{u} + \boldsymbol{t}_a$ でもかまいません）。SAMB に分解する前に、演算子を模型の格子ベクトルに移し、
$$
O^{\rm MP}_{ab}(\boldsymbol{R}+\boldsymbol{t}_b-\boldsymbol{t}_a) = O_{ab}(\boldsymbol{R}),
$$
対称化した行列（`*_sym`）は元に戻します。こうして、対称化した行列も、他の行列と同じく `seedname.nnkp` の格子ベクトル `irvec` を持ちます（`tb_gauge = true` のとき、その軌道の位置は $\boldsymbol{\tau}^{\rm MP}_a + \boldsymbol{t}_a$ です）。SAMB の係数は、原子をどこに置くかによりません。

## フーリエ変換

ブロッホ基底は $|\phi_{b}(\boldsymbol{k})\rangle = \sum_{\boldsymbol{R}} e^{i\boldsymbol{k}\cdot\boldsymbol{R}}|\phi_{b}(\boldsymbol{R})\rangle$ で、位相に軌道の位置を含みません。Closest Wannier の構成が `seedname.amn` と `seedname.eig` から $H(\boldsymbol{k})$、$S(\boldsymbol{k})$ などを与えるのはこの基底で、wannier90 の規約（`_hr.dat`、`_tb.dat`）と同じです。このとき

$$
O_{ab}(\boldsymbol{k}) = \sum_{\boldsymbol{R}} \frac{1}{N_{\boldsymbol{R}}} e^{i\boldsymbol{k}\cdot\boldsymbol{R}} O_{ab}(\boldsymbol{R}), \qquad
O_{ab}(\boldsymbol{R}) = \frac{1}{N_k}\sum_{\boldsymbol{k}} e^{-i\boldsymbol{k}\cdot\boldsymbol{R}} O_{ab}(\boldsymbol{k}),
$$

で、$N_{\boldsymbol{R}}$ = `ndegen`、和は DFT の k メッシュについてとります（`atoms_frac` なしの `fourier_transform_r_to_k` / `fourier_transform_k_to_r`）。

## tb_gauge

`tb_gauge = true` のとき、ブロッホ基底は軌道の位置 $\boldsymbol{\tau}_b$ を含み（`seedname.nnkp` にある軌道 $b$ の射影の中心、`util/utility.py:orbital_positions`。原子への射影ではその原子の位置）、$|\phi^{\rm tb}_{b}(\boldsymbol{k})\rangle = e^{i\boldsymbol{k}\cdot\boldsymbol{\tau}_b}|\phi_{b}(\boldsymbol{k})\rangle$ です。k 空間の行列は
$$
O^{\rm tb}_{ab}(\boldsymbol{k}) = \sum_{\boldsymbol{R}} \frac{1}{N_{\boldsymbol{R}}} e^{i\boldsymbol{k}\cdot(\boldsymbol{R}+\boldsymbol{\tau}_b-\boldsymbol{\tau}_a)} O^{\rm tb}_{ab}(\boldsymbol{R})
$$
になります（`fourier_transform_r_to_k` の `atoms_frac` 引数）。基底の変更は k に依存するユニタリー変換なので、固有値は変わりません。固有ベクトルと、ベリー接続や k 微分のワニエゲージの行列は変わります。

`pw2cw` が計算して書き出す実空間の行列（`Hr`、`Sr`、`*_sym`、`seedname_hr.dat.cw`、`seedname_r.dat.cw`、`seedname_v.dat.cw`、`seedname.s.cw` など）は、`tb_gauge` によらず、すべて wannier90 の規約です。`seedname_tb.dat.cw` は `tb_gauge = false` のときだけ書き出します。

`tb_gauge = true` の `postcw` では、`Response` は演算子を wannier90 の規約のまま持ち、計算に渡すときに変換します（`util/get_oper_R.py:to_tb_gauge`、`Response.operators` から呼ぶ）。$\boldsymbol{\tau}$ はデカルト座標、$m$、$n$ は行と列の軌道です。

| 演算子 | 変換 |
| --- | --- |
| `HH_R`、`SS_R`、`SH_R` | 変えない |
| `AA_R`（$\boldsymbol{r}$） | $A^{\rm tb}_{a,mn}(\boldsymbol{R}) = A_{a,mn}(\boldsymbol{R}) - \tau_{n,a}\delta_{mn}\delta_{\boldsymbol{R}\boldsymbol{0}}$ |
| `BB_R`（$H\boldsymbol{r}$） | $B^{\rm tb}_{a}(\boldsymbol{R}) = B_{a}(\boldsymbol{R}) - \tau_{n,a} H(\boldsymbol{R})$ |
| `CC_R`（$\boldsymbol{r}H\boldsymbol{r}$） | $C^{\rm tb}_{ab} = C_{ab} - \tau_{m,a} B_{b} - \tau_{n,b} B^{\dagger}_{a} + \tau_{m,a}\tau_{n,b} H$、ただし $B^{\dagger}_{a,mn}(\boldsymbol{R}) = B_{a,nm}(-\boldsymbol{R})^{*}$ |
| `SR_R`（$\sigma\boldsymbol{r}$）、`SHR_R`（$\sigma H\boldsymbol{r}$） | $SR^{\rm tb}_{ab}(\boldsymbol{R}) = SR_{ab}(\boldsymbol{R}) - \tau_{n,b} SS_{a}(\boldsymbol{R})$、$SHR^{\rm tb}_{ab}(\boldsymbol{R}) = SHR_{ab}(\boldsymbol{R}) - \tau_{n,b} SH_{a}(\boldsymbol{R})$（$a$：スピンの成分） |

k 微分も同じ基底でとります。$\partial_{\boldsymbol{k}} O^{\rm tb}(\boldsymbol{k}) = \sum_{\boldsymbol{R}} i(\boldsymbol{R}+\boldsymbol{\tau}_b-\boldsymbol{\tau}_a)\,e^{i\boldsymbol{k}\cdot(\boldsymbol{R}+\boldsymbol{\tau}_b-\boldsymbol{\tau}_a)} O^{\rm tb}_{ab}(\boldsymbol{R})/N_{\boldsymbol{R}}$（`pseudo=True` の `fourier_transform_r_to_k_new`、`fourier_transform_r_to_k_vec`、`atoms_frac` を与えた `get_v_R`）。

`to_tb_gauge` は `ndegen(-R) = ndegen(R)` を仮定しています。これは Wigner–Seitz 超格子では成り立ちます。

これらの変換により、どちらのゲージの演算子も同じ物理量を表すので、ワニエ補間の式は、各 k 点で同じベリー曲率、軌道磁化（−2Im f、g、h）、スピン流を与えます。J0、J1、J2 の項への分け方だけがゲージによります（`tests/test_tb_gauge.py` で確認）。これは、`HH_R` が `BB_R` と `CC_R` を作ったハミルトニアンであるときに成り立ちます。Qiao の方法のスピン流でも、テストの模型のように $SH(\boldsymbol{k}) = SS(\boldsymbol{k})\,H(\boldsymbol{k})$（k 空間の行列の積、$\boldsymbol{R}$ では畳み込み）なら、2 つのゲージは一致します。`SH_R` はすべてのブロッホバンドから $U^\dagger s H U$ として作るので、$U^\dagger s U\,U^\dagger H U$ とは $U^\dagger s(1-UU^\dagger)HU$ だけ違います。これは disentangle した模型では一般に 0 でなく、そのとき Qiao の方法のスピンホール伝導度は `tb_gauge` によることがあります（`symcw-wberri` のテストの disentangle したスピノルの模型がそうです）。wannier90 は `tb_gauge = false` の規約を使います。`postcw` は別のハミルトニアン（対称化したときの `Hr_sym`、`hr_input`）を使うことがあります。そのときも `BB_R` と `CC_R` はそれ自身のハミルトニアンで変換しますが、それらを与えたハミルトニアンと組み合わせる量（軌道磁化、ジャイロトロピックな K）は、どちらのゲージでも近似になり、2 つのハミルトニアンの差の程度だけゲージによって違います。

固有値だけを使う量（バンド分散、状態密度、Lindhard 関数）は `tb_gauge` によりません。位置演算子を無視するところでは、結果は `tb_gauge` によります。`use_tb_approximation = true` で使う速度 $\partial_{\boldsymbol{k}} H/\hbar$（`get_v_R`）のバンド間の行列要素は、軌道の位相によるからです。
