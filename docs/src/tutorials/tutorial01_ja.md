---
orphan: true
---

## CH4：Closest Wannier 強束縛模型

英語版は[こちら](tutorial01.md)です。

- 概要：水素様原子の $s$ 軌道と $p$ 軌道を使い、Closest Wannier 法で ${\rm CH}_{4}$ 分子の Closest Wannier 模型と、ワニエ補間したバンド構造を計算します。
    理論的な背景の詳細は、文献 [@ozaki-prb24] を参照してください。

- ディレクトリ：`tutorials/tutorial01/` *ファイルは
    [こちら](https://github.com/CMT-MU/SymClosestWannier/tree/main/tutorials/tutorial01)からダウンロードできます*


- 入力ファイル
    - `ch4.cwin`： *`pw2cw` の入力ファイル*

    - `ch4.win`： *`wannier90.x` の入力ファイル*

    - `ch4.nnkp`： *`wannier90.x` が作る結晶のデータの出力ファイル*
    
    - `ch4.eig, ch4.amn`： *`pw2wannier.x`、`VASP2WANNIER`、`WIEN2WANNIER` が作る出力ファイル* 

&nbsp;

1. [Quantum ESPRESSO](https://www.quantum-espresso.org/) や [VASP](https://www.vasp.at/) などの平面波（PW）の DFT 計算のソフトウェアで、コーン-シャム軌道と擬原子軌道の重なりを計算します。
次のステップでは、次のファイルを使います。
   - seedname.win、seedname.nnkp、seedname.eig、seedname.amn（[Wannier90](https://wannier.org/) が作るもの）
    （追加のファイル：seedname.mmn、seedname.spn、seedname.uHu、seedname.uIu など）

2. `pw2cw` を実行して、Closest Wannier 強束縛模型を作ります。

    ```bash title="Terminal"
    pw2cw ch4
    ```

    ${\rm CH}_{4}$ は分子なので disentanglement は必要なく、`ch4.cwin` で `disentangle` を次のように指定できます。

    ```vi title="Input file"
    disentangle = false
    ```

    実空間の Closest Wannier ハミルトニアンの行列要素を得るには、`ch4.cwin` で `write_hr` を次のように指定します。

    ```vi title="Input file"
    write_hr = true
    ```
    
- 出力ファイル：
    - `ch4.cwout`： *標準出力*
    - `ch4.hdf5`： *CWModel の計算の情報とデータ（`write_info_data = true` のとき）*
    - `ch4_hr.dat.cw`： *実空間の Closest Wannier ハミルトニアンの行列要素 $H(R)$（この場合は $H(0)$）*
    - `ch4_hr_R_dep.dat.cw`： *ハミルトニアンのフロベニウスノルムの、ボンドの長さ ||R||（格子ベクトルの 2 ノルム）への依存性*
    - `ch4_nr.dat.cw`： *実空間の Closest Wannier の電子密度行列の要素 $n(R)$*
    - `ch4/`
        - `ch4_hr_sym.dat.cw`： *実空間の、対称性に適合した Closest Wannier ハミルトニアンの行列要素*
        - `ch4_z.dat.cw`： *ハミルトニアン行列を SAMB の線形結合で表したときの展開係数*
        - `ch4_s.dat.cw`： *重なり行列を SAMB の線形結合で表したときの展開係数*
        - `ch4_n.dat.cw`： *電子密度行列を SAMB の線形結合で表したときの展開係数*

    ファイル形式の詳細は次のとおりです。
    - [seedname.cwin](../format/cwin_ja.md)
    - [seedname.cwout](../format/cwout_ja.md)
    - [seedname.hdf5](../format/hdf5_ja.md)
    - [seedname_hr.dat.cw](../format/hr_ja.md)
    - [mp_seedname_z.dat.cw](../format/z_ja.md)
    - [mp_seedname_s.dat.cw](../format/s_ja.md)
    - [mp_seedname_n.dat.cw](../format/n_ja.md)
