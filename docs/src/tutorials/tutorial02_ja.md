---
orphan: true
---

# グラフェン：Closest Wannier 強束縛模型

英語版は[こちら](tutorial02.md)です。

- 概要：水素様原子の $p_{z}$ 軌道を使い、Closest Wannier 法でグラフェンの Closest Wannier 模型と、ワニエ補間したバンド構造を計算します。
    理論的な背景の詳細は、文献 {footcite}`ozaki-prb24` を参照してください。

- ディレクトリ：`others/tutorials/tutorial02/` *ファイルは
    [こちら](https://github.com/CMT-MU/SymClosestWannier/tree/main/others/tutorials/tutorial02)からダウンロードできます*

- 入力ファイル
    - `graphene.cwin`： *`pw2cw` の入力ファイル*

    - `graphene.win`： *`wannier90.x` の入力ファイル*

    - `graphene.nnkp`： *`wannier90.x` が作る結晶のデータの出力ファイル*

    - `graphene.eig, graphene.amn`： *`pw2wannier.x`、`VASP2WANNIER`、`WIEN2WANNIER` が作る出力ファイル*

    - `graphene.band.gnu`： *DFT のバンドのデータ*

&nbsp;

1. [Quantum ESPRESSO](https://www.quantum-espresso.org/) や [VASP](https://www.vasp.at/) などの平面波（PW）の DFT 計算のソフトウェアで、コーン-シャム軌道と擬原子軌道の重なりを計算します。
次のステップでは、次のファイルを使います。
   - seedname.win、seedname.nnkp、seedname.eig、seedname.amn（[Wannier90](https://wannier.org/) が作るもの）
    （追加のファイル：seedname.mmn、seedname.spn、seedname.uHu、seedname.uIu など）

2. `pw2cw` を実行して、Closest Wannier 強束縛模型を作ります。
    ```bash
    pw2cw graphene
    ```

    Closest Wannier 法による disentanglement は、`graphene.cwin` で次のように指定して行います。

    ```text
    disentangle = true
    ```

    Closest Wannier 法では、ブロッホ状態と射影の重なりに、窓関数 $w(\varepsilon)$ を掛けます（文献 {footcite}`ozaki-prb24` の式 (4)）。

    $$
    & A_{mn}^{(\bf{k})} = w(\varepsilon_{m\bf{k}}) \langle \psi_{m{\bf k}}|g_{n}\rangle,
    \\
    & w(\varepsilon)
    = \frac{1}{e^{(\varepsilon - \mu_{\rm max})/\sigma_{\rm max}} + 1}
    + \frac{1}{e^{(\mu_{\rm min} - \varepsilon)/\sigma_{\rm min}} + 1}
    - 1 + \delta,
    $$

    ここで $\mu_{\rm min}$ と $\mu_{\rm max}$（$\mu_{\rm min} < \mu_{\rm max}$）はエネルギー窓の下端と上端を表し、$\sigma_{\rm min}$ と $\sigma_{\rm max}$ は $\mu_{\rm min}$ と $\mu_{\rm max}$ の周りのぼかしの程度を決めます。
    $\delta$ は、$A_{mn}^{(\bf{k})}$ からなる行列が悪条件にならないように加える小さな定数です。

    $\mu_{\rm max}$、$\mu_{\rm min}$、$\sigma_{\rm max}$、$\sigma_{\rm min}$、$\delta$ は、`graphene.cwin` で次のように指定します。

    ```text
    cwf_mu_max      = 10.0
    cwf_mu_min      = -30
    cwf_sigma_max = 3.0
    cwf_sigma_min = 0.0
    cwf_delta             = 1e-12
    ```

    また、

     ```text
    proj_min          = 0.1
    ```

    は射影度 $p_{m\bf{k}} = \sum_{n} |\langle \psi_{m{\bf k}}|g_{n}\rangle|^{2}$（窓関数なし）の最小値です。
    $p_{m\bf{k}}$ が `proj_min` より大きくないとき、その $m$ 番目の DFT のバンドはワニエ化から除きます。


    窓関数を導入することで、バンドの disentanglement が自然に取り込まれることに注意してください。
    ユニタリー行列は $\mathbf{U}^{(\mathbf{k})} = \mathbf{A}^{(\bf{k})}
    (\mathbf{A}^{(\bf{k}) \dagger} \mathbf{A}^{(\bf{k})})^{-1/2}$
    で得られ、バンドの disentanglement とワニエ化のための反復計算は要りません。

    $\mu_{\rm min}$、$\mu_{\rm max}$、$\sigma_{\rm min}$、$\sigma_{\rm max}$、$\delta$ を適切に選ぶと、ヒルベルト空間で初期推定に*最も近い*ワニエ関数が得られます（文献 {footcite}`ozaki-prb24` の式 (5)〜(17) を参照）。

    窓関数を適用した射影 $w(\varepsilon_{m{\bf k}}) \langle \psi_{m{\bf k}}|g_{n}\rangle$ を `graphene.amn.cw` に書き出すには（`proj_min` で除いたバンドは 0、`disentangle` しないときは $w = 1$）、`graphene.cwin` で `write_windowed_amn` を次のように指定します。

    ```text
    write_windowed_amn = true
    ```

    Wannier90 の `seedname.amn` の形式で、直交化せずに書き出します。
    直交化した後のユニタリー行列は、`write_u_matrices = true` で書き出します。

    実空間の Closest Wannier ハミルトニアンの行列要素を得るには、`graphene.cwin` で `write_hr` を次のように指定します。

    ```text
    write_hr = true
    ```

    出力ファイル `graphene.cwout` を見てください。
    CWF は反復計算なしに得られ、計算コストを大きく減らせます。
    eta\_ $\nu$ と eta\_ $\nu$ \_max（$\nu = 0, 2, 5$）は、DFT のバンドと Closest Wannier のバンドの間のバンド距離で、次のように定義します。

    $$
    \eta_{\nu} = \sqrt{\frac{\sum_{n\bf{k}}(\epsilon_{n\bf{k}}^{\rm DFT} - \epsilon_{n\bf{k}}^{\rm CW})^{2} \tilde{f}_{n\bf{k}} }{\sum_{n\bf{k}}\tilde{f}_{n\bf{k}}}}
    $$

    $$
    \tilde{f}_{n\bf{k}} = \sqrt{f_{n\bf{k}}^{\rm DFT}(\nu, \tau) f_{n\bf{k}}^{\rm CW}(\nu, \tau)}
    $$

    ここで $f_{n\bf{k}}(\nu, \tau)$ は、エネルギーをずらしたフェルミ-ディラック分布関数です。
    $$
    (\nu = 0)\, \epsilon_{\rm F} + 0.0\, [{\rm eV}] \\
    (\nu = 2)\, \epsilon_{\rm F} + 2.0\, [{\rm eV}] \\
    (\nu = 5)\, \epsilon_{\rm F} + 5.0\, [{\rm eV}]
    $$

    有効温度は $\tau = 0.0$ です。


- 出力ファイル：
    - `graphene.cwout`： *標準出力*
    - `graphene.hdf5`： *CWModel の計算の情報とデータ（`write_info_data = true` のとき）*
    - `graphene_hr.dat.cw`： *実空間の Closest Wannier ハミルトニアンの行列要素 $H(R)$*
    - `graphene_hr_R_dep.dat.cw`： *ハミルトニアンのフロベニウスノルムの、ボンドの長さ ||R||（格子ベクトルの 2 ノルム）への依存性と、指数関数のフィット $||H(R)|| = ||H(0)|| \exp(-||R||/\tau)$ で定義した減衰長 $\tau$ [Å]*
    - `graphene_nr.dat.cw`： *実空間の Closest Wannier の電子密度行列の要素 $n(R)$*
    - `graphene.amn.cw`： *窓関数を適用した、ブロッホ状態の試行軌道への射影（`write_windowed_amn = true` のとき）*
    - `graphene_band.txt`： Closest Wannier 強束縛模型のバンド分散のデータ。
    - `graphene_band_detail.txt`： 軌道の寄与を含む、Closest Wannier 強束縛模型のバンド分散のデータ。
    - `plot_band.gnu`： *DFT と Closest Wannier のバンドを比べる gnuplot のスクリプト*
    - `plot_band_detail.gnu`： *DFT と Closest Wannier のバンドを比べる gnuplot のスクリプト*
    - `graphene_band.pdf(eps)`： *DFT と Closest Wannier のバンドの比較*
    - `graphene_band_detail.pdf(eps)`： *DFT と Closest Wannier のバンドの比較*
    - `graphene/`
        - `graphene_hr_sym.dat.cw`： *実空間の、対称性に適合した Closest Wannier ハミルトニアンの行列要素*
        - `graphene_z.dat.cw`： *ハミルトニアン行列を SAMB の線形結合で表したときの展開係数*
        - `graphene_s.dat.cw`： *重なり行列を SAMB の線形結合で表したときの展開係数*
        - `graphene_n.dat.cw`： *電子密度行列を SAMB の線形結合で表したときの展開係数*
        - `graphene_band.txt`： 対称性に適合した Closest Wannier 強束縛模型のバンド分散のデータ。
        - `graphene_band_detail.txt`： 軌道の寄与を含む、対称性に適合した Closest Wannier 強束縛模型のバンド分散のデータ。
        - `plot_band.gnu`： *Closest Wannier と対称性に適合した Closest Wannier のバンドを比べる gnuplot のスクリプト*
        - `plot_band_detail.gnu`： *Closest Wannier と対称性に適合した Closest Wannier のバンドを比べる gnuplot のスクリプト*
        - `graphene_band.pdf(eps)`： *Closest Wannier と対称性に適合した Closest Wannier のバンドの比較*
        - `graphene_band_detail.pdf(eps)`： *Closest Wannier と対称性に適合した Closest Wannier のバンドの比較*

    ファイル形式の詳細は次のとおりです。
    - [seedname.cwin](../format/cwin_ja.md)
    - [seedname.cwout](../format/cwout_ja.md)
    - [seedname.hdf5](../format/hdf5_ja.md)
    - [seedname_hr.dat.cw](../format/hr_ja.md)
    - [mp_seedname_z.dat.cw](../format/z_ja.md)
    - [mp_seedname_s.dat.cw](../format/s_ja.md)
    - [mp_seedname_n.dat.cw](../format/n_ja.md)

## 参考文献

```{footbibliography}
```
