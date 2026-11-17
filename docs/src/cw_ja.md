---
orphan: true
---

# Closest Wannier 法

英語版は[こちら](cw.md)です。

Closest Wannier 法を使うと、ヒルベルト空間で初期推定に最も近いワニエ関数（closest Wannier functions、CWF）を、反復計算なしに得られ、計算コストを大きく減らせます。
理論的な背景の詳細は、文献 {footcite}`ozaki-prb24` を参照してください。

Closest Wannier 法の主なアイデアは、ブロッホ状態 $|\psi_{n\bf{k}}\rangle$ を試行局在軌道 $|g_{n}\rangle$ へ射影するときに、窓関数 $w(\varepsilon)$（文献 {footcite}`ozaki-prb24` の式 (4)）を導入することです。

$$
& A_{mn}^{(\bf{k})} = w(\varepsilon_{m\bf{k}}) \langle \psi_{m{\bf k}}|g_{n}\rangle,
\\
& w(\varepsilon) = \frac{1}{e^{(\varepsilon - \mu_{\rm max})/\sigma_{\rm max}} + 1}
+ \frac{1}{e^{(\mu_{\rm min} - \varepsilon)/\sigma_{\rm min}} + 1} - 1 + \delta.
$$

$\mu_{\rm min}$ と $\mu_{\rm max}$（$\mu_{\rm min} < \mu_{\rm max}$）はエネルギー窓の下端と上端を表し、$\sigma_{\rm min}$ と $\sigma_{\rm max}$ は、それぞれ $\mu_{\rm min}$ と $\mu_{\rm max}$ の周りのぼかしの程度を決めます。
このぼかしによって、すべてのブロッホ状態がそれぞれの重みで射影に取り込まれます。重みは窓の内側（$\mu_{\rm min} < \varepsilon_{m\bf{k}} < \mu_{\rm max}$）で特に大きく、窓の外側では小さくなります。
$\delta$ は、$A_{mn}^{(\bf{k})}$ からなる行列が悪条件にならないように加える小さな定数です。

窓関数を導入することで、バンドの disentanglement が自然に取り込まれることに注意してください。
ユニタリー行列は $\mathbf{U}^{(\mathbf{k})} = \mathbf{A}^{(\bf{k})}
(\mathbf{A}^{(\bf{k}) \dagger} \mathbf{A}^{(\bf{k})})^{-1/2}$
で得られ、バンドの disentanglement とワニエ化のための反復計算は要りません（`dis_num_iter = 0`、`num_iter = 0`）。
$\mu_{\rm min}$、$\mu_{\rm max}$、$\sigma_{\rm min}$、$\sigma_{\rm max}$、$\delta$ を適切に選ぶと、ヒルベルト空間で初期推定に**最も近い**ワニエ関数が得られます（文献 {footcite}`ozaki-prb24` の式 (5)〜(17) を参照）。

```{note}
CWF は、disentanglement とワニエ化のパラメータを指定して最局在ワニエ関数（MLWF）を作るときの初期推定としても使えます。
```

## 参考文献

```{footbibliography}
```
