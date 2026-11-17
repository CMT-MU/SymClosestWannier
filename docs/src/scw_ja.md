---
orphan: true
---

# 対称性に適合した Closest Wannier 模型

英語版は[こちら](scw.md)です。

Closest Wannier 法により、Closest Wannier 関数（CWF）と Closest Wannier（CW）ハミルトニアン $H^{\rm (CW)}$ が得られます。
次に、文献 {footcite}`hk-prb23` の対称性に適合した模型化に従って、CW ハミルトニアンを対称性に適合した多極子基底（SAMB）の線形結合に分解します。

$$
H^{\rm (SymCW)} = \sum_{j} z_{j} \mathbb{Z}_{j}
$$

ここで $\{z_{j}\}$ は各 SAMB の重みで、模型のパラメータに当たり、次の式で得られます。

$$
z_{j} = \sum_{\bf{R}} \mathrm{Tr}\left[H^{\rm (CW)}(\bf{R}) \mathbb{Z}_{j}(-\bf{R}) \right]
$$

理論的な背景の詳細は、文献 {footcite}`ro-prb25` を参照してください。

## 参考文献

```{footbibliography}
```
