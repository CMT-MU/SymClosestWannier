---
orphan: true
---

# MultiPie による SAMB の作成

英語版は[こちら](multipie.md)です。

**注意**：対称化をしない場合、この手順は必要ありません。

1. 入力ファイルを用意します。[CH4](https://github.com/CMT-MU/SymClosestWannier/blob/main/others/tutorials/tutorial01/ch4.py) 分子と [グラフェン](https://github.com/CMT-MU/SymClosestWannier/blob/main/others/tutorials/tutorial02/graphene.py) の入力ファイルの例を示します。
    - CH4 分子
    ```{literalinclude} ../../others/tutorials/tutorial01/ch4.py
    ```
    - グラフェン
    ```{literalinclude} ../../others/tutorials/tutorial02/graphene.py
    ```

2. この 2 つの入力ファイルがあるフォルダで、次のように実行して SAMB を作ります。
詳しくは "**mp_create --help**" を実行してください。
    ```
    $ mp_create ch4 graphene
    ```

3. **ch4** と **graphene** のフォルダに、次のファイルが作られます。
    - ch4.pkl、ch4.qtdw、ch4.pdf
    - graphene.pkl、graphene.qtdw、graphene.pdf

    **.pdf** は [TeXLive](https://www.tug.org/texlive/) が入っていれば作られ、**.qtdw** は [QtDraw](https://github.com/CMT-MU/QtDraw) が入っていれば作られます。

    それぞれのファイルの中身は次のとおりです。
    - **.pkl**：SAMB を含む模型の情報（バイナリ）。`pw2cw` が `mp_seedname` を通して読みます。
    - **.pdf**：模型と SAMB の詳しい情報（PDF 形式）。
    - **.qtdw**：QtDraw 用の分子または結晶の構造のファイル。

模型は、SymClosestWannier と一緒に入れた MultiPie で作ってください。MultiPie Ver.2 で作った模型（**.pkl**）は Ver.3 では読めません（[Ver.2 からの変更点](https://cmt-mu.github.io/MultiPie/src/changes_ja.html)を参照。Ver.2 の模型は `mp_convert` で Ver.3 に変換できます）。SymClosestWannier はどちらのバージョンでも動きます。ワニエ関数の f 軌道の名前は、入っている MultiPie に従います（Ver.2 の `f3x`、`f3y`、`f2`、`f1` は、Ver.3 では `fx`、`fy`、`fc`、`fs`）。Ver.3 では、ボンドは **bond** だけで与えます（**max_neighbor** と **search_cell_range** は削除されました）。

ファイル形式の詳細は [MultiPie](https://github.com/CMT-MU/MultiPie) を参照してください。
