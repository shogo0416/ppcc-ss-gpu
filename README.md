# PPCC-SS-GPU

[第6回粒子物理コンピューティングサマースクール](https://wiki.kek.jp/display/PPCC/PPCC-SS-2023)
での共通講習「GPUプログラミング」のサンプルコードです。
本コードは、サマースクールで用意したGPU環境で動かすことを想定しています。

## アプリの内容
- SAXPY
- 乱数を使った円周率の計算

## コードのビルド

### コードの取得

```
$ git clone git@github.com:shogo0416/ppcc-ss-gpu
```

> **Note**
> GitHubのアカウントを持っていない場合はZIPファイルをダウンロードして、
> 計算環境にコピーして下さい。


### ビルドの実行
`build`ディレクトリを作り、そこでコードのビルドを実行するとします。

```
$ cd ppcc-ss-gpu
$ mkdir build && cd build
$ cmake3 ../
$ make install
```

> **Note**
> Cmakeのversionは`3.17`以上を想定しています。
> `cmake3`コマンドで`Makefile`を作成してください。

`make install`が済むと`bin`ディレクトリが作成され、
そこに実行ファイルがあります。

```
$ cd ../bin/
$ ls
cpp-pi  cpp-saxpy  cuda-pi  cuda-saxpy
```

|実行ファイル|説明|
|------------|----|
|cpp-saxpy|CPUで実行するSAXPYのプログラム|
|cpp-pi|CPUで実行する円周率の計算(マルチスレッド対応)|
|cuda-saxpy|GPUで実行するSAXPYのプログラム|
|cuda-pi|GPUで実行する円周率の計算|

> **Note**
> ご自身でGPU環境を持っていてそこで走らせる場合は、
> cmakeを実行する際に`-DPPCC_ENV=False`を入れて下さい。
> ```
> $ cmake3 -DPPCC_ENV=False ../
> ```

## アプリを実行する上での注意点

サマースクールで用意した環境`xxxx`と`yyyy`にはそれぞれGPUが7台があります。
nvtop コマンドで空いているGPUを確認の上、アプリを実行して下さい。

### 前準備
ログイン後、`.bashrc` を開いて次の行を加えて下さい。
これが無いとGPUデバイスを選択出来ません。

```
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
```

### GPUデバイスを指定してアプリを実行
CUDAのアプリについては`-g <ID>`オプションを入れて実行して下さい。
`<ID>`はGPUデバイスの識別子で0-6を取ります。

```
./cuda-pi -g 1
```

### 各アプリの実行時のオプションについて
アプリの実行時のオプションは `-h` で表示されます。

```
./cuda-pi -h
```
