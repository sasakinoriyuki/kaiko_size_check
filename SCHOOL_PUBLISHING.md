# 学校版の更新・配布手順

更新先ブランチは `school` です。通常版の開発は `codex/windows-prototype` で続け、学校版に必要な変更をschoolへ取り込みます。

## ソース更新

- 学校版の起動に使うのは `bootstrap_school_windows.ps1` です。
- 指定Pythonは `X:\python 3.12\python.exe`、保存先は `SizeCheckSchool` です。
- 共通の処理はbootstrap.pyなどを使います。通常版と学校版の更新を混同しないよう、配布マニフェストの `installation_flavor` はschoolにします。
- 確認した変更をschoolブランチへコミットします。起動や更新の処理を変更した場合は、該当テストを実行します。

## 学校用ZIPの作成

配布担当者の開発環境で、自作モデル6ファイル（3種類の.ptと対応する.json）を持つフォルダを指定します。モデルはGitへ追加せず、外部の作業フォルダから読み込みます。

```powershell
py -3.12 build_one_click_package.py --school --public-update --output dist/kaiko_school.zip --models "<モデルのフォルダ>" --release-sequence <新しい配布番号> --version "<新しいバージョン>"
```

`--school` により、学校用PowerShellを配布内の `_internal/bootstrap_windows.ps1` へ配置し、学校用の案内を同梱します。学校版の更新フィードは空のままです。`--public-update` は画像を含めず、更新パッケージの展開・検証が通るかをローカルで確認する指定です。このコマンド自体はネットへ公開しません。

配布番号はschool_release.jsonの値より大きくします。現時点の配布番号は9です。

## 配布情報の更新

1. ZIPにBATが1本だけ入っていることを確認し、ファイルサイズとSHA256を記録します。
2. 学校用のGoogle Driveファイルの内容を更新します。現在のファイルIDは `1bZ5cXSgn9LUENCZuID4UPICrWKKSJZZu` です。
3. 配布先のファイル名・サイズを確認してから、school_release.jsonとREADME.mdのバージョン・検証内容を更新します。
4. その変更もschoolブランチへコミットします。

学校用ZIPへのリンクはREADME.mdに置きます。配布ファイルの共有範囲はGoogle Drive側の設定に従います。GitHubブランチへのソース公開によって、アプリの自動更新配信が始まることはありません。学校版の配信を追加する際は、学校版専用の署名付きフィードとschool形式のパッケージを使用します。
