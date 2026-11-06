# 学校版の更新・配布手順

更新先ブランチは `school` です。通常版の開発は `codex/windows-prototype` で続け、必要な変更を学校版へ取り込みます。

## 学校版の構成

- 指定Python：`X:\python 3.12\python.exe`
- 起動：配布ZIP内の `蚕チェック.bat` 1本
- 起動処理：`bootstrap_school_windows.ps1` を配布内の `_internal/bootstrap_windows.ps1` に配置
- 保存先：展開したフォルダそのもの。記録は `data`、ライブラリ・キャッシュ・一時ファイルは `runtime`、ログは `logs`
- マニフェスト：`installation_flavor=school`、`storage_mode=portable`
- 以前のAppData内の学校版は初回の記録コピー元として読み取る。そこへ新しい版をインストールしない

## 作成と確認

ソース変更に応じたテストを実行します。モデルはGitへ追加せず、配布担当者の外部作業フォルダから読み込みます。必要なのは自作モデル3種類の.ptと対応する.jsonです。

```powershell
py -3.12 build_one_click_package.py --school --public-update --output dist/kaiko_school.zip --models "<モデルのフォルダ>" --release-sequence <新しい配布番号> --version "<新しいバージョン>"
```

配布番号はschool_release.jsonの値より大きくします。現時点の配布番号は10です。`--public-update` は画像を含めず、更新パッケージの展開・検証をローカルで行う指定です。このコマンド自体はネットへ公開しません。

## 配布情報の更新

1. ZIP内のBATが1本で、撮影画像・記録・秘密鍵が含まれないことを確認し、サイズとSHA256を記録します。
2. 学校用のGoogle Driveファイル `1bZ5cXSgn9LUENCZuID4UPICrWKKSJZZu` の内容を更新します。
3. 配布先のサイズとファイル名を確認した後、school_release.jsonとREADME.mdのバージョン・検証内容を更新します。
4. 変更をschoolブランチへ反映します。

アプリの保存や依存環境を変更した場合は、実GUIでフォルダ内起動と保存先を確認します。Windows実機で未確認の部分は分かるように記載します。

配布ファイルの共有範囲はGoogle Drive側の設定に従います。GitHubへのソース公開によってアプリの自動更新配信が始まることはありません。配信を追加する際は学校版専用の署名付きフィードとschool形式のパッケージを使用します。
