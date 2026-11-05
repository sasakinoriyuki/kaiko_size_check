# 蚕CHANNEL 体長チェック — 学校用

学校用のコードと更新履歴は、この **`school` ブランチ**で管理します。

## 学校PCで使う

[学校用ZIPをダウンロード（Google Drive）](https://drive.google.com/file/d/1bZ5cXSgn9LUENCZuID4UPICrWKKSJZZu/view?usp=drivesdk)

1. ZIPを右クリックして「すべて展開」します。
2. **蚕チェック.bat**をダブルクリックします。
3. 初回の準備が終わると、アプリが起動します。次回も同じBATか、デスクトップの「蚕CHANNEL 体長チェック（学校用）」を使います。

使用するPythonは **`X:\python 3.12\python.exe`** です。指定のPythonから学校用の専用環境を作り、ライブラリ・アプリ・記録を `%LOCALAPPDATA%\KaikoChannel\SizeCheckSchool` に保存します。初回はネット接続が必要です。起動するたびにX:ドライブへアクセスできる必要があります。

詳しい操作とエラーの確認方法は [SCHOOL.md](SCHOOL.md) を参照してください。

## 現在の配布版

- バージョン：`2026.11.05-school1`（配布番号9）
- 自作の認識モデル：v7
- 学校用ZIPに入っているBAT：1本
- 検証：120件の自動テスト、学校版のGUI起動・重複起動の抑止を確認
- 学校のWindows PC・実際のX:ドライブでの動作は未確認
- 自動体長・頭数は未完成
- 学校版のアプリ自動更新配信は未有効

配布ZIPのハッシュなどは [school_release.json](school_release.json) に記録しています。

## コードの更新と配布

今後の学校版の変更は、このブランチへ反映します。配布ZIPはソースと自作モデルをまとめて作ります。GitHubの「Code → Download ZIP」で取得するものは開発用ソースです。アプリを使う場合は上の学校用ZIPを取得してください。

GitHubにはソース・説明・配布情報を置きます。撮影画像、教師ラベル、測定記録、モデルの重み、署名用の秘密鍵は含めません。

配布作業を行う人向けの手順は [SCHOOL_PUBLISHING.md](SCHOOL_PUBLISHING.md) にあります。
