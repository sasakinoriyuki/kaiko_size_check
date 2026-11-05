$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$kaikoPackage = Split-Path -Parent $PSScriptRoot
$kaikoLocal = [Environment]::GetFolderPath('LocalApplicationData')
$kaikoInstall = Join-Path $kaikoLocal 'KaikoChannel\SizeCheckSchool'
$kaikoPython = 'X:\python 3.12\python.exe'
$kaikoLogs = Join-Path $kaikoInstall 'logs'
New-Item -ItemType Directory -Force -Path $kaikoLogs | Out-Null
$kaikoLog = Join-Path $kaikoLogs ('setup_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.txt')
function Write-KaikoLog([string]$Text) {
    Write-Host $Text
    Add-Content -LiteralPath $kaikoLog -Value $Text -Encoding UTF8
}
try {
    Write-KaikoLog '蚕チェック（学校用）を準備しています。'
    Write-KaikoLog ('使用するPython: ' + $kaikoPython)
    if (-not (Test-Path -LiteralPath $kaikoPython -PathType Leaf)) {
        throw ('指定のPythonが見つかりません: ' + $kaikoPython + '。X:ドライブに接続できているか確認してください。')
    }
    # Single quotes inside Python code survive Windows PowerShell 5.1 argument passing.
    $kaikoProbe = "import sys,struct;assert sys.version_info[:2]==(3,12),'Python 3.12 is required';assert struct.calcsize('P')==8,'64-bit Python is required';import tkinter,venv,ssl,ensurepip;print('Python 3.12 (64-bit): OK')"
    $kaikoProbeOutput = & $kaikoPython -I -X utf8 -c $kaikoProbe 2>&1
    $kaikoProbeCode = $LASTEXITCODE
    foreach ($kaikoLine in @($kaikoProbeOutput)) { Write-KaikoLog ([string]$kaikoLine) }
    if ($kaikoProbeCode -ne 0) {
        throw '指定のPythonに必要な機能がありません。Python 3.12の64bit版とtkinter・venv・ssl・ensurepipが必要です。'
    }
    $env:KAIKO_SCHOOL_MODE = '1'
    & $kaikoPython -I -X utf8 (Join-Path $PSScriptRoot 'bootstrap.py') --package-root $kaikoPackage --install-root $kaikoInstall --log-file $kaikoLog
    $kaikoCode = $LASTEXITCODE
    if ($kaikoCode -ne 0) { throw '準備を完了できませんでした。上の内容と記録を確認してください。' }
    try {
        $kaikoDesktop = [Environment]::GetFolderPath('DesktopDirectory')
        $kaikoShortcutPath = Join-Path $kaikoDesktop '蚕CHANNEL 体長チェック（学校用）.lnk'
        $kaikoShell = New-Object -ComObject WScript.Shell
        $kaikoShortcut = $kaikoShell.CreateShortcut($kaikoShortcutPath)
        $kaikoTarget = Join-Path $kaikoInstall '蚕チェック.bat'
        if ((-not (Test-Path -LiteralPath $kaikoShortcutPath)) -or ($kaikoShortcut.TargetPath -eq $kaikoTarget)) {
            $kaikoShortcut.TargetPath = $kaikoTarget
            $kaikoShortcut.WorkingDirectory = $kaikoInstall
            $kaikoShortcut.Description = '学校のPython 3.12で準備・起動します'
            $kaikoShortcut.Save()
        }
    } catch { Write-KaikoLog 'デスクトップの項目は作成できませんでした。蚕チェック.batから起動できます。' }
    exit 0
} catch {
    Write-KaikoLog ('準備を完了できませんでした: ' + $_.Exception.Message)
    Write-KaikoLog ('記録の保存先: ' + $kaikoLog)
    exit 1
}
