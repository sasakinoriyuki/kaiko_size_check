$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$kaikoPackage = Split-Path -Parent $PSScriptRoot
$kaikoInstall = $kaikoPackage
if ($env:KAIKO_INSTALL_ROOT) { $kaikoInstall = $env:KAIKO_INSTALL_ROOT }
$kaikoRuntime = Join-Path $kaikoInstall 'runtime\Python312'
$kaikoCache = Join-Path $kaikoInstall 'runtime\Downloads'
$kaikoTemporary = Join-Path $kaikoInstall 'runtime\tmp'
New-Item -ItemType Directory -Force -Path $kaikoTemporary | Out-Null
$env:TEMP = $kaikoTemporary
$env:TMP = $kaikoTemporary
$env:TMPDIR = $kaikoTemporary
$kaikoLogs = Join-Path $kaikoInstall 'logs'
New-Item -ItemType Directory -Force -Path $kaikoLogs | Out-Null
$kaikoLog = Join-Path $kaikoLogs ('setup_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.txt')
function Write-KaikoLog([string]$Text) {
    Write-Host $Text
    Add-Content -LiteralPath $kaikoLog -Value $Text -Encoding UTF8
}
function Test-KaikoPython([string]$Executable, [string[]]$Prefix = @()) {
    if (-not $Executable) { return $null }
    if ($Executable -like '*\WindowsApps\*') { return $null }
    $kaikoProbe = "import sys,struct,tkinter,venv,ssl,ensurepip;assert sys.version_info[:2]==(3,12) and struct.calcsize('P')==8;print(getattr(sys,'_base_executable',sys.executable))"
    try {
        $kaikoLines = & $Executable @Prefix -I -X utf8 -c $kaikoProbe 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        $kaikoResult = (@($kaikoLines) | Select-Object -Last 1)
        if ($kaikoResult -and (Test-Path -LiteralPath $kaikoResult.Trim() -PathType Leaf)) { return $kaikoResult.Trim() }
    } catch { }
    return $null
}
try {
    Write-KaikoLog '蚕チェックを準備しています。初回だけ必要なものを自動で入れます。'
    $kaikoPython = Test-KaikoPython (Join-Path $kaikoRuntime 'python.exe')
    if (-not $kaikoPython) {
        $kaikoRegistry = @('HKCU:\Software\Python\PythonCore\3.12\InstallPath', 'HKLM:\Software\Python\PythonCore\3.12\InstallPath')
        foreach ($kaikoKey in $kaikoRegistry) {
            if (Test-Path -LiteralPath $kaikoKey) {
                $kaikoProperties = Get-ItemProperty -LiteralPath $kaikoKey
                if ($kaikoProperties.ExecutablePath) { $kaikoPython = Test-KaikoPython $kaikoProperties.ExecutablePath }
                if ($kaikoPython) { break }
            }
        }
    }
    if (-not $kaikoPython) {
        $kaikoLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
        if ($kaikoLauncher) { $kaikoPython = Test-KaikoPython $kaikoLauncher.Source @('-3.12') }
    }
    if (-not $kaikoPython) {
        $kaikoCommand = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($kaikoCommand) { $kaikoPython = Test-KaikoPython $kaikoCommand.Source }
    }
    if (-not $kaikoPython) {
        if (-not [Environment]::Is64BitOperatingSystem) { throw '64bit版のWindowsが必要です。' }
        Write-KaikoLog 'Python 3.12を公式サイトから取得しています。管理者として実行する必要はありません。'
        New-Item -ItemType Directory -Force -Path $kaikoCache | Out-Null
        $kaikoInstaller = Join-Path $kaikoCache 'python-3.12.10-amd64.exe'
        $kaikoExpected = '67B5635E80EA51072B87941312D00EC8927C4DB9BA18938F7AD2D27B328B95FB'
        $kaikoValid = (Test-Path -LiteralPath $kaikoInstaller -PathType Leaf) -and ((Get-FileHash -LiteralPath $kaikoInstaller -Algorithm SHA256).Hash -eq $kaikoExpected)
        if (-not $kaikoValid) {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            $kaikoTemporary = Join-Path $kaikoCache ([Guid]::NewGuid().ToString() + '.download')
            try {
                Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile $kaikoTemporary
                if ((Get-FileHash -LiteralPath $kaikoTemporary -Algorithm SHA256).Hash -ne $kaikoExpected) { throw 'Pythonのダウンロード内容が一致しません。実行せず停止しました。' }
                Move-Item -LiteralPath $kaikoTemporary -Destination $kaikoInstaller -Force
            } finally {
                if (Test-Path -LiteralPath $kaikoTemporary) { Remove-Item -LiteralPath $kaikoTemporary -Force }
            }
        }
        $kaikoArguments = @('/quiet', '/norestart', 'InstallAllUsers=0', ('TargetDir="' + $kaikoRuntime + '"'), 'PrependPath=0', 'Include_launcher=0', 'InstallLauncherAllUsers=0', 'Include_tcltk=1', 'Include_pip=1', 'Include_test=0', 'Include_doc=0', 'Shortcuts=0', 'AssociateFiles=0')
        $kaikoProcess = Start-Process -FilePath $kaikoInstaller -ArgumentList ($kaikoArguments -join ' ') -PassThru -Wait
        if ($kaikoProcess.ExitCode -notin @(0,3010)) { throw ('Pythonの準備に失敗しました。終了番号: ' + $kaikoProcess.ExitCode) }
        $kaikoPython = Test-KaikoPython (Join-Path $kaikoRuntime 'python.exe')
        if (-not $kaikoPython) { throw 'Pythonを起動できませんでした。記録を確認してください。' }
    }
    & $kaikoPython -I -X utf8 (Join-Path $PSScriptRoot 'bootstrap.py') --package-root $kaikoPackage --install-root $kaikoInstall --log-file $kaikoLog
    $kaikoCode = $LASTEXITCODE
    if ($kaikoCode -ne 0) { throw '準備を完了できませんでした。上の内容と記録を確認してください。' }
    exit 0
} catch {
    Write-KaikoLog ('準備を完了できませんでした: ' + $_.Exception.Message)
    Write-KaikoLog ('記録の保存先: ' + $kaikoLog)
    exit 1
}
