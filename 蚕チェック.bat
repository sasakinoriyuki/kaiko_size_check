@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
set "KAIKO_LAUNCHER_FILE=%~f0"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$s=Get-Content -LiteralPath $env:KAIKO_LAUNCHER_FILE -Raw -Encoding UTF8; $m=('# KAIKO_'+'STARTER'); & ([ScriptBlock]::Create($s.Substring($s.IndexOf($m)+$m.Length)))"
set "KAIKO_EXIT_CODE=%ERRORLEVEL%"
if not "%KAIKO_EXIT_CODE%"=="0" pause
exit /b %KAIKO_EXIT_CODE%
# KAIKO_STARTER
param([switch]$FunctionsOnly)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$kaikoChannel = 'parsonal'
$kaikoModulus = 'sIWjVY8yBJdVSLke1bI2TPq7grFA9RK3JoIUYo11eSTiFKcIBMoWDx8JKe01g2Il7bQt3ZwqYAApiu1C/kZz6DIbHUocveCaYno1T0qG01KWlSTYpYZl9HnEYB6s2iEApFlUGOKZ7kQUbtyiySTObA/rSIhUmmguTDl6FicIxqU5iMnwDM2InnvtHmNXBvu5jCIUhZYE9tv/Wy8smD8SxFFPYkrTBW5DGybikwYa+AWcah4Fwxfw7o4ZUQsfOuGEH47RGlVvXvWcedvDFj5IzxsX7W+IxJPUOWPZaj+Zy8b6MCCsUjNg0ZEhXXjwyL5an9c6+XNJdqgAyIKd2Vc8Bcb58CD7SkD0EWW7Gk2ACRuv/5zWcnjl5DKn/bNFIYLpPnwgWgaTDbb+7dFDR9AXtRM81jxP7j+JzqAmSBgoj9/hLDFP4k4qo9mnGEI3c+IlB58cvf502rDxtiIXUqNpQakI0cHVDZZrJl7wMU8v+CcDAJQ9NhbpBtaS1UpucDlJ'
$kaikoFeed = 'https://raw.githubusercontent.com/sasakinoriyuki/kaiko_size_check/parsonal/.updates/channel.json'

function Get-KaikoBytes([string]$Url, [long]$Limit, [int]$Timeout = 12) {
    for ($redirect = 0; $redirect -lt 6; $redirect++) {
        $uri = [Uri]$Url
        if ($uri.Scheme -ne 'https' -or $uri.Host -ne 'raw.githubusercontent.com' -or $uri.Port -ne 443 -or $uri.UserInfo) { throw '許可されていない配布先です。' }
        $request = [Net.HttpWebRequest]::Create($uri)
        $request.AllowAutoRedirect = $false
        $request.Timeout = $Timeout * 1000
        $request.ReadWriteTimeout = $Timeout * 1000
        $request.UserAgent = 'KaikoSizeCheck/2'
        $request.Headers['Cache-Control'] = 'no-cache'
        $response = $null
        $stream = $null
        $memory = New-Object IO.MemoryStream
        try {
            $response = $request.GetResponse()
            if ([int]$response.StatusCode -in @(301,302,303,307,308)) {
                $Url = (New-Object Uri($uri, $response.Headers['Location'])).AbsoluteUri
                continue
            }
            $stream = $response.GetResponseStream()
            $buffer = New-Object byte[] 16384
            while (($count = $stream.Read($buffer,0,$buffer.Length)) -gt 0) {
                if ($memory.Length + $count -gt $Limit) { throw '配布ファイルが大きすぎます。' }
                $memory.Write($buffer,0,$count)
            }
            return ,($memory.ToArray())
        } finally {
            if ($stream) { $stream.Dispose() }
            if ($response) { $response.Dispose() }
            $memory.Dispose()
        }
    }
    throw '配布先の転送回数が多すぎます。'
}

function Read-KaikoEnvelope([byte[]]$Raw, [string]$Channel, [string]$Modulus) {
    if ($Raw.Length -gt 65536) { throw '更新案内が大きすぎます。' }
    $envelope = [Text.Encoding]::UTF8.GetString($Raw) | ConvertFrom-Json
    $payload = [Convert]::FromBase64String($envelope.payload)
    $signature = [Convert]::FromBase64String($envelope.signature)
    $rsa = [Security.Cryptography.RSA]::Create()
    try {
        $parameters = New-Object Security.Cryptography.RSAParameters
        $parameters.Modulus = [Convert]::FromBase64String($Modulus)
        $parameters.Exponent = [Convert]::FromBase64String('AQAB')
        $rsa.ImportParameters($parameters)
        if (-not $rsa.VerifyData($payload,$signature,[Security.Cryptography.HashAlgorithmName]::SHA256,[Security.Cryptography.RSASignaturePadding]::Pkcs1)) { throw '更新の署名が一致しません。' }
    } finally { $rsa.Dispose() }
    $info = [Text.Encoding]::UTF8.GetString($payload) | ConvertFrom-Json
    if ($info.schema -ne 1 -or $info.minimum_bootstrap -gt 2 -or $info.channel -ne $Channel -or $info.encoding -ne 'base64') { throw 'このBATに対応していない更新案内です。' }
    if ($info.bytes -lt 1 -or $info.bytes -gt 33554432 -or $info.release_sequence -lt 1 -or $info.sha256 -cnotmatch '^[a-f0-9]{64}$') { throw '更新案内の値が不正です。' }
    return $info
}

function Resolve-KaikoPath([string]$Root, [string]$Name) {
    if (-not $Name -or $Name -match '^[\\/]' -or $Name -match '[:<>"|?*\x00-\x1f]') { throw '更新ファイルのパスが不正です。' }
    $parts = $Name.Replace('\','/').Split('/')
    foreach ($part in $parts) {
        if (-not $part -or $part -in @('.','..') -or $part -match '[ .]$' -or $part -match '^(?i:con|prn|aux|nul|conin\$|conout\$|com[1-9¹²³]|lpt[1-9¹²³])(\.|$)') { throw '不正なWindowsファイル名です。' }
        if ($part -in @('data','runtime','logs','updates','backups','.venv','output','_kaiko_backup','examples')) { throw '記録や環境を更新対象にはできません。' }
    }
    $prefix = [IO.Path]::GetFullPath($Root).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $path = [IO.Path]::GetFullPath([IO.Path]::Combine($Root,($parts -join [IO.Path]::DirectorySeparatorChar)))
    if (-not $path.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw '更新先がフォルダ外です。' }
    $parent = $path
    while ($parent -and $parent.Length -ge $prefix.Length-1) {
        if (([IO.File]::Exists($parent) -or [IO.Directory]::Exists($parent)) -and (([IO.File]::GetAttributes($parent) -band [IO.FileAttributes]::ReparsePoint) -ne 0)) { throw 'リンクされた更新先は使えません。' }
        $parent = [IO.Path]::GetDirectoryName($parent)
    }
    return $path
}

function Get-KaikoHash([byte[]]$Bytes) {
    $hash = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hash.ComputeHash($Bytes))).Replace('-','').ToLowerInvariant() }
    finally { $hash.Dispose() }
}

function Test-KaikoFiles([string]$Root, [string]$Channel) {
    $manifestPath = Resolve-KaikoPath $Root '_internal/update_manifest.json'
    $manifest = [IO.File]::ReadAllText($manifestPath,[Text.Encoding]::UTF8) | ConvertFrom-Json
    $flavor = 'desktop'
    if ($Channel -eq 'school') { $flavor = 'school' }
    if ($manifest.layout -ne 'one_click' -or $manifest.schema_version -ne 2 -or $manifest.entrypoint -ne 'external_launcher' -or $manifest.installation_flavor -ne $flavor) { throw '配布版の種類が違います。' }
    $files = @($manifest.files.PSObject.Properties)
    if ($files.Count -lt 3 -or $files.Count -gt 200 -or -not $manifest.files.PSObject.Properties['_internal/app.py']) { throw '配布内容が不正です。' }
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($file in $files) {
        $path = Resolve-KaikoPath $Root $file.Name
        if ($file.Name -ieq '_internal/update_manifest.json' -or $file.Name -match '(?i)\.bat$' -or -not $seen.Add($file.Name.Replace('\','/'))) { throw '配布ファイルが重複しています。' }
        if ($file.Value -cnotmatch '^[a-f0-9]{64}$' -or -not [IO.File]::Exists($path) -or (Get-KaikoHash ([IO.File]::ReadAllBytes($path))) -cne $file.Value) { throw ('不足・破損したファイル: ' + $file.Name) }
    }
    return $manifest
}

function Expand-KaikoPackage([byte[]]$Raw, $Info, [string]$Destination, [string]$Channel) {
    if ($Raw.Length -ne $Info.bytes -or (Get-KaikoHash $Raw) -cne $Info.sha256) { throw '更新ファイルの内容が一致しません。' }
    Add-Type -AssemblyName System.IO.Compression
    $memory = New-Object IO.MemoryStream(,$Raw)
    $archive = New-Object IO.Compression.ZipArchive($memory,[IO.Compression.ZipArchiveMode]::Read)
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    [long]$total = 0
    try {
        if ($archive.Entries.Count -lt 3 -or $archive.Entries.Count -gt 200) { throw '配布ファイルの数が不正です。' }
        foreach ($entry in $archive.Entries) {
            $name = $entry.FullName.Replace('\','/')
            $path = Resolve-KaikoPath $Destination $name
            $extension = [IO.Path]::GetExtension($name).ToLowerInvariant()
            if (-not $seen.Add($name) -or $name.EndsWith('/') -or (($entry.ExternalAttributes -shr 16) -band 61440) -eq 40960) { throw '重複やリンクを含む配布ファイルです。' }
            if ($extension -notin @('.py','.ps1','.json','.txt','.md','.pt') -and [IO.Path]::GetFileName($name) -cne 'LICENSE') { throw '許可されていない配布ファイルです。' }
            $total += $entry.Length
            if ($total -gt 67108864 -or $entry.Length -gt 33554432) { throw '展開後のサイズが大きすぎます。' }
            [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($path)) | Out-Null
            $inputStream = $entry.Open()
            $outputStream = [IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
            try {
                $buffer = New-Object byte[] 16384
                [long]$written = 0
                while (($count = $inputStream.Read($buffer,0,$buffer.Length)) -gt 0) {
                    $written += $count
                    if ($written -gt $entry.Length -or $written -gt 33554432) { throw '展開内容のサイズが一致しません。' }
                    $outputStream.Write($buffer,0,$count)
                }
                if ($written -ne $entry.Length) { throw '展開したファイルが不足しています。' }
            } finally { $outputStream.Dispose(); $inputStream.Dispose() }
        }
    } finally { $archive.Dispose(); $memory.Dispose() }
    $manifest = Test-KaikoFiles $Destination $Channel
    if ($manifest.release_sequence -ne $Info.release_sequence -or $seen.Count -ne @($manifest.files.PSObject.Properties).Count+1) { throw '更新案内と配布内容が違います。' }
    foreach ($file in $manifest.files.PSObject.Properties) { if (-not $seen.Contains($file.Name)) { throw '配布ファイルが不足しています。' } }
    return $manifest
}

function Start-KaikoOnline([string]$Root) {
    $Root = [IO.Path]::GetFullPath($Root)
    $stage = $null
    $lock = $null
    try {
        foreach ($folder in @($Root,(Join-Path $Root 'updates'),(Join-Path $Root 'logs'))) {
            if ((Test-Path -LiteralPath $folder) -and (([IO.File]::GetAttributes($folder) -band [IO.FileAttributes]::ReparsePoint) -ne 0)) { throw '保存先にリンクされたフォルダがあります。' }
        }
        $env:KAIKO_INSTALL_ROOT = $Root
        $env:KAIKO_SCHOOL_MODE = '0'
        if ($kaikoChannel -eq 'school') { $env:KAIKO_SCHOOL_MODE = '1' }
        $localReady = $false
        try { $null = Test-KaikoFiles $Root $kaikoChannel; $localReady = $true } catch { }
        if ($localReady) {
            & (Join-Path $Root '_internal\bootstrap_windows.ps1') | Out-Host
            return [int]$LASTEXITCODE
        }
        [IO.Directory]::CreateDirectory((Join-Path $Root 'updates')) | Out-Null
        try { $lock = [IO.File]::Open((Join-Path $Root 'updates\starter.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None) }
        catch { Write-Host '別の画面で初回の準備が進んでいます。'; return 0 }
        Write-Host 'GitHubから蚕チェックを取得しています。初回は必要なライブラリも準備します。'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $info = Read-KaikoEnvelope (Get-KaikoBytes $kaikoFeed 65536) $kaikoChannel $kaikoModulus
        $limit = [long]([Math]::Ceiling($info.bytes/3.0)*4+2)
        $encoded = Get-KaikoBytes $info.url $limit 60
        $raw = [Convert]::FromBase64String([Text.Encoding]::ASCII.GetString($encoded).Trim())
        $stage = Join-Path $Root ('updates\starter_' + [Guid]::NewGuid().ToString('N'))
        $null = Expand-KaikoPackage $raw $info $stage $kaikoChannel
        & (Join-Path $stage '_internal\bootstrap_windows.ps1') | Out-Host
        return [int]$LASTEXITCODE
    } catch {
        Write-Host ('起動の準備を完了できませんでした: ' + $_.Exception.Message)
        Write-Host 'ネット接続と、BATを置いたフォルダに書き込めるかを確認してください。'
        return 1
    } finally {
        if ($lock) { $lock.Dispose() }
        if ($stage -and (Test-Path -LiteralPath $stage)) { Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue }
    }
}

if ($FunctionsOnly) { return }
$kaikoResult = Start-KaikoOnline (Split-Path -Parent $env:KAIKO_LAUNCHER_FILE)
exit $kaikoResult
