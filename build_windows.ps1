param([string]$FfmpegDirectory = "", [string]$FfmpegLicenseFile = "", [string]$FfmpegReadmeFile = "", [string]$ReleaseName = "TomatoTracker-0.7.36",
      [string]$DistPath = "", [string]$WorkPath = "", [string]$PythonExecutable = "")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if ($ReleaseName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$') { throw 'Invalid release folder name.' }
if ($DistPath) {
    $taskDistRoot = [IO.Path]::GetFullPath($DistPath)
} else {
    $taskDistRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'dist'))
}
$taskRelease = [IO.Path]::GetFullPath((Join-Path $taskDistRoot $ReleaseName))
if (-not $taskRelease.StartsWith($taskDistRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Release path must be inside this project dist directory.'
}
$taskExistingExe = Join-Path $taskRelease ($ReleaseName + '.exe')
if (Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $taskExistingExe }) {
    throw 'The selected release is running. Close that version before rebuilding it, or choose a new ReleaseName.'
}
$taskPython = if ($PythonExecutable) { [IO.Path]::GetFullPath($PythonExecutable) } else { Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Create .venv and install requirements-dev.txt first.' }
if (-not $FfmpegDirectory) {
    $taskFfmpeg = Get-Command ffmpeg -ErrorAction Stop
    $FfmpegDirectory = Split-Path -Parent $taskFfmpeg.Source
}
foreach ($toolName in @('ffmpeg.exe', 'ffprobe.exe')) {
    if (-not (Test-Path -LiteralPath (Join-Path $FfmpegDirectory $toolName))) { throw "Missing $toolName" }
}
$taskFfmpegRoot = Split-Path -Parent $FfmpegDirectory
if (-not $FfmpegLicenseFile) { $FfmpegLicenseFile = Join-Path $taskFfmpegRoot 'LICENSE' }
if (-not $FfmpegReadmeFile) { $FfmpegReadmeFile = Join-Path $taskFfmpegRoot 'README.txt' }
foreach ($taskNotice in @($FfmpegLicenseFile, $FfmpegReadmeFile)) {
    if (-not (Test-Path -LiteralPath $taskNotice -PathType Leaf)) {
        throw 'FFmpeg notices missing. Specify -FfmpegLicenseFile and -FfmpegReadmeFile from the same build.'
    }
}
if (-not (Test-Path -LiteralPath 'assets\demo.mp4')) {
    & $taskPython -c "from lockframe.demo import create_demo; create_demo('assets/demo.mp4')"
    if ($LASTEXITCODE -ne 0) { throw 'Demo creation failed.' }
}
$taskArguments = @(
    '-m', 'PyInstaller', '--noconfirm', '--clean', '--windowed',
    '--name', $ReleaseName, '--paths', $PSScriptRoot,
    '--distpath', $taskDistRoot,
    '--icon', (Join-Path $PSScriptRoot 'assets\tomato.ico'),
    '--add-data', ((Join-Path $PSScriptRoot 'assets') + ';assets'),
    '--add-data', ((Join-Path $PSScriptRoot 'LICENSE') + ';.'),
    '--add-data', ((Join-Path $PSScriptRoot 'THIRD_PARTY_NOTICES.md') + ';.'),
    '--copy-metadata', 'numpy',
    '--copy-metadata', 'opencv-python-headless',
    '--copy-metadata', 'PySide6_Essentials',
    '--copy-metadata', 'shiboken6',
    '--exclude-module', 'torch',
    '--exclude-module', 'rife_ncnn_vulkan_python',
    '--exclude-module', 'PIL',
    '--add-binary', ((Join-Path $FfmpegDirectory 'ffmpeg.exe') + ';bin'),
    '--add-binary', ((Join-Path $FfmpegDirectory 'ffprobe.exe') + ';bin'),
    '--exclude-module', 'PySide6.QtWebEngineCore',
    '--exclude-module', 'PySide6.QtWebEngineWidgets',
    '--exclude-module', 'PySide6.QtQml',
    '--exclude-module', 'PySide6.QtQuick',
    (Join-Path $PSScriptRoot 'main.py')
)
$taskSpecPath = if ($WorkPath) {
    Join-Path (Split-Path -Parent ([IO.Path]::GetFullPath($WorkPath))) 'spec'
} else {
    Join-Path $taskDistRoot '_spec'
}
New-Item -ItemType Directory -Path $taskSpecPath -Force | Out-Null
$taskArguments += @('--specpath', $taskSpecPath)
if ($WorkPath) {
    $taskArguments += @('--workpath', [IO.Path]::GetFullPath($WorkPath))
}
# Isolate DLL discovery from unrelated tools on the workstation PATH (e.g.
# Poppler's ICU DLL has the same basename as Windows ICU but incompatible exports).
$taskOriginalPath = $env:PATH
$taskOriginalPyInstallerConfig = $env:PYINSTALLER_CONFIG_DIR
$taskBuildExit = 0
try {
    $env:PYINSTALLER_CONFIG_DIR = Join-Path (Split-Path -Parent $taskSpecPath) 'pyinstaller-cache'
    $env:PATH = @((Join-Path $env:WINDIR 'System32'), $env:WINDIR,
                  (Split-Path -Parent $taskPython),
                  (Join-Path $PSScriptRoot '.venv\Lib\site-packages\PySide6'),
                  $FfmpegDirectory) -join ';'
    & $taskPython @taskArguments
    $taskBuildExit = $LASTEXITCODE
} finally {
    $env:PATH = $taskOriginalPath
    if ($null -eq $taskOriginalPyInstallerConfig) {
        Remove-Item Env:PYINSTALLER_CONFIG_DIR -ErrorAction SilentlyContinue
    } else {
        $env:PYINSTALLER_CONFIG_DIR = $taskOriginalPyInstallerConfig
    }
}
if ($taskBuildExit -ne 0) { throw 'PyInstaller build failed.' }
Copy-Item -LiteralPath 'README.md','VALIDATION.md','THIRD_PARTY_NOTICES.md','LICENSE','documentation','docs' -Destination $taskRelease -Recurse -Force
$taskSources = Join-Path $taskRelease 'source'
New-Item -ItemType Directory -Path $taskSources -Force | Out-Null
Copy-Item -LiteralPath 'assets','main.py','pytest.ini','requirements.txt','requirements-dev.txt','requirements-lock.txt','build_windows.ps1','启动源码版.cmd','README.md','VALIDATION.md','THIRD_PARTY_NOTICES.md','LICENSE','documentation','docs' -Destination $taskSources -Recurse -Force
# Copy source files without retaining bytecode for modules deleted in upgrades.
foreach ($taskTree in @('lockframe', 'tests')) {
    $taskTreeRoot = (Resolve-Path -LiteralPath $taskTree).Path
    Get-ChildItem -LiteralPath $taskTreeRoot -Recurse -File | Where-Object {
        $_.FullName -notmatch '[\\/](__pycache__|\.pytest_cache|\.git)[\\/]'
    } | ForEach-Object {
        $taskRelative = $_.FullName.Substring($taskTreeRoot.Length).TrimStart([char]92, [char]47)
        $taskDestination = Join-Path (Join-Path $taskSources $taskTree) $taskRelative
        New-Item -ItemType Directory -Path (Split-Path -Parent $taskDestination) -Force | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $taskDestination -Force
    }
}
foreach ($taskProject in @('lockOn', 'StableFlowPro')) {
    $taskOrigin = Join-Path $PSScriptRoot ('third_party\' + $taskProject)
    $taskCopy = Join-Path $taskSources ('third_party\' + $taskProject)
    New-Item -ItemType Directory -Path $taskCopy -Force | Out-Null
    Get-ChildItem -LiteralPath $taskOrigin -Force | Where-Object { $_.Name -ne '.git' } | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $taskCopy -Recurse -Force
    }
}
$taskLicenses = Join-Path $taskRelease 'licenses'
New-Item -ItemType Directory -Path $taskLicenses -Force | Out-Null
Copy-Item -LiteralPath 'third_party\lockOn\LICENSE.md' -Destination (Join-Path $taskLicenses 'lockOn-GPL3.txt') -Force
Copy-Item -LiteralPath 'third_party\StableFlowPro\LICENSE' -Destination (Join-Path $taskLicenses 'StableFlowPro-MIT.txt') -Force
Copy-Item -LiteralPath $FfmpegLicenseFile -Destination (Join-Path $taskLicenses 'FFmpeg-LICENSE.txt') -Force
Copy-Item -LiteralPath $FfmpegReadmeFile -Destination (Join-Path $taskLicenses 'FFmpeg-README.txt') -Force
$taskLauncher = "@echo off`r`nsetlocal`r`nif not exist `"%~dp0$ReleaseName.exe`" (echo Please extract the entire ZIP first. & pause & exit /b 1)`r`nstart `"`" /D `"%~dp0`" `"%~dp0$ReleaseName.exe`" %*`r`n"
[IO.File]::WriteAllText((Join-Path $taskRelease '启动番茄跟踪.cmd'), $taskLauncher, [Text.Encoding]::ASCII)
Write-Output "READY: $taskRelease"
