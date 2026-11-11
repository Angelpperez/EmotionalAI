param(
    [string]$RuntimeRoot = 'D:\EmotionRecognitionRuntime\affectnet_replication',
    [string]$PythonVersion = '3.11.18'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot
$runtimePath = [System.IO.Path]::GetFullPath($RuntimeRoot)
New-Item -ItemType Directory -Path (Join-Path $runtimePath 'temp') -Force | Out-Null
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
$previousCache = $env:PIP_CACHE_DIR
try {
    $env:TEMP = Join-Path $runtimePath 'temp'
    $env:TMP = $env:TEMP
    $env:PIP_CACHE_DIR = Join-Path $runtimePath 'cache\pip'
    $uvExe = Join-Path $projectRoot '.python\uv-tool\bin\uv.exe'
    if (-not (Test-Path -LiteralPath $uvExe)) {
        python -m pip install --target '.python\uv-tool' 'uv==0.13.7'
        if ($LASTEXITCODE -ne 0) { throw 'uv bootstrap failed.' }
    }
    $managedPythonRoot = Join-Path $runtimePath 'python'
    & $uvExe python install $PythonVersion --install-dir $managedPythonRoot --no-bin --no-registry --cache-dir (Join-Path $runtimePath 'cache\uv')
    if ($LASTEXITCODE -ne 0) { throw 'Portable Python installation failed.' }
    $managedPython = Join-Path $managedPythonRoot "cpython-$PythonVersion-windows-x86_64-none\python.exe"
    $projectEnvironment = Join-Path $projectRoot '.venv311'
    if (-not (Test-Path -LiteralPath $projectEnvironment)) {
        $runtimeEnvironment = Join-Path $runtimePath '.venv311'
        if (-not (Test-Path -LiteralPath $runtimeEnvironment)) {
            & $uvExe venv $runtimeEnvironment --python $managedPython --seed --cache-dir (Join-Path $runtimePath 'cache\uv')
            if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
        }
        New-Item -ItemType Junction -Path $projectEnvironment -Target $runtimeEnvironment | Out-Null
    }
    $venvPython = Join-Path $projectEnvironment 'Scripts\python.exe'
    & $venvPython -c 'import sys; assert sys.version_info[:2] == (3,11), sys.version; print(sys.executable, sys.version)'
    if ($LASTEXITCODE -ne 0) { throw 'The existing environment is not Python 3.11.' }
    if (Test-Path -LiteralPath (Join-Path $projectRoot 'requirements_freeze.txt')) {
        & $venvPython -m pip install -r requirements_freeze.txt --extra-index-url https://download.pytorch.org/whl/cu126
        if ($LASTEXITCODE -ne 0) { throw 'Frozen dependency installation failed.' }
    } else {
        & $venvPython -m pip install 'torch==2.6.0' 'torchvision==0.21.0' --index-url https://download.pytorch.org/whl/cu126
        if ($LASTEXITCODE -ne 0) { throw 'PyTorch CUDA installation failed.' }
        & $venvPython -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    }
    & $venvPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed.' }
} finally {
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
    $env:PIP_CACHE_DIR = $previousCache
}
