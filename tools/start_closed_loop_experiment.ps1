param(
    [int]$DurationSeconds = 600,
    [int]$Camera = 0,
    [string]$CameraSource = "",
    [int]$Width = 640,
    [int]$Height = 480,
    [ValidateSet(0, 90, 180, 270)]
    [int]$Rotation = 0,
    [ValidateSet("mediapipe", "haar")]
    [string]$EyeDetector = "mediapipe",
    [int]$Port = 5055,
    [switch]$NoMirror,
    [switch]$KeepBlenderOpen,
    [string]$Scene = "D:\emotion3dgs_results\closed_loop_e2e\level_01_closed_loop_difficulty_v4.blend"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = if ($EyeDetector -eq "mediapipe") {
    "D:\emotion_eye_env312\Scripts\python.exe"
} else {
    "D:\emotion3dgs_env\Scripts\python.exe"
}
$blender = "C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"
$controller = Join-Path $repo "tools\blender_udp_difficulty_controller_experiment.py"
$runtime = Join-Path $repo "tools\live_closed_loop_end_to_end.py"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = "D:\emotion3dgs_results\closed_loop_e2e\run_$timestamp"

if (-not (Test-Path -LiteralPath $Scene)) {
    throw "No existe la escena experimental: $Scene"
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "No existe Python: $python"
}
if (-not (Test-Path -LiteralPath $blender)) {
    throw "No existe Blender: $blender"
}
$existingListener = Get-NetUDPEndpoint -LocalPort $Port -ErrorAction SilentlyContinue
if ($existingListener) {
    throw "El puerto UDP $Port esta ocupado. Use otro puerto o cierre el listener anterior."
}

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$env:CLOSED_LOOP_EXPERIMENT_LOG = Join-Path $runDir "blender_difficulty_events.csv"
$env:CLOSED_LOOP_EXPERIMENT_PORT = [string]$Port
$env:TEMP = "D:\emotion3dgs_temp"
$env:TMP = "D:\emotion3dgs_temp"

$blenderProcess = Start-Process `
    -FilePath $blender `
    -ArgumentList @($Scene, "--python", $controller) `
    -PassThru

Write-Host "Blender iniciado. Esperando el listener UDP..."
Start-Sleep -Seconds 7
if ($blenderProcess.HasExited) {
    throw "Blender se cerro antes de iniciar la corrida."
}

Write-Host "Iniciando corrida webcam de $DurationSeconds segundos."
Write-Host "Mire hacia la camara y mueva la mirada de forma natural. Presione Q para detener."

Push-Location $repo
try {
    $runtimeArgs = @(
        $runtime,
        "--camera", $Camera,
        "--width", $Width,
        "--height", $Height,
        "--rotation", $Rotation,
        "--eye-detector", $EyeDetector,
        "--port", $Port,
        "--duration-seconds", $DurationSeconds,
        "--output-dir", $runDir
    )
    if ($CameraSource) {
        $runtimeArgs += @("--camera-source", $CameraSource)
    }
    if ($NoMirror) {
        $runtimeArgs += "--no-mirror"
    }
    & $python @runtimeArgs
}
finally {
    Pop-Location
    if (-not $KeepBlenderOpen -and -not $blenderProcess.HasExited) {
        $null = $blenderProcess.CloseMainWindow()
        if (-not $blenderProcess.WaitForExit(5000)) {
            Stop-Process -Id $blenderProcess.Id
        }
    }
}

Write-Host ""
Write-Host "Corrida terminada: $runDir"
if ($KeepBlenderOpen) {
    Write-Host "Blender permanece abierto porque se uso -KeepBlenderOpen."
} else {
    Write-Host "Blender experimental cerrado para liberar el puerto UDP."
}
