[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SshHost,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 65535)]
    [int]$SshPort,

    [string]$SshUser = "root",
    [string]$RemoteRepo = "~/FlashAV2AV",

    [ValidateRange(1, 65535)]
    [int]$LocalPort = 6008,

    [ValidateRange(1, 65535)]
    [int]$RemotePort = 7860,

    [string]$ExpectedEngineVersion = "FLASHAV2AV_0.1.0",
    [string]$IdentityFile = "",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

function ConvertTo-PosixSingleQuoted {
    param([Parameter(Mandatory = $true)][string]$Value)
    $singleQuoteEscape = "'" + '"' + "'" + '"' + "'"
    return "'" + $Value.Replace("'", $singleQuoteEscape) + "'"
}

function ConvertTo-WindowsCommandLineArgument {
    param([Parameter(Mandatory = $true)][string]$Value)

    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') {
        return $Value
    }

    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes++
            continue
        }
        if ($character -eq '"') {
            [void]$builder.Append(('\' * (2 * $backslashes + 1)))
            [void]$builder.Append('"')
            $backslashes = 0
            continue
        }
        if ($backslashes -gt 0) {
            [void]$builder.Append(('\' * $backslashes))
            $backslashes = 0
        }
        [void]$builder.Append($character)
    }
    if ($backslashes -gt 0) {
        [void]$builder.Append(('\' * (2 * $backslashes)))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

$sshCommand = Get-Command ssh.exe -ErrorAction SilentlyContinue
if ($null -eq $sshCommand) {
    throw "Windows OpenSSH client ssh.exe was not found. Install OpenSSH Client first."
}

$existingListener = Get-NetTCPConnection `
    -State Listen `
    -LocalPort $LocalPort `
    -ErrorAction SilentlyContinue
if ($null -ne $existingListener) {
    throw "Local port $LocalPort is occupied. This script will not terminate an unknown process; close the old tunnel or choose -LocalPort."
}

if ($IdentityFile) {
    $resolvedIdentity = (Resolve-Path -LiteralPath $IdentityFile).Path
}

$quotedRepo = ConvertTo-PosixSingleQuoted -Value $RemoteRepo
$quotedExpectedEngineVersion = ConvertTo-PosixSingleQuoted -Value $ExpectedEngineVersion
$launchToken = [Guid]::NewGuid().ToString("N")
$quotedLaunchToken = ConvertTo-PosixSingleQuoted -Value $launchToken
$remotePayload = @"
cd -- $quotedRepo && DEMO_LAUNCH_TOKEN=$quotedLaunchToken EXPECTED_ENGINE_VERSION=$quotedExpectedEngineVersion PORT=$RemotePort bash scripts/run_demo.sh start && printf '\nREMOTE_DEMO_READY\n' && exec bash -c 'trap "exit 0" INT TERM HUP; while true; do sleep 3600; done'
"@.Trim()

$sshArguments = @(
    "-tt",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=20",
    "-o", "ServerAliveCountMax=3",
    "-p", $SshPort.ToString(),
    "-L", "127.0.0.1:${LocalPort}:127.0.0.1:${RemotePort}"
)
if ($IdentityFile) {
    $sshArguments += @("-i", $resolvedIdentity)
}
$sshArguments += @("${SshUser}@${SshHost}", $remotePayload)

$argumentLine = ($sshArguments | ForEach-Object {
    ConvertTo-WindowsCommandLineArgument -Value ([string]$_)
}) -join " "

Write-Host "Opening an SSH window. Enter the server password there unless -IdentityFile is set."
$sshProcess = Start-Process `
    -FilePath $sshCommand.Source `
    -ArgumentList $argumentLine `
    -PassThru

$healthUrl = "http://127.0.0.1:$LocalPort/health"
$demoUrl = "http://127.0.0.1:$LocalPort/?v=$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
$deadline = [DateTime]::UtcNow.AddMinutes(6)
$ready = $false

try {
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($sshProcess.HasExited) {
            throw "SSH exited with code $($sshProcess.ExitCode). Inspect the SSH window for the error."
        }
        try {
            $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 3
            $ready = (
                $health.status -eq "ok" -and
                $health.engine_version -eq $ExpectedEngineVersion -and
                $health.launch_ready -eq $true -and
                $health.launch_token -eq $launchToken -and
                $health.dialog_backend -eq "pipecat" -and
                $health.frame_ready -eq $true -and
                $health.workers.motion_alive -eq $true -and
                $health.workers.render_alive -eq $true -and
                $health.dialog_session.ready -eq $true -and
                $health.dialog_session.mode -eq "native_s2s" -and
                $health.dialog_session.s2s_ready -eq $true
            )
            if ($ready) {
                break
            }
        }
        catch {
            # GPU cold start and tunnel establishment normally take time.
        }
        Start-Sleep -Seconds 2
    }

    if (-not $ready) {
        throw "The service did not pass health checks within 6 minutes. SSH pid=$($sshProcess.Id); inspect remote logs/pipecat_mse.log."
    }
}
finally {
    if (-not $ready -and -not $sshProcess.HasExited) {
        Stop-Process -Id $sshProcess.Id -ErrorAction SilentlyContinue
    }
}

Write-Host "DEMO_READY tunnel_pid=$($sshProcess.Id) url=$demoUrl"
Write-Host "Keep only one demo page open. Close the SSH window to stop the tunnel."
if (-not $NoBrowser) {
    Start-Process $demoUrl
}
