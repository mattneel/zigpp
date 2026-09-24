# ppup installs and manages Zig++ toolchains on Windows.
#
#     irm https://zigpp.lol/ppup.ps1 | iex
#
# A toolchain is a release archive unpacked into $env:PPUP_HOME\toolchains
# \<version>, and the default toolchain is the one $env:PPUP_HOME\current
# points at. `ppup help` lists the commands.

# The path of this file, when ppup runs from a file; empty when it came in on a
# pipe. It is read outside the wrapper below, because inside it myCommand is
# the wrapper.
$ppupScriptPath = $MyInvocation.MyCommand.Path

# The implementation is one scriptblock, so that `irm | iex` (and dot-sourcing)
# does not leave helper functions in the caller's session.
& {
    param([string[]] $CommandLine)

    $PpupVersion = '0.1.0'

    # The releases of mattneel/zigpp, and the public copy of this script.
    $ReleasesUrl = 'https://github.com/mattneel/zigpp/releases'
    $PpupUrl = 'https://zigpp.lol/ppup.ps1'

    # The targets Zig++ publishes; see doc/book/src/installing.md.
    $Targets = 'x86_64-linux aarch64-linux aarch64-macos x86_64-windows'

    # Windows PowerShell 5.1 asks for TLS 1.0 by default, which GitHub and
    # zigpp.lol refuse.
    try {
        [Net.ServicePointManager]::SecurityProtocol =
            [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    } catch {
        # A host without TLS 1.2 support: the downloads below will say so.
    }

    $ProgressPreference = 'SilentlyContinue'

    # Cmdlet failures stop the command instead of printing and carrying on.
    $ErrorActionPreference = 'Stop'

    # A scratch directory inside PPUP_HOME, which every failure cleans up. The
    # flags live in a hashtable because functions can write to a hashtable
    # without reaching into a caller's variables; Dir is set once PPUP_HOME is
    # known.
    $Scratch = @{
        Dir  = ''
        Made = $false
    }

    function New-TempDir {
        if (-not $Scratch.Made) {
            New-Item -ItemType Directory -Force -Path $Scratch.Dir | Out-Null
            $Scratch.Made = $true
        }
    }

    function Remove-TempDir {
        if ($Scratch.Made -and (Test-Path -LiteralPath $Scratch.Dir)) {
            Remove-Item -LiteralPath $Scratch.Dir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    # Status messages go to the host; the values a caller may capture (the
    # default version, the toolchain list, help, --version) go to the output
    # stream through Write-Output, as a command's results should.
    function Write-Info([string] $Message) {
        Write-Host $Message
    }

    function Write-Warn([string] $Message) {
        [Console]::Error.WriteLine("ppup: $Message")
    }

    function Fail([string] $Message) {
        # Through a pipe there is no script file, and `exit` would close the
        # caller's session instead of reporting an error.
        Remove-TempDir
        if ($null -eq $ScriptFile) {
            throw "ppup: $Message"
        }
        [Console]::Error.WriteLine("ppup: $Message")
        exit 1
    }

    function Show-Usage {
        Write-Output (@'
ppup {0}: install and manage Zig++ toolchains.

Usage: ppup [<command>] [<option>...]

With no command, ppup installs the newest Zig++ release as the default
toolchain, installs itself as %PPUP_HOME%\bin\ppup.ps1, and puts
%PPUP_HOME%\current and %PPUP_HOME%\bin on the user PATH.

Commands:
  install <version|latest>
                    Install a toolchain. It becomes the default if no
                    toolchain is the default yet. <version> is as in the
                    archive names, e.g. 0.17.0-dev.2380+zigpp.add158e97.
  update            Install the newest release and make it the default
  default [<version>]
                    Show the default toolchain, or make <version> the default
  list              List the installed toolchains, marking the default
  uninstall <version>
                    Remove a toolchain, unless it is the default
  self update       Replace ppup with the newest one from {1}
  self uninstall    Remove every toolchain, ppup, and its PATH entries
  help              Show this help
  --version         Show the ppup version

Options:
  --no-modify-path  Leave the user PATH alone. The environment variable
                    PPUP_NO_MODIFY_PATH=1 does the same.

Environment:
  PPUP_HOME         Where toolchains live (default: the home that this ppup
                    is installed in, else %LOCALAPPDATA%\zigpp)

Releases are published for: {2}
'@ -f $PpupVersion, $PpupUrl, $Targets)
    }

    # ------------------------------------------------------------ environment

    $ScriptFile = $null
    if ($ppupScriptPath) {
        $ScriptFile = $ppupScriptPath
    }

    # Whether ppup may edit the user PATH; --no-modify-path and
    # PPUP_NO_MODIFY_PATH=1 turn that off.
    $ModifyPath = $true
    if ($env:PPUP_NO_MODIFY_PATH -and $env:PPUP_NO_MODIFY_PATH -ne '0') {
        $ModifyPath = $false
    }

    # An installed ppup is %PPUP_HOME%\bin\ppup.ps1, and it keeps to that home
    # when the environment names none: nothing else records a PPUP_HOME chosen
    # at install.
    $PpupHome = $env:PPUP_HOME
    if (-not $PpupHome -and $ScriptFile) {
        $scriptDir = Split-Path -Parent $ScriptFile
        $installedHome = Split-Path -Parent $scriptDir
        if ((Split-Path -Leaf $ScriptFile) -eq 'ppup.ps1' -and
            (Split-Path -Leaf $scriptDir) -eq 'bin' -and
            $installedHome -and
            (Test-Path -LiteralPath (Join-Path $installedHome 'toolchains') -PathType Container)) {
            $PpupHome = $installedHome
        }
    }
    if (-not $PpupHome) {
        if (-not $env:LOCALAPPDATA) {
            Fail 'LOCALAPPDATA is not set, so the default PPUP_HOME cannot be guessed; set PPUP_HOME'
        }
        $PpupHome = Join-Path $env:LOCALAPPDATA 'zigpp'
    }
    try {
        $PpupHome = [IO.Path]::GetFullPath($PpupHome)
    } catch {
        Fail "PPUP_HOME is not a usable path: $PpupHome"
    }
    $PpupHome = $PpupHome.TrimEnd('\', '/')
    if (-not $PpupHome) {
        Fail 'PPUP_HOME is empty'
    }

    $BinDir = Join-Path $PpupHome 'bin'
    $ToolchainsDir = Join-Path $PpupHome 'toolchains'
    $CurrentLink = Join-Path $PpupHome 'current'
    $PpupScript = Join-Path $BinDir 'ppup.ps1'
    $PpupShim = Join-Path $BinDir 'ppup.cmd'

    # Whether this is the installed ppup, bin\ppup.ps1, rather than the
    # installer: the script arriving through `irm | iex`, or run from a
    # download.
    $IsInstalledCopy = [bool]($ScriptFile -and
        ([IO.Path]::GetFullPath($ScriptFile) -eq [IO.Path]::GetFullPath($PpupScript)))

    $Scratch.Dir = Join-Path $PpupHome ('tmp-' + [IO.Path]::GetRandomFileName())

    # ------------------------------------------------------------------ hosts

    function Get-Target {
        $arch = $env:PROCESSOR_ARCHITEW6432
        if (-not $arch) {
            $arch = $env:PROCESSOR_ARCHITECTURE
        }
        switch ($arch) {
            'AMD64' { return 'x86_64-windows' }
            'ARM64' {
                Write-Info 'Zig++ publishes no ARM64 build for Windows, so ppup installs the x86_64 one, which runs under emulation.'
                return 'x86_64-windows'
            }
            default {
                Fail "no Zig++ release is published for Windows/$arch; Zig++ publishes: $Targets"
            }
        }
    }

    function Get-File([string] $Url, [string] $Path) {
        $client = New-Object System.Net.WebClient
        $client.Headers.Add('User-Agent', "ppup/$PpupVersion")
        try {
            $client.DownloadFile($Url, $Path)
        } finally {
            $client.Dispose()
        }
    }

    function Get-Sha256([string] $Path) {
        (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    }

    function Expand-Zip([string] $Archive, [string] $Destination) {
        try {
            Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction Stop
            [System.IO.Compression.ZipFile]::ExtractToDirectory($Archive, $Destination)
        } catch {
            Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
        }
    }

    # --------------------------------------------------------------- releases

    # Resolve-Release <version|latest> reads the release's SHA256SUMS and
    # returns the version, the tag, the archive name, its SHA-256, and its URL.
    function Resolve-Release([string] $Request) {
        $target = Get-Target
        if (-not $Request -or $Request -eq 'latest') {
            $sumsUrl = "$ReleasesUrl/latest/download/SHA256SUMS"
        } else {
            $sumsUrl = "$ReleasesUrl/download/zigpp-$($Request.Split('+')[0])/SHA256SUMS"
        }

        New-TempDir
        $sums = Join-Path $Scratch.Dir 'SHA256SUMS'
        try {
            Get-File $sumsUrl $sums
        } catch {
            Fail "cannot download $sumsUrl (no such release, or no network?): $($_.Exception.Message)"
        }

        $prefix = "zig-$target-"
        $pattern = '^[0-9a-f]{64}  ' + [regex]::Escape($prefix) + '.*\.zip$'
        $line = Get-Content -LiteralPath $sums |
            Where-Object { $_ -match $pattern } |
            Select-Object -First 1
        if (-not $line) {
            Fail "$sumsUrl has no $target archive; Zig++ publishes: $Targets"
        }

        # Each line is "<sha256>  <file name>".
        $fields = $line -split '\s+'
        $sha256 = $fields[0]
        $name = $fields[1]
        if (-not $name.StartsWith($prefix) -or -not $name.EndsWith('.zip')) {
            Fail "$sumsUrl names an archive ppup does not understand: $name"
        }
        $version = $name.Substring($prefix.Length, $name.Length - $prefix.Length - '.zip'.Length)

        if ($Request -and $Request -ne 'latest' -and $version.Split('+')[0] -ne $Request.Split('+')[0]) {
            Fail "release $sumsUrl does not contain $Request"
        }

        $tag = "zigpp-$($version.Split('+')[0])"
        [pscustomobject]@{
            Version = $version
            Name    = $name
            Sha256  = $sha256
            Url     = "$ReleasesUrl/download/$tag/$name"
        }
    }

    function Install-Toolchain([string] $Request) {
        $release = Resolve-Release $Request
        $aim = Join-Path $ToolchainsDir $release.Version
        if (Test-Path -LiteralPath $aim) {
            Write-Info "Zig++ $($release.Version) is already installed"
            return $release.Version
        }

        Write-Info "downloading Zig++ $($release.Version) for $(Get-Target)"
        $archive = Join-Path $Scratch.Dir $release.Name
        try {
            Get-File $release.Url $archive
        } catch {
            Fail "cannot download $($release.Url): $($_.Exception.Message)"
        }

        $got = Get-Sha256 $archive
        if ($got -ne $release.Sha256) {
            Fail "checksum mismatch for $($release.Name): expected $($release.Sha256), got $got"
        }

        $unpacked = Join-Path $Scratch.Dir 'unpacked'
        New-Item -ItemType Directory -Force -Path $unpacked | Out-Null
        try {
            Expand-Zip $archive $unpacked
        } catch {
            Fail "cannot unpack $($release.Name): $($_.Exception.Message)"
        }

        $src = Join-Path $unpacked "zig-$(Get-Target)-$($release.Version)"
        if (-not (Test-Path -LiteralPath $src)) {
            # Fall back to the archive's single top-level directory.
            $dirs = @(Get-ChildItem -LiteralPath $unpacked -Directory)
            if ($dirs.Count -ne 1) {
                Fail "$($release.Name) does not hold one top-level directory"
            }
            $src = $dirs[0].FullName
        }
        if (-not (Test-Path -LiteralPath (Join-Path $src 'zig.exe'))) {
            Fail "$($release.Name) does not hold a zig.exe"
        }

        New-Item -ItemType Directory -Force -Path $ToolchainsDir | Out-Null
        Move-Item -LiteralPath $src -Destination $aim
        Write-Info "installed Zig++ $($release.Version) in $aim"
        return $release.Version
    }

    # ------------------------------------------------------------- toolchains

    function Get-DefaultVersion {
        if (-not (Test-Path -LiteralPath $CurrentLink)) {
            return $null
        }
        $item = Get-Item -LiteralPath $CurrentLink -Force
        $target = @($item.Target)[0]
        if (-not $target) {
            return $null
        }
        $target = [string] $target
        if ((Split-Path -Leaf (Split-Path -Parent $target)) -ne 'toolchains') {
            return $null
        }
        return (Split-Path -Leaf $target)
    }

    # Whether the current junction is there, even when its toolchain is gone:
    # such a junction does not show up in Directory.Exists, but the parent
    # directory still lists it.
    function Test-LinkPresent {
        if ([System.IO.Directory]::Exists($CurrentLink)) {
            return $true
        }
        $parent = Split-Path -Parent $CurrentLink
        if (-not [System.IO.Directory]::Exists($parent)) {
            return $false
        }
        $leaf = Split-Path -Leaf $CurrentLink
        foreach ($entry in [System.IO.Directory]::GetFileSystemEntries($parent)) {
            if ((Split-Path -Leaf $entry) -eq $leaf) {
                return $true
            }
        }
        return $false
    }

    function Remove-Junction {
        if (-not (Test-LinkPresent)) {
            return
        }
        try {
            # Deletes the junction itself, never its target.
            [System.IO.Directory]::Delete($CurrentLink, $false)
        } catch {
            # A junction whose toolchain is gone needs rmdir.
            $previous = $ErrorActionPreference
            $ErrorActionPreference = 'SilentlyContinue'
            cmd /c rmdir "$CurrentLink" | Out-Null
            $ErrorActionPreference = $previous
        }
    }

    # The installed toolchain named $Version, which may leave out the build
    # metadata: 0.17.0-dev.2380 finds 0.17.0-dev.2380+zigpp.add158e97.
    function Resolve-Installed([string] $Version) {
        if (Test-Path -LiteralPath (Join-Path (Join-Path $ToolchainsDir $Version) 'zig.exe')) {
            return $Version
        }
        $found = @(Get-InstalledVersions | Where-Object { $_ -like "$Version+*" })
        if ($found.Count -gt 1) {
            Fail "several installed toolchains match ${Version}: use the full version from 'ppup list'"
        }
        if ($found.Count -eq 1) {
            return $found[0]
        }
        Fail "no such toolchain: $Version (see: ppup list)"
    }

    function Set-DefaultVersion([string] $Version) {
        $resolved = Resolve-Installed $Version
        $dir = Join-Path $ToolchainsDir $resolved
        New-Item -ItemType Directory -Force -Path $PpupHome | Out-Null
        Remove-Junction
        New-Item -ItemType Junction -Path $CurrentLink -Target $dir | Out-Null
    }

    function Get-InstalledVersions {
        if (-not (Test-Path -LiteralPath $ToolchainsDir)) {
            return @()
        }
        $versions = @()
        foreach ($dir in @(Get-ChildItem -LiteralPath $ToolchainsDir -Directory)) {
            $versions += $dir.Name
        }
        return @($versions | Sort-Object -Descending)
    }

    # ------------------------------------------------------------------- PATH

    function Get-UserPath {
        $raw = [Environment]::GetEnvironmentVariable('Path', 'User')
        if ($null -eq $raw) {
            return ''
        }
        return $raw
    }

    function Set-UserPath([string] $Value) {
        if (-not (Test-Path 'HKCU:\Environment')) {
            New-Item -Path 'HKCU:\Environment' -Force | Out-Null
        }
        # Keep REG_EXPAND_SZ, so that entries like %USERPROFILE%\... still
        # expand.
        $kind = 'String'
        try {
            if ((Get-Item 'HKCU:\Environment').GetValueKind('Path') -eq 'ExpandString') {
                $kind = 'ExpandString'
            }
        } catch {
            # No Path value yet.
        }
        Set-ItemProperty -Path 'HKCU:\Environment' -Name 'Path' -Value $Value -Type $kind
    }

    function Add-ToPath {
        # The user PATH is the installer's business: the installed ppup leaves
        # it as the install, or its --no-modify-path, left it.
        if (-not $ModifyPath -or $IsInstalledCopy) {
            return
        }
        $entries = @($CurrentLink, $BinDir)

        $userPath = Get-UserPath
        $parts = @($userPath -split ';' | Where-Object { $_ })
        $missing = @($entries | Where-Object { $parts -notcontains $_ })
        if ($missing.Count -gt 0) {
            Set-UserPath (@($missing + $parts) -join ';')
            Write-Info "added $($missing -join ' and ') to the user PATH"
        }

        foreach ($entry in $entries) {
            if (@($env:Path -split ';' | Where-Object { $_ }) -notcontains $entry) {
                $env:Path = "$entry;$env:Path"
            }
        }
    }

    function Remove-FromPath {
        $entries = @($CurrentLink, $BinDir)

        $raw = Get-UserPath
        $parts = @($raw -split ';' | Where-Object { $_ -and ($entries -notcontains $_) })
        $new = ($parts -join ';')
        if ($new -ne $raw) {
            Set-UserPath $new
            Write-Info 'removed the Zig++ entries from the user PATH'
        }

        $env:Path = (@($env:Path -split ';' | Where-Object { $_ -and ($entries -notcontains $_) }) -join ';')
    }

    # ---------------------------------------------------------- ppup itself

    function Write-Shim {
        $shim = "@echo off`r`npowershell -NoProfile -ExecutionPolicy Bypass -File `"%~dp0ppup.ps1`" %*`r`n"
        [IO.File]::WriteAllText($PpupShim, $shim, (New-Object System.Text.ASCIIEncoding))
    }

    function Install-Ppup {
        New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

        if ($ScriptFile -and -not $IsInstalledCopy) {
            # Running from a file: install that file.
            Copy-Item -LiteralPath $ScriptFile -Destination $PpupScript -Force
            Write-Info "installed ppup $PpupVersion as $PpupScript"
        } elseif (-not (Test-Path -LiteralPath $PpupScript)) {
            # Running from a pipe: PowerShell does not keep the text of what
            # `iex` was given, so fetch the published script.
            try {
                Get-File $PpupUrl $PpupScript
                Write-Info "installed ppup $PpupVersion as $PpupScript"
            } catch {
                Write-Warn "could not download $PpupUrl, so the ppup command was not installed; zig works, and 'ppup self update' can get it later"
            }
        }
        Write-Shim
    }

    # --------------------------------------------------------------- commands

    function Invoke-InstallDefault {
        $version = Install-Toolchain 'latest'
        Install-Ppup
        Set-DefaultVersion $version
        Write-Info "Zig++ $version is now the default toolchain"
        Add-ToPath
        Show-HowToRun
    }

    function Invoke-Install([string] $Request) {
        if (-not $Request) {
            Fail "install needs a version, or 'latest'"
        }
        $version = Install-Toolchain $Request
        Install-Ppup
        $default = Get-DefaultVersion
        if ($default -and $default -ne $version) {
            Write-Info "Zig++ $version is not the default: $default is"
        } elseif (-not $default) {
            Set-DefaultVersion $version
            Write-Info "Zig++ $version is now the default toolchain"
        }
        Add-ToPath
        Show-HowToRun
    }

    function Invoke-Update {
        $version = Install-Toolchain 'latest'
        Install-Ppup
        if ((Get-DefaultVersion) -eq $version) {
            Write-Info "Zig++ $version is the newest release, and it is already the default"
        } else {
            Set-DefaultVersion $version
            Write-Info "Zig++ $version is now the default toolchain"
        }
        Add-ToPath
        Show-HowToRun
    }

    function Show-HowToRun {
        if ($ModifyPath -and (@($env:Path -split ';' | Where-Object { $_ }) -notcontains $CurrentLink)) {
            Write-Info ''
            Write-Info 'To use zig in this shell, run:'
            Write-Info "    `$env:Path = `"$CurrentLink;`$env:Path`""
        }
        Write-Info ''
        Write-Info "Run 'zig version' to check the toolchain, or 'ppup help' for the rest."
    }

    function Invoke-Default([string] $Version) {
        if (-not $Version) {
            $default = Get-DefaultVersion
            if ($default) {
                Write-Output $default
            } else {
                # Like the Unix ppup: the answer is that there is no default.
                Write-Output 'no default toolchain; set one with: ppup default <version>'
            }
            return
        }
        Set-DefaultVersion $Version
        Write-Info "$(Get-DefaultVersion) is now the default toolchain"
    }

    function Show-List {
        $versions = @(Get-InstalledVersions)
        if ($versions.Count -eq 0) {
            Write-Output 'no toolchains installed; install one with: ppup install <version>'
            return
        }
        $default = Get-DefaultVersion
        foreach ($version in $versions) {
            if ($version -eq $default) {
                Write-Output "* $version (default)"
            } else {
                Write-Output "  $version"
            }
        }
    }

    function Invoke-Uninstall([string] $Version) {
        if (-not $Version) {
            Fail 'uninstall needs a version (see: ppup list)'
        }
        $resolved = Resolve-Installed $Version
        if ((Get-DefaultVersion) -eq $resolved) {
            Fail "$resolved is the default toolchain; make another one the default first: ppup default <version>"
        }
        Remove-Item -LiteralPath (Join-Path $ToolchainsDir $resolved) -Recurse -Force
        Write-Info "removed Zig++ $resolved"
    }

    function Invoke-SelfUpdate {
        New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
        New-TempDir
        $new = Join-Path $Scratch.Dir 'ppup.ps1'
        Write-Info "downloading ppup from $PpupUrl"
        try {
            Get-File $PpupUrl $new
        } catch {
            Fail "cannot download ${PpupUrl}: $($_.Exception.Message)"
        }
        if (-not (Select-String -LiteralPath $new -Pattern 'PPUP_HOME' -Quiet)) {
            Fail "$PpupUrl does not look like ppup; leaving the installed one alone"
        }
        Copy-Item -LiteralPath $new -Destination $PpupScript -Force
        Write-Shim
        Write-Info "updated $PpupScript"
    }

    function Invoke-SelfUninstall {
        Remove-Junction
        if ($ModifyPath) {
            Remove-FromPath
        }
        if (Test-Path -LiteralPath $PpupHome) {
            Remove-Item -LiteralPath $PpupHome -Recurse -Force
            Write-Info "removed $PpupHome"
        } else {
            Write-Info "nothing to remove: $PpupHome does not exist"
        }
        Write-Info 'Zig++ toolchains and ppup are gone; restart your shell to drop the old PATH entries'
    }

    # --------------------------------------------------------------- dispatch

    $Command = ''
    $Argument = ''
    $ShowHelp = $false
    $ShowVersion = $false
    $positional = @()

    foreach ($a in @($CommandLine)) {
        if ($a -match '^--?no-modify-path$' -or $a -match '^-NoModifyPath$') {
            $ModifyPath = $false
        } elseif ($a -eq '--') {
            # Consumed by callers that pass it, e.g. `powershell -Command`.
        } elseif ($a -match '^(-h|--help)$') {
            $ShowHelp = $true
        } elseif ($a -match '^(-V|-v|--version)$') {
            $ShowVersion = $true
        } elseif ($a -match '^-') {
            Fail "unknown option: $a (try: ppup help)"
        } else {
            $positional += $a
        }
    }

    if ($positional.Count -gt 0) {
        $Command = $positional[0]
    }
    if ($positional.Count -gt 1) {
        $Argument = $positional[1]
    }
    if ($positional.Count -gt 2) {
        Fail "unexpected argument: $($positional[2]) (try: ppup help)"
    }

    try {
        if ($ShowHelp -or $Command -eq 'help') {
            Show-Usage
        } elseif ($ShowVersion) {
            Write-Output "ppup $PpupVersion"
        } elseif ($Command -eq '') {
            Invoke-InstallDefault
        } elseif ($Command -eq 'install') {
            Invoke-Install $Argument
        } elseif ($Command -eq 'update') {
            Invoke-Update
        } elseif ($Command -eq 'default') {
            Invoke-Default $Argument
        } elseif ($Command -eq 'list') {
            Show-List
        } elseif ($Command -eq 'uninstall') {
            Invoke-Uninstall $Argument
        } elseif ($Command -eq 'self') {
            if ($Argument -eq 'update') {
                Invoke-SelfUpdate
            } elseif ($Argument -eq 'uninstall') {
                Invoke-SelfUninstall
            } elseif (-not $Argument) {
                Fail 'self needs a command: ppup self update, or ppup self uninstall'
            } else {
                Fail "unknown self command: $Argument (try: ppup help)"
            }
        } else {
            Fail "unknown command: $Command (try: ppup help)"
        }
    } finally {
        Remove-TempDir
    }
} @($args)
