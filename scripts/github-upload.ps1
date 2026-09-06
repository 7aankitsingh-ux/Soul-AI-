<#
.SYNOPSIS
  One-command upload of Soul AI to GitHub.

.DESCRIPTION
  Initializes git if needed, makes the first commit, adds a "main" branch,
  wires the given GitHub remote, and pushes. After this you can just run
  git add -A; git commit -m "..."; git push

.EXAMPLE
  .\scripts\github-upload.ps1 -Repo https://github.com/you/soul-ai
#>
param(
  [Parameter(Mandatory = $true)]
  [string]$Repo,
  [string]$Branch = "main",
  [string]$CommitMessage = "Initial commit"
)

$ErrorActionPreference = "Stop"

function Run-Git {
  param([string[]]$Args)
  & git @Args
  if ($LASTEXITCODE -ne 0) { throw "git $($Args -join ' ') failed" }
}

git --version | Out-Null
if ($LASTEXITCODE -ne 0) { throw "git is not installed." }

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$userName = git config user.name
$userEmail = git config user.email
if (-not $userName -or -not $userEmail) {
  Write-Host "`nGit identity is not set. Run these once:"
  Write-Host "  git config --global user.name `"Your Name`""
  Write-Host "  git config --global user.email `"you@example.com`""
  throw "Aborting: set your git user.name / user.email first."
}

if (-not (Test-Path ".git")) {
  Write-Host "Initializing repository..."
  Run-Git @("init", "-b", $Branch)
} else {
  Write-Host "Repository already initialized."
}

Run-Git @("add", "-A")

$status = git status --porcelain
if ($status) {
  Run-Git @("commit", "-m", $CommitMessage)
} else {
  Write-Host "Nothing to commit."
}

$remotes = git remote
if ($remotes -notmatch "^origin$") {
  Run-Git @("remote", "add", "origin", $Repo)
} else {
  Write-Host "Remote 'origin' already exists."
  git remote set-url origin $Repo
}

Write-Host "Pushing to $Repo ..."
Run-Git @("push", "-u", "origin", $Branch)

Write-Host "`nDone! Repo is live at $Repo"