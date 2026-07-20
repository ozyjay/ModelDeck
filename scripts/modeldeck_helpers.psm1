function Resolve-ModelDeckWorker {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$ManagementUrl,
        [string]$Worker,
        [string]$ModelId,
        [string]$Runtime
    )

    # Invoke-RestMethod deliberately writes a top-level JSON array as one pipeline object.
    # Enumerate it explicitly so Where-Object compares one Worker at a time.
    $Response = Invoke-RestMethod -Uri "$ManagementUrl/api/workers" -TimeoutSec 10
    $Workers = @($Response.GetEnumerator())
    $Matches = if ($Worker) {
        @($Workers | Where-Object { $_.id -eq $Worker -or $_.name -eq $Worker })
    }
    else {
        @($Workers | Where-Object {
            (-not $ModelId -or $_.model_id -eq $ModelId) -and
            (-not $Runtime -or $_.runtime -eq $Runtime)
        })
    }
    if ($Matches.Count -eq 0) {
        throw 'No configured Worker matches the requested identity. Create it in ModelDeck first.'
    }
    if ($Matches.Count -gt 1) {
        throw "More than one Worker matches. Supply -Worker with its editable name or internal ID: $($Matches.name -join ', ')"
    }
    return $Matches[0]
}

function Resolve-ModelDeckRoute {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$ManagementUrl,
        [Parameter(Mandatory)][string]$WorkerId,
        [string]$PublicName
    )

    $Live = Invoke-RestMethod -Uri "$ManagementUrl/api/live" -TimeoutSec 10
    $Matches = @($Live.routes | Where-Object {
        $_.worker_ids -contains $WorkerId -and (-not $PublicName -or $_.public_name -eq $PublicName)
    })
    if ($Matches.Count -eq 0) {
        throw 'The Worker is not assigned to a matching Route in the published Event.'
    }
    if ($Matches.Count -gt 1 -and -not $PublicName) {
        throw "The Worker serves several Routes. Supply -RouteName: $($Matches.public_name -join ', ')"
    }
    return $Matches[0]
}

Export-ModuleMember -Function Resolve-ModelDeckWorker, Resolve-ModelDeckRoute
