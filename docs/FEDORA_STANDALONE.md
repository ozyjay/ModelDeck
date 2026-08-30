# Fedora standalone ModelDeck

The Fedora standalone package provides a native GTK4/libadwaita ModelDeck window and keeps
the management service plus stable gateway available as user services after that window is
closed. It targets Fedora 44 x86_64 and the Framework Desktop ROCm configuration.

## Package contents and prerequisites

The RPM contains three immutable Python environments below `/usr/libexec/modeldeck`:

- `control` for the management API, gateway, and GTK desktop launcher;
- `rocm72` for the primary core ROCm Workers; and
- `rocm72-q4` for the self-contained DiffusionGemma Q4 Worker.

It does not include model weights, Hugging Face credentials, optional speech/Moshiko/llama.cpp
runtimes, or a downloader. Install the appropriate Fedora AMD/ROCm host support first and use
HuggingFacePull to acquire supported Models in the normal local Hugging Face cache. ModelDeck
reports the detected driver and ROCm versions; it does not silently claim the target stack.

The desktop package requires Fedora's `gtk4`, `libadwaita`, `python3-gobject`, and
`webkitgtk6.0` packages. Fedora's older `webkit2gtk4.1` binding is GTK3-only and is not used.
Its user services always bind management and gateway to loopback on
ports 3600 and 8600 respectively.

## Build an unsigned RPM

The default build is intentionally offline. The standalone wrapper can also prepare an empty
wheelhouse on a connected release machine with Python 3.12: it downloads pinned package releases,
builds wheels for Q4 dependencies that are published only as source archives, creates
`packaging/fedora/wheelhouse.sha256`, and then builds the RPM. Review the generated manifest before
signing or distributing the result. The manifest has one lower-case SHA-256 and filename per
wheel. The wrapper automatically discovers installed Python 3.12 interpreters, including
pyenv-managed versions; `-Python /path/to/python3.12` overrides that discovery. The build host
also needs `rpmbuild` and `patchelf`; the latter removes build-host RPATHs from the bundled Python
runtime before packaging.

```powershell
pwsh -NoProfile -File scripts/packaging/build_fedora_standalone.ps1 -PrepareWheelhouse
```

The standalone build script invokes the canonical RPM builder and prints each produced package
path. In its default mode it fails if a wheel is unlisted, missing, or has a different digest. It
uses `pip --no-index` and rejects the live ROCm URLs in the source requirements after replacing
them with their pinned package versions. No model download or package-manager installation occurs
during the build.

The application version is read from `backend/modeldeck/__init__.py`; the spec receives that value
only from the build script. For a packaging-only rebuild, keep the application version and pass a
new RPM release number, for example `-RpmRelease 2`. For a new application release, change
`__version__` and build with the default RPM release `1`.

## Sign, install, and launch

Signing is separate from building. Install Fedora's `rpm-sign` package and make the release GPG
private key available to the local signing agent; never add a key to this repository. The release
wrapper uses the only available private key, requires an explicit key ID if several are available,
and can create a new protected key on a fresh signing workstation.

```powershell
pwsh -NoProfile -File scripts/packaging/release_fedora_rpm.ps1 `
  -RpmPath dist/fedora/x86_64/modeldeck-0.1.0-1.fc44.x86_64.rpm `
  -CreateKey -SigningName 'ModelDeck Release' -SigningEmail 'release@example.com'
```

For an existing key, use its long ID or fingerprint:

```powershell
pwsh -NoProfile -File scripts/packaging/release_fedora_rpm.ps1 `
  -RpmPath dist/fedora/x86_64/modeldeck-0.1.0-1.fc44.x86_64.rpm `
  -KeyId <private-key-id-or-fingerprint>
rpm --checksig --verbose dist/fedora/x86_64/modeldeck-0.1.0-1.fc44.x86_64.rpm
sudo dnf install dist/fedora/x86_64/modeldeck-0.1.0-1.fc44.x86_64.rpm
modeldeck-desktop
```

The signing wrapper verifies its work in an isolated temporary RPM database, so it does not add a
key to the release workstation's system RPM database. Before another machine can verify or install
the RPM, distribute the public key and import it there:

```bash
gpg --armor --export <private-key-id-or-fingerprint> > modeldeck-release-signing-key.asc
sudo rpm --import modeldeck-release-signing-key.asc
```

Launching the window starts `modeldeck.target` with `systemctl --user start`. Closing the window
does not stop the target, so configured local applications may continue using
`http://127.0.0.1:8600/v1`. Use the window menu's **Stop ModelDeck services** command, which
requests graceful Worker shutdown, when that gateway should no longer run. The RPM does not
enable either service at login.

## Per-user state, export, and importing a development installation

The package writes state to `~/.local/share/modeldeck` and Worker logs to
`~/.local/state/modeldeck/logs/workers`. First use starts empty. Select **Import existing
state…** from the desktop window to copy an existing data directory such as
`/path/to/ModelDeck/.modeldeck`.

Import stops services, checks SQLite integrity and schema version 4, copies the selected state
without modifying the source, and backs up existing packaged-app state before replacement. It
preserves configured Workers, routing profiles, compatibility evidence, thermal state, and
trusted runtime manifests. Older v2/v3 databases must use the documented migrations before
import.

Select **Export state…** to choose a parent folder for an import-compatible copy. ModelDeck stops
services while it copies state and creates a new timestamped `modeldeck-state-…` directory there;
it never replaces an existing export or modifies the active state. The command-line equivalent is
`modeldeck-export-state ~/.local/share/modeldeck /path/to/new-export-directory` and should be run
only after stopping ModelDeck services.

## Updating

An RPM update does not interrupt a running inference request. On the next desktop launch,
ModelDeck compares its installed release metadata with the management service build ID. If they
differ, it offers a restart and explains that active Workers and requests will be interrupted.
