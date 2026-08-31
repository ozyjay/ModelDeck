Name:           modeldeck
Version:        %{modeldeck_version}
Release:        %{modeldeck_release}%{?dist}
Summary:        Local-first model runtime manager and stable capability gateway
License:        Apache-2.0
BuildArch:      x86_64
Source0:        modeldeck-payload.tar.gz
AutoReqProv:    no

# The payload contains reviewed, prebuilt Python and ROCm wheel binaries. Preserve those
# artefacts exactly instead of attempting Fedora debuginfo extraction, ELF stripping,
# shebang rewriting, or deterministic archive rewriting within installed packages.
%global debug_package %{nil}
%global __strip /bin/true
%global __brp_mangle_shebangs %{nil}
%global __os_install_post_build_reproducibility %{nil}
%global __brp_check_rpaths %{nil}

Requires:       gtk4
Requires:       libadwaita
Requires:       python3
Requires:       python3-gobject
Requires:       webkitgtk6.0
Requires:       systemd

%description
ModelDeck is a local-first model runtime manager and stable gateway for local
desktop applications. This package contains the core control-plane, ROCm and
Q4 runtime environments, but never includes model weights.

%prep
%setup -q -c -T
tar -xzf %{SOURCE0}

%install
mkdir -p %{buildroot}
cp -a usr %{buildroot}/

%files
%license /usr/share/doc/modeldeck/APACHE-2.0.txt
/usr/bin/modeldeck-desktop
/usr/libexec/modeldeck
/usr/lib/systemd/user/modeldeck.target
/usr/lib/systemd/user/modeldeck-management.service
/usr/lib/systemd/user/modeldeck-gateway.service
/usr/share/applications/com.modeldeck.ModelDeck.desktop
/usr/share/icons/hicolor/scalable/apps/modeldeck.svg
/usr/share/modeldeck/release.json

%changelog
* Mon Aug 31 2026 ModelDeck maintainers <maintainers@modeldeck.local> - 2.0.0-1
- Add the guided capability setup and publication workflow
- Strengthen identity, qualification, thermal and routing evidence boundaries

* Sat Aug 29 2026 ModelDeck maintainers <maintainers@modeldeck.local> - 0.1.0-1
- Initial Fedora standalone package
