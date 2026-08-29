Name:           modeldeck
Version:        0.1.0
Release:        1%{?dist}
Summary:        Local-first model runtime manager and stable capability gateway
License:        Apache-2.0
BuildArch:      x86_64
Source0:        modeldeck-payload.tar.gz

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
* Sat Aug 29 2026 ModelDeck maintainers <maintainers@modeldeck.local> - 0.1.0-1
- Initial Fedora standalone package
