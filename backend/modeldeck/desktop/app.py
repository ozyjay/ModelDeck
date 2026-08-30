"""GTK4/libadwaita shell for packaged and development ModelDeck sessions.

Imports of GI are intentionally delayed until ``main`` so control-plane tests and
headless service use do not require a graphical stack.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modeldeck.desktop.controller import DesktopServiceError, ServiceController, should_prompt_for_restart

APP_ID = "com.modeldeck.ModelDeck"
CONSOLE_URI = "http://127.0.0.1:3600/"
RELEASE_METADATA = Path("/usr/share/modeldeck/release.json")
IMPORTER = Path("/usr/libexec/modeldeck/control/bin/modeldeck-import-state")
EXPORTER = Path("/usr/libexec/modeldeck/control/bin/modeldeck-export-state")
DEVELOPMENT_MODE_ENV = "MODELDECK_DESKTOP_DEVELOPMENT"
BUILD_ID_ENV = "MODELDECK_DESKTOP_BUILD_ID"


def _desktop_data_dir() -> Path:
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / "modeldeck"


def read_installed_build_id(path: Path = RELEASE_METADATA) -> str:
    development_build_id = os.environ.get(BUILD_ID_ENV)
    if development_build_id:
        return development_build_id
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise DesktopServiceError("The installed ModelDeck release metadata is unreadable") from error
    build_id = payload.get("build_id") if isinstance(payload, dict) else None
    if not isinstance(build_id, str) or not build_id:
        raise DesktopServiceError("The installed ModelDeck release metadata has no build ID")
    return build_id


def main() -> None:
    try:
        import gi

        gi.require_version("Adw", "1")
        gi.require_version("Gtk", "4.0")
        gi.require_version("WebKit", "6.0")
        from gi.repository import Adw, Gio, GLib, Gtk, WebKit
    except (ImportError, ValueError) as error:
        raise SystemExit("ModelDeck Desktop requires GTK4, libadwaita, and WebKitGTK 4.1") from error

    class ModelDeckDesktop(Adw.Application):
        def __init__(self) -> None:
            self.development_mode = os.environ.get(DEVELOPMENT_MODE_ENV) == "1"
            application_flags = (
                Gio.ApplicationFlags.NON_UNIQUE
                if self.development_mode
                else Gio.ApplicationFlags.DEFAULT_FLAGS
            )
            super().__init__(application_id=APP_ID, flags=application_flags)
            self.controller = ServiceController()
            self.window: Any | None = None
            self.content: Any | None = None
            self.webview: Any | None = None
            self.installed_build_id: str | None = None
            self._add_actions(Gio)

        def _add_actions(self, gio: Any) -> None:
            actions = (
                ("restart", self._restart),
                ("stop", self._confirm_stop),
                ("import", self._import),
                ("export", self._export),
            )
            for name, callback in actions:
                action = gio.SimpleAction.new(name, None)
                action.connect("activate", callback)
                self.add_action(action)

        def do_activate(self) -> None:
            if self.window is not None:
                self.window.present()
                return
            self.window = Adw.ApplicationWindow(
                application=self,
                title="ModelDeck",
                default_width=1280,
                default_height=840,
            )
            toolbar = Adw.ToolbarView()
            header = Adw.HeaderBar()
            if not self.development_mode:
                menu = Gio.Menu()
                menu.append("Restart services", "app.restart")
                menu.append("Import existing state…", "app.import")
                menu.append("Export state…", "app.export")
                menu.append("Stop ModelDeck services…", "app.stop")
                menu_button = Gtk.MenuButton(
                    icon_name="open-menu-symbolic",
                    menu_model=menu,
                    tooltip_text="ModelDeck services",
                )
                header.pack_end(menu_button)
            toolbar.add_top_bar(header)
            self.content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            toolbar.set_content(self.content)
            self.window.set_content(toolbar)
            self.window.present()
            try:
                self.installed_build_id = read_installed_build_id()
            except DesktopServiceError as error:
                self._show_recovery(str(error))
                return
            self._start_in_background(restart=False)

        def _set_content(self, widget: Any) -> None:
            assert self.content is not None
            child = self.content.get_first_child()
            if child is not None:
                self.content.remove(child)
            self.content.append(widget)

        def _show_status(self, title: str, detail: str) -> None:
            page = Adw.StatusPage(title=title, description=detail, icon_name="modeldeck-symbolic")
            self._set_content(page)

        def _show_recovery(self, detail: str) -> None:
            page = Adw.StatusPage(
                title="ModelDeck is not ready",
                description=detail,
                icon_name="dialog-error-symbolic",
            )
            retry = Gtk.Button(label="Retry", css_classes=["suggested-action"])
            retry.connect("clicked", lambda *_: self._start_in_background(restart=False))
            page.set_child(retry)
            self._set_content(page)

        def _start_in_background(self, *, restart: bool) -> None:
            if self.development_mode:
                self._show_status("Connecting to ModelDeck", "Waiting for local development services…")
            else:
                self._show_status("Starting ModelDeck", "Starting local management and gateway services…")

            def work() -> None:
                try:
                    if restart:
                        self.controller.restart()
                    elif not self.development_mode:
                        self.controller.start()
                    health = self.controller.wait_until_ready()
                except DesktopServiceError as error:
                    GLib.idle_add(self._show_recovery, str(error))
                    return
                GLib.idle_add(self._handle_ready, health)

            threading.Thread(target=work, daemon=True).start()

        def _handle_ready(self, health: Any) -> None:
            if self.installed_build_id and should_prompt_for_restart(
                installed_build_id=self.installed_build_id,
                running_build_id=health.build_id,
            ):
                self._prompt_for_update_restart()
                return
            self._show_console()

        def _prompt_for_update_restart(self) -> None:
            dialog = Adw.AlertDialog(
                heading="An updated ModelDeck is installed",
                body=(
                    "Restart services now to use it. This stops active Workers and interrupts local requests."
                ),
            )
            dialog.add_response("keep", "Keep current services")
            dialog.add_response("restart", "Restart services")
            dialog.set_response_appearance("restart", Adw.ResponseAppearance.DESTRUCTIVE)
            dialog.set_default_response("keep")
            dialog.choose(self.window, None, self._finish_update_prompt)

        def _finish_update_prompt(self, dialog: Any, result: Any) -> None:
            try:
                response = dialog.choose_finish(result)
            except GLib.Error:
                response = "keep"
            if response == "restart":
                self._start_in_background(restart=True)
            else:
                self._show_console()

        def _show_console(self) -> None:
            self.webview = WebKit.WebView()
            self.webview.set_hexpand(True)
            self.webview.set_vexpand(True)
            self.webview.load_uri(CONSOLE_URI)
            self._set_content(self.webview)

        def _restart(self, *_args: Any) -> None:
            self._start_in_background(restart=True)

        def _confirm_stop(self, *_args: Any) -> None:
            dialog = Adw.AlertDialog(
                heading="Stop ModelDeck services?",
                body=(
                    "This gracefully stops Workers and makes the local gateway unavailable to other "
                    "applications."
                ),
            )
            dialog.add_response("cancel", "Cancel")
            dialog.add_response("stop", "Stop services")
            dialog.set_response_appearance("stop", Adw.ResponseAppearance.DESTRUCTIVE)
            dialog.set_default_response("cancel")
            dialog.choose(self.window, None, self._finish_stop_prompt)

        def _finish_stop_prompt(self, dialog: Any, result: Any) -> None:
            try:
                response = dialog.choose_finish(result)
            except GLib.Error:
                response = "cancel"
            if response != "stop":
                return
            self._show_status("Stopping ModelDeck", "Gracefully stopping local Workers and services…")

            def work() -> None:
                try:
                    self.controller.stop()
                except DesktopServiceError as error:
                    GLib.idle_add(self._show_recovery, str(error))
                    return
                GLib.idle_add(
                    self._show_status,
                    "ModelDeck services stopped",
                    "Open ModelDeck again to start the local management service and gateway.",
                )

            threading.Thread(target=work, daemon=True).start()

        def _import(self, *_args: Any) -> None:
            chooser = Gtk.FileDialog(title="Select ModelDeck state export")
            chooser.open(self.window, None, self._finish_select_import_source)

        def _finish_select_import_source(self, chooser: Any, result: Any) -> None:
            try:
                source = chooser.open_finish(result).get_path()
            except GLib.Error:
                return
            if not source:
                return
            dialog = Adw.AlertDialog(
                heading="Import existing ModelDeck state?",
                body=(
                    "Services will stop. Existing packaged-app state will be backed up before the selected "
                    "archive is imported."
                ),
            )
            dialog.add_response("cancel", "Cancel")
            dialog.add_response("import", "Import state")
            dialog.set_response_appearance("import", Adw.ResponseAppearance.DESTRUCTIVE)
            dialog.set_default_response("cancel")
            dialog.choose(
                self.window,
                None,
                lambda prompt, chosen: self._finish_import_prompt(prompt, chosen, source),
            )

        def _finish_import_prompt(self, dialog: Any, result: Any, source: str) -> None:
            try:
                response = dialog.choose_finish(result)
            except GLib.Error:
                response = "cancel"
            if response != "import":
                return
            self._show_status(
                "Importing ModelDeck state",
                "Stopping services and validating the selected data…",
            )

            def work() -> None:
                try:
                    self.controller.stop()
                    completed = subprocess.run(
                        (str(IMPORTER), source, str(_desktop_data_dir()), "--replace-existing"),
                        capture_output=True,
                        check=True,
                        text=True,
                        timeout=120,
                    )
                    del completed
                    self.controller.start()
                    self.controller.wait_until_ready()
                except (
                    DesktopServiceError,
                    OSError,
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as error:
                    GLib.idle_add(self._show_recovery, f"State import failed: {error}")
                    return
                GLib.idle_add(self._show_console)

            threading.Thread(target=work, daemon=True).start()

        def _export(self, *_args: Any) -> None:
            chooser = Gtk.FileDialog(title="Save ModelDeck state export")
            export_name = f"modeldeck-state-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.tar"
            chooser.set_initial_name(export_name)
            chooser.save(self.window, None, self._finish_select_export_destination)

        def _finish_select_export_destination(self, chooser: Any, result: Any) -> None:
            try:
                destination = chooser.save_finish(result).get_path()
            except GLib.Error:
                return
            if not destination:
                return
            export_name = Path(destination).name
            dialog = Adw.AlertDialog(
                heading="Export ModelDeck state?",
                body=(
                    f"Services will stop while state is archived to {export_name}. "
                    "The export can later be selected with Import existing state…"
                ),
            )
            dialog.add_response("cancel", "Cancel")
            dialog.add_response("export", "Export state")
            dialog.set_response_appearance("export", Adw.ResponseAppearance.DESTRUCTIVE)
            dialog.set_default_response("cancel")
            dialog.choose(
                self.window,
                None,
                lambda prompt, chosen: self._finish_export_prompt(prompt, chosen, destination),
            )

        def _finish_export_prompt(self, dialog: Any, result: Any, destination: str) -> None:
            try:
                response = dialog.choose_finish(result)
            except GLib.Error:
                response = "cancel"
            if response != "export":
                return
            self._show_status("Exporting ModelDeck state", "Stopping services and copying local state…")

            def work() -> None:
                export_error: Exception | None = None
                try:
                    self.controller.stop()
                    subprocess.run(
                        (str(EXPORTER), str(_desktop_data_dir()), destination),
                        capture_output=True,
                        check=True,
                        text=True,
                        timeout=120,
                    )
                except (
                    DesktopServiceError,
                    OSError,
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as error:
                    export_error = error
                try:
                    self.controller.start()
                    self.controller.wait_until_ready()
                except DesktopServiceError as error:
                    detail = f"ModelDeck services could not restart: {error}"
                    if export_error is not None:
                        detail = f"State export failed: {export_error}. {detail}"
                    GLib.idle_add(self._show_recovery, detail)
                    return
                if export_error is not None:
                    GLib.idle_add(
                        self._show_status,
                        "State export failed",
                        f"{export_error} ModelDeck services have restarted.",
                    )
                    return
                GLib.idle_add(self._show_console)

            threading.Thread(target=work, daemon=True).start()

    app = ModelDeckDesktop()
    raise SystemExit(app.run(sys.argv))


if __name__ == "__main__":
    main()
