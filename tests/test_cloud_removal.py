"""Removing cloud controls must also retire credentials without touching tracking preferences."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_upgrade_removes_cloud_settings_and_page_but_keeps_local_preferences(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from lockframe import app
    from lockframe.media import configure_cache_preferences
    application = QApplication.instance() or QApplication([])
    preferences = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    preferences.setValue("ai/api_key_encrypted", "old-encrypted-placeholder")
    preferences.setValue("ai/qwen-cn/api_key_encrypted", "other-encrypted-placeholder")
    preferences.setValue("ai/provider", "qwen-cn")
    preferences.setValue("planar_region", True)
    preferences.setValue("adaptive_reference", True)
    preferences.setValue("last_folder", str(tmp_path))
    monkeypatch.setattr(app, "QSettings", lambda *args: preferences)
    window = app.MainWindow()
    try:
        assert [window.inspector_tabs.tabText(i) for i in range(window.inspector_tabs.count())] == ["构图", "区域", "性能", "导出"]
        assert not hasattr(window, "ai_key")
        assert not any(key.startswith("ai/") for key in preferences.allKeys())
        assert window.settings().planar_region and window.settings().adaptive_reference
        assert preferences.value("last_folder") == str(tmp_path)
    finally:
        window.dirty = False
        window.close()
        configure_cache_preferences(None)
