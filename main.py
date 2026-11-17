if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    import os
    from pathlib import Path
    import sys
    import traceback

    try:
        from lockframe.logging import configure_logging, get_logger, install_excepthook
        configure_logging()
        install_excepthook()
        get_logger("main").info("启动番茄跟踪 frozen=%s 参数=%s",
                                getattr(sys, "frozen", False), sys.argv[1:])
        from lockframe.app import main
        raise SystemExit(main())
    except Exception:
        details = traceback.format_exc()
        try:
            get_logger("main").error("启动失败\n%s", details)
        except Exception:
            pass
        try:
            from lockframe.logging import startup_error
            log = startup_error(details)
        except Exception:
            import tempfile
            folder=Path(tempfile.gettempdir())/'tomato-tracker-logs'
            folder.mkdir(parents=True,exist_ok=True)
            log=folder/'startup-error.log'
            log.write_text('Startup failed; logging unavailable.',encoding='utf-8')
        if "--self-check" in sys.argv:
            index = sys.argv.index("--self-check")
            if index + 1 < len(sys.argv):
                import json
                destination = Path(sys.argv[index + 1])
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "report.json").write_text(json.dumps({"ok": False, "error": details},
                                                                   ensure_ascii=False, indent=2), encoding="utf-8")
        elif os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, f"软件启动失败，诊断记录已保存：\n{log}", "番茄跟踪 · 启动提示", 0x10)
        raise SystemExit(1)
