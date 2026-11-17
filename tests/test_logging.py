from pathlib import Path

from lockframe.logging import configure_logging, get_logger, log_folder


def test_logging_writes_to_configured_folder(tmp_path):
    logger = configure_logging(tmp_path)
    get_logger("test").info("hello 番茄 %d", 42)
    for handler in logger.handlers:
        handler.flush()
    files = list(tmp_path.glob("*.log"))
    assert files
    content = files[0].read_text(encoding="utf-8")
    assert "hello 番茄 42" in content
    assert log_folder() == tmp_path


def test_get_logger_is_namespaced():
    assert get_logger("demo").name == "lockframe.demo"
