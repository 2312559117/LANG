"""Read real device counters outside the GUI thread; unsupported counters stay blank."""
import shutil
import subprocess
import threading

from PySide6.QtCore import QObject, QTimer, Signal

from .media import NO_WINDOW
from .performance import memory_status


class HardwareMonitor(QObject):
    updated = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._stopped = False
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.poll)
        self.timer.start()
        QTimer.singleShot(0, self.poll)

    def poll(self):
        if self._running or self._stopped:
            return
        self._running = True
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        result = {"memory": memory_status()}
        try:
            executable = shutil.which("nvidia-smi")
            if executable:
                output = subprocess.run([executable, "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                                         "--format=csv,noheader,nounits"], capture_output=True,
                                        timeout=2, creationflags=NO_WINDOW)
                if output.returncode == 0:
                    fields = output.stdout.decode("utf-8", "replace").splitlines()[0].split(",")
                    result["gpu"] = {"name": fields[0].strip(), "percent": float(fields[1]),
                                     "used_mb": float(fields[2]), "total_mb": float(fields[3])}
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        finally:
            self._running = False
        if not self._stopped:
            try:
                self.updated.emit(result)
            except RuntimeError:
                pass

    def stop(self):
        self._stopped = True
        self.timer.stop()
