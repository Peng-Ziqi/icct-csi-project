# utils/logger.py
import os
import sys
import datetime

class Tee:
    """把print同时写到终端和文件"""
    def __init__(self, file_path, mode="a", encoding="utf-8"):
        self.file = open(file_path, mode, encoding=encoding)
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except:
            pass

def setup_log_redirect(log_path: str, with_time_header: bool = True):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    tee = Tee(log_path, mode="a", encoding="utf-8")
    sys.stdout = tee
    sys.stderr = tee
    if with_time_header:
        print("\n" + "=" * 80)
        print(f"[Log Start] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[Log File ] {log_path}")
        print("=" * 80)
    return tee
