import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import ctypes


class M(ctypes.Structure):
    _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong),
                ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                ("totalpg", ctypes.c_ulonglong), ("availpg", ctypes.c_ulonglong),
                ("totalv", ctypes.c_ulonglong), ("availv", ctypes.c_ulonglong),
                ("ex", ctypes.c_ulonglong)]


if __name__ == "__main__":
    m = M()
    m.l = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    print("ram_toplamGB: %.1f ram_bosGB: %.1f yuk%%: %d" % (m.total / 1e9, m.avail / 1e9, m.load))
    print("commit_limitGB: %.1f commit_bosGB: %.1f" % (m.totalpg / 1e9, m.availpg / 1e9))
