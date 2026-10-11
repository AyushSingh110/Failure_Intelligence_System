"""
E51 connectivity control. Standard library only. Imports nothing from fie or any installed package.

    <launcher> -I -S -B egress_selftest.py --expect open|blocked --target 151.101.0.223:443 [--target ...]

Run through EVERY measurement launcher, with -I -S -B so that no site-packages code and no .pth
file executes. Opens a TCP connection to each literal address, sends nothing, and closes.
Prints one JSON object. Exit 0 only when every target matches the expectation.

`open`    the positive control, taken while the network is up: proves this launcher can reach
          the network and that this test detects it.
`blocked` the negative control, taken with the network off.
"""
import argparse
import ctypes
import json
import socket
import sys


def image_path() -> str:
    """The process image as the operating system reports it, not as Python believes it."""
    try:
        buf = ctypes.create_unicode_buffer(32768)
        ctypes.windll.kernel32.GetModuleFileNameW(None, buf, len(buf))
        return buf.value
    except Exception as exc:                                   # non-Windows, or no ctypes
        return f"<unavailable: {type(exc).__name__}>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", choices=["open", "blocked"], required=True)
    ap.add_argument("--target", action="append", required=True)
    ap.add_argument("--timeout", type=float, default=6.0)
    args = ap.parse_args()

    results = []
    for target in args.target:
        host, port = target.rsplit(":", 1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(args.timeout)
        try:
            sock.connect((host, int(port)))                    # literal address: no DNS involved
            results.append({"target": target, "connected": True, "error": None})
        except OSError as exc:
            results.append({"target": target, "connected": False,
                            "error": f"{type(exc).__name__}:{getattr(exc, 'winerror', None) or exc.errno}"})
        finally:
            sock.close()

    want = args.expect == "open"
    ok = all(r["connected"] is want for r in results)
    print(json.dumps({
        "expect": args.expect, "ok": ok, "results": results,
        "sys_executable": sys.executable, "base_executable": getattr(sys, "_base_executable", None),
        "process_image": image_path(), "python": sys.version.split()[0],
        "flags": {"isolated": sys.flags.isolated, "no_site": sys.flags.no_site},
        "site_packages_on_path": [p for p in sys.path if "site-packages" in p],
    }))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
