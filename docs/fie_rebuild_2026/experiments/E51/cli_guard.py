"""
E51 CLI wrapper. Runs the INSTALLED `python -m fie ...` with the probe's instrumentation loaded
first, so the command line is exercised under the same pickle tripwire, socket recorder and
process recorder as the API. Standard library only.

    <env>\\Scripts\\python.exe -I -B cli_guard.py --probe <probe.py> --verified <json> --out <dir> -- detect "..." --output json

stdout and stderr are the command's own. `cli_guard.json` in --out records what the
instrumentation saw. The exit code is the command's, or the probe's 90/97/98.
"""
import argparse
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--verified", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv[:cut])
    cli_args = argv[cut + 1:]

    spec = importlib.util.spec_from_file_location("e51_probe", args.probe)
    probe = importlib.util.module_from_spec(spec)
    sys.modules["e51_probe"] = probe
    spec.loader.exec_module(probe)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    probe._out_dir = out
    probe._env_root = sys.prefix
    probe._verified = set(json.loads(Path(args.verified).read_text(encoding="utf-8")))
    import platform
    platform.uname()                                   # see probe.py: before the process recorder is armed
    probe.install_instrumentation()
    probe._phase = "cli"

    import fie
    site = os.path.normcase(os.path.join(sys.prefix, "Lib", "site-packages"))
    if not os.path.normcase(os.path.abspath(fie.__file__)).startswith(site):
        return probe.EXIT_NOT_INSTALLED_COPY

    code = 0
    sys.argv = ["fie", *cli_args]
    try:
        runpy.run_module("fie", run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException as exc:                       # the result, not a crash of the wrapper
        code = 70
        probe._flush("cli_exception.json", {"exception": type(exc).__name__})
    sys.stdout.flush()
    probe._flush("cli_guard.json", {
        "argv_shape": [a if a.startswith("-") or a in ("detect", "explain") else "<prompt>" for a in cli_args],
        "exit_code": code, "models_opened": probe._opened_models,
        "egress_attempts_python_level": len(probe._egress),
        "egress_targets": sorted({(e["kind"], e["target"], e["caller"]) for e in probe._egress}),
        "process_violations": probe._violations})
    return code


if __name__ == "__main__":
    sys.exit(main())
