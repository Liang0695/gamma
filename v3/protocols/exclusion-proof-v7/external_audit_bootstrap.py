#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emit a sequenced, acknowledged audit stream for one wrapped Python process."""

from __future__ import annotations

import json
import os
import runpy
import socket
import sys
import threading
import traceback


endpoint = os.environ.get("K27_EXTERNAL_AUDIT_ENDPOINT", "")
sock = None
role = os.environ.get("K27_EXTERNAL_AUDIT_ROLE", "child")
sequence = 0
send_guard = threading.local()
process_status = 0

if not endpoint:
    # Never execute the requested payload outside the audited runner path.
    print("AUDIT_ENDPOINT_REQUIRED", file=sys.stderr)
    raise SystemExit(78)


def send(event, **fields):
    """Send one framed event. Any transport error must fail this process."""
    global sequence
    if sock is None:
        return
    if getattr(send_guard, "active", False):
        raise RuntimeError("recursive audit transport call")
    send_guard.active = True
    try:
        item = {"event": event, "pid": os.getpid(), "ppid": os.getppid(),
                "role": role, "seq": sequence, **fields}
        wire = (json.dumps(item, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        sock.sendall(wire)
        sequence += 1
    finally:
        send_guard.active = False


host, port = endpoint.rsplit(":", 1)
# Do not run the wrapped command unless the parent collector is reachable.
sock = socket.create_connection((host, int(port)), timeout=5)
sock.settimeout(5)
send("process_start")


def hook(event, args):
    if event == "open" and sock is not None and not getattr(send_guard, "active", False):
        # Raising on send failure aborts the open and makes the wrapped case fail.
        send("open", path=os.path.abspath(str(args[0])))


if sock is not None:
    sys.addaudithook(hook)


def run_wrapped():
    global process_status
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit("missing wrapped command")
    kind = argv.pop(0)
    while True:
        if os.path.normcase(os.path.abspath(kind)) == os.path.normcase(os.path.abspath(sys.executable)):
            # Nested commands pass [python, flags..., bootstrap, python, target...].
            while argv and argv[0] in ("-X", "-W"):
                if len(argv) < 2:
                    raise SystemExit("missing interpreter option value")
                del argv[:2]
            while argv and argv[0] in ("-B", "-E", "-I", "-s", "-S", "-u"):
                argv.pop(0)
            if not argv:
                raise SystemExit("missing wrapped Python target")
            kind = argv.pop(0)
            continue
        if os.path.normcase(os.path.abspath(kind)) == os.path.normcase(os.path.abspath(__file__)):
            # The child already entered through this bootstrap; don't install a
            # second event stream inside the same OS process.
            if not argv:
                raise SystemExit("missing wrapped command after bootstrap")
            kind = argv.pop(0)
            continue
        break
    if kind == "-m":
        module = argv.pop(0)
        sys.argv = [module] + argv
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    elif kind == "-c":
        code = argv.pop(0)
        sys.argv = ["-c"] + argv
        exec(compile(code, "<externally-audited-child>", "exec"),
             {"__name__": "__main__", "__file__": "<externally-audited-child>"})
    else:
        sys.argv = [kind] + argv
        runpy.run_path(kind, run_name="__main__")


if __name__ == "__main__":
    try:
        run_wrapped()
    except SystemExit as exc:
        code = exc.code
        process_status = code if isinstance(code, int) else (0 if code is None else 1)
        if code is not None and not isinstance(code, int):
            print(code, file=sys.stderr)
    except BaseException:
        process_status = 1
        traceback.print_exc()
    if sock is not None:
        try:
            send("process_end", exit_code=process_status)
            sock.shutdown(socket.SHUT_WR)
        except Exception as exc:
            process_status = 1
            print("external audit stream incomplete: %s" % type(exc).__name__, file=sys.stderr)
        finally:
            sock.close()
    raise SystemExit(process_status)
