"""Lifetime control for this runner's exact trusted process group/job only."""
import os
import signal
import subprocess
import time


class OwnedProcess:
    def __init__(self, argv, *, cwd, env, stdout, stderr, gate):
        self.proc = None
        self.job = None
        self.registered = []
        self.children_stopped = True
        self.job_empty = True
        self.assigned = False
        self._stopped = False
        self.gate = gate
        if os.name == 'nt':
            self._windows_setup()
        try:
            self.proc = subprocess.Popen(
                argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                start_new_session=os.name != 'nt',
            )
            self.registered.append(self.proc.pid)
            if os.name == 'nt' and not self.api.AssignProcessToJobObject(self.job, int(self.proc._handle)):
                raise OSError('owned_job_assignment_failed')
            self.assigned = True
            # Trusted driver may only spawn children after successful ownership.
            gate.write_bytes(b'owned')
        except BaseException:
            self.stop(cwd)
            raise

    def _windows_setup(self):
        import ctypes as c
        from ctypes import wintypes as w
        class Basic(c.Structure):
            _fields_ = [('process_time', c.c_int64), ('job_time', c.c_int64),
                        ('flags', w.DWORD), ('min_ws', c.c_size_t), ('max_ws', c.c_size_t),
                        ('active_limit', w.DWORD), ('affinity', c.c_size_t),
                        ('priority', w.DWORD), ('scheduling', w.DWORD)]
        class Io(c.Structure):
            _fields_ = [(name, c.c_uint64) for name in ('r_ops', 'w_ops', 'o_ops', 'r_bytes', 'w_bytes', 'o_bytes')]
        class Extended(c.Structure):
            _fields_ = [('basic', Basic), ('io', Io), ('process_memory', c.c_size_t),
                        ('job_memory', c.c_size_t), ('peak_process', c.c_size_t), ('peak_job', c.c_size_t)]
        class Accounting(c.Structure):
            _fields_ = [('user', c.c_int64), ('kernel', c.c_int64), ('period_user', c.c_int64),
                        ('period_kernel', c.c_int64), ('page_faults', w.DWORD),
                        ('total', w.DWORD), ('active', w.DWORD), ('terminated', w.DWORD)]
        self.ctypes, self.Accounting = c, Accounting
        self.api = c.WinDLL('kernel32', use_last_error=True)
        specs = {
            'CreateJobObjectW': ([c.c_void_p, w.LPCWSTR], w.HANDLE),
            'SetInformationJobObject': ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
            'AssignProcessToJobObject': ([w.HANDLE, w.HANDLE], w.BOOL),
            'TerminateJobObject': ([w.HANDLE, w.UINT], w.BOOL),
            'QueryInformationJobObject': ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p], w.BOOL),
            'OpenProcess': ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            'IsProcessInJob': ([w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)], w.BOOL),
            'WaitForSingleObject': ([w.HANDLE, w.DWORD], w.DWORD),
            'CloseHandle': ([w.HANDLE], w.BOOL),
        }
        for name, (args, result) in specs.items():
            fn = getattr(self.api, name)
            fn.argtypes, fn.restype = args, result
        self.job = self.api.CreateJobObjectW(None, None)
        limit = Extended()
        limit.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.job or not self.api.SetInformationJobObject(self.job, 9, c.byref(limit), c.sizeof(limit)):
            if self.job:
                self.api.CloseHandle(self.job)
                self.job = None
            raise OSError('owned_job_creation_failed')

    def wait(self, timeout):
        return self.proc.wait(timeout=timeout)

    def stop(self, cwd):
        """Terminate only our job/session; no executable-name or global PID search."""
        if self._stopped:
            return
        self._stopped = True
        if self.proc is not None and not self.assigned:
            # Startup gate prevents children on ownership failure.
            self.proc.kill()
            self.proc.wait(timeout=3)
        handles = []
        marker = cwd / 'child.pid'
        child_pid = None
        if marker.is_file():
            try:
                child_pid = int(marker.read_text())
            except (ValueError, OSError):
                self.children_stopped = False
        if os.name == 'nt' and self.job:
            if child_pid:
                handle = self.api.OpenProcess(0x100000 | 0x1000, False, child_pid)
                if handle:
                    from ctypes import wintypes as w
                    belongs = w.BOOL()
                    if self.api.IsProcessInJob(handle, self.job, self.ctypes.byref(belongs)) and belongs.value:
                        handles.append(handle)
                        self.registered.append(child_pid)
                    else:
                        self.children_stopped = False
                        self.api.CloseHandle(handle)  # Never terminate an unowned PID.
            self.api.TerminateJobObject(self.job, 124)
            if self.proc is not None:
                self.proc.wait(timeout=3)
            for handle in handles:
                self.children_stopped &= self.api.WaitForSingleObject(handle, 3000) == 0
                self.api.CloseHandle(handle)
            info = self.Accounting()
            until = time.monotonic() + 3
            while time.monotonic() < until:
                ok = self.api.QueryInformationJobObject(self.job, 1, self.ctypes.byref(info),
                                                       self.ctypes.sizeof(info), None)
                if ok and info.active == 0:
                    break
                time.sleep(0.01)
            self.job_empty = bool(ok and info.active == 0)
            self.api.CloseHandle(self.job)
            self.job = None
        elif self.proc is not None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=3)
            if child_pid:
                self.registered.append(child_pid)
                # Linux orphan zombies have stopped executing, even before reaping.
                state = '/proc/%d/stat' % child_pid
                try:
                    from pathlib import Path
                    self.children_stopped = Path(state).read_text().split(') ', 1)[1][0] == 'Z'
                except FileNotFoundError:
                    self.children_stopped = True
                except OSError:
                    self.children_stopped = False
