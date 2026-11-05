"""
Stop Windows suspending a training run that nobody is watching.

Measured on the lab A4000: `powercfg /requests` listed AnyDesk as the only
process holding a SYSTEM power request. The training process asserted
nothing, so the moment the remote session disconnected the last request
dropped and the machine slept. Epochs that take 3.6 minutes were arriving
51 minutes apart, and a four-day tuning run produced one trial.

A scheduled task running in session 0 does not fix this. Session isolation
stops the lock screen from killing the job; it does nothing about the
machine going to sleep underneath it.

`powercfg /requestsoverride` is the documented remedy, but an override is
invisible to `powercfg /requests`, so there is no way to confirm it took
effect short of disconnecting and waiting. Calling SetThreadExecutionState
is checkable: the process appears in `powercfg /requests` under SYSTEM
within seconds, which turns a four-day guess into a one-line verification.

ES_DISPLAY_REQUIRED is deliberately not set. Keeping the panel lit for a
headless run wastes power and does not keep the CPU awake.
"""
import os

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake(verbose=True):
    """Ask Windows to stay awake until this process exits. No-op elsewhere.

    ES_CONTINUOUS makes the request persist rather than resetting one idle
    timer, and the request dies with the process, so a crashed or killed
    run does not leave the machine permanently awake.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        ok = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        if verbose:
            print("  keep-awake: " + ("asserted, confirm with "
                                      "`powercfg /requests`" if ok else
                                      "REFUSED by Windows"), flush=True)
        return bool(ok)
    except Exception as e:                      # pragma: no cover
        if verbose:
            print(f"  keep-awake unavailable: {e}", flush=True)
        return False


def release(verbose=False):
    """Drop the request early. Rarely needed; process exit does this."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        if verbose:
            print("  keep-awake released", flush=True)
    except Exception:
        pass
