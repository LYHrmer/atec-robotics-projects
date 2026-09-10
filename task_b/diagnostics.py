"""Instance-local, read-only capture immediately before an official auto-reset.

The callback is an audit sink, never a controller input. It must only read state
and return data; this helper does not step, render, compute observations or
change reset arguments. No Isaac or GPU dependency is imported here.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy


_ABSENT = object()


class PreResetRecorder:
    """Temporarily observe ``env._reset_idx`` during explicit step scopes.

    Example::

        with PreResetRecorder(task, lambda env_ids: diagnostic_dict()) as recorder:
            env.reset()                 # deliberately not captured
            with recorder.step():
                result = env.step(action)
            record = recorder.last_record  # None if no reset occurred

    ``callback(env_ids)`` receives a copy of the reset IDs. Its returned value
    is deep-copied before the real reset can mutate shared buffers. Each record
    contains ``snapshot`` or ``observer_error``; an observer failure never skips
    the real reset. The original reset's return value and exceptions propagate.
    The callback itself remains responsible for being read-only.
    """

    def __init__(self, env, callback):
        if not callable(callback):
            raise TypeError("callback must be callable")
        self.env = env
        self.callback = callback
        self.step_records = []
        self.observer_errors = []
        self.total_step_resets = 0
        self.ignored_resets = 0
        self.step_index = 0
        self._active = False
        self._inside_step = False
        self._original_reset = None
        self._original_instance_value = _ABSENT
        self._wrapper = None

    @property
    def last_record(self):
        """Last reset in this step, or None; values already own their storage."""
        return self.step_records[-1] if self.step_records else None

    def __enter__(self):
        if self._active:
            raise RuntimeError("PreResetRecorder is already installed")
        original = self.env._reset_idx
        if not callable(original):
            raise TypeError("env._reset_idx must be callable")
        self._original_reset = original
        self._original_instance_value = vars(self.env).get("_reset_idx", _ABSENT)

        def observe_reset(*args, **kwargs):
            if self._inside_step:
                self.total_step_resets += 1
                record = {
                    "step_index": self.step_index,
                    "reset_index_in_step": len(self.step_records),
                    "phase": "terminal_pre_reset",
                    "env_ids": None,
                    "snapshot": None,
                    "observer_error": None,
                }
                try:
                    env_ids = args[0] if args else kwargs.get("env_ids")
                    record["env_ids"] = deepcopy(env_ids)
                    # A callback cannot accidentally modify the actual reset IDs.
                    value = self.callback(deepcopy(env_ids))
                    record["snapshot"] = deepcopy(value)
                except BaseException as error:
                    # Audit failures must not alter official reset execution.
                    record["observer_error"] = {
                        "type": type(error).__name__, "message": str(error),
                    }
                    self.observer_errors.append({
                        "step_index": self.step_index,
                        "reset_index_in_step": record["reset_index_in_step"],
                        **record["observer_error"],
                    })
                self.step_records.append(record)
            else:
                self.ignored_resets += 1
            # Exactly one untouched call; do not catch or replace its exception.
            return original(*args, **kwargs)

        self._wrapper = observe_reset
        self.env._reset_idx = observe_reset
        self._active = True
        return self

    @contextmanager
    def step(self):
        """Mark one real env.step call; initial/manual resets stay outside."""
        if not self._active:
            raise RuntimeError("Enter PreResetRecorder before opening a step scope")
        if self._inside_step:
            raise RuntimeError("Nested step scopes are not supported")
        self.step_index += 1
        self.step_records = []
        self._inside_step = True
        try:
            yield self
        finally:
            self._inside_step = False

    def close(self):
        """Restore the prior instance attribute, or its original class lookup."""
        if not self._active:
            return
        if self._original_instance_value is _ABSENT:
            delattr(self.env, "_reset_idx")
        else:
            self.env._reset_idx = self._original_instance_value
        self._active = False
        self._inside_step = False

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
