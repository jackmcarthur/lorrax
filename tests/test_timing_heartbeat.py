"""Deterministic cadence/lifecycle contracts for long timing sections."""

from __future__ import annotations

import threading
import time

import pytest

from common import timing


def _capture_trace(monkeypatch):
    messages: list[str] = []
    heartbeat = threading.Event()

    def trace(message: str) -> None:
        messages.append(message)
        if " still running; " in message:
            heartbeat.set()

    monkeypatch.setattr(timing, "_trace", trace)
    monkeypatch.setattr(timing, "_rank0", lambda: True)
    return messages, heartbeat


def _force_tick(collector: timing.TimingCollector) -> None:
    with collector._heartbeat_condition:
        for section in collector._heartbeat_sections:
            collector._heartbeat_sections[section] = 0.0
        collector._heartbeat_condition.notify_all()


def test_announced_section_heartbeats_until_completion(monkeypatch):
    messages, heartbeat = _capture_trace(monkeypatch)
    collector = timing.TimingCollector()

    with collector.section("long", announce=True, label="long kernel"):
        _force_tick(collector)
        assert heartbeat.wait(timeout=2.0)

    assert messages[0] == "-> long kernel"
    assert any(".. long kernel still running" in line for line in messages)
    assert messages[-1].startswith("<- long kernel  ")
    assert not collector._heartbeat_sections


def test_nested_announced_section_has_one_deepest_speaker(monkeypatch):
    messages, heartbeat = _capture_trace(monkeypatch)
    collector = timing.TimingCollector()

    with collector.section("parent", announce=True):
        with collector.section("child", announce=True):
            _force_tick(collector)
            assert heartbeat.wait(timeout=2.0)

    heartbeats = [line for line in messages if "still running" in line]
    assert len(heartbeats) == 1
    assert heartbeats[0].startswith("  .. child still running")


def test_quiet_child_does_not_change_parent_heartbeat_indent(monkeypatch):
    messages, heartbeat = _capture_trace(monkeypatch)
    monkeypatch.setattr(timing, "_trace_flag", lambda: False)
    collector = timing.TimingCollector()

    with collector.section("parent", announce=True):
        with collector.section("quiet child"):
            _force_tick(collector)
            assert heartbeat.wait(timeout=2.0)

    line = next(line for line in messages if "still running" in line)
    assert line.startswith(".. parent still running")


def test_watcher_exception_cleans_registration_and_stack(monkeypatch):
    messages, _ = _capture_trace(monkeypatch)
    collector = timing.TimingCollector()

    def fail() -> None:
        raise RuntimeError("device failure")

    with pytest.raises(RuntimeError, match="device failure"):
        with collector.section("watch", announce=True) as section:
            section.watch(fail)

    assert not collector._heartbeat_sections
    assert collector._stack() == []
    assert messages[-1].endswith("[EXC]")


def test_handled_outer_exception_does_not_mark_success_failed(monkeypatch):
    messages, _ = _capture_trace(monkeypatch)
    collector = timing.TimingCollector()

    try:
        raise ValueError("already handled")
    except ValueError:
        with collector.section("success", announce=True):
            pass

    assert messages[-1].startswith("<- success  ")
    assert not messages[-1].endswith("[EXC]")


def test_trace_failure_does_not_mask_watcher_failure(monkeypatch):
    monkeypatch.setattr(timing, "_rank0", lambda: True)
    monkeypatch.setattr(
        timing, "_trace",
        lambda message: (_ for _ in ()).throw(BrokenPipeError("closed")))
    collector = timing.TimingCollector()

    def device_failure() -> None:
        raise RuntimeError("device failure")

    with pytest.raises(RuntimeError, match="device failure"):
        with collector.section("watch", announce=True) as section:
            section.watch(device_failure)

    assert not collector._active_sections
    assert not collector._heartbeat_sections


def test_heartbeat_remains_live_during_blocking_watcher(monkeypatch):
    messages, heartbeat = _capture_trace(monkeypatch)
    collector = timing.TimingCollector()
    watcher_entered = threading.Event()
    release_watcher = threading.Event()
    finished = threading.Event()

    def watcher() -> None:
        watcher_entered.set()
        assert release_watcher.wait(timeout=2.0)

    def run() -> None:
        with collector.section("async", announce=True) as section:
            section.watch(watcher)
        finished.set()

    worker = threading.Thread(target=run)
    worker.start()
    assert watcher_entered.wait(timeout=2.0)
    _force_tick(collector)
    assert heartbeat.wait(timeout=2.0)
    assert not finished.is_set()
    release_watcher.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert messages[-1].startswith("<- async  ")


def test_body_exception_cleans_registration_and_stack(monkeypatch):
    messages, _ = _capture_trace(monkeypatch)
    collector = timing.TimingCollector()

    with pytest.raises(ValueError, match="body"):
        with collector.section("body", announce=True):
            raise ValueError("body")

    assert not collector._heartbeat_sections
    assert collector._stack() == []
    assert messages[-1].endswith("[EXC]")


def test_blocked_heartbeat_cannot_overtake_exit_or_inflate_record(monkeypatch):
    heartbeat_entered = threading.Event()
    release_heartbeat = threading.Event()
    body_may_exit = threading.Event()
    finished = threading.Event()
    messages: list[str] = []

    def trace(message: str) -> None:
        if "still running" in message:
            heartbeat_entered.set()
            assert release_heartbeat.wait(timeout=2.0)
        messages.append(message)

    monkeypatch.setattr(timing, "_trace", trace)
    monkeypatch.setattr(timing, "_rank0", lambda: True)
    collector = timing.TimingCollector()

    def run() -> None:
        with collector.section("blocked output", announce=True):
            _force_tick(collector)
            assert heartbeat_entered.wait(timeout=2.0)
            body_may_exit.set()
        finished.set()

    worker = threading.Thread(target=run)
    worker.start()
    assert body_may_exit.wait(timeout=2.0)
    time.sleep(0.02)  # only proves the exiting worker is lock-blocked
    assert not finished.is_set()
    release_heartbeat.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert "still running" in messages[-2]
    assert messages[-1].startswith("<- blocked output  ")
    assert collector.records()[0]["inclusive"] < 0.5


def test_rank_nonzero_starts_no_scheduler(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr(timing, "_trace", messages.append)
    monkeypatch.setattr(timing, "_rank0", lambda: False)
    monkeypatch.setattr(timing, "_trace_flag", lambda: False)
    collector = timing.TimingCollector()

    with collector.section("rank one", announce=True):
        pass

    assert messages == []
    assert collector._heartbeat_thread is None
    assert not collector._heartbeat_sections


def test_reset_refuses_without_mutating_an_active_tree(monkeypatch):
    _capture_trace(monkeypatch)
    collector = timing.TimingCollector()
    root = collector._root

    with collector.section("active"):
        with pytest.raises(RuntimeError, match="cannot reset timing"):
            collector.reset()
        assert collector._root is root

    assert collector.records()[0]["name"] == "active"


def test_reset_refuses_quiet_section_active_on_another_thread(monkeypatch):
    monkeypatch.setattr(timing, "_trace_flag", lambda: False)
    collector = timing.TimingCollector()
    entered = threading.Event()
    release = threading.Event()

    def run() -> None:
        with collector.section("quiet worker"):
            entered.set()
            assert release.wait(timeout=2.0)

    worker = threading.Thread(target=run)
    worker.start()
    assert entered.wait(timeout=2.0)
    root = collector._root
    with pytest.raises(RuntimeError, match="cannot reset timing"):
        collector.reset()
    assert collector._root is root
    release.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert collector.records()[0]["name"] == "quiet worker"


def test_nonannounced_section_has_no_mandatory_heartbeat(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr(timing, "_trace", messages.append)
    monkeypatch.setattr(timing, "_rank0", lambda: True)
    monkeypatch.setattr(timing, "_trace_flag", lambda: False)
    collector = timing.TimingCollector()

    with collector.section("quiet"):
        pass

    assert messages == []
    assert collector._heartbeat_thread is None
