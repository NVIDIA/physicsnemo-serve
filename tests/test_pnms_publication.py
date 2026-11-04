"""Concurrent publishers cannot roll back another completed build or image."""

from copy import deepcopy
import errno
import multiprocessing
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "application-cli/src"))
from pnms_app import build, environment_lock  # noqa: E402
from pnms_app.common import digest, read_json, write_json  # noqa: E402


@pytest.mark.parametrize("fail_publication", [False, True])
def test_windows_lock_waits_and_releases_without_fcntl(
    tmp_path, monkeypatch, fail_publication
):
    project = {"root": tmp_path}
    path = tmp_path / ".pnms/publication.lock"
    path.parent.mkdir()
    path.write_bytes(b"retained lock inode")
    identity = path.stat().st_ino
    calls, waits = [], []

    def locking(fd, mode, size):
        assert os.lseek(fd, 0, os.SEEK_CUR) == 0
        assert size == 1
        calls.append(mode)
        if mode == "acquire" and len(calls) <= 11:
            raise OSError(errno.EACCES, "another publisher holds the lock")

    monkeypatch.setattr(build, "os", SimpleNamespace(name="nt"))
    monkeypatch.setitem(sys.modules, "fcntl", None)
    monkeypatch.setitem(sys.modules, "time", SimpleNamespace(sleep=waits.append))
    monkeypatch.setitem(
        sys.modules,
        "msvcrt",
        SimpleNamespace(locking=locking, LK_NBLCK="acquire", LK_UNLCK="release"),
    )

    def publish():
        with build.publication_lock(project):
            assert calls == ["acquire"] * 12
            assert len(waits) == 11 and all(delay > 0 for delay in waits)
            if fail_publication:
                raise RuntimeError("publication failed")

    if fail_publication:
        with pytest.raises(RuntimeError, match="publication failed"):
            publish()
    else:
        publish()
    assert calls == ["acquire"] * 12 + ["release"]
    assert path.stat().st_ino == identity
    assert path.read_bytes() == b"retained lock inode"


def test_windows_lock_does_not_retry_unrelated_errors(tmp_path, monkeypatch):
    failure = OSError(errno.EBADF, "invalid descriptor")

    def locking(*args):
        raise failure

    monkeypatch.setattr(build, "os", SimpleNamespace(name="nt"))
    monkeypatch.setitem(sys.modules, "fcntl", None)
    monkeypatch.setitem(
        sys.modules,
        "msvcrt",
        SimpleNamespace(locking=locking, LK_NBLCK="acquire", LK_UNLCK="release"),
    )
    with pytest.raises(OSError) as raised:
        with build.publication_lock({"root": tmp_path}):
            pytest.fail("publication must not run without acquiring the lock")
    assert raised.value is failure


def _actor(
    project,
    root,
    result,
    previous,
    called,
    entered,
    release,
    done,
    outcomes,
    *,
    pause=None,
    package=False,
):
    """Use separate interpreter processes and independent file descriptors."""
    original_commit = environment_lock.commit
    original_image = build.publish_image

    def commit(*args, **kwargs):
        entered.set()
        if pause == "commit" and not release.wait(15):
            raise RuntimeError("publication test commit barrier timed out")
        return original_commit(*args, **kwargs)

    def publish_image(*args, **kwargs):
        if pause == "image":
            entered.set()
            if not release.wait(15):
                raise RuntimeError("publication test image barrier timed out")
            raise OSError("simulated image pointer failure")
        original_image(*args, **kwargs)
        if package:
            entered.set()

    environment_lock.commit = commit
    build.publish_image = publish_image
    try:
        called.set()
        if package:
            from pnms_app import images

            images.build_image = lambda *args, **kwargs: result["image"]
            build.package_build(
                root / "build.json", project=project, repository=ROOT, image="app:new"
            )
        else:
            build.publish_success(project, root, result, previous_lock=previous)
        outcomes.put((root.name, "complete"))
    except Exception as error:
        outcomes.put((root.name, str(error)))
    finally:
        done.set()


def _fixture(tmp_path):
    app = tmp_path / "app"
    project = {"root": app, "runtime": "python", "name": "app"}
    snapshot = environment_lock.snapshot(
        project, {}, python={"version": "3.12", "distributions": []}
    )
    write_json(app / "pnms.lock", snapshot)
    write_json(app / ".pnms/last-build.json", {"receipt": "prior-build"})
    write_json(app / ".pnms/last-image.json", {"image": {"image_id": "prior-image"}})
    previous = (app / "pnms.lock").read_bytes()

    def result(name):
        selected = deepcopy(snapshot)
        selected["environment"]["python"]["distributions"] = [
            {"name": "example", "version": name}
        ]
        record = {"schema_version": 1, "digest": digest(selected), "snapshot": selected}
        root = app / ".pnms/builds" / name
        value = {
            "status": "complete",
            "name": "app",
            "runtime": "python",
            "build_id": name,
            "environment_lock": record,
            "image": {"image_id": name + "-image"},
        }
        write_json(root / "build.json", value)
        return root, value

    return project, previous, result


def _run_race(
    project, previous, first, second, *, fail_first=False, package_second=False
):
    context = multiprocessing.get_context("spawn")
    release = context.Event()
    outcomes = context.Queue()
    first_events = [context.Event() for _ in range(3)]
    second_events = [context.Event() for _ in range(3)]
    # event ordering: called, entered critical section, done.
    left = context.Process(
        target=_actor,
        args=(
            project,
            *first,
            previous,
            *first_events[:2],
            release,
            first_events[2],
            outcomes,
        ),
        kwargs={"pause": "image" if fail_first else "commit"},
    )
    right = context.Process(
        target=_actor,
        args=(
            project,
            *second,
            previous,
            *second_events[:2],
            release,
            second_events[2],
            outcomes,
        ),
        kwargs={"package": package_second},
    )
    try:
        left.start()
        assert first_events[1].wait(10), (
            "first publisher must reach its critical section"
        )
        right.start()
        assert second_events[0].wait(10), "second process must attempt publication"
        # The unfixed code lets the second writer fully publish while the first
        # retains its old rollback snapshot. Serialization makes it wait here.
        if second_events[1].wait(1):
            assert second_events[2].wait(10), (
                "second publisher must finish before releasing the stale writer"
            )
        release.set()
        left.join(10)
        right.join(10)
        assert left.exitcode == right.exitcode == 0
        return dict(outcomes.get(timeout=5) for _ in range(2))
    finally:
        release.set()
        for process in (left, right):
            if process.pid is not None:
                process.join(2)
                if process.is_alive():
                    process.terminate()
                    process.join(2)


def test_stale_publisher_preserves_completed_winner_across_processes(tmp_path):
    project, previous, result = _fixture(tmp_path)
    first, second = result("first"), result("second")
    outcomes = _run_race(project, previous, first, second)
    winners = [name for name, status in outcomes.items() if status == "complete"]
    assert len(winners) == 1
    winner = winners[0]
    assert any("pnms.lock changed" in status for status in outcomes.values())
    lock = read_json(project["root"] / "pnms.lock")
    expected_lock = dict(
        (root.name, receipt["environment_lock"]["snapshot"])
        for root, receipt in (first, second)
    )[winner]
    assert lock == expected_lock, "stale rollback must preserve the winner's lock"
    state = project["root"] / ".pnms"
    assert (
        read_json(state / "last-build.json")["receipt"] == f"builds/{winner}/build.json"
    )
    assert (
        read_json(state / "last-image.json")["image"]["image_id"] == winner + "-image"
    )


def test_failed_build_cannot_rollback_concurrent_from_build_image_publication(tmp_path):
    project, previous, result = _fixture(tmp_path)
    first, second = result("first"), result("packaged")
    outcomes = _run_race(
        project, previous, first, second, fail_first=True, package_second=True
    )
    assert outcomes["packaged"] == "complete"
    assert "simulated image pointer failure" in outcomes["first"]
    state = project["root"] / ".pnms"
    assert (project["root"] / "pnms.lock").read_bytes() == previous
    assert read_json(state / "last-build.json") == {"receipt": "prior-build"}
    assert (
        read_json(state / "last-image.json")["image"]["image_id"] == "packaged-image"
    ), "build rollback must preserve the separately completed image publication"
