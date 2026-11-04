"""Exercise production Runner locking scopes with a CPU-only backend boundary."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]


class ProcessMathGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("a C++ compiler is required")
        source = (WORKFLOW_ROOT / "src" / "runner.cpp").read_text()

        def before(start, end):
            begin = source.index(start)
            return source[begin : source.index(end, begin)]

        # Keep the real gate, RAII restoration, and constructor/request/startup
        # lock scopes. Only Torch math flags and the SDK load/execute boundary
        # are substitutes, allowing a loader to pause deterministically on CPU.
        common = before("using Json =", "void configure_determinism()")
        restore = before(
            "struct RestoreMathPolicy {", "\nMathPolicy initialize_process"
        )
        loading = before(
            "    const auto load_start =", "    if (!torch::cuda::is_available())"
        )
        startup = before(
            "  std::string self_test(const CancellationToken& cancellation) {",
            "    check_cancelled(cancellation);",
        ).split("{", 1)[1]
        running = before(
            "  RunnerResult run(const RunnerRequest& request, const CancellationToken& cancellation) {",
            "    check_cancelled(cancellation);",
        ).split("{", 1)[1]
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        root = Path(temporary.name)
        probe = root / "process_gate.cpp"
        probe.write_text(
            r"""
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
namespace nlohmann { class json; }
using namespace std::chrono_literals;
std::atomic<bool> tf32{true};
struct Schedule {
  std::promise<void> loaded, restore_started, release_load, release_restore;
  std::shared_future<void> load_release{release_load.get_future()};
  std::shared_future<void> restore_release{release_restore.get_future()};
  bool fail{false};
};
Schedule* active = nullptr;
struct MathPolicy {
  void apply() const {
    active->restore_started.set_value();
    active->restore_release.wait();
    tf32.store(true);
  }
};
MathPolicy initialize_process(const std::string&) { return {}; }
"""
            + common
            + restore
            + r"""
class Host {
 public:
  void load(Schedule& schedule) {
"""
            + loading
            + r"""
    (void)load_start;
    tf32.store(false);  // The AOTI loader's temporary process-wide override.
    schedule.loaded.set_value();
    schedule.load_release.wait();
    if (schedule.fail) throw std::runtime_error("expected load failure");
  }
  bool run(const std::function<bool()>& body) {
"""
            + running
            + r"""
    return body();
  }
  bool self_test(const std::function<bool()>& body) {
"""
            + startup
            + r"""
    return body();
  }
 private:
  struct { std::string math_profile{"provider-default-v1"}; } config;
  std::mutex request_mutex;
};
void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void loading_case(bool startup, bool fail) {
  Schedule schedule;
  schedule.fail = fail;
  active = &schedule;
  Host resident, incoming;
  auto loaded = schedule.loaded.get_future();
  auto restoring = schedule.restore_started.get_future();
  auto loader = std::async(std::launch::async, [&] {
    try { incoming.load(schedule); }
    catch (const std::runtime_error& error) {
      if (!fail || std::string(error.what()) != "expected load failure") throw;
      return;
    }
    check(!fail, "expected constructor failure did not propagate");
  });
  const auto did_load = loaded.wait_for(3s) == std::future_status::ready;
  std::promise<void> requested, entered;
  auto requested_future = requested.get_future();
  auto entered_future = entered.get_future();
  auto request = std::async(std::launch::async, [&] {
    requested.set_value();
    const auto operation = [&] { entered.set_value(); return tf32.load(); };
    return startup ? resident.self_test(operation) : resident.run(operation);
  });
  const auto did_request = requested_future.wait_for(3s) == std::future_status::ready;
  const auto blocked_during_load = entered_future.wait_for(100ms) == std::future_status::timeout;
  schedule.release_load.set_value();
  const auto did_restore = restoring.wait_for(3s) == std::future_status::ready;
  const auto blocked_during_restore = entered_future.wait_for(100ms) == std::future_status::timeout;
  schedule.release_restore.set_value();
  loader.get();
  const auto actual_math = request.get();
  check(did_load && did_request && did_restore, "test scheduling did not reach the backend boundary");
  check(blocked_during_load, "resident execution entered while another host changed process math");
  check(blocked_during_restore, "process gate was released before restoring the math policy");
  check(actual_math, "resident execution observed the loader's temporary TF32 override");
}
void concurrent_readers() {
  Host first, second;
  std::promise<void> first_entered, second_entered, release;
  auto first_ready = first_entered.get_future();
  auto second_ready = second_entered.get_future();
  const auto done = release.get_future().share();
  auto a = std::async(std::launch::async, [&] {
    return first.run([&] { first_entered.set_value(); done.wait(); return true; });
  });
  const auto first_started = first_ready.wait_for(3s) == std::future_status::ready;
  auto b = std::async(std::launch::async, [&] {
    return second.self_test([&] { second_entered.set_value(); done.wait(); return true; });
  });
  const auto overlapped = second_ready.wait_for(3s) == std::future_status::ready;
  release.set_value();
  check(a.get() && b.get() && first_started && overlapped,
        "independent resident executions must retain concurrent shared access");
}
int main(int argc, char** argv) {
  try {
    if (argc != 2) return 2;
    const std::string mode(argv[1]);
    if (mode == "readers") concurrent_readers();
    else loading_case(mode.starts_with("startup"), mode.ends_with("failure"));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""
        )
        cls.executable = root / "process_gate"
        result = subprocess.run(
            [compiler, "-std=c++20", "-pthread", str(probe), "-o", str(cls.executable)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def invoke(self, mode):
        result = subprocess.run(
            [str(self.executable), mode], capture_output=True, text=True, timeout=15
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_request_waits_for_loading_and_restoration(self):
        self.invoke("run")

    def test_startup_waits_for_loading_and_restoration(self):
        self.invoke("startup")

    def test_request_waits_for_restoration_after_load_failure(self):
        self.invoke("run-failure")

    def test_startup_waits_for_restoration_after_load_failure(self):
        self.invoke("startup-failure")

    def test_independent_resident_calls_can_overlap(self):
        self.invoke("readers")


if __name__ == "__main__":
    unittest.main()
