"""
Run every MediTrack AI test suite and print a combined report (Week 4).

    python -m tests.run_all

No API key or network required - nothing here calls an LLM.
"""

from __future__ import annotations

import io
import logging
import time
from contextlib import contextmanager, redirect_stdout

SUITES = (
    ("Week 2 - knowledge base & query handling", "tests.test_knowledge_base"),
    ("Week 3 - orders, inventory & escalation", "tests.test_orders"),
    # Boundary, security, concurrency and recovery cases. The three suites
    # above only ever ask "does the good case work?"; this one asks what
    # happens when it doesn't.
    ("Week 4 - negative, security & recovery", "tests.test_negative"),
)


@contextmanager
def _quiet_logging():
    """
    Raise the root log level for the duration of one test.

    The recovery tests in test_negative deliberately feed the app corrupt
    files and expect it to log the failure at ERROR and carry on - that's
    the behaviour under test, not a problem. Without this, the console
    report is unreadable, and a real failure's traceback is lost in it.
    """
    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        root.setLevel(previous)


def run_suite(module_name: str) -> tuple[int, int, list[str], float]:
    """Import a suite, run every test_* function, and count outcomes."""
    module = __import__(module_name, fromlist=["*"])
    tests = [
        value
        for name, value in sorted(vars(module).items())
        if name.startswith("test_") and callable(value)
    ]
    passed, failures = 0, []
    started = time.perf_counter()
    for test in tests:
        try:
            # Suites print nothing, but a stray print would corrupt the report.
            with redirect_stdout(io.StringIO()), _quiet_logging():
                test()
            passed += 1
        except AssertionError as exc:
            failures.append(f"{test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report, don't abort the run
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
    return passed, len(tests), failures, time.perf_counter() - started


def main() -> None:
    print("=" * 78)
    print("MediTrack AI - full test report")
    print("=" * 78)
    print(f"{'Suite':<48}{'Tests':>7}{'Passed':>8}{'Time':>10}")
    print("-" * 78)

    total_tests = total_passed = 0
    all_failures: list[str] = []

    for label, module_name in SUITES:
        passed, total, failures, elapsed = run_suite(module_name)
        total_tests += total
        total_passed += passed
        all_failures += [f"[{label}] {f}" for f in failures]
        print(f"{label:<48}{total:>7}{passed:>8}{elapsed:>9.2f}s")

    # The validation benchmark reports accuracy rather than pass/fail counts.
    from tests.test_validation import run_validation

    started = time.perf_counter()
    metrics = run_validation()
    elapsed = time.perf_counter() - started
    cases = sum(m.total for m in metrics)
    correct = sum(m.passed for m in metrics)
    total_tests += cases
    total_passed += correct
    all_failures += [f"[validation: {m.name}] {f}" for m in metrics for f in m.failures]
    print(f"{'Week 4 - MVP validation benchmark':<48}{cases:>7}{correct:>8}{elapsed:>9.2f}s")

    print("-" * 78)
    rate = total_passed / total_tests * 100 if total_tests else 0.0
    print(f"{'TOTAL':<48}{total_tests:>7}{total_passed:>8}{rate:>9.1f}%")

    if all_failures:
        print("\nFailures:")
        for failure in all_failures:
            print(f"  {failure}")
        raise SystemExit(1)

    print("\nAll suites passed.")


if __name__ == "__main__":
    main()
