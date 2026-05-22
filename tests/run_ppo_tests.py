import importlib
import inspect
import pathlib
import sys
import traceback


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tests_dir = pathlib.Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))

    failures = 0
    total = 0
    for path in sorted(tests_dir.glob("test_*.py")):
        module = importlib.import_module(path.stem)
        for name, func in sorted(inspect.getmembers(module, inspect.isfunction)):
            if not name.startswith("test_"):
                continue
            total += 1
            test_id = f"{path.stem}.{name}"
            try:
                func()
            except Exception:
                failures += 1
                print(f"FAIL {test_id}")
                traceback.print_exc()
            else:
                print(f"PASS {test_id}")

    print(f"{total - failures}/{total} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
