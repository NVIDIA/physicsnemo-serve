from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
SEPARATOR = "-" * 76
HEADER = (
    "NVIDIA PhysicsNeMo Serve Third-Party Open Source Software Notices\n"
    + "=" * 76
    + "\n\nNVIDIA PhysicsNeMo Serve makes use of various open source software.\n"
    "The specific open source components used and their licenses are listed below.\n\n"
)
ENTRY = re.compile(
    r"^-{76}\n (\d+)\. (.*?) version: (.*?)\n-{76}\n"
    r" homepage: ([^\n]*)\n ([^\n]*)\n\n",
    re.MULTILINE,
)
LEGAL_FILE = re.compile(
    r"^(licen[cs]e|copying|copyright|notice|unlicense)([._-].*)?$", re.IGNORECASE
)
NAME_ALIASES = {
    "click - python command line utility": "click",
    "hyper http library": "hyper",
    "pygments - python syntax highlighter": "pygments",
    "filesystem-spec": "fsspec",
    "py-filelock": "filelock",
    "python3-rich": "rich",
    "rust-openssl": "openssl-sys",
    "redox-os": "redox-syscall",
    "criterion-rs": "criterion",
    "futures-rs": "futures",
    "static-assertions-rs": "static-assertions",
    "half-rs": "half",
    "spin-rs": "spin",
    "vcpkg-rs": "vcpkg",
    "cc-rs": "cc",
}
VERSION_ALIASES = {
    ("rayon", "1.13.0"): "rayon-core",
    ("wasm-bindgen", "0.4.63"): "wasm-bindgen-futures",
}


@dataclass
class Notice:
    ecosystem: str
    name: str
    version: str
    homepage: str
    license: str
    texts: list[str]
    sources: list[str]

    @property
    def key(self) -> tuple[str, str, str]:
        return self.ecosystem, normalize(self.name), self.version


@dataclass
class Existing:
    number: int
    name: str
    version: str
    homepage: str
    license: str
    body: str
    block: str


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def legacy_name(entry: Existing) -> str:
    name = normalize(entry.name.rsplit("/", 1)[-1])
    name = NAME_ALIASES.get(name, name)
    name = VERSION_ALIASES.get((name, entry.version), name)
    return re.sub(r"^(python-|rust-)", "", name)


def versions_match(version: str, notice: Notice) -> bool:
    if version == notice.version:
        return True
    if notice.ecosystem != "Python" or not all(
        re.fullmatch(r"\d+(\.\d+)*", value) for value in (version, notice.version)
    ):
        return False

    def parts(value: str) -> tuple[int, ...]:
        result = [int(part) for part in value.split(".")]
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return tuple(result)

    return parts(version) == parts(notice.version)


def parse_existing(text: str) -> tuple[str, list[Existing]]:
    matches = list(ENTRY.finditer(text))
    if not matches:
        raise ValueError("Existing inventory has no component sections")
    result = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result.append(
            Existing(
                int(match[1]),
                match[2],
                match[3],
                match[4],
                match[5],
                text[match.end() : end],
                text[match.start() : end],
            )
        )
    if len({entry.number for entry in result}) != len(result):
        raise ValueError("Existing inventory has duplicate entry numbers")
    return text[: matches[0].start()], result


def matching_entries(
    notices: list[Notice], existing: list[Existing]
) -> dict[tuple[str, str, str], Existing]:
    result = {}
    remaining = list(existing)
    for strategy in ("name", "alias", "homepage"):
        for notice in notices:
            if notice.key in result:
                continue

            wanted = (
                notice.homepage.rstrip("/")
                if strategy == "homepage"
                else normalize(notice.name)
            )

            def identity(entry: Existing) -> str:
                if strategy == "homepage":
                    return entry.homepage.rstrip("/")
                if strategy == "alias":
                    return legacy_name(entry)
                return normalize(entry.name.rsplit("/", 1)[-1])

            candidates = [
                entry
                for entry in remaining
                if versions_match(entry.version, notice) and identity(entry) == wanted
            ]
            competing = [
                item
                for item in notices
                if item.version == notice.version
                and (
                    item.homepage.rstrip("/")
                    if strategy == "homepage"
                    else normalize(item.name)
                )
                == wanted
            ]
            if wanted and len(candidates) == len(competing) == 1:
                result[notice.key] = candidates[0]
                remaining.remove(candidates[0])
    return result


def run_json(command: list[str], cwd: Path, env: dict[str, str] | None = None):
    return json.loads(subprocess.check_output(command, cwd=cwd, env=env, text=True))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()


def legal_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and LEGAL_FILE.match(path.name)
    )


def collect_rust(
    root: Path,
    executable: str,
    existing: list[Existing],
    fallback_existing: list[Existing] | None = None,
):
    lock = root / "Cargo.lock"
    before = lock.read_bytes()
    env = {**os.environ, "CARGO_NET_OFFLINE": "true"}
    metadata = run_json(
        [
            "cargo",
            "metadata",
            "--locked",
            "--offline",
            "--all-features",
            "--format-version",
            "1",
        ],
        root,
        env,
    )
    inventory = run_json([executable, "--json", "--all-features"], root, env)
    if lock.read_bytes() != before:
        raise ValueError("cargo-license changed Cargo.lock; generation stopped")
    tree = subprocess.check_output(
        [
            "cargo",
            "tree",
            "--workspace",
            "--all-features",
            "--target",
            "all",
            "--locked",
            "--offline",
            "--prefix",
            "none",
            "--format",
            "{p}",
        ],
        cwd=root,
        env=env,
        text=True,
    )
    selected = set(re.findall(r"^(\S+) v(\S+)", tree, re.MULTILINE))
    packages = {}
    for package in metadata["packages"]:
        key = package["name"], package["version"]
        if key in packages:
            raise ValueError(f"Ambiguous Rust package identity: {key}")
        packages[key] = package
    notices = []
    for item in inventory:
        key = item["name"], item["version"]
        if key not in selected:
            continue
        package = packages[key]
        folder = Path(package["manifest_path"]).parent
        paths = legal_files(folder)
        if package.get("license_file"):
            declared = Path(package["license_file"])
            declared = declared if declared.is_absolute() else folder / declared
            if not declared.is_file():
                raise ValueError(f"Missing declared Rust license file: {declared}")
            paths = sorted(set(paths + [declared]))
        if not paths and package.get("source") is None:
            workspace_license = tomllib.loads((root / "Cargo.toml").read_text())
            workspace_license = (
                workspace_license.get("workspace", {}).get("package", {}).get("license")
            )
            if (
                package.get("license") == workspace_license
                and (root / "LICENSE").is_file()
            ):
                paths = [root / "LICENSE"]
        texts = list(dict.fromkeys(read_text(path) for path in paths))
        texts = [text for text in texts if text]
        if not item.get("license"):
            raise ValueError(f"Unknown Rust license expression: {key}")
        notices.append(
            Notice(
                "Rust",
                item["name"],
                item["version"],
                item.get("repository") or f"https://crates.io/crates/{item['name']}",
                item["license"],
                texts,
                [
                    str(path.relative_to(folder))
                    if path.is_relative_to(folder)
                    else str(path)
                    for path in paths
                ],
            )
        )
    if {(notice.name, notice.version) for notice in notices} != selected:
        raise ValueError(
            "cargo-license inventory does not cover the resolved Rust tree"
        )
    matches = matching_entries(notices, existing)
    fallback_matches = matching_entries(notices, fallback_existing or [])
    reused = []
    for notice in notices:
        if not notice.texts:
            previous = matches.get(notice.key) or fallback_matches.get(notice.key)
            if previous is None or not previous.body.strip():
                raise ValueError(
                    f"No Rust license text for {notice.name} {notice.version}; supply a reviewed notice first"
                )
            notice.texts = [previous.body.strip()]
            notice.license = previous.license
            notice.sources = [f"existing notice entry {previous.number}"]
            reused.append(f"{notice.name} {notice.version}")
    return notices, reused


def wheel_rank(wheel: dict) -> tuple[int, str]:
    filename = wheel["url"].rsplit("/", 1)[-1]
    if filename.endswith("-none-any.whl"):
        rank = 0
    elif "manylinux" in filename and "x86_64" in filename:
        rank = 1 if "cp312" in filename else 2 if "abi3" in filename else 3
    else:
        rank = 4 if "linux" in filename and "x86_64" in filename else 5
    return rank, filename


def cached_download(wheel: dict, cache: Path, offline: bool) -> Path:
    algorithm, expected = wheel["hash"].split(":", 1)
    if algorithm != "sha256":
        raise ValueError(f"Unsupported archive hash: {algorithm}")
    target = cache / expected
    if not target.exists():
        if offline:
            raise ValueError(f"Archive not cached: {wheel['url']}")
        with urllib.request.urlopen(wheel["url"], timeout=90) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Archive hash mismatch: {wheel['url']}")
        target.write_bytes(payload)
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Cached archive hash mismatch: {wheel['url']}")
    return target


def stage_metadata(archive: Path, destination: Path) -> Path:
    with zipfile.ZipFile(archive) as wheel:
        members = [
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(members) != 1 or len(PurePosixPath(members[0]).parts) != 2:
            raise ValueError(f"Invalid wheel metadata: {archive}")
        prefix = members[0].removesuffix("METADATA")
        folder = destination / prefix.rstrip("/")
        if folder.exists():
            raise ValueError(f"Duplicate wheel metadata directory: {folder.name}")
        for name in wheel.namelist():
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe wheel path: {name}")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(wheel.read(name))
        return folder


def collect_python(root: Path, python: str, cache: Path, offline: bool):
    import email.parser

    lock = tomllib.loads((root / "uv.lock").read_text())
    packages = []
    for package in lock["package"]:
        source = package.get("source", {})
        if "registry" in source:
            packages.append(package)
        elif not ("virtual" in source or "editable" in source):
            raise ValueError(f"Unsupported Python dependency source: {package['name']}")
    expected = {
        (normalize(package["name"]), package["version"]) for package in packages
    }
    if len(expected) != len(packages):
        raise ValueError("Duplicate locked Python distribution identity")
    cache.mkdir(parents=True, exist_ok=True)
    notices = {}
    with tempfile.TemporaryDirectory(
        prefix="physicsnemo-license-metadata-"
    ) as temporary:
        stage = Path(temporary)
        for package in packages:
            if not package.get("wheels"):
                raise ValueError(
                    f"No locked wheel for {package['name']} {package['version']}"
                )
            wheel = min(package["wheels"], key=wheel_rank)
            folder = stage_metadata(cached_download(wheel, cache, offline), stage)
            metadata = email.parser.BytesParser().parsebytes(
                (folder / "METADATA").read_bytes()
            )
            key = normalize(metadata["Name"]), metadata["Version"]
            if key != (normalize(package["name"]), package["version"]):
                raise ValueError(f"Wheel identity mismatch: {wheel['url']}")
            declared = set(metadata.get_all("License-File", []))
            paths = sorted(
                path
                for path in folder.rglob("*")
                if path.is_file()
                and (
                    path.relative_to(folder).parts[0] == "licenses"
                    or path.relative_to(folder).as_posix() in declared
                    or LEGAL_FILE.match(path.name)
                    or path.name.lower().startswith("authors")
                )
            )
            texts = list(dict.fromkeys(read_text(path) for path in paths))
            texts = [text for text in texts if text]
            if not texts:
                raise ValueError(f"No Python license text: {key}")
            notices[key] = Notice(
                "Python",
                package["name"],
                package["version"],
                "",
                "",
                texts,
                [wheel["url"], wheel["hash"]]
                + [path.relative_to(folder).as_posix() for path in paths],
            )
        inventory = run_json(
            [
                python,
                "-m",
                "piplicenses",
                "--format=json",
                "--with-urls",
                "--with-license-file",
                "--with-notice-file",
                "--no-license-path",
                "--with-system",
                "--packages",
                *sorted({package["name"] for package in packages}),
            ],
            root,
            {**os.environ, "PYTHONPATH": str(stage)},
        )
        actual = set()
        for item in inventory:
            key = normalize(item["Name"]), item["Version"]
            if key not in expected or key in actual:
                raise ValueError(
                    f"Unexpected or duplicate pip-licenses distribution: {key}"
                )
            actual.add(key)
            notice = notices[key]
            notice.homepage = item.get("URL") or "UNKNOWN"
            if notice.homepage == "UNKNOWN":
                notice.homepage = (
                    f"https://pypi.org/project/{notice.name}/{notice.version}/"
                )
            notice.license = " ".join(item.get("License", "UNKNOWN").split())
            if notice.license == "UNKNOWN" or len(notice.license) > 200:
                notice.license = "See license files below"
            for field in ("LicenseText", "NoticeText"):
                content = item.get(field, "UNKNOWN").strip()
                if content and content != "UNKNOWN" and content not in notice.texts:
                    notice.texts.append(content)
        if actual != expected:
            raise ValueError(
                f"pip-licenses omitted locked distributions: {sorted(expected - actual)}"
            )
    return list(notices.values())


def render(
    notices: list[Notice],
    header: str,
    existing: list[Existing],
    refresh_existing: bool = False,
):
    if len({notice.key for notice in notices}) != len(notices):
        raise ValueError("Duplicate component identity")
    matches = matching_entries(notices, existing)
    canonical_names = {normalize(notice.name) for notice in notices}
    retained = []
    appended = []
    for notice in notices:
        previous = matches.get(notice.key)
        if previous and not refresh_existing:
            retained.append(previous)
            continue
        name = notice.name
        if previous and (
            normalize(previous.name) == normalize(name)
            or normalize(previous.name) not in canonical_names
        ):
            name = previous.name
        body = "\n\n".join(notice.texts).strip()
        if (
            previous
            and name == previous.name
            and notice.homepage == previous.homepage
            and notice.license == previous.license
            and body.split() == previous.body.split()
        ):
            retained.append(previous)
        else:
            appended.append((name, notice, body))
    chunks = [header]
    chunks.extend(
        entry.block for entry in sorted(retained, key=lambda entry: entry.number)
    )
    if appended and not chunks[-1].endswith("\n\n"):
        chunks.append("\n")
    next_number = max((entry.number for entry in existing), default=0) + 1
    for name, notice, body in sorted(
        appended,
        key=lambda item: (item[0].casefold(), item[1].version, item[1].ecosystem),
    ):
        block = (
            f"{SEPARATOR}\n {next_number}. {name} version: {notice.version}\n"
            f"{SEPARATOR}\n homepage: {notice.homepage}\n {notice.license}\n\n{body}\n\n"
        )
        chunks.append("\n".join(line.rstrip() for line in block.splitlines()) + "\n")
        next_number += 1
    return "".join(chunks).rstrip("\n") + "\n", len(retained)


def collect_supplemental(root: Path) -> list[Notice]:
    """Read reviewed ML/native notices outside the root dependency locks."""
    manifest = root / ".licenses/ml-native.json"
    if not manifest.exists():
        return []
    result = []
    for item in json.loads(manifest.read_text())["components"]:
        texts = [
            "Distribution role: " + item["distribution_role"],
            "Version evidence: " + "; ".join(item["version_evidence"]),
            "Upstream notice sources:\n" + "\n".join(item["sources"]),
            item["copyright"],
        ]
        for source in item["license_files"]:
            path = (root / source["path"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Supplemental license path escapes repository")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise ValueError(f"Supplemental license hash mismatch: {path}")
            text = data.decode("utf-8").strip()
            if not text:
                raise ValueError(f"Empty supplemental license: {path}")
            texts.append(text)
        if not item["license_files"]:
            raise ValueError(f"Missing supplemental license: {item['name']}")
        result.append(
            Notice(
                "ML/native",
                item["name"],
                item["version"],
                item["homepage"],
                item["license"],
                texts,
                item["sources"],
            )
        )
    return result


def atomic_write(path: Path, text: str):
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Merge cargo-license and pip-licenses notices in the existing format."
    )
    parser.add_argument("--cargo-license", default="cargo-license")
    parser.add_argument(
        "--python", default=sys.executable, help="Python with pip-licenses installed"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache/physicsnemo-serve/license-wheels",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Require already cached Python wheels"
    )
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / ".licenses/licenses.txt"
    )
    parser.add_argument("--output", type=Path, default=ROOT / ".licenses/licenses.txt")
    parser.add_argument("--report", type=Path, help="Optional JSON provenance report")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Regenerate matching same-version notices instead of preserving them",
    )
    parser.add_argument(
        "--reuse-notices",
        type=Path,
        default=ROOT / ".licenses/licenses.txt",
        help="Additional existing notices to reuse when a Rust archive omits texts",
    )
    args = parser.parse_args()
    executable = shutil.which(args.cargo_license)
    if executable is None:
        parser.error("cargo-license is not installed or not on PATH")
    try:
        header, existing = (
            parse_existing(args.baseline.read_text())
            if args.baseline.exists()
            else (HEADER, [])
        )
        fallback_existing = (
            parse_existing(args.reuse_notices.read_text())[1]
            if args.reuse_notices.exists() and args.reuse_notices != args.baseline
            else []
        )
        rust, reused = collect_rust(ROOT, executable, existing, fallback_existing)
        python = collect_python(ROOT, args.python, args.cache_dir, args.offline)
        supplemental = collect_supplemental(ROOT)
        notices = rust + python + supplemental
        result, retained = render(notices, header, existing, args.refresh_existing)
        if args.report:
            if args.report.resolve() == args.output.resolve():
                raise ValueError("Report and inventory paths must differ")
            report = {
                "rust_components": len(rust),
                "python_distributions": len(python),
                "ml_native_components": len(supplemental),
                "retained_sections": retained,
                "reused_rust_notices": reused,
                "refresh_existing": args.refresh_existing,
                "components": [vars(notice) for notice in notices],
            }
            atomic_write(args.report, json.dumps(report, indent=2) + "\n")
        atomic_write(args.output, result)
    except (
        ValueError,
        OSError,
        subprocess.CalledProcessError,
        zipfile.BadZipFile,
    ) as error:
        parser.exit(1, f"License generation failed: {error}\n")
    print(
        f"Wrote {len(rust)} Rust, {len(python)} Python and {len(supplemental)} ML/native entries; "
        f"kept {retained} sections unchanged."
    )
    if reused:
        print(
            f"Reused {len(reused)} existing Rust notices whose source archives omit license files:"
        )
        print("\n".join(f"  {name}" for name in reused))


if __name__ == "__main__":
    main()
