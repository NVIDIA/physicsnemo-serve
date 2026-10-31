def test_cfd_reuse_keeps_existing_package_and_skips_builder(tmp_path):
    import importlib

    module = importlib.import_module("qa.scripts.run_native_cfd_job")
    source = tmp_path / "qualified"
    source.mkdir()
    (source / "package.txt").write_text("qualified package")
    invoked = []
    helper = getattr(
        module, "build_or_reuse", lambda *args, **kwargs: invoked.append("builder")
    )
    target = tmp_path / "new-job-build"
    helper(
        source,
        target,
        lambda: invoked.append("builder"),
        verify=lambda path: str(path / "package.txt"),
    )
    assert invoked == [], "unchanged compiled bytes must not invoke Builder"
    assert (target / "package.txt").read_text() == "qualified package"
