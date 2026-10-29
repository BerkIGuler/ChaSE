import subprocess

import chase.provenance as P


def test_code_state_of_this_checkout():
    s = P.code_state()
    head = subprocess.run(["git", "-C", str(P.REPO_DIR), "rev-parse", "HEAD"], capture_output=True, text=True)
    assert s["version"] and s["commit"] == (head.stdout.strip() or None)
    assert ("dirty" in s) == (s["commit"] is not None)


def test_no_commit_inside_another_repository(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    pkg = tmp_path / "site-packages"
    pkg.mkdir()
    monkeypatch.setattr(P, "REPO_DIR", pkg)
    assert P.code_state()["commit"] is None


def test_sha256(tmp_path):
    f = tmp_path / "a"
    f.write_bytes(b"abc")
    assert P.sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
