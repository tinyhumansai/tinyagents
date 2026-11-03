"""Exercise the Release workflow: a release is a tag on main, never a bot commit.

Run: python3 -B .github/scripts/test-release-workflow.py -v
"""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/release.yml"
VERIFY_STEP = "Verify the release tag is on main and matches the crate version"

# Stands in for `cargo metadata`: reports the version in the checkout's VERSION file.
FAKE_CARGO = """#!/usr/bin/env bash
printf '{"packages":[{"name":"tinyagents-harness","version":"%s"}]}\\n' "$(cat VERSION)"
"""


def step_script(workflow, name):
    match = re.search(
        rf"(?ms)^      - name: {re.escape(name)}\n.*?^        run: \|\n((?:^          [^\n]*\n|^\n)+)",
        workflow,
    )
    if not match:
        raise AssertionError(f"step {name!r} was not found")
    return "\n".join(
        line[10:] if line.startswith("          ") else "" for line in match[1].splitlines()
    )


def git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


class ReleaseWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = WORKFLOW.read_text()
        cls.script = step_script(cls.workflow, VERIFY_STEP)

    def test_release_never_commits_tags_or_pushes(self):
        for forbidden in [
            "git commit",
            "git tag",
            "git push",
            "cargo update",
            "github-actions[bot]",
            "user.name",
            "next-release-version",
        ]:
            self.assertNotIn(forbidden, self.workflow)
        self.assertIn("permissions:\n  contents: read\n", self.workflow)
        self.assertNotIn("contents: write", self.workflow)

    def test_release_is_triggered_by_a_tag(self):
        self.assertIn("  push:\n    tags: ['v*']\n", self.workflow)
        dispatch = re.search(r"(?ms)^  workflow_dispatch:\n(?P<body>.*?)(?=^\S|^concurrency:)", self.workflow)
        self.assertIsNotNone(dispatch)
        self.assertEqual(re.findall(r"^      ([a-z_-]+):$", dispatch["body"], re.MULTILINE), ["tag"])
        self.assertIn("ref: refs/tags/${{ env.RELEASE_TAG }}", self.workflow)
        # The verification runs before anything is built or published.
        self.assertLess(self.workflow.index(VERIFY_STEP), self.workflow.index("- name: Run tests"))
        self.assertLess(self.workflow.index(VERIFY_STEP), self.workflow.index("cargo publish"))
        # A retry skips publishing a version crates.io already has.
        self.assertIn('"https://crates.io/api/v1/crates/${CRATE_NAME}/${RELEASE_TAG#v}"', self.workflow)
        self.assertLess(
            self.workflow.index("api/v1/crates"), self.workflow.index("- name: Publish to crates.io")
        )

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        base = Path(self.directory.name)
        bin_dir = base / "bin"
        bin_dir.mkdir()
        (bin_dir / "cargo").write_text(FAKE_CARGO)
        (bin_dir / "cargo").chmod(0o755)
        self.path = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"

        # origin: main carries 3.1.4 (tagged v3.1.4, and v9.9.9 as a mismatch);
        # a commit off main carries 3.1.5, tagged v3.1.5 like the old bot release.
        work = base / "work"
        work.mkdir()
        git(work, "init", "-q", "-b", "main")
        (work / "VERSION").write_text("3.1.4")
        git(work, "add", "VERSION")
        git(work, "commit", "-q", "-m", "bump to 3.1.4")
        git(work, "tag", "v3.1.4")
        git(work, "tag", "v9.9.9")
        git(work, "tag", "v3.1.4-rc1")
        git(work, "checkout", "-q", "-b", "off-main")
        (work / "VERSION").write_text("3.1.5")
        git(work, "commit", "-q", "-am", "Release v3.1.5")
        git(work, "tag", "v3.1.5")
        git(work, "checkout", "-q", "main")
        git(work, "branch", "-q", "-D", "off-main")
        self.origin = base / "origin.git"
        git(base, "clone", "-q", "--bare", str(work), str(self.origin))

    def verify(self, tag, event="push", ref=None):
        checkout = Path(self.directory.name) / f"checkout-{tag}-{event}"
        git(Path(self.directory.name), "clone", "-q", str(self.origin), str(checkout))
        git(checkout, "checkout", "-q", f"refs/tags/{tag}")
        return subprocess.run(
            ["bash", "-c", self.script],
            cwd=checkout,
            env={
                **os.environ,
                "PATH": self.path,
                "CRATE_NAME": "tinyagents-harness",
                "RELEASE_TAG": tag,
                "GITHUB_EVENT_NAME": event,
                "GITHUB_REF": ref or f"refs/tags/{tag}",
            },
            capture_output=True,
            text=True,
        )

    def test_tag_on_main_matching_the_crate_version_is_released(self):
        result = self.verify("v3.1.4")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.verify("v3.1.4", event="workflow_dispatch", ref="refs/heads/main")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tag_off_main_is_refused(self):
        result = self.verify("v3.1.5")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is not on main", result.stderr)

    def test_tag_that_differs_from_the_crate_version_is_refused(self):
        result = self.verify("v9.9.9")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match tinyagents-harness version 3.1.4", result.stderr)

    def test_dispatch_from_another_branch_is_refused(self):
        result = self.verify("v3.1.4", event="workflow_dispatch", ref="refs/heads/epic/release")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dispatched from main", result.stderr)

    def test_non_release_tag_is_refused(self):
        result = self.verify("v3.1.4-rc1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is not vX.Y.Z", result.stderr)


if __name__ == "__main__":
    unittest.main()
