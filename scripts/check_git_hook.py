"""Exercise staged-file protections in disposable Git repositories."""

from pathlib import Path
import subprocess
import tempfile
import unittest

HOOK = Path(__file__).resolve().parents[1] / ".githooks/pre-commit"


class GitHookTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="metalvla-hook-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.git("init", "-q")

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, capture_output=True, check=True).stdout

    def stage(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        self.git("add", "--", name)
        return path

    def run_hook(self):
        return subprocess.run([str(HOOK)], cwd=self.root, capture_output=True)

    def commit_fixture(self):
        tree = self.git("write-tree").decode().strip()
        commit = self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                          "commit-tree", tree, "-m", "fixture").decode().strip()
        self.git("update-ref", "HEAD", commit)

    def test_large_staged_blob_rejected_after_worktree_shrinks_or_disappears(self):
        path = self.stage("large.bin", b"x" * (6 * 1024 * 1024))
        path.write_bytes(b"x")
        self.assertNotEqual(self.run_hook().returncode, 0)
        path.unlink()
        self.assertNotEqual(self.run_hook().returncode, 0)

    def test_small_staged_blob_ignores_large_worktree_copy(self):
        path = self.stage("small.txt", b"fixture\n")
        path.write_bytes(b"x" * (6 * 1024 * 1024))
        self.assertEqual(self.run_hook().returncode, 0)

    def test_rename_into_data_is_rejected(self):
        self.stage("small.txt", b"fixture\n")
        self.commit_fixture()
        (self.root / "data").mkdir()
        self.git("mv", "--", "small.txt", "data/small.txt")
        self.assertIn(b"R100", self.git("diff", "--cached", "--name-status"))
        self.assertNotEqual(self.run_hook().returncode, 0)

    def test_special_paths_preserve_staged_secret_checks(self):
        for name in ("space name.txt", 'quote"name.txt', "line\nbreak.txt", "-leading.txt"):
            with self.subTest(name=name):
                path = self.stage(name, ("sk-" + "a" * 24).encode())
                path.write_bytes(b"safe working copy")
                self.assertNotEqual(self.run_hook().returncode, 0)
                self.git("rm", "--cached", "-f", "--", name)

    def test_small_file_and_deletion_are_allowed(self):
        self.stage("small.txt", b"fixture\n")
        self.assertEqual(self.run_hook().returncode, 0)
        self.commit_fixture()
        self.git("rm", "--", "small.txt")
        self.assertEqual(self.run_hook().returncode, 0)

    def test_type_change_checks_the_new_staged_blob(self):
        (self.root / "changed").symlink_to("missing-target")
        self.git("add", "--", "changed")
        self.commit_fixture()
        (self.root / "changed").unlink()
        self.stage("changed", b"x" * (6 * 1024 * 1024))
        self.assertNotEqual(self.run_hook().returncode, 0)


if __name__ == "__main__":
    unittest.main()
