"""Tests for LocalBackend."""

from pathlib import Path

import pytest

from pydantic_ai_backends import LocalBackend


class TestLocalBackendInit:
    """Test LocalBackend initialization."""

    def test_init_with_root_dir(self, tmp_path: Path):
        """Test initialization with root_dir."""
        backend = LocalBackend(root_dir=tmp_path)
        assert backend.root_dir == tmp_path
        assert backend.execute_enabled is True

    def test_init_with_allowed_directories(self, tmp_path: Path):
        """Test initialization with allowed_directories."""
        dir1 = tmp_path / "project"
        dir2 = tmp_path / "libs"

        backend = LocalBackend(allowed_directories=[str(dir1), str(dir2)])

        # Directories should be created
        assert dir1.exists()
        assert dir2.exists()
        # root_dir should default to first allowed directory
        assert backend.root_dir == dir1

    def test_init_with_explicit_root_dir_and_allowed(self, tmp_path: Path):
        """Test that explicit root_dir takes precedence."""
        dir1 = tmp_path / "project"
        work = tmp_path / "work"
        work.mkdir()

        backend = LocalBackend(
            root_dir=work,
            allowed_directories=[str(dir1)],
        )

        assert backend.root_dir == work

    def test_init_with_execute_disabled(self, tmp_path: Path):
        """Test initialization with execute disabled."""
        backend = LocalBackend(root_dir=tmp_path, enable_execute=False)
        assert backend.execute_enabled is False

    def test_init_creates_sandbox_id(self, tmp_path: Path):
        """Test that sandbox_id is generated."""
        backend = LocalBackend(root_dir=tmp_path)
        assert backend.id is not None
        assert len(backend.id) > 0

    def test_init_with_custom_sandbox_id(self, tmp_path: Path):
        """Test initialization with custom sandbox_id."""
        backend = LocalBackend(root_dir=tmp_path, sandbox_id="my-sandbox")
        assert backend.id == "my-sandbox"


class TestLocalBackendFileOps:
    """Test LocalBackend file operations."""

    def test_write_and_read(self, tmp_path: Path):
        """Test writing and reading files."""
        backend = LocalBackend(root_dir=tmp_path)

        result = backend.write("test.txt", "hello world")
        assert result.error is None

        content = backend.read("test.txt")
        assert "hello world" in content
        assert "1\t" in content  # Line number

    def test_write_creates_directories(self, tmp_path: Path):
        """Test that write creates parent directories."""
        backend = LocalBackend(root_dir=tmp_path)

        result = backend.write("deep/nested/file.txt", "content")
        assert result.error is None
        assert (tmp_path / "deep" / "nested" / "file.txt").exists()

    def test_read_nonexistent_file(self, tmp_path: Path):
        """Test reading a nonexistent file."""
        backend = LocalBackend(root_dir=tmp_path)

        content = backend.read("nonexistent.txt")
        assert "Error" in content
        assert "not found" in content

    def test_edit_file(self, tmp_path: Path):
        """Test editing a file."""
        backend = LocalBackend(root_dir=tmp_path)

        backend.write("test.txt", "hello world")
        result = backend.edit("test.txt", "hello", "goodbye")

        assert result.error is None
        assert result.occurrences == 1

        content = backend.read("test.txt")
        assert "goodbye world" in content

    def test_edit_file_not_found(self, tmp_path: Path):
        """Test editing a nonexistent file."""
        backend = LocalBackend(root_dir=tmp_path)

        result = backend.edit("nonexistent.txt", "old", "new")
        assert result.error is not None
        assert "not found" in result.error

    def test_ls_info(self, tmp_path: Path):
        """Test listing directory contents."""
        backend = LocalBackend(root_dir=tmp_path)

        backend.write("file1.txt", "a")
        backend.write("file2.txt", "b")
        (tmp_path / "subdir").mkdir()

        entries = backend.ls_info(".")
        assert len(entries) == 3
        names = [e["name"] for e in entries]
        assert "file1.txt" in names
        assert "file2.txt" in names
        assert "subdir" in names

    def test_glob_info(self, tmp_path: Path):
        """Test finding files with glob pattern."""
        backend = LocalBackend(root_dir=tmp_path)

        backend.write("test1.py", "# python")
        backend.write("test2.py", "# python")
        backend.write("readme.md", "# markdown")

        entries = backend.glob_info("*.py")
        assert len(entries) == 2

    def test_grep_raw(self, tmp_path: Path):
        """Test searching files with grep."""
        backend = LocalBackend(root_dir=tmp_path)

        backend.write("test.txt", "hello world\nfoo bar\nhello again")

        result = backend.grep_raw("hello")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_grep_raw_hidden_files_ignored_by_default(self, tmp_path: Path):
        """Hidden files should be excluded when ignore_hidden is True."""
        backend = LocalBackend(root_dir=tmp_path)

        backend.write(".secret.txt", "hidden content")
        backend.write("visible.txt", "visible content")
        backend.write(".venv/test.txt", "another hidden content")

        matches_default = backend.grep_raw("content")
        matches_explicit = backend.grep_raw("content", ignore_hidden=True)

        for matches in (matches_default, matches_explicit):
            paths = {match["path"] for match in matches}
            assert paths == {str(tmp_path / "visible.txt")}

    def test_grep_raw_hidden_files_included_when_requested(self, tmp_path: Path):
        """Hidden files should be searched when ignore_hidden=False."""
        backend = LocalBackend(root_dir=tmp_path)

        backend.write(".secret.txt", "hidden content")
        backend.write("visible.txt", "visible content")
        backend.write(".venv/test.txt", "another hidden content")
        matches = backend.grep_raw("content", ignore_hidden=False)
        paths = {match["path"] for match in matches}
        assert paths == {
            str(tmp_path / "visible.txt"),
            str(tmp_path / ".secret.txt"),
            str(tmp_path / ".venv/test.txt"),
        }


class TestLocalBackendAllowedDirectories:
    """Test LocalBackend with allowed_directories restriction."""

    def test_read_within_allowed(self, tmp_path: Path):
        """Test read() works within allowed directory."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        test_file = allowed / "test.txt"
        test_file.write_text("hello world")

        backend = LocalBackend(allowed_directories=[str(allowed)])

        result = backend.read(str(test_file))
        assert "hello world" in result

    def test_read_outside_allowed(self, tmp_path: Path):
        """Test read() fails outside allowed directory."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()

        backend = LocalBackend(allowed_directories=[str(allowed)])

        result = backend.read("/etc/passwd")
        assert "Error" in result
        assert "Access denied" in result

    def test_write_within_allowed(self, tmp_path: Path):
        """Test write() works within allowed directory."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()

        backend = LocalBackend(allowed_directories=[str(allowed)])

        result = backend.write(str(allowed / "new_file.txt"), "content")
        assert result.error is None
        assert (allowed / "new_file.txt").read_text() == "content"

    def test_write_outside_allowed(self, tmp_path: Path):
        """Test write() fails outside allowed directory."""
        allowed = tmp_path / "allowed"
        outside = tmp_path / "outside"
        allowed.mkdir()
        outside.mkdir()

        backend = LocalBackend(allowed_directories=[str(allowed)])

        result = backend.write(str(outside / "file.txt"), "content")
        assert result.error is not None
        assert "Access denied" in result.error

    def test_path_traversal_blocked(self, tmp_path: Path):
        """Test that path traversal attempts are blocked."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()

        backend = LocalBackend(allowed_directories=[str(allowed)])

        # Try to escape using ..
        result = backend.read(str(allowed / ".." / "etc" / "passwd"))
        assert "Error" in result


class TestLocalBackendExecute:
    """Test LocalBackend shell execution."""

    async def test_execute_basic(self, tmp_path: Path):
        """Test basic command execution."""
        backend = LocalBackend(root_dir=tmp_path)

        result = await backend.execute("echo 'hello'")
        assert result.exit_code == 0
        assert "hello" in result.output

    async def test_execute_with_timeout(self, tmp_path: Path):
        """Test command execution with timeout."""
        backend = LocalBackend(root_dir=tmp_path)

        result = await backend.execute("sleep 0.1 && echo done", timeout=10)
        assert result.exit_code == 0
        assert "done" in result.output

    async def test_execute_timeout_exceeded(self, tmp_path: Path):
        """Test command execution timeout."""
        backend = LocalBackend(root_dir=tmp_path)

        result = await backend.execute("sleep 10", timeout=1)
        assert result.exit_code == 124
        assert "timed out" in result.output

    async def test_execute_disabled_raises(self, tmp_path: Path):
        """Test that execute raises when disabled."""
        backend = LocalBackend(root_dir=tmp_path, enable_execute=False)

        with pytest.raises(RuntimeError) as exc_info:
            await backend.execute("echo 'hello'")

        assert "disabled" in str(exc_info.value)

    async def test_execute_working_dir(self, tmp_path: Path):
        """Test that execute uses root_dir as working directory."""
        backend = LocalBackend(root_dir=tmp_path)

        result = await backend.execute("pwd")
        assert result.exit_code == 0
        assert str(tmp_path) in result.output


class TestLocalBackendPathResolution:
    """Test LocalBackend path resolution and non-existent file handling."""

    def test_read_file_absolute_path(self, tmp_path: Path):
        """Read using a full absolute path."""
        backend = LocalBackend(root_dir=tmp_path)
        (tmp_path / "abs_test.txt").write_text("absolute content")

        content = backend.read(str(tmp_path / "abs_test.txt"))
        assert "absolute content" in content

    def test_read_file_relative_path(self, tmp_path: Path):
        """Read using a relative path, resolved from root_dir."""
        backend = LocalBackend(root_dir=tmp_path)
        (tmp_path / "rel_test.txt").write_text("relative content")

        content = backend.read("rel_test.txt")
        assert "relative content" in content

    def test_read_file_nonexistent_no_crash(self, tmp_path: Path):
        """Reading a non-existent file returns a graceful error string."""
        backend = LocalBackend(root_dir=tmp_path)

        content = backend.read("does_not_exist.txt")
        assert "Error" in content
        assert "not found" in content

    def test_read_bytes_nonexistent(self, tmp_path: Path):
        """_read_bytes returns empty bytes for a non-existent file."""
        backend = LocalBackend(root_dir=tmp_path)

        result = backend._read_bytes("no_such_file.bin")
        assert result == b""

    def test_read_bytes_nonexistent_absolute(self, tmp_path: Path):
        """_read_bytes returns empty bytes for a non-existent absolute path."""
        backend = LocalBackend(root_dir=tmp_path)

        result = backend._read_bytes(str(tmp_path / "no_such_file.bin"))
        assert result == b""

    def test_write_and_read_relative_path(self, tmp_path: Path):
        """Write and read using relative paths resolved from root_dir."""
        backend = LocalBackend(root_dir=tmp_path)

        result = backend.write("new_file.txt", "hello world")
        assert result.error is None

        content = backend.read("new_file.txt")
        assert "hello world" in content

    def test_custom_root_dir(self, tmp_path: Path):
        """Paths resolve against a custom root_dir."""
        custom_root = tmp_path / "custom" / "root"
        custom_root.mkdir(parents=True)

        backend = LocalBackend(root_dir=custom_root)
        backend.write("test.txt", "custom root content")

        assert (custom_root / "test.txt").read_text() == "custom root content"
        content = backend.read("test.txt")
        assert "custom root content" in content
