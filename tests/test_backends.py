"""Tests for backend implementations."""

from pydantic_ai_backends import CompositeBackend, LocalBackend, StateBackend


class TestStateBackend:
    """Tests for StateBackend."""

    async def test_write_and_read(self):
        """Test writing and reading a file."""
        backend = StateBackend()

        result = await backend.write("/test.txt", "Hello, World!")
        assert result.error is None
        assert result.path == "/test.txt"

        content = await backend.read("/test.txt")
        assert "Hello, World!" in content
        assert "1\t" in content  # Line number

    async def test_write_multiline(self):
        """Test writing multiline content."""
        backend = StateBackend()

        content = "Line 1\nLine 2\nLine 3"
        await backend.write("/multi.txt", content)

        result = await backend.read("/multi.txt")
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    async def test_read_nonexistent(self):
        """Test reading a file that doesn't exist."""
        backend = StateBackend()

        result = await backend.read("/nonexistent.txt")
        assert "Error:" in result
        assert "not found" in result

    async def test_edit_file(self):
        """Test editing a file."""
        backend = StateBackend()

        await backend.write("/test.txt", "Hello, World!")
        result = await backend.edit("/test.txt", "World", "Universe")

        assert result.error is None
        assert result.occurrences == 1

        content = await backend.read("/test.txt")
        assert "Universe" in content
        assert "World" not in content

    async def test_edit_multiple_occurrences(self):
        """Test editing with multiple occurrences."""
        backend = StateBackend()

        await backend.write("/test.txt", "foo bar foo baz foo")

        # Should fail without replace_all
        result = await backend.edit("/test.txt", "foo", "qux")
        assert result.error is not None
        assert "3 times" in result.error

        # Should succeed with replace_all
        result = await backend.edit("/test.txt", "foo", "qux", replace_all=True)
        assert result.error is None
        assert result.occurrences == 3

    async def test_ls_info(self):
        """Test listing directory contents."""
        backend = StateBackend()

        await backend.write("/dir/file1.txt", "content1")
        await backend.write("/dir/file2.txt", "content2")
        await backend.write("/dir/subdir/file3.txt", "content3")

        entries = await backend.ls_info("/dir")
        names = [e["name"] for e in entries]

        assert "file1.txt" in names
        assert "file2.txt" in names
        assert "subdir" in names

    async def test_glob_info(self):
        """Test glob pattern matching."""
        backend = StateBackend()

        await backend.write("/src/main.py", "# main")
        await backend.write("/src/utils.py", "# utils")
        await backend.write("/src/test.js", "// test")
        await backend.write("/lib/helper.py", "# helper")

        # Find all Python files
        results = await backend.glob_info("**/*.py")
        paths = [r["path"] for r in results]

        assert "/src/main.py" in paths
        assert "/src/utils.py" in paths
        assert "/lib/helper.py" in paths
        assert "/src/test.js" not in paths

    async def test_grep_raw(self):
        """Test grep searching."""
        backend = StateBackend()

        await backend.write("/test.py", "def hello():\n    print('Hello')")
        await backend.write("/other.py", "def goodbye():\n    print('Goodbye')")

        results = await backend.grep_raw("hello")
        assert isinstance(results, list)
        assert len(results) > 0
        assert any(r["path"] == "/test.py" for r in results)

    async def test_path_validation(self):
        """Test path validation security."""
        backend = StateBackend()

        # Test .. traversal
        result = await backend.write("/../etc/passwd", "hack")
        assert result.error is not None

        # Test ~ expansion
        result = await backend.write("~/secret", "hack")
        assert result.error is not None

    async def test_write_bytes(self):
        """Test writing bytes to StateBackend."""
        backend = StateBackend()

        # Write bytes content
        result = await backend.write("/binary.txt", b"Hello, bytes!")
        assert result.error is None
        assert result.path == "/binary.txt"

        # Read it back
        content = await backend.read("/binary.txt")
        assert "Hello, bytes!" in content

    async def test_read_bytes(self):
        """Test reading raw bytes from files.

        Note: StateBackend stores content as text lines, so binary data
        with invalid UTF-8 sequences will be converted with errors='replace'.
        This is expected behavior for a text-based in-memory backend.
        """
        backend = StateBackend()

        # Write text content
        await backend.write("/text.txt", "Hello, World!")
        data = await backend._read_bytes("/text.txt")
        assert isinstance(data, bytes)
        assert data == b"Hello, World!"

        # Write valid UTF-8 bytes content (as text will round-trip correctly)
        await backend.write("/valid_utf8.txt", "Hello 世界 🌍")
        data = await backend._read_bytes("/valid_utf8.txt")
        assert isinstance(data, bytes)
        assert data == "Hello 世界 🌍".encode()

        # Test multiline content
        await backend.write("/multi.txt", "Line 1\nLine 2\nLine 3")
        data = await backend._read_bytes("/multi.txt")
        assert isinstance(data, bytes)
        assert data == b"Line 1\nLine 2\nLine 3"

        # Test non-existent file
        data = await backend._read_bytes("/nonexistent.txt")
        assert isinstance(data, bytes)
        assert data == b""

        # Test empty file
        await backend.write("/empty.txt", "")
        data = await backend._read_bytes("/empty.txt")
        assert isinstance(data, bytes)
        assert data == b""

    async def test_edge_case_filenames(self):
        """Test handling of edge case filenames with special characters."""
        backend = StateBackend()

        # Filename with spaces
        result = await backend.write("/file with spaces.txt", "Content with spaces")
        assert result.error is None
        content = await backend.read("/file with spaces.txt")
        assert "Content with spaces" in content

        # Filename with special characters
        result = await backend.write("/file-name_test.123.txt", "Special chars")
        assert result.error is None
        content = await backend.read("/file-name_test.123.txt")
        assert "Special chars" in content

        # Filename with unicode
        result = await backend.write("/файл.txt", "Unicode filename")
        assert result.error is None
        content = await backend.read("/файл.txt")
        assert "Unicode filename" in content

        # Filename with emoji
        result = await backend.write("/file_🚀.txt", "Emoji filename")
        assert result.error is None
        content = await backend.read("/file_🚀.txt")
        assert "Emoji filename" in content

        # Deep nested path
        result = await backend.write("/very/deep/nested/path/to/file.txt", "Deep nested")
        assert result.error is None
        content = await backend.read("/very/deep/nested/path/to/file.txt")
        assert "Deep nested" in content

        # Path with dots
        result = await backend.write("/path/to/../file.txt", "Dots path")
        assert result.error is not None  # Should fail due to .. validation

        # Multiple extensions
        result = await backend.write("/archive.tar.gz", "Archive content")
        assert result.error is None
        content = await backend.read("/archive.tar.gz")
        assert "Archive content" in content

        # Filename with parentheses and brackets
        result = await backend.write("/file (copy) [1].txt", "Brackets and parens")
        assert result.error is None
        content = await backend.read("/file (copy) [1].txt")
        assert "Brackets and parens" in content

        # Filename with quotes (single and double)
        result = await backend.write("/file's-name.txt", "Single quote")
        assert result.error is None
        content = await backend.read("/file's-name.txt")
        assert "Single quote" in content


class TestLocalBackend:
    """Tests for LocalBackend."""

    async def test_write_and_read(self, tmp_path):
        """Test writing and reading a file."""
        backend = LocalBackend(root_dir=tmp_path)

        result = await backend.write("test.txt", "Hello, World!")
        assert result.error is None

        content = await backend.read("test.txt")
        assert "Hello, World!" in content

    async def test_write_bytes(self, tmp_path):
        """Test writing bytes to a file."""
        backend = LocalBackend(root_dir=tmp_path)

        result = await backend.write("binary.dat", b"\x80\x81\x82")
        assert result.error is None

        # Verify file was written as bytes
        full_path = tmp_path / "binary.dat"
        assert full_path.exists()
        assert full_path.read_bytes() == b"\x80\x81\x82"

    def test_creates_directory(self, tmp_path):
        """Test that LocalBackend creates directory automatically."""
        new_dir = tmp_path / "new_dir"
        assert not new_dir.exists()

        _ = LocalBackend(root_dir=new_dir)
        assert new_dir.exists()

    async def test_edit_file(self, tmp_path):
        """Test editing a file."""
        backend = LocalBackend(root_dir=tmp_path)

        await backend.write("test.txt", "Hello, World!")
        result = await backend.edit("test.txt", "World", "Universe")

        assert result.error is None

        content = await backend.read("test.txt")
        assert "Universe" in content

    async def test_glob_info(self, tmp_path):
        """Test glob pattern matching."""
        backend = LocalBackend(root_dir=tmp_path)

        await backend.write("src/main.py", "# main")
        await backend.write("src/utils.py", "# utils")

        results = await backend.glob_info("**/*.py")
        assert len(results) == 2

    async def test_read_bytes(self, tmp_path):
        """Test reading raw bytes from files."""
        backend = LocalBackend(root_dir=tmp_path)

        # Write text content
        await backend.write("text.txt", "Hello, World!")
        data = await backend._read_bytes("text.txt")
        assert isinstance(data, bytes)
        assert data == b"Hello, World!"

        # Write binary content
        await backend.write("binary.dat", b"\x00\x01\x02\xff\xfe")
        data = await backend._read_bytes("binary.dat")
        assert isinstance(data, bytes)
        assert data == b"\x00\x01\x02\xff\xfe"

        # Test multiline content
        await backend.write("multi.txt", "Line 1\nLine 2\nLine 3")
        data = await backend._read_bytes("multi.txt")
        assert isinstance(data, bytes)
        assert data == b"Line 1\nLine 2\nLine 3"

        # Test non-existent file
        data = await backend._read_bytes("nonexistent.txt")
        assert isinstance(data, bytes)
        assert data == b""

        # Test UTF-8 content
        await backend.write("unicode.txt", "Hello 世界 🌍")
        data = await backend._read_bytes("unicode.txt")
        assert isinstance(data, bytes)
        assert data == "Hello 世界 🌍".encode()

        # Test empty file
        await backend.write("empty.txt", "")
        data = await backend._read_bytes("empty.txt")
        assert isinstance(data, bytes)
        assert data == b""

    async def test_edge_case_filenames(self, tmp_path):
        """Test handling of edge case filenames with special characters."""
        backend = LocalBackend(root_dir=tmp_path)

        # Filename with spaces
        result = await backend.write("file with spaces.txt", "Content with spaces")
        assert result.error is None
        content = await backend.read("file with spaces.txt")
        assert "Content with spaces" in content
        # Verify file actually exists on filesystem
        assert (tmp_path / "file with spaces.txt").exists()

        # Filename with special characters
        result = await backend.write("file-name_test.123.txt", "Special chars")
        assert result.error is None
        content = await backend.read("file-name_test.123.txt")
        assert "Special chars" in content

        # Filename with unicode
        result = await backend.write("файл.txt", "Unicode filename")
        assert result.error is None
        content = await backend.read("файл.txt")
        assert "Unicode filename" in content
        assert (tmp_path / "файл.txt").exists()

        # Filename with emoji
        result = await backend.write("file_🚀.txt", "Emoji filename")
        assert result.error is None
        content = await backend.read("file_🚀.txt")
        assert "Emoji filename" in content

        # Deep nested path
        result = await backend.write("very/deep/nested/path/to/file.txt", "Deep nested")
        assert result.error is None
        content = await backend.read("very/deep/nested/path/to/file.txt")
        assert "Deep nested" in content
        assert (tmp_path / "very" / "deep" / "nested" / "path" / "to" / "file.txt").exists()

        # Multiple extensions
        result = await backend.write("archive.tar.gz", "Archive content")
        assert result.error is None
        content = await backend.read("archive.tar.gz")
        assert "Archive content" in content

        # Filename with parentheses and brackets
        result = await backend.write("file (copy) [1].txt", "Brackets and parens")
        assert result.error is None
        content = await backend.read("file (copy) [1].txt")
        assert "Brackets and parens" in content
        assert (tmp_path / "file (copy) [1].txt").exists()

        # Filename with single quote
        result = await backend.write("file's-name.txt", "Single quote")
        assert result.error is None
        content = await backend.read("file's-name.txt")
        assert "Single quote" in content

        # Test editing file with spaces
        edit_result = await backend.edit("file with spaces.txt", "Content", "Modified")
        assert edit_result.error is None
        content = await backend.read("file with spaces.txt")
        assert "Modified with spaces" in content

        # Binary file with special name
        result = await backend.write("data (binary) [v2].bin", b"\x00\x01\x02\x03")
        assert result.error is None
        data = await backend._read_bytes("data (binary) [v2].bin")
        assert data == b"\x00\x01\x02\x03"

    async def test_execute(self, tmp_path):
        """Test executing shell commands."""
        backend = LocalBackend(root_dir=tmp_path)

        result = await backend.execute("echo 'Hello, World!'")
        assert result.exit_code == 0
        assert "Hello, World!" in result.output

    async def test_execute_disabled(self, tmp_path):
        """Test that execute raises error when disabled."""
        import pytest

        backend = LocalBackend(root_dir=tmp_path, enable_execute=False)

        with pytest.raises(RuntimeError, match="Shell execution is disabled"):
            await backend.execute("echo 'test'")


class TestCompositeBackend:
    """Tests for CompositeBackend."""

    async def test_routing(self):
        """Test that operations are routed to correct backend."""
        memory_backend = StateBackend()
        temp_backend = StateBackend()

        composite = CompositeBackend(
            default=memory_backend,
            routes={"/temp/": temp_backend},
        )

        # Write to default backend
        await composite.write("/data.txt", "default data")
        assert "/data.txt" in memory_backend.files
        assert "/data.txt" not in temp_backend.files

        # Write to temp backend
        await composite.write("/temp/cache.txt", "temp data")
        assert "/temp/cache.txt" in temp_backend.files
        assert "/temp/cache.txt" not in memory_backend.files

    async def test_aggregated_ls(self):
        """Test that root ls aggregates from all backends."""
        backend1 = StateBackend()
        backend2 = StateBackend()

        await backend1.write("/file1.txt", "content1")
        await backend2.write("/special/file2.txt", "content2")

        composite = CompositeBackend(
            default=backend1,
            routes={"/special/": backend2},
        )

        entries = await composite.ls_info("/")
        names = [e["name"] for e in entries]

        assert "file1.txt" in names
        assert "special" in names  # Virtual directory for route

    async def test_read_bytes(self):
        """Test reading raw bytes through composite backend.

        Note: Using StateBackend instances which store text, so binary
        data will be converted. This tests routing, not binary handling.
        """
        backend1 = StateBackend()
        backend2 = StateBackend()

        composite = CompositeBackend(
            default=backend1,
            routes={"/special/": backend2},
        )

        # Write to default backend
        await composite.write("/default.txt", "default content")
        data = await composite._read_bytes("/default.txt")
        assert isinstance(data, bytes)
        assert data == b"default content"

        # Write to routed backend
        await composite.write("/special/routed.txt", "routed content")
        data = await composite._read_bytes("/special/routed.txt")
        assert isinstance(data, bytes)
        assert data == b"routed content"

        # Write UTF-8 text to routed backend
        await composite.write("/special/unicode.txt", "Hello 世界")
        data = await composite._read_bytes("/special/unicode.txt")
        assert isinstance(data, bytes)
        assert data == "Hello 世界".encode()

        # Test non-existent file
        data = await composite._read_bytes("/nonexistent.txt")
        assert isinstance(data, bytes)
        assert data == b""

    async def test_edge_case_filenames_routing(self):
        """Test routing with edge case filenames across different backends."""
        # Use StateBackend for virtual path handling
        default_backend = StateBackend()
        special_backend = StateBackend()

        composite = CompositeBackend(
            default=default_backend,
            routes={"/special/": special_backend},
        )

        # File with spaces in default backend
        result = await composite.write("/file with spaces.txt", "default backend")
        assert result.error is None
        content = await composite.read("/file with spaces.txt")
        assert "default backend" in content

        # File with spaces in routed backend
        result = await composite.write("/special/file with spaces.txt", "routed backend")
        assert result.error is None
        content = await composite.read("/special/file with spaces.txt")
        assert "routed backend" in content

        # Unicode filenames in different backends
        result = await composite.write("/файл.txt", "default unicode")
        assert result.error is None
        result = await composite.write("/special/文件.txt", "routed unicode")
        assert result.error is None

        # Emoji filenames
        result = await composite.write("/emoji_😀.txt", "default emoji")
        assert result.error is None
        result = await composite.write("/special/emoji_🚀.txt", "routed emoji")
        assert result.error is None

        # Complex filenames with special chars
        result = await composite.write("/report (final) [v2.1].txt", "default complex")
        assert result.error is None
        result = await composite.write("/special/data [backup] (2024).txt", "routed complex")
        assert result.error is None

        # Verify routing is correct
        assert "default complex" in await composite.read("/report (final) [v2.1].txt")
        assert "routed complex" in await composite.read("/special/data [backup] (2024).txt")

        # Deep nested paths with special chars
        result = await composite.write("/deep/path with spaces/file.txt", "default nested")
        assert result.error is None
        result = await composite.write("/special/deep/path (v2)/file's.txt", "routed nested")
        assert result.error is None

        # Test editing files with special names across backends
        edit_result = await composite.edit("/file with spaces.txt", "default", "modified")
        assert edit_result.error is None
        assert "modified backend" in await composite.read("/file with spaces.txt")

        edit_result = await composite.edit("/special/file with spaces.txt", "routed", "updated")
        assert edit_result.error is None
        assert "updated backend" in await composite.read("/special/file with spaces.txt")
