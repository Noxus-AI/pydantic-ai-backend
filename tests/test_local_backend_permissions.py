"""Tests for LocalBackend permission integration."""

from pathlib import Path

import pytest

from pydantic_ai_backends import LocalBackend
from pydantic_ai_backends.permissions import (
    OperationPermissions,
    PermissionChecker,
    PermissionError,
    PermissionRule,
    PermissionRuleset,
)


class TestLocalBackendPermissionsInit:
    """Tests for LocalBackend initialization with permissions."""

    def test_init_without_permissions(self, tmp_path: Path):
        """Test that permissions are optional."""
        backend = LocalBackend(root_dir=tmp_path)

        assert backend.permissions is None
        assert backend.permission_checker is None

    def test_init_with_permissions(self, tmp_path: Path):
        """Test initialization with permissions."""
        ruleset = PermissionRuleset(default="deny")
        backend = LocalBackend(root_dir=tmp_path, permissions=ruleset)

        assert backend.permissions is ruleset
        assert backend.permission_checker is not None
        assert isinstance(backend.permission_checker, PermissionChecker)

    def test_init_with_ask_callback(self, tmp_path: Path):
        """Test initialization with ask callback."""

        async def my_callback(op: str, target: str, reason: str) -> bool:
            return True

        ruleset = PermissionRuleset(default="ask")
        backend = LocalBackend(
            root_dir=tmp_path,
            permissions=ruleset,
            ask_callback=my_callback,
        )

        assert backend.permission_checker is not None

    def test_init_with_ask_fallback(self, tmp_path: Path):
        """Test initialization with ask fallback."""
        ruleset = PermissionRuleset(default="ask")
        backend = LocalBackend(
            root_dir=tmp_path,
            permissions=ruleset,
            ask_fallback="deny",
        )

        assert backend.permission_checker is not None


class TestLocalBackendReadPermissions:
    """Tests for read operation permission checks."""

    async def test_read_allowed(self, tmp_path: Path):
        """Test read when permission allows."""
        ruleset = PermissionRuleset(
            read=OperationPermissions(default="allow"),
        )
        backend = LocalBackend(root_dir=tmp_path, permissions=ruleset)

        # Create and read a file
        (tmp_path / "test.txt").write_text("content")
        result = await backend.read("test.txt")

        assert "content" in result
        assert "Error" not in result

    async def test_read_denied(self, tmp_path: Path):
        """Test read when permission denies."""
        ruleset = PermissionRuleset(
            read=OperationPermissions(default="deny"),
        )
        backend = LocalBackend(root_dir=tmp_path, permissions=ruleset)

        (tmp_path / "test.txt").write_text("content")
        result = await backend.read("test.txt")

        assert "Error" in result
        assert "Permission denied" in result

    async def test_read_denied_by_rule(self, tmp_path: Path):
        """Test read denied by specific rule."""
        ruleset = PermissionRuleset(
            read=OperationPermissions(
                default="allow",
                rules=[
                    PermissionRule(
                        pattern="**/.env",
                        action="deny",
                        description="Protect env files",
                    )
                ],
            ),
        )
        backend = LocalBackend(root_dir=tmp_path, permissions=ruleset)

        env_file = tmp_path / ".env"
        env_file.write_text("SECRET=123")

        result = await backend.read(".env")

        assert "Error" in result
        assert "Protect env files" in result

    async def test_read_ask_with_deny_fallback(self, tmp_path: Path):
        """Test read with ask action and deny fallback."""
        ruleset = PermissionRuleset(
            read=OperationPermissions(default="ask"),
        )
        backend = LocalBackend(
            root_dir=tmp_path,
            permissions=ruleset,
            ask_fallback="deny",
        )

        (tmp_path / "test.txt").write_text("content")
        result = await backend.read("test.txt")

        assert "Error" in result
        assert "approval required" in result

    async def test_read_ask_with_error_fallback(self, tmp_path: Path):
        """Test read with ask action and error fallback."""
        ruleset = PermissionRuleset(
            read=OperationPermissions(default="ask"),
        )
        backend = LocalBackend(
            root_dir=tmp_path,
            permissions=ruleset,
            ask_fallback="error",
        )

        (tmp_path / "test.txt").write_text("content")

        with pytest.raises(PermissionError):
            await backend.read("test.txt")


class TestLocalBackendWritePermissions:
    """Tests for write operation permission checks."""

    async def test_write_allowed(self, tmp_path: Path):
        """Test write when permission allows."""
        ruleset = PermissionRuleset(
            write=OperationPermissions(default="allow"),
        )
        backend = LocalBackend(root_dir=tmp_path, permissions=ruleset)

        result = await backend.write("test.txt", "content")

        assert result.error is None
        assert (tmp_path / "test.txt").read_text() == "content"

    async def test_write_denied(self, tmp_path: Path):
        """Test write when permission denies."""
        ruleset = PermissionRuleset(
            write=OperationPermissions(default="deny"),
        )
        backend = LocalBackend(root_dir=tmp_path, permissions=ruleset)

        result = await backend.write("test.txt", "content")

        assert result.error is not None
        assert "Permission denied" in result.error
        assert not (tmp_path / "test.txt").exists()

    async def test_write_denied_by_rule(self, tmp_path: Path):
        """Test write denied by specific rule."""
        ruleset = PermissionRuleset(
            write=OperationPermissions(
                default="allow",
                rules=[
                    PermissionRule(
                        pattern="**/sensitive.txt",
                        action="deny",
                        description="Protect sensitive file",
                    )
                ],
            ),
        )
        backend = LocalBackend(root_dir=tmp_path, permissions=ruleset)

        result = await backend.write("sensitive.txt", "content")

        assert result.error is not None
        assert "Protect sensitive file" in result.error


class TestLocalBackendEditPermissions:
    """Tests for edit operation permission checks."""

    async def test_edit_allowed(self, tmp_path: Path):
        """Test edit when permission allows."""
        ruleset = PermissionRuleset(
            edit=OperationPermissions(default="allow"),
        )
        backend = LocalBackend(root_dir=tmp_path, permissions=ruleset)

        (tmp_path / "test.txt").write_text("hello world")
        result = await backend.edit("test.txt", "hello", "goodbye")

        assert result.error is None
        assert (tmp_path / "test.txt").read_text() == "goodbye world"

    async def test_edit_denied(self, tmp_path: Path):
        """Test edit when permission denies."""
        ruleset = PermissionRuleset(
            edit=OperationPermissions(default="deny"),
        )
        backend = LocalBackend(root_dir=tmp_path, permissions=ruleset)

        (tmp_path / "test.txt").write_text("hello world")
        result = await backend.edit("test.txt", "hello", "goodbye")

        assert result.error is not None
        assert "Permission denied" in result.error
        # Content should be unchanged
        assert (tmp_path / "test.txt").read_text() == "hello world"


class TestLocalBackendExecutePermissions:
    """Tests for execute operation permission checks."""

    async def test_execute_allowed(self, tmp_path: Path):
        """Test execute when permission allows."""
        ruleset = PermissionRuleset(
            execute=OperationPermissions(default="allow"),
        )
        backend = LocalBackend(root_dir=tmp_path, permissions=ruleset)

        result = await backend.execute("echo hello")

        assert result.exit_code == 0
        assert "hello" in result.output

    async def test_execute_denied(self, tmp_path: Path):
        """Test execute when permission denies."""
        ruleset = PermissionRuleset(
            execute=OperationPermissions(default="deny"),
        )
        backend = LocalBackend(root_dir=tmp_path, permissions=ruleset)

        result = await backend.execute("echo hello")

        assert result.exit_code == 1
        assert "Error" in result.output
        assert "Permission denied" in result.output

    async def test_execute_denied_by_rule(self, tmp_path: Path):
        """Test execute denied by specific rule."""
        ruleset = PermissionRuleset(
            execute=OperationPermissions(
                default="allow",
                rules=[
                    PermissionRule(
                        pattern="rm *",
                        action="deny",
                        description="Block rm commands",
                    )
                ],
            ),
        )
        backend = LocalBackend(root_dir=tmp_path, permissions=ruleset)

        result = await backend.execute("rm file.txt")

        assert result.exit_code == 1
        assert "Block rm commands" in result.output


class TestLocalBackendPermissionsWithAllowedDirectories:
    """Tests for permissions combined with allowed_directories."""

    async def test_allowed_directories_checked_first(self, tmp_path: Path):
        """Test that allowed_directories are checked before permissions."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        ruleset = PermissionRuleset(
            read=OperationPermissions(default="allow"),
        )
        backend = LocalBackend(
            allowed_directories=[str(project_dir)],
            permissions=ruleset,
        )

        # Create file outside allowed directories
        outside = tmp_path / "outside.txt"
        outside.write_text("content")

        # Should fail due to allowed_directories, not permissions
        result = await backend.read(str(outside))

        assert "Error" in result
        assert "outside allowed directories" in result

    async def test_both_checks_pass(self, tmp_path: Path):
        """Test when both checks pass."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        ruleset = PermissionRuleset(
            read=OperationPermissions(default="allow"),
        )
        backend = LocalBackend(
            allowed_directories=[str(project_dir)],
            permissions=ruleset,
        )

        (project_dir / "test.txt").write_text("content")
        result = await backend.read("test.txt")

        assert "content" in result
        assert "Error" not in result
