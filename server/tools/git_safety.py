"""Git safety tool with protection against dangerous operations.

This tool wraps git commands with safety checks to prevent:
- Force pushes to protected branches (main, master)
- Hard resets that lose uncommitted changes
- Amending commits that have been pushed
- Skipping hooks without explicit permission
- Other destructive operations
"""

import json
import subprocess
import re
import os
from mcp.server.fastmcp import FastMCP

# Protected branches that should never receive force pushes
PROTECTED_BRANCHES = ["main", "master", "develop", "production", "release"]

# Dangerous command patterns
DANGEROUS_PATTERNS = [
    # Force push patterns
    (
        r"push\s+.*--force(?!-with-lease)",
        "force_push",
        "Force push can overwrite remote history. Use --force-with-lease instead for safer force pushes.",
    ),
    (
        r"push\s+.*-f(?!\w)",
        "force_push",
        "Force push (-f) can overwrite remote history. Use --force-with-lease instead.",
    ),
    # Hard reset patterns
    (
        r"reset\s+--hard",
        "hard_reset",
        "Hard reset will discard all uncommitted changes permanently.",
    ),
    # Clean patterns
    (
        r"clean\s+.*-f",
        "clean_force",
        "git clean -f will permanently delete untracked files.",
    ),
    (
        r"clean\s+.*-d.*-f|clean\s+.*-f.*-d",
        "clean_force_dirs",
        "git clean -fd will permanently delete untracked files AND directories.",
    ),
    # Branch deletion
    (
        r"branch\s+.*-D",
        "force_delete_branch",
        "Force delete (-D) will delete the branch even if not fully merged.",
    ),
    (r"push\s+.*:\s*\w+", "delete_remote_branch", "This will delete a remote branch."),
    # Rebase on public branches
    (
        r"rebase\s+.*(?:main|master|develop)",
        "rebase_public",
        "Rebasing onto public branches can cause issues for collaborators.",
    ),
    # Checkout with force
    (
        r"checkout\s+.*-f",
        "force_checkout",
        "Force checkout will discard local changes.",
    ),
    # Skip hooks
    (
        r"commit\s+.*--no-verify",
        "skip_hooks",
        "Skipping pre-commit hooks may bypass important checks.",
    ),
    (
        r"push\s+.*--no-verify",
        "skip_push_hooks",
        "Skipping push hooks may bypass important checks.",
    ),
    # Amend
    (
        r"commit\s+.*--amend",
        "amend",
        "Amending commits that have been pushed requires force push.",
    ),
]

# Commands that are always blocked
BLOCKED_COMMANDS = [
    (
        r"push\s+(-f|--force)\s+(origin\s+)?(main|master)\b",
        "Force push to main/master is blocked. This is almost never what you want.",
    ),
    (
        r"push\s+(origin\s+)?(main|master)\s+(-f|--force)",
        "Force push to main/master is blocked. This is almost never what you want.",
    ),
]


def check_dangerous_command(command: str) -> dict:
    """Check if a git command is dangerous and return details."""
    command_lower = command.lower().strip()

    # Check for completely blocked commands
    for pattern, message in BLOCKED_COMMANDS:
        if re.search(pattern, command_lower):
            return {"blocked": True, "reason": message, "command": command}

    # Check for dangerous patterns that need warnings
    warnings = []
    for pattern, danger_type, message in DANGEROUS_PATTERNS:
        if re.search(pattern, command_lower):
            warnings.append({"type": danger_type, "message": message})

    if warnings:
        return {
            "blocked": False,
            "dangerous": True,
            "warnings": warnings,
            "command": command,
        }

    return {"blocked": False, "dangerous": False, "command": command}


def get_current_branch() -> str:
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def is_branch_pushed() -> bool:
    """Check if current branch has been pushed to remote."""
    try:
        result = subprocess.run(
            ["git", "status", "-sb"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # If output contains "ahead", branch has unpushed commits
            # If it doesn't mention origin, branch might not be tracking
            return "ahead" not in result.stdout and "origin" in result.stdout
    except Exception:
        pass
    return True  # Assume pushed if we can't determine


def get_last_commit_author() -> tuple:
    """Get the author name and email of the last commit."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%an %ae"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def check_amend_safety() -> dict:
    """Check if amending the last commit is safe."""
    branch = get_current_branch()
    is_pushed = is_branch_pushed()
    author = get_last_commit_author()

    issues = []

    if branch in PROTECTED_BRANCHES:
        issues.append(f"You're on protected branch '{branch}'. Amending here is risky.")

    if is_pushed:
        issues.append(
            "The last commit appears to have been pushed. Amending will require force push."
        )

    return {
        "safe": len(issues) == 0,
        "branch": branch,
        "is_pushed": is_pushed,
        "author": author,
        "issues": issues,
    }


def register_git_safety_tools(mcp: FastMCP) -> None:
    """Register git safety tools.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    async def git_safe(
        command: str, allow_dangerous: bool = False, reason: str = ""
    ) -> str:
        """Execute git commands with safety checks.

        This tool wraps git commands with safety validations to prevent
        accidental data loss or repository corruption.

        Args:
            command: The git command to execute (without 'git' prefix)
                     Example: "status", "commit -m 'message'", "push origin main"
            allow_dangerous: Set to True to allow dangerous operations (requires reason)
            reason: Required explanation when allow_dangerous=True

        Returns:
            JSON string with command output or safety warnings

        Safety Features:
            - Blocks force push to main/master
            - Warns about hard resets, force operations
            - Checks amend safety (is commit pushed?)
            - Warns about skipping hooks
            - Validates destructive operations

        Examples:
            git_safe(command="status")
            git_safe(command="commit -m 'Add feature'")
            git_safe(command="push origin feature-branch")
            git_safe(command="reset --hard HEAD~1", allow_dangerous=True, reason="Discarding failed experiment")
        """
        # Validate command
        if not command or not command.strip():
            return json.dumps({"error": True, "message": "No git command provided"})

        command = command.strip()

        # Remove 'git' prefix if user included it
        if command.lower().startswith("git "):
            command = command[4:].strip()

        # Check for dangerous commands
        safety_check = check_dangerous_command(command)

        # Blocked commands cannot be overridden
        if safety_check.get("blocked"):
            return json.dumps(
                {
                    "error": True,
                    "blocked": True,
                    "message": safety_check["reason"],
                    "command": f"git {command}",
                    "suggestion": "If you really need to do this, use execute_bash directly with extreme caution.",
                },
                indent=2,
            )

        # Dangerous commands need explicit permission
        if safety_check.get("dangerous") and not allow_dangerous:
            warnings = safety_check.get("warnings", [])

            # Special handling for amend - add extra context
            if any(w["type"] == "amend" for w in warnings):
                amend_check = check_amend_safety()
                if not amend_check["safe"]:
                    warnings.extend(
                        [
                            {"type": "amend_context", "message": issue}
                            for issue in amend_check["issues"]
                        ]
                    )

            return json.dumps(
                {
                    "error": True,
                    "requires_confirmation": True,
                    "message": "This command has potential risks",
                    "warnings": warnings,
                    "command": f"git {command}",
                    "next_step": "To proceed, call git_safe with allow_dangerous=True and provide a reason explaining why this is intentional.",
                },
                indent=2,
            )

        # If allowing dangerous operation, require reason
        if safety_check.get("dangerous") and allow_dangerous and not reason:
            return json.dumps(
                {
                    "error": True,
                    "message": "When using allow_dangerous=True, you must provide a reason",
                    "command": f"git {command}",
                    "example": f'git_safe(command="{command}", allow_dangerous=True, reason="Explanation here")',
                },
                indent=2,
            )

        # Execute the command
        try:
            result = subprocess.run(
                ["git"] + command.split(),
                capture_output=True,
                text=True,
                timeout=60,
                cwd=os.getcwd(),
            )

            output = {
                "success": result.returncode == 0,
                "command": f"git {command}",
                "stdout": result.stdout.strip() if result.stdout else "",
                "stderr": result.stderr.strip() if result.stderr else "",
                "return_code": result.returncode,
            }

            # Add warning acknowledgment if dangerous command was allowed
            if safety_check.get("dangerous") and allow_dangerous:
                output["warning_acknowledged"] = True
                output["reason_provided"] = reason

            return json.dumps(output, indent=2)

        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "error": True,
                    "message": f"Command timed out after 60 seconds: git {command}",
                    "suggestion": "The command may be waiting for input or running a long operation",
                },
                indent=2,
            )
        except FileNotFoundError:
            return json.dumps(
                {
                    "error": True,
                    "message": "Git is not installed or not in PATH",
                    "suggestion": "Install git and ensure it's available in your system PATH",
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {
                    "error": True,
                    "message": f"Error executing git command: {str(e)}",
                    "command": f"git {command}",
                },
                indent=2,
            )

    @mcp.tool()
    async def git_status_safe() -> str:
        """Get comprehensive git status with safety information.

        Returns current branch, uncommitted changes, push/pull status,
        and any potential issues.

        Returns:
            JSON string with detailed repository status
        """
        try:
            status_info = {}

            # Get current branch
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if branch_result.returncode == 0:
                status_info["branch"] = branch_result.stdout.strip()
                status_info["is_protected"] = (
                    status_info["branch"] in PROTECTED_BRANCHES
                )

            # Get status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if status_result.returncode == 0:
                changes = (
                    status_result.stdout.strip().split("\n")
                    if status_result.stdout.strip()
                    else []
                )
                status_info["uncommitted_changes"] = len(changes)
                status_info["has_changes"] = len(changes) > 0
                if changes and changes[0]:
                    status_info["changed_files"] = changes[:10]  # First 10 files
                    if len(changes) > 10:
                        status_info["more_files"] = len(changes) - 10

            # Get ahead/behind status
            tracking_result = subprocess.run(
                ["git", "status", "-sb"], capture_output=True, text=True, timeout=10
            )
            if tracking_result.returncode == 0:
                status_line = tracking_result.stdout.split("\n")[0]
                if "ahead" in status_line:
                    match = re.search(r"ahead (\d+)", status_line)
                    if match:
                        status_info["commits_ahead"] = int(match.group(1))
                if "behind" in status_line:
                    match = re.search(r"behind (\d+)", status_line)
                    if match:
                        status_info["commits_behind"] = int(match.group(1))
                status_info["has_remote"] = (
                    "origin" in status_line or "[" in status_line
                )

            # Get last commit info
            log_result = subprocess.run(
                ["git", "log", "-1", "--format=%h %s (%an, %ar)"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if log_result.returncode == 0:
                status_info["last_commit"] = log_result.stdout.strip()

            # Safety warnings
            warnings = []
            if status_info.get("is_protected"):
                warnings.append(
                    f"On protected branch '{status_info['branch']}' - be careful with commits"
                )
            if status_info.get("commits_behind", 0) > 0:
                warnings.append(
                    f"Branch is {status_info['commits_behind']} commits behind remote - consider pulling"
                )
            if (
                status_info.get("has_changes")
                and status_info.get("commits_ahead", 0) > 0
            ):
                warnings.append(
                    "You have both uncommitted changes and unpushed commits"
                )

            status_info["warnings"] = warnings
            status_info["success"] = True

            return json.dumps(status_info, indent=2)

        except subprocess.TimeoutExpired:
            return json.dumps(
                {"error": True, "message": "Git status command timed out"}, indent=2
            )
        except Exception as e:
            return json.dumps(
                {"error": True, "message": f"Error getting git status: {str(e)}"},
                indent=2,
            )

    @mcp.tool()
    async def git_commit_safe(
        message: str,
        add_all: bool = False,
        add_files: str = "",
        amend: bool = False,
        allow_empty: bool = False,
    ) -> str:
        """Safely create a git commit with validations.

        This tool provides a safer way to create commits with built-in
        checks and validations.

        Args:
            message: Commit message (required)
            add_all: If True, stages all changes (git add -A) before commit
            add_files: Space-separated list of specific files to add before commit
            amend: If True, amends the last commit (with safety checks)
            allow_empty: If True, allows creating empty commits

        Returns:
            JSON string with commit result or safety warnings

        Examples:
            git_commit_safe(message="Add new feature")
            git_commit_safe(message="Fix bug", add_all=True)
            git_commit_safe(message="Update docs", add_files="README.md docs/")
        """
        if not message or not message.strip():
            return json.dumps(
                {"error": True, "message": "Commit message is required"}, indent=2
            )

        try:
            results = []

            # Handle file staging
            if add_all:
                add_result = subprocess.run(
                    ["git", "add", "-A"], capture_output=True, text=True, timeout=30
                )
                results.append(
                    {
                        "step": "add_all",
                        "success": add_result.returncode == 0,
                        "output": (
                            add_result.stderr.strip()
                            if add_result.stderr
                            else "All changes staged"
                        ),
                    }
                )
            elif add_files:
                files = add_files.strip().split()
                add_result = subprocess.run(
                    ["git", "add"] + files, capture_output=True, text=True, timeout=30
                )
                results.append(
                    {
                        "step": "add_files",
                        "success": add_result.returncode == 0,
                        "files": files,
                        "output": (
                            add_result.stderr.strip()
                            if add_result.stderr
                            else "Files staged"
                        ),
                    }
                )

            # Check for amend safety
            if amend:
                amend_check = check_amend_safety()
                if not amend_check["safe"]:
                    return json.dumps(
                        {
                            "error": True,
                            "requires_confirmation": True,
                            "message": "Amending this commit may cause issues",
                            "issues": amend_check["issues"],
                            "branch": amend_check["branch"],
                            "is_pushed": amend_check["is_pushed"],
                            "suggestion": "If the commit has been pushed, you'll need to force push after amending. Consider creating a new commit instead.",
                        },
                        indent=2,
                    )

            # Build commit command
            commit_cmd = ["git", "commit", "-m", message]
            if amend:
                commit_cmd.append("--amend")
            if allow_empty:
                commit_cmd.append("--allow-empty")

            # Execute commit
            commit_result = subprocess.run(
                commit_cmd, capture_output=True, text=True, timeout=30
            )

            if commit_result.returncode == 0:
                # Get the new commit hash
                hash_result = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                commit_hash = (
                    hash_result.stdout.strip()
                    if hash_result.returncode == 0
                    else "unknown"
                )

                return json.dumps(
                    {
                        "success": True,
                        "message": "Commit created successfully",
                        "commit_hash": commit_hash,
                        "commit_message": message,
                        "amended": amend,
                        "steps": results,
                        "output": commit_result.stdout.strip(),
                    },
                    indent=2,
                )
            else:
                error_msg = commit_result.stderr.strip() or commit_result.stdout.strip()

                # Provide helpful suggestions based on common errors
                suggestions = []
                if "nothing to commit" in error_msg.lower():
                    suggestions.append(
                        "No changes are staged. Use add_all=True or add_files='...' to stage changes first."
                    )
                if "please tell me who you are" in error_msg.lower():
                    suggestions.append(
                        "Git user not configured. Run: git config --global user.name 'Your Name' && git config --global user.email 'your@email.com'"
                    )

                return json.dumps(
                    {
                        "error": True,
                        "message": "Commit failed",
                        "error_output": error_msg,
                        "suggestions": suggestions,
                        "steps": results,
                    },
                    indent=2,
                )

        except subprocess.TimeoutExpired:
            return json.dumps(
                {"error": True, "message": "Commit command timed out"}, indent=2
            )
        except Exception as e:
            return json.dumps(
                {"error": True, "message": f"Error creating commit: {str(e)}"}, indent=2
            )
