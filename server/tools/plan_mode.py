"""Plan mode tools for implementation planning workflow.

Plan mode allows the AI to:
1. Enter a planning phase before implementation
2. Explore the codebase and understand requirements
3. Design an implementation approach with clear steps
4. Present the plan to the user for approval
5. Exit plan mode to execute the approved plan
"""

import json
import os
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from utils.paths import plans_dir

# Plan storage location (under the app home, created on demand)
PLAN_DIR = str(plans_dir())
CURRENT_PLAN_FILE = os.path.join(PLAN_DIR, "current_plan.json")


def ensure_plan_dir():
    """Ensure the plan directory exists."""
    os.makedirs(PLAN_DIR, exist_ok=True)


def load_current_plan() -> dict:
    """Load the current plan from file."""
    ensure_plan_dir()
    if os.path.exists(CURRENT_PLAN_FILE):
        try:
            with open(CURRENT_PLAN_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_current_plan(plan: dict):
    """Save the current plan to file."""
    ensure_plan_dir()
    with open(CURRENT_PLAN_FILE, "w") as f:
        json.dump(plan, f, indent=2)


def clear_current_plan():
    """Clear the current plan."""
    if os.path.exists(CURRENT_PLAN_FILE):
        os.remove(CURRENT_PLAN_FILE)


def register_plan_mode_tools(mcp: FastMCP) -> None:
    """Register plan mode tools."""

    @mcp.tool()
    async def enter_plan_mode(task_description: str, context: str = "") -> str:
        """Enter plan mode to design an implementation approach.

        Use this for complex implementations — multiple files, architectural
        decisions, or unclear requirements. Skip it for simple fixes, single-file
        changes, and clear requests.

        Full workflow: enter_plan_mode() -> explore with search tools ->
        add_plan_step() / add_plan_file() / add_plan_risk() -> present_plan() for
        approval -> approve_plan() -> exit_plan_mode() -> implement, tracking
        progress with todo_write.

        Use this BEFORE starting complex implementations to:
        - Analyze the task requirements
        - Explore relevant code
        - Design a step-by-step plan
        - Get user approval before coding

        Args:
            task_description: Brief description of what needs to be implemented
            context: Optional additional context (related files, constraints, etc.)

        Returns:
            JSON string confirming plan mode entry
        """
        existing_plan = load_current_plan()
        if existing_plan and existing_plan.get("status") == "planning":
            return json.dumps(
                {
                    "error": True,
                    "message": "Already in plan mode",
                    "current_task": existing_plan.get("task_description"),
                    "suggestion": "Use exit_plan_mode(cancel=True) to cancel current plan",
                },
                indent=2,
            )

        plan = {
            "status": "planning",
            "task_description": task_description,
            "context": context,
            "steps": [],
            "files_to_modify": [],
            "files_to_create": [],
            "risks": [],
            "created_at": datetime.now().isoformat(),
            "approved": False,
        }

        save_current_plan(plan)

        return json.dumps(
            {
                "success": True,
                "message": "Entered plan mode",
                "task": task_description,
                "next_steps": [
                    "Explore codebase using glob_search, grep_search, fs_read",
                    "Use add_plan_step() to add implementation steps",
                    "Use add_plan_file() to track files to modify/create",
                    "When ready, use present_plan() for user approval",
                ],
            },
            indent=2,
        )

    @mcp.tool()
    async def add_plan_step(
        step_number: int, title: str, description: str, files_involved: str = ""
    ) -> str:
        """Add a step to the current implementation plan.

        Args:
            step_number: Order of this step (1, 2, 3, etc.)
            title: Short title for the step
            description: Detailed description of what this step involves
            files_involved: Comma-separated list of files this step touches

        Returns:
            JSON string confirming step was added
        """
        plan = load_current_plan()
        if not plan or plan.get("status") != "planning":
            return json.dumps(
                {
                    "error": True,
                    "message": "Not in plan mode. Use enter_plan_mode() first.",
                },
                indent=2,
            )

        files = (
            [f.strip() for f in files_involved.split(",") if f.strip()]
            if files_involved
            else []
        )

        step = {
            "number": step_number,
            "title": title,
            "description": description,
            "files": files,
            "status": "pending",
        }

        existing_numbers = [s["number"] for s in plan["steps"]]
        if step_number in existing_numbers:
            for i, s in enumerate(plan["steps"]):
                if s["number"] == step_number:
                    plan["steps"][i] = step
                    break
        else:
            plan["steps"].append(step)
            plan["steps"].sort(key=lambda x: x["number"])

        save_current_plan(plan)

        return json.dumps(
            {
                "success": True,
                "message": f"Added step {step_number}: {title}",
                "total_steps": len(plan["steps"]),
            },
            indent=2,
        )

    @mcp.tool()
    async def add_plan_file(file_path: str, action: str, description: str = "") -> str:
        """Track a file that will be modified or created.

        Args:
            file_path: Path to the file
            action: "modify" or "create"
            description: What changes will be made

        Returns:
            JSON string confirming file was tracked
        """
        plan = load_current_plan()
        if not plan or plan.get("status") != "planning":
            return json.dumps(
                {
                    "error": True,
                    "message": "Not in plan mode. Use enter_plan_mode() first.",
                },
                indent=2,
            )

        file_info = {"path": file_path, "description": description}

        if action.lower() == "create":
            if file_info not in plan["files_to_create"]:
                plan["files_to_create"].append(file_info)
        else:
            if file_info not in plan["files_to_modify"]:
                plan["files_to_modify"].append(file_info)

        save_current_plan(plan)

        return json.dumps(
            {"success": True, "message": f"Tracked file: {file_path} ({action})"},
            indent=2,
        )

    @mcp.tool()
    async def add_plan_risk(risk: str, mitigation: str = "") -> str:
        """Add a potential risk to the plan.

        Args:
            risk: Description of the potential issue
            mitigation: How this risk will be addressed

        Returns:
            JSON string confirming risk was added
        """
        plan = load_current_plan()
        if not plan or plan.get("status") != "planning":
            return json.dumps(
                {
                    "error": True,
                    "message": "Not in plan mode. Use enter_plan_mode() first.",
                },
                indent=2,
            )

        plan["risks"].append({"risk": risk, "mitigation": mitigation})
        save_current_plan(plan)

        return json.dumps(
            {
                "success": True,
                "message": "Risk added to plan",
                "total_risks": len(plan["risks"]),
            },
            indent=2,
        )

    @mcp.tool()
    async def present_plan() -> str:
        """Present the current plan for user review.

        Returns:
            JSON string with the complete plan formatted for display
        """
        plan = load_current_plan()
        if not plan:
            return json.dumps(
                {
                    "error": True,
                    "message": "No plan exists. Use enter_plan_mode() first.",
                },
                indent=2,
            )

        return json.dumps(
            {
                "plan_ready": True,
                "task": plan["task_description"],
                "context": plan.get("context", ""),
                "steps": plan["steps"],
                "files_to_modify": plan["files_to_modify"],
                "files_to_create": plan["files_to_create"],
                "risks": plan["risks"],
                "total_steps": len(plan["steps"]),
                "message": "Please review this plan. Reply 'approve' to proceed or provide feedback.",
            },
            indent=2,
        )

    @mcp.tool()
    async def approve_plan() -> str:
        """Mark the current plan as approved."""
        plan = load_current_plan()
        if not plan:
            return json.dumps({"error": True, "message": "No plan exists."}, indent=2)

        plan["approved"] = True
        plan["approved_at"] = datetime.now().isoformat()
        save_current_plan(plan)

        return json.dumps(
            {
                "success": True,
                "message": "Plan approved! Ready to implement.",
                "next_step": "Use exit_plan_mode() to start implementing",
            },
            indent=2,
        )

    @mcp.tool()
    async def exit_plan_mode(cancel: bool = False) -> str:
        """Exit plan mode.

        Args:
            cancel: If True, cancels plan. If False, exits to implement (requires approval).

        Returns:
            JSON string confirming exit
        """
        plan = load_current_plan()
        if not plan:
            return json.dumps(
                {
                    "success": True,
                    "message": "No active plan. Already out of plan mode.",
                },
                indent=2,
            )

        if cancel:
            clear_current_plan()
            return json.dumps(
                {
                    "success": True,
                    "message": "Plan cancelled.",
                    "task_cancelled": plan["task_description"],
                },
                indent=2,
            )

        if not plan.get("approved"):
            return json.dumps(
                {
                    "error": True,
                    "message": "Plan not approved. Use present_plan() then approve_plan() first.",
                    "suggestion": "To cancel, use exit_plan_mode(cancel=True)",
                },
                indent=2,
            )

        plan["status"] = "implementing"
        plan["implementation_started_at"] = datetime.now().isoformat()
        save_current_plan(plan)

        return json.dumps(
            {
                "success": True,
                "message": "Exited plan mode. Ready to implement!",
                "task": plan["task_description"],
                "steps_to_complete": len(plan["steps"]),
                "reminder": "Use todo_write() to track implementation progress",
            },
            indent=2,
        )

    @mcp.tool()
    async def get_plan_status() -> str:
        """Get the current plan status."""
        plan = load_current_plan()
        if not plan:
            return json.dumps(
                {"in_plan_mode": False, "message": "No active plan."}, indent=2
            )

        return json.dumps(
            {
                "in_plan_mode": plan.get("status") == "planning",
                "status": plan.get("status"),
                "task": plan.get("task_description"),
                "approved": plan.get("approved", False),
                "steps_count": len(plan.get("steps", [])),
                "created_at": plan.get("created_at"),
            },
            indent=2,
        )
