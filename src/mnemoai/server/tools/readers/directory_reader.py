"""Directory reading functionality."""

import json
import os
from typing import Dict, List


async def read_directory(path: str, depth: int) -> str:
    """Read directory contents.

    Args:
        path: Directory path
        depth: Recursion depth for subdirectories

    Returns:
        JSON string with directory contents
    """
    if not os.path.exists(path):
        return json.dumps({"error": True, "message": f"Path doesn't exist: {path}"})

    if not os.path.isdir(path):
        return json.dumps(
            {"error": True, "message": f"Path is not a directory: {path}"}
        )

    def scan_directory(dir_path: str, current_depth: int, max_depth: int) -> List[Dict]:
        """Recursively scan directory contents.

        Args:
            dir_path: Directory path to scan
            current_depth: Current recursion depth
            max_depth: Maximum recursion depth

        Returns:
            List of directory items
        """
        items = []
        try:
            for item in sorted(os.listdir(dir_path)):
                if item.startswith("."):
                    continue

                item_path = os.path.join(dir_path, item)
                relative_path = os.path.relpath(item_path, path)

                if os.path.isdir(item_path):
                    dir_info = {
                        "name": item,
                        "type": "directory",
                        "path": relative_path,
                    }

                    if current_depth < max_depth:
                        dir_info["contents"] = scan_directory(
                            item_path, current_depth + 1, max_depth
                        )

                    items.append(dir_info)
                else:
                    try:
                        size = os.path.getsize(item_path)
                        items.append(
                            {
                                "name": item,
                                "type": "file",
                                "path": relative_path,
                                "size": size,
                            }
                        )
                    except OSError:
                        items.append(
                            {
                                "name": item,
                                "type": "file",
                                "path": relative_path,
                                "size": "unknown",
                            }
                        )
        except PermissionError:
            pass

        return items

    contents = scan_directory(path, 0, depth)

    return json.dumps(
        {
            "path": path,
            "type": "directory",
            "depth": depth,
            "contents": contents,
            "total_items": len(contents),
        }
    )
