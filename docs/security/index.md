---
hide:
  - toc
---

# Security

Decide what MnemoAI may do and understand where your data goes.
Permissions are enforced by the application alongside the model's instructions.

<div class="grid cards" markdown>

- **Permissions and safety**

    Understand approval prompts, auto tiers, plan mode, git protections, and file-write checks.

    [Control actions →](../guides/safety.md)

- **Data and privacy**

    Find local conversation and memory files, and understand provider and tool connections.

    [Understand data handling →](data.md)

</div>

## Common decisions

- [Inspect a project before allowing changes](../guides/safety.md#investigate-before-changing-anything)
- [Choose an auto-approval tier for this session](../guides/safety.md#stop-being-asked-for-a-while)
- [Understand the safety floor](../guides/safety.md#understand-the-safety-floor)
- [Diagnose a refused file write](../guides/safety.md#why-a-write-was-refused)
- [Review dependency security](../guides/safety.md#dependency-security)

External [MCP servers](../guides/mcp.md) and [tool hooks](../guides/hooks.md)
are additional programs you configure. Review what they run and what they receive.
