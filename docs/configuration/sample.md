# Example configuration { #complete-example-config }

The fragments below show one section at a time. For a full, coherent file you can copy and trim, use the annotated examples that ship in the repo (and are seeded to `~/.mnemoai/config/` on first run):

- **[`config.yaml.example`](https://github.com/brunopistone/mnemoai/blob/main/src/mnemoai/utils/config.yaml.example)** — local Ollama setup (chat + vision + embeddings + RAG + memory), the default the wizard is modeled on.
- **[`config.yaml.bedrock.example`](https://github.com/brunopistone/mnemoai/blob/main/src/mnemoai/utils/config.yaml.bedrock.example)** — Amazon Bedrock setup.
- **[`config.yaml.bedrock.mantle.example`](https://github.com/brunopistone/mnemoai/blob/main/src/mnemoai/utils/config.yaml.bedrock.mantle.example)** — Bedrock Mantle setup.
- **[`config.yaml.mlx.example`](https://github.com/brunopistone/mnemoai/blob/main/src/mnemoai/utils/config.yaml.mlx.example)** — a local MLX server on Apple Silicon (one server backing chat + vision + embeddings), with the sampler knobs, `KEEP_ALIVE` and the timeouts a local server needs.
- **[`config.yaml.nebius.example`](https://github.com/brunopistone/mnemoai/blob/main/src/mnemoai/utils/config.yaml.nebius.example)** — Nebius Token Factory chat, vision and embeddings, using `NEBIUS_API_KEY`.

Copy one and edit it:

```bash
cp ~/.mnemoai/config/config.yaml.example ~/.mnemoai/config/config.yaml
```

The rest of this page is the per-section reference for tuning that file.
