# Environment variables { #environment-variables }

Mnemo AI reads a handful of environment variables. Provider API keys can be set either in your shell or, more conveniently, under the config `ENV:` block — every key there is exported as an environment variable at startup.

| Variable                     | Purpose                                                    | Notes                                                                                                                                                           |
| ---------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MNEMOAI_CONFIG`             | Explicit path to `config.yaml`                             | Highest-priority config location; overrides the normal resolution order                                                                                         |
| `MNEMOAI_HOME`               | Override the app home (default `~/.mnemoai`)               | Moves config, prompts, plans, tasks, and per-profile/per-model state together                                                                                   |
| `MNEMOAI_PROMPTS`            | Explicit path to `prompts.yaml`                            | Overrides the normal prompts resolution order                                                                                                                   |
| `LOG_LEVEL`                  | Log verbosity: `DEBUG` / `INFO` / `WARNING`                | Default `WARNING`; one line per record on stderr, full records (tracebacks included) in `~/.mnemoai/logs/mnemoai.log`. `DEBUG` also prints tracebacks on screen |
| `OPENAI_API_KEY`             | OpenAI auth (`TYPE: openai`)                               | Or set `MODEL_ID.API_KEY`                                                                                                                                       |
| `ANTHROPIC_API_KEY`          | Anthropic auth (`TYPE: anthropic`)                         | Or set `MODEL_ID.API_KEY`                                                                                                                                       |
| `AWS_PROFILE` / `AWS_REGION` | AWS profile / region (Bedrock, Mantle, SageMaker)          | Standard boto3 chain; often set via the `ENV:` block                                                                                                            |
| `AWS_BEARER_TOKEN_BEDROCK`   | Bedrock API key for **standard** Bedrock (`TYPE: bedrock`) | Read automatically by `langchain-aws`                                                                                                                           |
| `BEDROCK_API_KEY`            | Bedrock API key for **Mantle** (`TYPE: mantle`)            | Or `MODEL_ID.API_KEY`; else a token is minted from AWS creds                                                                                                    |
| `BRAVE_API_KEY`              | Brave Search API key for web search                        | Can also be the top-level `BRAVE_API_KEY` config key                                                                                                            |
| `TAVILY_API_KEY`             | Tavily search and Extract/Crawl/Map key                    | Select `WEB_SEARCH_PROVIDER: tavily`; can also be the top-level `TAVILY_API_KEY` config key                                                                     |

```yaml
# Set provider keys/vars without touching your shell:
ENV:
  AWS_PROFILE: my-bedrock-profile
  BEDROCK_API_KEY: bedrock-api-key-XXXXXXXX
```
