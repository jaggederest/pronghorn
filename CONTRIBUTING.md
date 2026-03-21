# Contributing to Pronghorn

This project is developed entirely by AI agents. To contribute, you should too.

## Setup

```bash
git clone git@github.com:jaggederest/pronghorn.git
cd pronghorn
cargo test --workspace  # make sure it builds
```

## Making Changes

```bash
claude
```

Then tell it what you want. It has a [CLAUDE.md](CLAUDE.md) that gives it full context.

For example:
- "Add Opus codec support to pronghorn-audio"
- "Implement the faster-whisper WebSocket STT backend"
- "The jitter buffer doesn't handle packet loss gracefully when WiFi drops — fix it"

## Guidelines

1. **Let the agent read CLAUDE.md first.** It contains the build commands, architecture overview, and code style. The agent will pick it up automatically.

2. **All code must pass the gate:**
   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   ```

3. **Write tests.** If the agent doesn't write tests, ask it to. Every crate has established test patterns to follow.

4. **Keep the satellite thin.** `pronghorn-satellite` must not depend on `pronghorn-pipeline`. The Pi Zero 2 has 512MB of RAM and our dignity to protect.

5. **Latency is the metric.** If a design choice adds latency, it needs a very good reason.

## Architecture Decisions

If you're making a significant architectural change, use plan mode:

```
claude --plan
```

Describe what you want to build. The agent will explore the codebase, ask questions, and produce a plan for your approval before writing code.

## Commit Messages

The agent writes these. They include `Co-Authored-By: Claude` in the trailer. This is fine. This is the way.
