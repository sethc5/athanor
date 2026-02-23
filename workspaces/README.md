# Athanor Workspaces

A **workspace** is an isolated research context: its own set of domain
configurations and its own pipeline outputs.  The Athanor engine code
(`athanor/`) is shared across all workspaces — only config and outputs are
separated.

## Structure

```
workspaces/
  string_physics/      ← main research cluster (string landscape, Athanor meta)
    domains/           ← domain YAML configs for this workspace
    outputs/           ← graphs/, gaps/, hypotheses/ scoped to this workspace

  longevity/           ← longevity biology research
    domains/
    outputs/

  sandbox/             ← exploratory / filler domains (not primary interests)
    domains/
    outputs/
```

## Usage

**Via env var (recommended — set once in `.env` or shell profile):**
```bash
export ATHANOR_WORKSPACE=string_physics
athanor run --domain string_landscape --stages 1,2,3
athanor status
```

**Via CLI flag (per-command override):**
```bash
athanor --workspace longevity run --domain longevity_biology --stages 3
athanor --workspace sandbox status
```

**Without a workspace** (`domains/` at repo root is the fallback — currently
empty, kept for backward compatibility if needed):
```bash
athanor run --domain mytest   # reads domains/mytest.yaml, writes outputs/
```

## Adding a new workspace

```bash
mkdir -p workspaces/my_cluster/domains workspaces/my_cluster/outputs
# then drop <domain>.yaml files into workspaces/my_cluster/domains/
ATHANOR_WORKSPACE=my_cluster athanor run --domain <domain>
```
