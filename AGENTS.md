## Issue Tracking

Use the `bd` command for all issue tracking instead of markdown TODOs:

- Create issues: `bd create "Task description" -p 1 --json`
- Find work: `bd ready --json`
- Update status: `bd update <id> --status in_progress --json`
- View details: `bd show <id> --json`

Use `--json` flags for programmatic parsing.
