# diterm-detox

Terminal detox + AI bullshit detector. Flags rogue agents before they nuke your DB.

## Overview

`diterm-detox` is a command-line tool that analyzes terminal commands for potentially dangerous operations. It helps prevent destructive commands from being executed, particularly useful when working with AI agents that have terminal access.

## Features

- 🛡️ Detects dangerous command patterns (rm -rf /, DROP DATABASE, etc.)
- 📊 Calculates risk scores for commands
- 🚨 Provides clear warnings before dangerous operations
- 🔍 Extensible pattern matching system

## Installation

```bash
pip install -e .
```

## Usage

Check a command for dangers:

```bash
diterm-detox "rm -rf /"
```

Safe command example:

```bash
diterm-detox "ls -la"
```

## Development

Install development dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

## Detected Patterns

The tool currently detects:
- Recursive deletions of root directory
- Dangerous dd commands
- Database DROP/DELETE/TRUNCATE operations
- Fork bombs
- Direct writes to disk devices
- Recursive chmod 777 on root
- And more...

## License

MIT
