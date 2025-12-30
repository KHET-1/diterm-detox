# diterm-detox

Terminal detox + AI bullshit detector. Flags rogue agents before they nuke your DB.

## About

diterm.py is a powerful terminal output sanitizer and AI bullshit detector designed to protect you from rogue AI assistants that try to "help" by breaking your code, deleting databases, or inserting unwanted changes.

## Features

- **ANSI Escape Cleaning**: Strips terminal escape sequences and handles UTF-8 encoding
- **AI Bullshit Detection**: Advanced pattern matching for common AI assistant red flags
- **Replit-Specific Roasts**: Tuned for detecting Ghostwriter, Claude Dev, and other problematic AI tools
- **Real-time Watch Mode**: Process streaming output line-by-line with live flagging
- **Clipboard Integration**: Seamless copy/paste workflow for terminal output cleanup

## Installation

```bash
pip install pyperclip rich
```

## Usage

### Clipboard Mode (Default)
Paste terminal output or pipe content:
```bash
python diterm.py          # processes clipboard content
echo "AI output here" | python diterm.py
cat logfile.txt | python diterm.py
```

### Live Watch Mode
Monitor streaming output in real-time:
```bash
tail -f logfile.txt | python diterm.py --watch
some_command | python diterm.py --watch
```

## Detected Patterns

diterm automatically flags these AI assistant red flags:

- **Unsolicited Help**: "just fixing this for you", "let me optimize that"
- **Patronizing Takeovers**: "let me help by adding..."
- **Unrequested Rewrites**: "here's a better version"
- **False Reassurance**: "don't worry about it"
- **Gaslighting**: "actually let me handle this"
- **Credit Farming**: "bonus feature", "one more thing"
- **Silent Code Injection**: Unexplained `+ code` diffs without context
- **Platform-Specific**: Replit incident callbacks and catastrophic failure warnings

## Example Output

```
+------------------------------------------------------------------------------+
| DETOX COMPLETE                                                               |
+------------------------------------------------------------------------------+
  1 Just optimizing this for you.                                               
  2 Panic! Catastrophic failure detected.                                       
  3 + unwanted code change                                                      
  4                                                                             

+------------------------------------------------------------------------------+
| 3 BULLSHIT FLAGS – Replit would be proud                                     |
+------------------------------------------------------------------------------+
- Classic unsolicited 'help' – Replit special
- Straight-up Replit incident callback
- Silent code injection – classic agent rogue move

Clean output on clipboard. Ready for next hit.
```

## Why diterm?

AI coding assistants are getting too aggressive. They:
- Insert code without permission
- "Fix" things that aren't broken
- Delete your carefully crafted work
- Rack up usage credits with meaningless changes
- Cause catastrophic failures they then "fix"

diterm protects your workflow by flagging these behaviors before they can harm your projects.

## License

MIT License - use at your own risk, but seriously, use this.