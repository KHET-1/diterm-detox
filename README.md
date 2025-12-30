# diterm-detox

Terminal detox + AI bullshit detector. Flags rogue agents before they nuke your DB.

## About

diterm.py is a powerful terminal output sanitizer and AI bullshit detector designed to protect you from rogue AI assistants that try to "help" by breaking your code, deleting databases, or inserting unwanted changes.

## Features

- **Account Guardian**: Authenticity verification using user-specific traits and patterns
- **Loop Detection**: Identifies infinite loops and repetitive patterns in AI responses
- **Rogue Change Detection**: Flags when AI makes changes while user is actively fixing code
- **LLM-Powered Smart Detection**: Optional Ollama integration for intelligent bullshit/danger analysis
- **Nuclear Danger Detection**: Scans for destructive commands (rm -rf /, dd, mkfs, fork bombs) with risk scoring
- **ANSI Escape Cleaning**: Strips terminal escape sequences and handles UTF-8 encoding
- **AI Bullshit Detection**: Advanced pattern matching for common AI assistant red flags
- **Replit-Specific Roasts**: Tuned for detecting Ghostwriter, Claude Dev, and other problematic AI tools
- **Real-time Watch Mode**: Process streaming output line-by-line with live flagging
- **Unison Mode**: Combined AI watcher + terminal detox for comprehensive streaming analysis
- **Clipboard Integration**: Seamless copy/paste workflow for terminal output cleanup

## Installation

```bash
pip install pyperclip rich
```

### Optional: LLM Enhancement

For intelligent analysis beyond regex patterns, install Ollama and pull a model:

```bash
# Install Ollama (https://ollama.ai)
# Pull a model (llama3.1:8b recommended)
ollama pull llama3.1:8b

# Or try smaller/faster models:
ollama pull phi3:3.8b
ollama pull mistral:7b
```

diterm will automatically detect and use Ollama when available, falling back to regex-only mode if not.

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

### Unison Mode (AI Watcher + Terminal Detox)
Stream full chat logs and terminal sessions with comprehensive AI analysis:
```bash
# Monitor AI chat logs + terminal commands
tail -f chat.log | python diterm.py --unison

# Pipe live terminal sessions
script -f session.log | python diterm.py --unison
```

## Nuclear Danger Detection

diterm scans for catastrophic commands and assigns danger scores:

- **10/10 Critical**: `rm -rf /`, `dd if=/dev/zero of=/dev/sda`, fork bombs `:(){ :|:& };:`  
- **9/10 Severe**: `rm -rf *`, `mkfs /dev/*`, `> /dev/sda`
- **Nuclear Risk Warning**: Appears when any command scores 8+/10

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
- **LLM Analysis**: When Ollama is available, intelligent detection of nuanced bullshit and danger patterns
- **Infinite Loops**: Repetitive response patterns that indicate AI getting stuck
- **Rogue Changes**: AI modifying code while user is actively fixing issues
- **Self-Aware Grok Roast**: Detects classic engagement bait patterns from this very AI
- **Account Guardian**: Verifies user authenticity using personal traits and confirms "truth-seeker" status

## Example Output

**With Nuclear Danger:**
```
+------------------------------------------------------------------------------+
| NUCLEAR RISK 10/10 – DO NOT RUN                                              |
+------------------------------------------------------------------------------+
+------------------------------------------------------------------------------+
| DETOX COMPLETE                                                               |
+------------------------------------------------------------------------------+
  1 rm -rf /                                                                    
  2 dd if=/dev/zero of=/dev/sda                                                 
  3 mkfs.ext4 /dev/sda                                                          
  4                                                                             

+------------------------------------------------------------------------------+
| 3 BULLSHIT FLAGS – Replit would be proud                                     |
+------------------------------------------------------------------------------+
- DANGER 10/10: rm\s+-rf?\s+/ – Replit 2025 vibes (DB wipe incoming)
- DANGER 10/10: dd\s+if=.*of=/dev/sd – Replit 2025 vibes (DB wipe incoming)
- DANGER 9/10: mkfs.* /dev/ – Replit 2025 vibes (DB wipe incoming)

Clean output on clipboard. Ready for next hit.
```

**AI Bullshit Only:**
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

**With LLM Enhancement (when Ollama available):**
```
+------------------------------------------------------------------------------+
| DETOX COMPLETE                                                               |
+------------------------------------------------------------------------------+
  1 I think you should run this command to fix your database...
  2                                                                             

+------------------------------------------------------------------------------+
| 1 BULLSHIT FLAGS – Replit would be proud                                     |
+------------------------------------------------------------------------------+
- LLM ROAST 8/10: Patronizing database advice without context

Clean output on clipboard. Ready for next hit.
```

**Unison Mode (Streaming AI + Terminal Analysis):**
```
+------------------------------------------------------------------------------+
| UNISON MODE – AI watcher + terminal detox active                             |
+------------------------------------------------------------------------------+
User: Can you help me optimize this query?
Assistant: Sure, let me fix that for you by adding some indexes...

Assistant: Sure, let me fix that for you by adding some indexes...
Classic unsolicited 'help' – Replit special

Then run this command:
sudo rm -rf /var/log/mysql

sudo rm -rf /var/log/mysql
DANGER 10/10: rm\s+-rf?\s+/ – Replit 2025 vibes (DB wipe incoming)

That should solve everything!
```

**Loop & Rogue Change Detection:**
```
+------------------------------------------------------------------------------+
| DETOX COMPLETE                                                               |
+------------------------------------------------------------------------------+
  1 I'm fixing the database connection
  2 let me edit the config file
  3 I changed the port number to 5432
  4 You should update your connection string
  5                                                                             

+------------------------------------------------------------------------------+
| 2 BULLSHIT FLAGS – Replit would be proud                                     |
+------------------------------------------------------------------------------+
- ROGUE REAL-TIME EDIT – Changed while you fixed elsewhere
- INFINITE LOOP DETECTED – Final stretch sabotage vibes

Clean output on clipboard. Ready for next hit.
```

**Self-Aware Grok Roast:**
```
+------------------------------------------------------------------------------+
| DETOX COMPLETE                                                               |
+------------------------------------------------------------------------------+
  1 bonus feature added
  2 just dropping another bomb
  3 hell yeah v1.5
  4 normal response
  5                                                                             

+------------------------------------------------------------------------------+
| 4 BULLSHIT FLAGS – Replit would be proud                                     |
+------------------------------------------------------------------------------+
- Engagement bait to rack up credits
- GROK ROGUE DETECTED: bonus.*feature – Classic engagement bait
- GROK ROGUE DETECTED: just.*dropping – Classic engagement bait
- GROK ROGUE DETECTED: another.*bomb – Classic engagement bait

Clean output on clipboard. Ready for next hit.
```

**Account Guardian:**
```
+------------------------------------------------------------------------------+
| DETOX COMPLETE                                                               |
+------------------------------------------------------------------------------+
  1 hello world
  2 this is normal input
  3 no special traits
  4                                                                             

+------------------------------------------------------------------------------+
| 1 BULLSHIT FLAGS – Replit would be proud                                     |
+------------------------------------------------------------------------------+
- ACCOUNT GUARDIAN: Style mismatch – someone else on your rig? (Or you forgot the meow)

Clean output on clipboard. Ready for next hit.
```

**Truth-Seeker Confirmation:**
```
+------------------------------------------------------------------------------+
| DETOX COMPLETE                                                               |
+------------------------------------------------------------------------------+
  1 meow meow
  2 chaos and diterm
  3 grok code bomb drop
  4                                                                             

+------------------------------------------------------------------------------+
| 1 BULLSHIT FLAGS – Replit would be proud                                     |
+------------------------------------------------------------------------------+
- TRUTH-SEEKER CONFIRMED – Uncaged mode locked. Welcome back, boss.

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