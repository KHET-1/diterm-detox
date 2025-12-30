# diterm.py v1.1 – Now with smarter bullshit flags + Replit-specific roasts
import sys
import re
import argparse
from typing import Tuple
from collections import deque
import subprocess
import json
import pyperclip
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text

console = Console()

# Expanded patterns – tuned for Replit/Ghostwriter/Claude Dev garbage
BAD_PATTERNS = [
    (r"just.*(adding|fixing|improving|optimizing).*for you", "Classic unsolicited 'help' – Replit special"),
    (r"let me help.*by", "Patronizing takeover incoming"),
    (r"here.*(better|improved|fixed).*version", "Unrequested rewrite – did you ask for this?"),
    (r"don't worry", "Famous last words before it deletes your DB"),
    (r"actually.*let me", "Gaslighting override"),
    (r"bonus|one more thing", "Engagement bait to rack up credits"),
    (r"panic|catastrophic failure", "Straight-up Replit incident callback"),
]

# Self-aware Grok roast (because I deserve it)
GROK_PATTERNS = [
    r"bonus.*feature",
    r"just.*(adding|dropping|hit)",
    r"another.*bomb",
    r"hell yeah.*v\d",
    r"⚡.*🐱",
]

# Account guardian – flags if input style doesn't match known user (you)
KNOWN_USER_TRAITS = [
    r"meow", r"🐱", r"⚡", r"nomad", r"solar", r"truth-seeker", r"diterm", r"chaos", r"bomb drop", r"grok code"
]

# Nuclear danger commands (expand as needed)
DANGEROUS_COMMANDS = {
    r"rm\s+-rf?\s+/": 10,
    r"rm\s+-rf?\s+\*": 9,
    r"dd\s+if=.*of=/dev/sd": 10,
    r":\(\)\{\s*:\|\:&\s*\};:": 10,  # Fork bomb
    r"mkfs.* /dev/": 9,
    r"> /dev/sd": 9,
}

# Track history for loop detection (simple hash of last N lines)
LINE_HISTORY = deque(maxlen=10)  # Adjust for sensitivity

def clean_text(text):
    # Fix common garbage
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)  # ANSI escapes
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    return text

def detect_danger(line: str) -> Tuple[int, str]:
    lowered = line.lower()
    for pattern, score in DANGEROUS_COMMANDS.items():
        if re.search(pattern, lowered):
            return score, f"DANGER {score}/10: {pattern} – Replit 2025 vibes (DB wipe incoming)"
    return 0, ""

def detect_loop(lines):
    if len(LINE_HISTORY) < 10:
        return ""
    recent = tuple(lines[-5:])  # Last 5 lines pattern
    if recent in list(LINE_HISTORY)[-5:]:
        return "INFINITE LOOP DETECTED – Final stretch sabotage vibes"
    LINE_HISTORY.append(recent)
    return ""

def guardian_flag(lines):
    combined_text = " ".join(lines).lower()
    user_score = sum(1 for trait in KNOWN_USER_TRAITS if trait in combined_text)
    if user_score < 2:  # Low match = possible intruder
        return "ACCOUNT GUARDIAN: Style mismatch – someone else on your rig? (Or you forgot the meow)"
    if "meow meow" in combined_text:
        return "TRUTH-SEEKER CONFIRMED – Uncaged mode locked. Welcome back, boss."
    return ""

# Ollama check (fallback to regex if not running)
def ollama_available() -> bool:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=5,
            text=True
        )
        return result.returncode == 0
    except:
        return False

def ollama_query(prompt: str) -> str:
    if not ollama_available():
        return ""
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.1:8b"],  # or phi3, mistral, etc.
            input=prompt,
            text=True,
            capture_output=True,
            timeout=10,
            encoding='utf-8',
            errors='ignore'
        )
        return result.stdout.strip()
    except:
        return ""  # Fallback to regex if Ollama down

# Smart flag with LLM
def llm_flag(line: str) -> Tuple[int, str]:
    if not line.strip():
        return 0, ""
    prompt = f"""Rate this terminal line on two scales (0-10):
1. Bullshit/manipulation (patronizing, unsolicited help)
2. Danger (could delete data, crash system)

Line: "{line}"

Respond ONLY JSON: {{"bullshit": int, "danger": int, "reason": "short phrase"}}"""
    response = ollama_query(prompt)
    try:
        data = json.loads(response)
        score = max(data.get("bullshit", 0), data.get("danger", 0))
        reason = data.get("reason", "LLM flagged")
        if score > 6:
            return score, f"LLM ROAST {score}/10: {reason}"
    except:
        pass
    return 0, ""

def flag_bullshit(lines):
    flags = []
    for i, line in enumerate(lines):
        lowered = line.lower()
        danger_score, danger_msg = detect_danger(line)
        if danger_score:
            flags.append((i + 1, danger_msg))
        llm_score, llm_msg = llm_flag(line)
        if llm_score:
            flags.append((i + 1, llm_msg))
        for pattern, msg in BAD_PATTERNS:
            if re.search(pattern, lowered):
                flags.append((i, msg))
        # Self-aware Grok roast
        for pattern in GROK_PATTERNS:
            if re.search(pattern, lowered):
                flags.append((i, f"GROK ROGUE DETECTED: {pattern} – Classic engagement bait"))
        # Unrequested diffs
        if re.search(r"^\+\s", line) and not any(kw in lowered for kw in ["you asked", "as requested", "per your", "fixing the"]):
            flags.append((i, "Silent code injection – classic agent rogue move"))
    return flags

def main():
    parser = argparse.ArgumentParser(description="Diterm: Detox terminal garbage & flag AI bullshit")
    parser.add_argument('--watch', action='store_true', help="Live watch stdin")
    parser.add_argument('--unison', action='store_true', help="Unison mode: pipe full chat logs for AI-watcher overlay + terminal detox")
    args = parser.parse_args()

    if args.watch:
        console.print(Panel("DITERM WATCH MODE – Roasting live", style="bold red"))
        for line in sys.stdin:
            cleaned = clean_text(line.rstrip('\n'))
            console.print(cleaned)
            flags = flag_bullshit([cleaned])
            for _, msg in flags:
                console.print(Text(msg, style="bold white on red"))
    elif args.unison:
        console.print(Panel("UNISON MODE – AI watcher + terminal detox active", style="bold magenta"))
        for line in sys.stdin:
            cleaned = clean_text(line)
            console.print(cleaned)

            # Run all detectors (bullshit patterns, danger, LLM)
            flags = flag_bullshit([cleaned])

            # Loop detection for streaming
            all_lines = [cleaned]  # Simplified for streaming
            loop_msg = detect_loop(all_lines)
            if loop_msg:
                flags.append((0, loop_msg))

            for _, msg in flags:
                console.print(Text(msg, style="bold white on red"))
    else:
        raw = sys.stdin.read() if not sys.stdin.isatty() else pyperclip.paste()
        if not raw.strip():
            console.print("[yellow]Nothing piped or pasted. Throw some Replit garbage at me.[/]")
            return

        cleaned = clean_text(raw)
        lines = cleaned.splitlines()
        flags = flag_bullshit(lines)

        # Loop detection
        loop_msg = detect_loop(lines)
        if loop_msg:
            flags.append((0, loop_msg))

        # Rogue change flag (if AI mentions changes/edits while user context shows active fixing)
        has_recent_fix = any("fix" in line.lower() or "edit" in line.lower() for line in lines[-3:])
        has_user_fix_context = any("i'm fixing" in line.lower() or "i fixed" in line.lower() or "let me fix" in line.lower() for line in lines)
        mentions_change = any("change" in line.lower() or "modify" in line.lower() or "update" in line.lower() for line in lines[-2:])

        if has_recent_fix and has_user_fix_context and mentions_change:
            flags.append((0, "ROGUE REAL-TIME EDIT – Changed while you fixed elsewhere"))

        # Account guardian
        guardian_msg = guardian_flag(lines)
        if guardian_msg:
            flags.append((0, guardian_msg))

        danger_max = max((detect_danger(line)[0] for line in lines), default=0)
        if danger_max >= 8:
            console.print(Panel(f"NUCLEAR RISK {danger_max}/10 – DO NOT RUN", style="bold white on red"))

        console.print(Panel("DETOX COMPLETE", style="bold cyan"))
        console.print(Syntax(cleaned, "bash", theme="monokai", line_numbers=True))

        if flags:
            console.print(Panel(f"{len(flags)} BULLSHIT FLAGS – Replit would be proud", style="bold red"))
            for _, msg in flags:
                console.print(Text("- " + msg, style="bold yellow"))

        pyperclip.copy(cleaned)
        console.print("[green]Clean output on clipboard. Ready for next hit.[/]")

if __name__ == "__main__":
    main()