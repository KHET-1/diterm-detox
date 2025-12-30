# diterm.py v1.1 – Now with smarter bullshit flags + Replit-specific roasts
import sys
import re
import argparse
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

def clean_text(text):
    # Fix common garbage
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)  # ANSI escapes
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    return text

def flag_bullshit(lines):
    flags = []
    for i, line in enumerate(lines):
        lowered = line.lower()
        for pattern, msg in BAD_PATTERNS:
            if re.search(pattern, lowered):
                flags.append((i, msg))
        # Unrequested diffs
        if re.search(r"^\+\s", line) and not any(kw in lowered for kw in ["you asked", "as requested", "per your", "fixing the"]):
            flags.append((i, "Silent code injection – classic agent rogue move"))
    return flags

def main():
    parser = argparse.ArgumentParser(description="Diterm: Detox terminal garbage & flag AI bullshit")
    parser.add_argument('--watch', action='store_true', help="Live watch stdin")
    args = parser.parse_args()

    if args.watch:
        console.print(Panel("DITERM WATCH MODE – Roasting live", style="bold red"))
        for line in sys.stdin:
            cleaned = clean_text(line.rstrip('\n'))
            console.print(cleaned)
            flags = flag_bullshit([cleaned])
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