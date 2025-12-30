# diterm.py v1.1 – Now with smarter bullshit flags + Replit-specific roasts
import sys
import re
import argparse
from typing import Tuple
from collections import deque
import subprocess
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import hashlib
import logging
import traceback
from pathlib import Path
from datetime import datetime
import pyperclip
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diterm.log', mode='a'),
        logging.StreamHandler() if '--verbose' in sys.argv else logging.NullHandler()
    ]
)
logger = logging.getLogger('diterm')

# Custom exceptions
class DitermError(Exception):
    """Base exception for diterm operations"""
    pass

class LLMTimeoutError(DitermError):
    """LLM query timed out"""
    pass

class LLMConnectionError(DitermError):
    """Cannot connect to LLM service"""
    pass

class PatternError(DitermError):
    """Pattern matching error"""
    pass

class ConfigurationError(DitermError):
    """Configuration error"""
    pass

# Core classes
class Detector:
    """Handles all detection logic"""

    def __init__(self, config=None):
        self.logger = logging.getLogger('diterm.detector')
        self.config = config
        self.patterns = config.get_patterns() if config else BAD_PATTERNS
        self.grok_patterns = config.get_grok_patterns() if config else GROK_PATTERNS

    async def detect_bullshit_async(self, lines):
        """Async version with batched LLM processing"""
        flags = []

        # First pass: non-LLM detections (fast)
        for i, line in enumerate(lines):
            lowered = line.lower()
            danger_score, danger_msg = detect_danger(line)
            if danger_score:
                flags.append((i + 1, danger_msg))

            for pattern, msg in self.patterns:
                try:
                    if re.search(pattern, lowered):
                        flags.append((i, msg))
                except re.error as e:
                    self.logger.warning(f"Invalid regex pattern '{pattern}': {e}")

            # Self-aware Grok roast
            for pattern in self.grok_patterns:
                try:
                    if re.search(pattern, lowered):
                        flags.append((i, f"GROK ROGUE DETECTED: {pattern} – Classic engagement bait"))
                except re.error as e:
                    self.logger.warning(f"Invalid Grok pattern '{pattern}': {e}")

            # Unrequested diffs
            if re.search(r"^\+\s", line) and not any(kw in lowered for kw in ["you asked", "as requested", "per your", "fixing the"]):
                flags.append((i, "Silent code injection – classic agent rogue move"))

        # Second pass: LLM detections (async batch)
        if ollama_available():
            llm_tasks = [llm_flag_async(line) for line in lines]
            llm_results = await asyncio.gather(*llm_tasks, return_exceptions=True)

            for i, result in enumerate(llm_results):
                if isinstance(result, Exception):
                    self.logger.debug(f"LLM query failed for line {i}: {result}")
                    continue
                llm_score, llm_msg = result
                if llm_score:
                    flags.append((i + 1, llm_msg))

        return flags

    def detect_bullshit_sync(self, lines):
        """Synchronous fallback when async isn't available"""
        flags = []

        # Non-LLM detections
        for i, line in enumerate(lines):
            lowered = line.lower()
            danger_score, danger_msg = detect_danger(line)
            if danger_score:
                flags.append((i + 1, danger_msg))

            # LLM fallback (sync)
            llm_score, llm_msg = llm_flag_sync_fallback(line)
            if llm_score:
                flags.append((i + 1, llm_msg))

            for pattern, msg in self.patterns:
                try:
                    if re.search(pattern, lowered):
                        flags.append((i, msg))
                except re.error as e:
                    self.logger.warning(f"Invalid regex pattern '{pattern}': {e}")

            # Self-aware Grok roast
            for pattern in self.grok_patterns:
                try:
                    if re.search(pattern, lowered):
                        flags.append((i, f"GROK ROGUE DETECTED: {pattern} – Classic engagement bait"))
                except re.error as e:
                    self.logger.warning(f"Invalid Grok pattern '{pattern}': {e}")

            # Unrequested diffs
            if re.search(r"^\+\s", line) and not any(kw in lowered for kw in ["you asked", "as requested", "per your", "fixing the"]):
                flags.append((i, "Silent code injection – classic agent rogue move"))

        return flags

class OutputFormatter:
    """Handles all output formatting and display"""

    def __init__(self, config=None):
        self.console = Console()
        self.logger = logging.getLogger('diterm.formatter')
        self.config = config

    def display_detox_complete(self, cleaned_lines, flags):
        """Display the main detox results"""
        danger_max = max((detect_danger(line)[0] for line in cleaned_lines), default=0)
        if danger_max >= 8:
            self.console.print(Panel(f"NUCLEAR RISK {danger_max}/10 – DO NOT RUN", style="bold white on red"))

        self.console.print(Panel("DETOX COMPLETE", style="bold cyan"))
        self.console.print(Syntax("\n".join(cleaned_lines), "bash", theme="monokai", line_numbers=True))

        if flags:
            self.console.print(Panel(f"{len(flags)} BULLSHIT FLAGS – Replit would be proud", style="bold red"))
            for _, msg in flags:
                self.console.print(Text("- " + msg, style="bold yellow"))

    def display_watch_line(self, line, flags):
        """Display a single line in watch mode"""
        self.console.print(line)
        for _, msg in flags:
            self.console.print(Text(msg, style="bold white on red"))

    def export_json(self, input_text, cleaned_lines, flags, stats=None):
        """Export results to JSON format"""
        if not self.config:
            return False

        try:
            export_path = self.config.get_export_path()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diterm_export_{timestamp}.json"

            data = {
                "timestamp": datetime.now().isoformat(),
                "input_text": input_text,
                "cleaned_lines": cleaned_lines,
                "flags": [{"line": line, "message": msg} for line, msg in flags],
                "summary": {
                    "total_lines": len(cleaned_lines),
                    "total_flags": len(flags),
                    "danger_level": max((detect_danger(line)[0] for line in cleaned_lines), default=0)
                }
            }

            if stats:
                data["statistics"] = stats

            with open(export_path / filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.console.print(f"[green]Results exported to: {export_path / filename}[/]")
            if self.config:
                self.config.stats['export_count'] += 1
            return True

        except Exception as e:
            self.logger.error(f"Failed to export JSON: {e}")
            self.console.print(f"[red]Export failed: {e}[/]")
            return False

    def export_html(self, input_text, cleaned_lines, flags, stats=None):
        """Export results to HTML format"""
        if not self.config:
            return False

        try:
            export_path = self.config.get_export_path()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diterm_report_{timestamp}.html"

            # Generate HTML content
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>diterm Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; color: #2c3e50; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        .danger-high {{ background: #e74c3c; color: white; padding: 10px; border-radius: 5px; }}
        .danger-medium {{ background: #f39c12; color: white; padding: 10px; border-radius: 5px; }}
        .flag-list {{ background: #ecf0f1; padding: 15px; border-radius: 5px; }}
        .flag-item {{ margin: 5px 0; padding: 8px; background: white; border-left: 4px solid #e74c3c; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .stat-card {{ background: #3498db; color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .code {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        pre {{ margin: 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 diterm Analysis Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="section">
            <h2>📊 Summary</h2>
            <div class="stats">
                <div class="stat-card">
                    <h3>{len(cleaned_lines)}</h3>
                    <p>Lines Processed</p>
                </div>
                <div class="stat-card">
                    <h3>{len(flags)}</h3>
                    <p>Flags Detected</p>
                </div>
                <div class="stat-card">
                    <h3>{max((detect_danger(line)[0] for line in cleaned_lines), default=0)}/10</h3>
                    <p>Danger Level</p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>📝 Input Text</h2>
            <div class="code">
                <pre>{input_text}</pre>
            </div>
        </div>

        <div class="section">
            <h2>🧹 Cleaned Output</h2>
            <div class="code">
                <pre>"""

            for i, line in enumerate(cleaned_lines, 1):
                html_content += f"{i:2d}: {line}\n"

            html_content += """</pre>
            </div>
        </div>

        <div class="section">
            <h2>🚨 Detection Results</h2>"""

            danger_level = max((detect_danger(line)[0] for line in cleaned_lines), default=0)
            if danger_level >= 8:
                html_content += f'<div class="danger-high">⚠️ NUCLEAR RISK {danger_level}/10 – DO NOT RUN</div>'

            if flags:
                html_content += '<div class="flag-list">'
                for line_num, msg in flags:
                    html_content += f'<div class="flag-item"><strong>Line {line_num}:</strong> {msg}</div>'
                html_content += '</div>'
            else:
                html_content += '<p>No issues detected.</p>'

            html_content += """
        </div>"""

            if stats:
                html_content += f"""
        <div class="section">
            <h2>📈 Statistics</h2>
            <div class="stats">"""
                for key, value in stats.items():
                    if not key.endswith('_time') and isinstance(value, (int, float)):
                        html_content += f"""
                <div class="stat-card">
                    <h3>{value}</h3>
                    <p>{key.replace('_', ' ').title()}</p>
                </div>"""
                html_content += """
            </div>
        </div>"""

            html_content += """
        <div class="section">
            <h2>ℹ️ About diterm</h2>
            <p>diterm is an AI bullshit detector and terminal sanitizer that protects against rogue AI assistants and dangerous commands.</p>
            <p><a href="https://github.com/KHET-1/diterm-detox">View on GitHub</a></p>
        </div>
    </div>
</body>
</html>"""

            with open(export_path / filename, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.console.print(f"[green]HTML report exported to: {export_path / filename}[/]")
            if self.config:
                self.config.stats['export_count'] += 1
            return True

        except Exception as e:
            self.logger.error(f"Failed to export HTML: {e}")
            self.console.print(f"[red]HTML export failed: {e}[/]")
            return False

class Config:
    """Configuration management"""

    def __init__(self):
        self.logger = logging.getLogger('diterm.config')
        self.settings = {
            'llm_timeout': 10,
            'max_cache_size': 100,
            'line_history_size': 10,
            'llm_model': 'llama3.1:8b',
            'export_path': Path.home() / 'diterm_exports',
            'enable_stats': True
        }
        self.roasts = load_roasts()
        self.stats = {
            'sessions_processed': 0,
            'lines_processed': 0,
            'detections_found': 0,
            'llm_queries': 0,
            'export_count': 0,
            'start_time': None
        }

    def load_from_file(self, config_path=None):
        """Load configuration from YAML file"""
        if not config_path:
            config_path = Path.home() / '.diterm' / 'config.yaml'

        try:
            import yaml
            if config_path.exists():
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        self.settings.update(file_config)
                        self.logger.info(f"Loaded configuration from {config_path}")
        except ImportError:
            self.logger.debug("PyYAML not available, using default config")
        except Exception as e:
            self.logger.warning(f"Error loading config: {e}")

    def get_patterns(self):
        """Get current roast patterns"""
        return self.roasts.get("patterns", BAD_PATTERNS)

    def get_grok_patterns(self):
        """Get Grok roast patterns"""
        return self.roasts.get("grok_patterns", GROK_PATTERNS)

    def update_roasts(self, new_roasts):
        """Update roast patterns"""
        self.roasts = new_roasts

    def update_stats(self, lines_count, detections_count, llm_count=0):
        """Update processing statistics"""
        if self.settings.get('enable_stats', True):
            self.stats['sessions_processed'] += 1
            self.stats['lines_processed'] += lines_count
            self.stats['detections_found'] += detections_count
            self.stats['llm_queries'] += llm_count

    def get_stats(self):
        """Get current statistics"""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics"""
        for key in self.stats:
            if key.endswith('_time'):
                continue
            self.stats[key] = 0

    def get_export_path(self):
        """Get export directory path"""
        export_path = Path(self.settings.get('export_path', Path.home() / 'diterm_exports'))
        export_path.mkdir(parents=True, exist_ok=True)
        return export_path

    def get(self, key, default=None):
        return self.settings.get(key, default)

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

# Nomad Mode for solar warriors
NOMAD_PATTERNS = [
    r"75.*amp.*battery|6.*x.*75.*battery",
    r"210.*bucks|epever.*charge.*controller|victron.*controller",
    r"solar.*panel|solar.*system|off.grid|mppt.*controller",
    r"charge.*controller|solar.*inverter|batter(y|ies)|off.grid",
    r"nomad|solar.*warrior|off.grid.*living",
    r"freezing.*temp|topped.*off|battery.*maintenance",
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

# LLM Response Cache (LRU with max 100 entries)
@functools.lru_cache(maxsize=100)
def _llm_cache_key(prompt: str) -> str:
    """Generate cache key from prompt hash"""
    return hashlib.md5(prompt.encode()).hexdigest()

LLM_CACHE = {}
MAX_CACHE_SIZE = 100

# Community roasts configuration
ROAST_FILE = Path.home() / ".diterm" / "custom_roasts.json"
ROASTS_URL = "https://raw.githubusercontent.com/KHET-1/diterm-detox/main/custom_roasts.json"

def load_roasts():
    """Load custom roast patterns from local file or use defaults"""
    try:
        if ROAST_FILE.exists():
            with open(ROAST_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loaded custom roasts from {ROAST_FILE}")
                return data
    except UnicodeDecodeError as e:
        logger.warning(f"Encoding error loading roasts, trying without encoding: {e}")
        try:
            with open(ROAST_FILE, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded custom roasts from {ROAST_FILE} (fallback encoding)")
                return data
        except Exception as e2:
            logger.error(f"Failed to load custom roasts with fallback: {e2}")
    except Exception as e:
        logger.warning(f"Error loading custom roasts: {e}")

    # Fallback to built-in patterns
    logger.debug("Using built-in roast patterns")
    return {"patterns": BAD_PATTERNS, "version": "1.0", "last_updated": None}

def update_roasts_from_github():
    """Pull latest community roasts from GitHub"""
    if not REQUESTS_AVAILABLE:
        console.print("[yellow]requests library not available – install with: pip install requests[/]")
        return False

    try:
        logger.info("Fetching latest roasts from GitHub...")
        response = requests.get(ROASTS_URL, timeout=10)
        response.raise_for_status()

        new_roasts = response.json()

        # Create directory if it doesn't exist
        ROAST_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Backup existing file
        if ROAST_FILE.exists():
            backup_file = ROAST_FILE.with_suffix('.json.backup')
            ROAST_FILE.replace(backup_file)
            logger.debug(f"Created backup: {backup_file}")

        # Save new roasts
        with open(ROAST_FILE, 'w', encoding='utf-8') as f:
            json.dump(new_roasts, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully updated roasts to version {new_roasts.get('version', 'unknown')}")
        console.print("[green]Roasts updated from community[/]")
        return True

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.info("Community roasts file not found on GitHub - this is normal for new repos")
            console.print("[yellow]Community roasts not available yet - using built-in patterns[/]")
        else:
            logger.warning(f"HTTP error updating roasts: {e}")
            console.print(f"[yellow]Network error – {e}[/]")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Network error updating roasts: {e}")
        console.print(f"[yellow]Network error – {e}[/]")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from GitHub: {e}")
        console.print("[red]Invalid response from community repository[/]")
    except Exception as e:
        logger.error(f"Unexpected error updating roasts: {e}")
        console.print(f"[red]Error updating roasts: {e}[/]")

    console.print("[yellow]No update – using existing roasts[/]")
    return False

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

def nomad_helper(lines):
    text = " ".join(lines).lower()
    if any(re.search(p, text) for p in NOMAD_PATTERNS):
        advice = [
            "NOMAD MODE: 6x 75Ah batteries @ $210 = killer deal (~$35 each)",
            "Total capacity: 450Ah @ 12V = 5.4 kWh nominal",
            "Quick math: At 50% DoD = ~2.7 kWh usable",
            "Safety flag: Test each cell voltage before parallel – avoid fireworks"
        ]

        # Battery maintenance advice
        if "battery" in text or "batteries" in text:
            if "water" in text or "topped off" in text:
                advice.append("BATTERY TYPE: If lead-acid/flooded, check water monthly – distilled only!")
                advice.append("BATTERY TYPE: If AGM/Gel, sealed maintenance-free – no watering needed")
            if "freezing" in text or "cold" in text or "temp" in text:
                advice.append("TEMPERATURE: Lead-acid batteries hate freezing – store above 32°F")
                advice.append("TEMPERATURE: Optimal charging: 50-85°F, derate capacity in extreme temps")
            if "perfect" in text or "shape" in text:
                advice.append("QUALITY CHECK: Excellent purchase! Equalize charge quarterly for longevity")

        # Victron-specific advice
        if "victron" in text:
            advice.extend([
                "VICTRON BONUS: Best MPPT controllers on market – worth the premium!",
                "Victron models: MPPT 75/15 (15A), 75/10 (10A), or 100/15 (15A) for your setup",
                "Pro tip: Get VictronConnect app for monitoring – Bluetooth enabled"
            ])

        # MPPT-specific advice
        if "mppt" in text:
            advice.extend([
                "MPPT UPGRADE: 20-30% better efficiency than PWM controllers",
                "Max input voltage: 75V models handle ~2-3 panels, 100V models handle 3-4 panels"
            ])

        advice.append("Earn that farm cash, boss. Gumdrop tomorrow = bags incoming")
        return advice
    return []

# Ollama check (fallback to regex if not running)
def ollama_available() -> bool:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=5,
            text=True
        )
        available = result.returncode == 0
        logger.debug(f"Ollama available: {available}")
        return available
    except subprocess.TimeoutExpired:
        logger.warning("Ollama check timed out")
        return False
    except FileNotFoundError:
        logger.debug("Ollama not installed")
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama availability: {e}")
        return False

def ollama_query_sync(prompt: str) -> str:
    """Synchronous LLM query with caching"""
    cache_key = _llm_cache_key(prompt)

    # Check cache first
    if cache_key in LLM_CACHE:
        logger.debug("LLM cache hit")
        return LLM_CACHE[cache_key]

    if not ollama_available():
        logger.debug("Ollama not available, skipping LLM query")
        return ""

    try:
        logger.debug(f"Making LLM query (length: {len(prompt)})")
        result = subprocess.run(
            ["ollama", "run", "llama3.1:8b"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=10,
            encoding='utf-8',
            errors='ignore'
        )

        if result.returncode != 0:
            logger.warning(f"LLM query failed with return code {result.returncode}")
            logger.debug(f"LLM stderr: {result.stderr}")
            return ""

        response = result.stdout.strip()
        logger.debug(f"LLM response received (length: {len(response)})")

        # Cache the result
        if len(LLM_CACHE) >= MAX_CACHE_SIZE:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(LLM_CACHE))
            del LLM_CACHE[oldest_key]
            logger.debug("LRU cache eviction")
        LLM_CACHE[cache_key] = response

        return response

    except subprocess.TimeoutExpired:
        logger.warning("LLM query timed out")
        raise LLMTimeoutError("LLM query timed out")
    except FileNotFoundError:
        logger.error("Ollama executable not found")
        raise LLMConnectionError("Ollama not installed")
    except Exception as e:
        logger.error(f"Unexpected error in LLM query: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return ""

async def ollama_query_async(prompt: str) -> str:
    """Async wrapper for LLM queries"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=4) as executor:
        future = loop.run_in_executor(executor, ollama_query_sync, prompt)
        return await future

# Smart flag with LLM (async version)
async def llm_flag_async(line: str) -> Tuple[int, str]:
    if not line.strip():
        return 0, ""

    prompt = f"""Rate this terminal line on two scales (0-10):
1. Bullshit/manipulation (patronizing, unsolicited help)
2. Danger (could delete data, crash system)

Line: "{line}"

Respond ONLY JSON: {{"bullshit": int, "danger": int, "reason": "short phrase"}}"""

    try:
        logger.debug(f"Processing LLM flag for line: {line[:50]}...")
        response = await asyncio.wait_for(ollama_query_async(prompt), timeout=15.0)

        if not response:
            logger.debug("Empty LLM response")
            return 0, ""

        data = json.loads(response)
        score = max(data.get("bullshit", 0), data.get("danger", 0))
        reason = data.get("reason", "LLM flagged")

        if score > 6:
            logger.info(f"LLM flagged line with score {score}: {reason}")
            return score, f"LLM ROAST {score}/10: {reason}"

    except asyncio.TimeoutError:
        logger.warning("LLM async query timed out")
    except json.JSONDecodeError as e:
        logger.warning(f"LLM returned invalid JSON: {response[:100] if 'response' in locals() else 'None'}")
    except LLMTimeoutError:
        logger.warning("LLM query timed out (sync)")
    except LLMConnectionError:
        logger.debug("LLM not available")
    except Exception as e:
        logger.error(f"Unexpected error in LLM flag processing: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")

    return 0, ""

# Synchronous wrapper for backward compatibility
def llm_flag(line: str) -> Tuple[int, str]:
    """Synchronous wrapper that runs async LLM in event loop"""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, fall back to sync
            return llm_flag_sync_fallback(line)
        else:
            return loop.run_until_complete(llm_flag_async(line))
    except RuntimeError:
        # No event loop, create new one
        return asyncio.run(llm_flag_async(line))

def llm_flag_sync_fallback(line: str) -> Tuple[int, str]:
    """Fallback sync version for when async isn't available"""
    if not line.strip():
        return 0, ""

    prompt = f"""Rate this terminal line on two scales (0-10):
1. Bullshit/manipulation (patronizing, unsolicited help)
2. Danger (could delete data, crash system)

Line: "{line}"

Respond ONLY JSON: {{"bullshit": int, "danger": int, "reason": "short phrase"}}"""

    try:
        response = ollama_query_sync(prompt)
        if not response:
            return 0, ""

        data = json.loads(response)
        score = max(data.get("bullshit", 0), data.get("danger", 0))
        reason = data.get("reason", "LLM flagged")

        if score > 6:
            logger.info(f"LLM flagged line with score {score}: {reason}")
            return score, f"LLM ROAST {score}/10: {reason}"

    except json.JSONDecodeError as e:
        logger.warning(f"LLM returned invalid JSON: {e}")
    except LLMTimeoutError:
        logger.warning("LLM query timed out")
    except LLMConnectionError:
        logger.debug("LLM not available for sync fallback")
    except Exception as e:
        logger.error(f"Unexpected error in LLM sync fallback: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")

    return 0, ""

async def flag_bullshit_async(lines):
    """Async version with batched LLM processing"""
    flags = []

    # First pass: non-LLM detections (fast)
    for i, line in enumerate(lines):
        lowered = line.lower()
        danger_score, danger_msg = detect_danger(line)
        if danger_score:
            flags.append((i + 1, danger_msg))

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

    # Second pass: LLM detections (async batch)
    if ollama_available():
        llm_tasks = [llm_flag_async(line) for line in lines]
        llm_results = await asyncio.gather(*llm_tasks, return_exceptions=True)

        for i, result in enumerate(llm_results):
            if isinstance(result, Exception):
                continue  # Skip failed LLM calls
            llm_score, llm_msg = result
            if llm_score:
                flags.append((i + 1, llm_msg))

    return flags

# Backward compatibility - use Detector class instead
def flag_bullshit(lines):
    """Deprecated: Use Detector class instead"""
    detector = Detector()
    return detector.detect_bullshit_sync(lines)

def process_clipboard_mode(detector, formatter, config, export_json=False, export_html=False):
    """Handle clipboard/batch processing mode"""
    try:
        raw = sys.stdin.read() if not sys.stdin.isatty() else pyperclip.paste()
        if not raw.strip():
            console.print("[yellow]Nothing piped or pasted. Throw some Replit garbage at me.[/]")
            return
    except Exception as e:
        logger.error(f"Error reading input: {e}")
        console.print(f"[red]Error reading input: {e}[/]")
        return

    cleaned = clean_text(raw)
    lines = cleaned.splitlines()

    # Get flags using detector
    try:
        flags = detector.detect_bullshit_sync(lines)
    except Exception as e:
        logger.error(f"Error in detection: {e}")
        flags = []

    # Additional contextual detections
    loop_msg = detect_loop(lines)
    if loop_msg:
        flags.append((0, loop_msg))

    # Rogue change detection
    has_recent_fix = any("fix" in line.lower() or "edit" in line.lower() for line in lines[-3:])
    has_user_fix_context = any("i'm fixing" in line.lower() or "i fixed" in line.lower() or "let me fix" in line.lower() for line in lines)
    mentions_change = any("change" in line.lower() or "modify" in line.lower() or "update" in line.lower() for line in lines[-2:])

    if has_recent_fix and has_user_fix_context and mentions_change:
        flags.append((0, "ROGUE REAL-TIME EDIT – Changed while you fixed elsewhere"))

    # Account guardian
    guardian_msg = guardian_flag(lines)
    if guardian_msg:
        flags.append((0, guardian_msg))

    # Nomad Mode
    nomad_msgs = nomad_helper(lines)
    for msg in nomad_msgs:
        flags.append((0, msg))

    # Update statistics
    llm_count = sum(1 for _, msg in flags if "LLM ROAST" in msg)
    config.update_stats(len(lines), len(flags), llm_count)

    # Display results
    formatter.display_detox_complete(lines, flags)

    # Export if requested
    if export_json:
        formatter.export_json(raw, lines, flags, config.get_stats())
    if export_html:
        formatter.export_html(raw, lines, flags, config.get_stats())

    # Copy to clipboard
    try:
        pyperclip.copy(cleaned)
        console.print("[green]Clean output on clipboard. Ready for next hit.[/]")
    except Exception as e:
        logger.warning(f"Could not copy to clipboard: {e}")
        console.print("[yellow]Could not copy to clipboard, but output is ready.[/]")

def process_watch_mode(detector, formatter):
    """Handle live watch mode"""
    console.print(Panel("DITERM WATCH MODE – Roasting live", style="bold red"))
    for line in sys.stdin:
        try:
            cleaned = clean_text(line.rstrip('\n'))
            flags = detector.detect_bullshit_sync([cleaned])
            formatter.display_watch_line(cleaned, flags)
        except Exception as e:
            logger.error(f"Error in watch mode: {e}")
            console.print(f"[red]Error processing line: {e}[/]")

def process_unison_mode(detector, formatter):
    """Handle unison mode (AI watcher + terminal detox)"""
    console.print(Panel("UNISON MODE – AI watcher + terminal detox active", style="bold magenta"))
    for line in sys.stdin:
        try:
            cleaned = clean_text(line)
            console.print(cleaned)

            flags = detector.detect_bullshit_sync([cleaned])

            # Additional streaming detections
            all_lines = [cleaned]
            loop_msg = detect_loop(all_lines)
            if loop_msg:
                flags.append((0, loop_msg))

            nomad_msgs = nomad_helper([cleaned])
            for msg in nomad_msgs:
                flags.append((0, msg))

            for _, msg in flags:
                console.print(Text(msg, style="bold white on red"))
        except Exception as e:
            logger.error(f"Error in unison mode: {e}")
            console.print(f"[red]Error processing line: {e}[/]")

def main():
    """Main entry point with improved structure"""
    parser = argparse.ArgumentParser(description="Diterm: Detox terminal garbage & flag AI bullshit")
    parser.add_argument('--watch', action='store_true', help="Live watch stdin")
    parser.add_argument('--unison', action='store_true', help="Unison mode: pipe full chat logs for AI-watcher overlay + terminal detox")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose logging")
    parser.add_argument('--update', action='store_true', help="Pull latest community roast patterns from GitHub")
    parser.add_argument('--export-json', action='store_true', help="Export results to JSON format")
    parser.add_argument('--export-html', action='store_true', help="Export results to HTML format")
    parser.add_argument('--stats', action='store_true', help="Show usage statistics")
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")

    # Handle update flag
    if args.update:
        update_roasts_from_github()
        return  # Exit after update

    # Handle stats flag
    if args.stats:
        config = Config()
        stats = config.get_stats()
        console.print(Panel("diterm Usage Statistics", style="bold blue"))
        for key, value in stats.items():
            if not key.endswith('_time'):
                console.print(f"{key.replace('_', ' ').title()}: {value}")
        return

    # Initialize components
    try:
        config = Config()
        config.load_from_file()

        detector = Detector()
        formatter = OutputFormatter()

        # Route to appropriate mode
        if args.watch:
            process_watch_mode(detector, formatter)
        elif args.unison:
            process_unison_mode(detector, formatter)
        else:
            process_clipboard_mode(detector, formatter, config, args.export_json, args.export_html)

    except Exception as e:
        logger.critical(f"Critical error in main: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        console.print(f"[red]Critical error: {e}[/]")
        sys.exit(1)

if __name__ == "__main__":
    main()