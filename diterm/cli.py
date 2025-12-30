"""
Command-line interface for diterm-detox.
"""

import sys
from .detector import DangerousCommandDetector


def main():
    """Main entry point for the CLI."""
    if len(sys.argv) < 2:
        print("Usage: diterm-detox <command>")
        print("Example: diterm-detox 'rm -rf /'")
        sys.exit(1)
    
    command = " ".join(sys.argv[1:])
    detector = DangerousCommandDetector()
    
    if detector.is_dangerous(command):
        risk_score = detector.get_risk_score(command)
        print(f"⚠️  DANGER DETECTED! Risk Score: {risk_score}/10")
        print(f"Command: {command}")
        print("\n🛑 This command may cause irreversible damage!")
        sys.exit(1)
    else:
        print(f"✅ Command appears safe: {command}")
        sys.exit(0)


if __name__ == "__main__":
    main()
