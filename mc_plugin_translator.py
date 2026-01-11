#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
translate_serowsour.py
Script created by SerowSour
A YAML translation and auto-repair tool.
"""

import argparse
import concurrent.futures
import logging
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

import yaml

# Attempt to import googletrans; fallback to deep-translator if available
try:
    from googletrans import Translator as GTTranslator
except Exception:
    GTTranslator = None

try:
    from deep_translator import GoogleTranslator as DTTranslator
except Exception:
    DTTranslator = None

# -----------------------
# Configuration
# -----------------------
DEFAULT_LANG = "it"
SPINNER_DELAY = 0.12
MAX_RETRIES = 3
TRANSLATE_TIMEOUT = 10  # seconds per translation call

# Terms that must never be translated (Minecraft-specific vocabulary)
MINECRAFT_TERMS = {
    "Land", "land", "Chunk", "Chunks", "chunk", "chunks", "Biome", "biomes",
    "Nether", "End", "Overworld", "PvP", "PVP", "Cooldown", "Cooldowns",
    "Claim", "Claims", "claim", "claims", "Unclaim", "unclaim", "Spawn",
    "Mob", "Mobs", "XP", "Health", "Mana", "Region", "Block", "Blocks",
    "Item", "Items", "Inventory", "Server", "Player", "Players", "World",
    "Worlds"
}
LOWER_TERMS = {t.lower() for t in MINECRAFT_TERMS}

# Regex for placeholders and protected tokens
PLACEHOLDER_RE = re.compile(r"(\{[^}]+\}|%[^%\s]+%|\$[A-Za-z0-9_]+)")

# Regex for Minecraft terms (case-insensitive)
TERMS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sorted(MINECRAFT_TERMS, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE
)

# -----------------------
# Spinner for console feedback
# -----------------------
_spinner_running = False
_spinner_thread = None


def spinner_start(message: str):
    """Start a console spinner with a status message."""
    global _spinner_running, _spinner_thread
    if _spinner_running:
        return
    _spinner_running = True

    def run():
        chars = "|/-\\"
        idx = 0
        while _spinner_running:
            sys.stdout.write(f"\r{message} {chars[idx % len(chars)]}")
            sys.stdout.flush()
            idx += 1
            time.sleep(SPINNER_DELAY)
        sys.stdout.write("\r" + " " * (len(message) + 4) + "\r")
        sys.stdout.flush()

    _spinner_thread = threading.Thread(target=run, daemon=True)
    _spinner_thread.start()


def spinner_stop(final_message: str = ""):
    """Stop the spinner and optionally print a final message."""
    global _spinner_running, _spinner_thread
    if not _spinner_running:
        if final_message:
            print(final_message)
        return
    _spinner_running = False
    if _spinner_thread:
        _spinner_thread.join(timeout=1)
    if final_message:
        print(final_message)


# -----------------------
# Logging
# -----------------------
logger = logging.getLogger("translate_serowsour")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
logger.addHandler(handler)


# -----------------------
# File helpers
# -----------------------
def find_file(path_str: str) -> Path:
    """Locate a file across common Termux/Android paths."""
    p = Path(path_str)
    if p.exists():
        return p
    candidates = [
        Path("/storage/emulated/0") / path_str,
        Path("/sdcard") / path_str,
        Path("/storage/emulated/0/Download") / path_str,
        Path("/sdcard/Download") / path_str,
        Path.cwd() / path_str,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def backup_file(path: Path) -> Path:
    """Create a backup copy of the file."""
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    return bak


# -----------------------
# YAML fixer (non-destructive)
# -----------------------
def fix_yaml_content(content: str) -> str:
    """
    Repair lines with unquoted values containing problematic characters.
    Attempts to preserve indentation and comments.
    """
    fixed_lines = []
    for line in content.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        # Preserve comments and lines without ':'
        if stripped.startswith("#") or ":" not in stripped:
            fixed_lines.append(line)
            continue

        # Split only on the first ':'
        key_part, val_part = stripped.split(":", 1)
        key = key_part.rstrip()
        val = val_part.lstrip()

        # Leave empty, quoted, or block scalar values unchanged
        if val == "" or val.startswith(("'", '"')) or val.startswith("|") or val.startswith(">"):
            fixed_lines.append(line)
            continue

        # Quote values containing problematic characters
        if any(c in val for c in ["&", ":", "'"]):
            if "'" in val and '"' not in val:
                safe_val = "'" + val.replace("'", "''") + "'"
            else:
                safe_val = '"' + val.replace('\\', '\\\\').replace('"', '\\"') + '"'
            new_line = f"{indent}{key}: {safe_val}"
            fixed_lines.append(new_line)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def load_yaml_with_fix(path: Path, make_backup: bool = True) -> Any:
    """Load YAML, attempting automatic repair if parsing fails."""
    raw = path.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as e:
        logger.warning("Invalid YAML: %s", e)
        if make_backup:
            bak = backup_file(path)
            logger.info("Backup created: %s", bak.name)
        logger.info("Attempting automatic YAML repair (pre-translation fix)...")
        fixed = fix_yaml_content(raw)
        try:
            return yaml.safe_load(fixed)
        except Exception as e2:
            logger.error("Automatic repair failed: %s", e2)
            raise


# -----------------------
# Masking / Unmasking
# -----------------------
def mask_text(text: str, mapping: Dict[str, str]) -> str:
    """Replace placeholders and protected terms with tokens."""
    token_index = len(mapping)

    def repl_ph(m):
        nonlocal token_index
        token = f"__PH{token_index}__"
        mapping[token] = m.group(0)
        token_index += 1
        return token

    text = PLACEHOLDER_RE.sub(repl_ph, text)

    def repl_term(m):
        nonlocal token_index
        original = m.group(0)
        if original.lower() in LOWER_TERMS:
            token = f"__MT{token_index}__"
            mapping[token] = original
            token_index += 1
            return token
        return original

    return TERMS_PATTERN.sub(repl_term, text)


def unmask_text(text: str, mapping: Dict[str, str]) -> str:
    """Restore masked placeholders and terms."""
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text


# -----------------------
# Translation engine with fallback and timeout
# -----------------------
def translate_via_googletrans(text: str, dest: str) -> str:
    if GTTranslator is None:
        raise RuntimeError("googletrans not available")
    t = GTTranslator()
    res = t.translate(text, dest=dest)
    return res.text


def translate_via_deep_translator(text: str, dest: str) -> str:
    if DTTranslator is None:
        raise RuntimeError("deep-translator not available")
    return DTTranslator(source="auto", target=dest).translate(text)


def translate_one(masked_text: str, dest: str, timeout: int = TRANSLATE_TIMEOUT) -> str:
    """
    Attempt translation with timeout and fallback.
    Returns the translated string or raises an exception.
    """
    def call_google():
        return translate_via_googletrans(masked_text, dest)

    def call_deep():
        return translate_via_deep_translator(masked_text, dest)

    attempts = []
    if GTTranslator is not None:
        attempts.append(call_google)
    if DTTranslator is not None:
        attempts.append(call_deep)

    if not attempts:
        raise RuntimeError("No translation engine available (install googletrans or deep-translator)")

    last_exc = None
    for fn in attempts:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fn)
                return fut.result(timeout=timeout)
        except Exception as e:
            last_exc = e
    raise last_exc


# -----------------------
# Translation functions with masking, retry, and logging
# -----------------------
PROGRESS = {"translated": 0, "skipped": 0}


def translate_string(s: str, dest: str, max_retries: int = MAX_RETRIES) -> str:
    """Translate a single string with masking and retry logic."""
    if not isinstance(s, str) or not s.strip():
        PROGRESS["skipped"] += 1
        return s

    mapping: Dict[str, str] = {}
    masked = mask_text(s, mapping)

    if re.fullmatch(r"(?:(?:__PH\d+__)|(?:__MT\d+__))+", masked or ""):
        PROGRESS["skipped"] += 1
        return unmask_text(masked, mapping)

    last_result = unmask_text(masked, mapping)
    for attempt in range(1, max_retries + 1):
        try:
            translated_masked = translate_one(masked, dest)
            final = unmask_text(translated_masked, mapping)
            PROGRESS["translated"] += 1
            return final
        except Exception as e:
            logger.debug("Attempt %d failed for string: %s", attempt, e)
            last_result = unmask_text(masked, mapping)
            time.sleep(1)

    PROGRESS["skipped"] += 1
    return last_result


def translate_value(val: Any, dest: str) -> Any:
    """Recursively translate strings inside YAML structures."""
    if isinstance(val, str):
        return translate_string(val, dest)
    if isinstance(val, dict):
        return {k: translate_value(v, dest) for k, v in val.items()}
    if isinstance(val, list):
        return [translate_value(x, dest) for x in val]
    return val


# -----------------------
# Post-translation fixes
# -----------------------
def post_fix_translated_content(obj: Any) -> Any:
    """Perform minimal non-destructive cleanup on translated content."""
    if isinstance(obj, str):
        s = obj
        s = s.replace("''", "’") if "''" in s else s
        words = s.split()
        for i, w in enumerate(words):
            lw = re.sub(r"[^\w]", "", w).lower()
            if lw in LOWER_TERMS:
                for orig in MINECRAFT_TERMS:
                    if orig.lower() == lw:
                        words[i] = re.sub(r"\b" + re.escape(re.sub(r"[^\w]", "", w)) + r"\b", orig, words[i])
                        break
        return " ".join(words)
    if isinstance(obj, dict):
        return {k: post_fix_translated_content(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [post_fix_translated_content(x) for x in obj]
    return obj


# -----------------------
# Final sanity check
# -----------------------
def final_sanity_check(obj: Any) -> bool:
    """Verify that no unresolved tokens remain."""
    problems = []

    def walk(x, path="root"):
        if x is None:
            problems.append(f"{path} is None")
            return
        if isinstance(x, str):
            if "__PH" in x or "__MT" in x:
                problems.append(f"{path} contains unresolved tokens")
            return
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}")
            return
        if isinstance(x, list):
            for idx, v in enumerate(x):
                walk(v, f"{path}[{idx}]")
            return

    walk(obj)
    if problems:
        logger.warning("Final sanity check: issues detected:")
        for p in problems:
            logger.warning(" - %s", p)
        return False
    return True


# -----------------------
# Main pipeline
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="translate_serowsour.py - YAML translator with intelligent repair")
    parser.add_argument("-i", "--input", required=True, help="Input YAML file")
    parser.add_argument("-o", "--output", default=None, help="Output YAML file (default: input_LANG.yml)")
    parser.add_argument("-l", "--lang", default=DEFAULT_LANG, help="Target language (e.g., it)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("-nobackup", action="store_true", help="Do not create automatic backup")
    args = parser.parse_args()

    print("Script created by SerowSour")

    if args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    input_path = find_file(args.input)
    if not input_path:
        print("❌ Input file not found:", args.input)
        sys.exit(2)

    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        suffix = input_path.suffix or ".yml"
        output_name = f"{stem}_{args.lang}{suffix}"
        output_path = input_path.with_name(output_name)

    # PHASE A
    phase_msg = "PHASE A — Pre-processing (validating and repairing source YAML)"
    spinner_start(phase_msg)
    try:
        data = load_yaml_with_fix(input_path, make_backup=not args.nobackup)
    except Exception as e:
        spinner_stop(f"{phase_msg} -> ERROR: unable to load YAML: {e}")
        sys.exit(3)
    spinner_stop(f"{phase_msg} -> OK")

    # PHASE B
    phase_msg = "PHASE B — Translation in progress"
    spinner_start(phase_msg)
    try:
        translated = translate_value(data, args.lang)
    except Exception as e:
        spinner_stop(f"{phase_msg} -> ERROR: {e}")
        sys.exit(4)
    spinner_stop(f"{phase_msg} -> OK (translated: {PROGRESS['translated']}, skipped: {PROGRESS['skipped']})")

    # PHASE C
    phase_msg = "PHASE C — Post-processing (partial cleanup)"
    spinner_start(phase_msg)
    try:
        translated = post_fix_translated_content(translated)
    except Exception as e:
        spinner_stop(f"{phase_msg} -> ERROR: {e}")
        sys.exit(5)
    spinner_stop(f"{phase_msg} -> OK")

    # PHASE D
    phase_msg = "PHASE D — Post-processing (final cleanup)"
    spinner_start(phase_msg)
    try:
        pass
    except Exception as e:
        spinner_stop(f"{phase_msg} -> ERROR: {e}")
        sys.exit(6)
    spinner_stop(f"{phase_msg} -> OK")

    # PHASE E
    phase_msg = "PHASE E — Final integrity check"
    spinner_start(phase_msg)
    ok = final_sanity_check(translated)
    spinner_stop(f"{phase_msg} -> {'OK' if ok else 'ISSUES DETECTED'}")
    if not ok:
        logger.warning("Final integrity check detected issues. Review the log for details.")

    # Save output
    try:
        output_path.write_text(yaml.dump(translated, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"Output file created: {output_path.name}")
    except Exception as e:
        logger.error("Output write error: %s", e)
        print("❌ Error writing output file:", e)
        sys.exit(7)

    print("Operation completed.")


if __name__ == "__main__":
    main()
