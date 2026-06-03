"""
Fix Unicode encoding issues in all Python files for Windows compatibility.
"""

from pathlib import Path
import re

replacements = {
    '->': '->',
    '[OK]': '[OK]',
    '-': '-',
    '*': '*',
    '[TROPHY]': '[TROPHY]',
    '[PKG]': '[PKG]',
    '[SOCCER]': '[SOCCER]',
}

python_files = Path('.').rglob('*.py')

for file in python_files:
    try:
        content = file.read_text(encoding='utf-8')
        original = content

        for unicode_char, ascii_replacement in replacements.items():
            content = content.replace(unicode_char, ascii_replacement)

        if content != original:
            file.write_text(content, encoding='utf-8')
            print(f"[FIXED] {file}")
        else:
            print(f"[OK] {file}")
    except Exception as e:
        print(f"[ERROR] {file}: {e}")

print("\nDone!")
