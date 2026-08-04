"""Fix all non-ASCII characters in convert_tflite.py for Windows cp1252 terminal."""
import re

with open('convert_tflite.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace known emoji/unicode chars
replacements = [
    ('\U0001f504', ''),   # rotating arrows
    ('\u2714', '[OK]'),   # checkmark
    ('\u274c', '[ERR]'),  # cross
    ('\u2705', '[OK]'),   # green check
    ('\u2500', '-'),      # box drawing
    ('\u2013', '-'),      # en dash
    ('\u2026', '...'),    # ellipsis
    ('\u2019', "'"),      # right quote
    ('\u2018', "'"),      # left quote
    ('\u2014', '--'),     # em dash
    ('\u00e2', ''),       # misc
]
for old, new in replacements:
    content = content.replace(old, new)

# Final safety: encode to ascii ignoring anything remaining
lines = content.split('\n')
clean_lines = []
for line in lines:
    try:
        line.encode('cp1252')
        clean_lines.append(line)
    except UnicodeEncodeError:
        # Replace non-cp1252 chars with closest ASCII
        clean_lines.append(line.encode('ascii', 'replace').decode('ascii'))

content = '\n'.join(clean_lines)

with open('convert_tflite.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('convert_tflite.py fixed successfully.')
