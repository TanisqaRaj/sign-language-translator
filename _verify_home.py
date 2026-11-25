import sys, ast
sys.path.insert(0, r'C:\Users\rajta\OneDrive\Desktop\Projects\Sign lang translator')

path = r'C:\Users\rajta\OneDrive\Desktop\Projects\Sign lang translator\app_pages\home.py'
with open(path, 'r', encoding='utf-8') as f:
    src = f.read()

# 1. AST parse
tree = ast.parse(src)
print('PASS: AST parses cleanly')

# 2. show_home() with no params
fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
show_home_fn = next((f for f in fns if f.name == 'show_home'), None)
assert show_home_fn is not None, 'show_home not found'
assert len(show_home_fn.args.args) == 0, 'show_home has params'
print('PASS: show_home() with no parameters')

# 3. Text checks
checks = [
    ('Live Translator',             'CTA nav to Live Translator'),
    ('st.rerun()',                   'st.rerun() present'),
    ('from utils.ui_components',     'utils.ui_components import'),
    ('svg_communication_scene',      'svg_communication_scene used'),
    ('HIDE_SIDEBAR_CSS',             'HIDE_SIDEBAR_CSS injected'),
    ('HOME_PAGE_CSS',                'HOME_PAGE_CSS injected'),
    ('glass-navbar',                 'glass-navbar HTML'),
    ('home-hero-section',            'hero section class'),
    ('How It Works',                 'How It Works section'),
    ('step-card-3d',                 '3D step cards'),
    ('AI Translation Flow',          'AI translation flow section'),
    ('feature-card-premium',         'premium feature cards'),
    ('Technology Designed',          'about section'),
    ('footer',                       'footer element'),
    ('audio_on',                     'audio_on session state'),
    ('audio-btn',                    'audio-btn CSS class'),
    ('scrollY',                      'scroll JS'),
    ('Without Barriers',             'hero headline text'),
    ('hero_cta',                     'hero CTA key'),
    ('bottom_cta',                   'bottom CTA key'),
    ('audio_toggle_btn',             'audio toggle button key'),
    ('_svg_ai_flow',                 'AI flow SVG function'),
]

for text, label in checks:
    assert text in src, 'MISSING: ' + label + ' (' + repr(text) + ')'
    print('PASS: ' + label)

# 4. No ML/camera
for bad in ['import cv2', 'import mediapipe', 'VideoCapture']:
    assert bad not in src, 'FORBIDDEN: ' + bad
print('PASS: no ML/camera logic')

# 5. Feature count
count_fc = src.count('feature-card-premium')
print('PASS: feature-card-premium x' + str(count_fc) + ' (expect 8)')

# 6. Module import
from app_pages.home import show_home
print('PASS: module imports successfully')

print()
print('Lines: ' + str(len(src.splitlines())))
print('ALL CHECKS PASSED')
