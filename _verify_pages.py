import ast, sys

errors = []

# ── analytics.py ──────────────────────────────────────────────────────────────
with open('app_pages/analytics.py', encoding='utf-8') as f:
    src_a = f.read()

try:
    ast.parse(src_a)
    print('analytics.py  → syntax OK')
except SyntaxError as e:
    errors.append(f'analytics.py SYNTAX ERROR: {e}')

checks_a = [
    'show_analytics',
    'analytics_key',
    'history_key',
    'use_plotly',
    'SessionAnalytics',
    'svg_analytics',
    'snap = analytics.snapshot()',
    'snap.total_gestures',
    'snap.recognition_accuracy',
    'snap.average_confidence',
    'snap.current_fps',
    'analytics.format_duration()',
    'snap.most_frequent_gesture',
    'snap.high_confidence_count',
    'snap.low_confidence_count',
    'snap.gesture_frequency',
    'snap.fps_history',
    'plotly.graph_objects',
    'go.Scatter',
    'go.Pie',
    'reset_analytics',
    'analytics.reset()',
    'st.rerun()',
    'conf-track',
    'conf-fill-high',
    'clay-card',
    '#83A6CE',
    '#C48CB3',
    'analytics-hero',
    'fade-in-up',
    'chart-glass',
]

for needle in checks_a:
    if needle not in src_a:
        errors.append(f'analytics.py MISSING: {needle!r}')

# ── settings.py ───────────────────────────────────────────────────────────────
with open('app_pages/settings.py', encoding='utf-8') as f:
    src_s = f.read()

try:
    ast.parse(src_s)
    print('settings.py   → syntax OK')
except SyntaxError as e:
    errors.append(f'settings.py SYNTAX ERROR: {e}')

checks_s = [
    'def show_settings_local(tts)',
    'def show_settings_cloud()',
    'CONFIDENCE_THRESHOLD',
    'STABLE_FRAME_COUNT',
    'CAMERA_INDEX',
    'DEFAULT_LANGUAGE',
    'DEFAULT_THEME',
    'DEFAULT_FONT_SIZE',
    'DEFAULT_HIGH_CONTRAST',
    'AUTO_SPEAK_DEFAULT',
    'SUBTITLE_ENABLED_DEFAULT',
    'SUPPORTED_LANGUAGES',
    'CHARACTER_CONFIDENCE_THRESHOLD',
    'CHARACTER_STABLE_FRAME_COUNT',
    'get_language_list',
    'tab_ui, tab_cam, tab_tts, tab_trans',
    'Interface',
    'Camera',
    'Speech',
    'Translation',
    # session state keys
    'session_state["theme"]',
    'session_state["large_font"]',
    'session_state["high_contrast"]',
    'session_state["subtitles_on"]',
    'session_state["camera_index"]',
    'session_state["detection_threshold"]',
    'session_state["auto_speak"]',
    'session_state["speech_rate"]',
    'session_state["voice_index"]',
    'session_state["target_language"]',
    'tts.get_voices()',
    'tts.set_voice_by_index',
    'session_state["language"]',
    'session_state["font_size"]',
    'session_state["conf_threshold"]',
    'session_state["stable_frames_cfg"]',
    'session_state["char_conf_threshold"]',
    'session_state["char_stable_frames_cfg"]',
    'reset_settings',
    'DEFAULT_LANGUAGE',
    'st.rerun()',
    'inject_css',
    'settings-hero',
    'settings-card',
    'theme-preview-row',
    'camera-diagram',
    'speed-bar-row',
    '#83A6CE',
    '#C48CB3',
    '_lang_flag',
    '_speed_bars_html',
]

for needle in checks_s:
    if needle not in src_s:
        errors.append(f'settings.py MISSING: {needle!r}')

print()
if errors:
    print(f'FAILURES ({len(errors)}):')
    for e in errors:
        print(' -', e)
    sys.exit(1)
else:
    total = len(checks_a) + len(checks_s)
    print(f'ALL {total} CHECKS PASSED ✓')
