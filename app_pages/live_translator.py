# =============================================================================
# app_pages/live_translator.py  –  Live Translator Page (LOCAL / app.py only)
# =============================================================================

from __future__ import annotations

import time

import cv2
import numpy as np
import streamlit as st

from config import (
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    STABLE_FRAME_COUNT,
    LOW_CONFIDENCE_THRESHOLD,
    GESTURE_LABELS,
    GESTURE_MEANINGS,
    DEFAULT_LANGUAGE,
    CHARACTER_CONFIDENCE_THRESHOLD,
    CHARACTER_STABLE_FRAME_COUNT,
    CHARACTER_DEBOUNCE_FRAMES,
)
from utils.ui_components import (
    render_ai_panel,
    render_confidence_meter,
    render_suggestions,
    render_detection_status,
    render_subtitle_strip,
    camera_placeholder_html,
)
from utils.translator import translate, translate_all_languages, get_language_list
from utils.gesture_history import GestureHistory
from utils.analytics import SessionAnalytics
from preprocess import normalise_landmarks
from inference import predict, predict_character, PredictionSmoother


# =============================================================================
# Frame-level helpers
# =============================================================================

def _process_frame(frame, interp, in_idx, out_idx, label_map, scaler,
                   detector, smoother):
    mhr, annotated = detector.find_all_landmarks(frame, draw=True)
    has_hand = mhr.dominant is not None

    if not has_hand:
        smoother.reset()
        return annotated, None, 0.0, False, {}, mhr

    raw_lm   = mhr.dominant.landmarks
    features = normalise_landmarks(raw_lm)
    raw_pred, conf = predict(features, interp, in_idx, out_idx, label_map, scaler)
    gesture, confidence = smoother.update(raw_pred, conf)

    all_probs: dict = {}
    if raw_pred:
        all_probs[raw_pred] = conf
        rest = max(0.0, 1.0 - conf) / max(len(GESTURE_LABELS) - 1, 1)
        for lbl in GESTURE_LABELS:
            if lbl != raw_pred:
                all_probs[lbl] = rest

    return annotated, gesture, confidence, has_hand, all_probs, mhr


def _draw_hand_debug_strip(frame: np.ndarray, mhr) -> None:
    h, w = frame.shape[:2]
    n     = mhr.count
    l_col = (0, 220, 0) if mhr.left_detected  else (80, 80, 200)
    r_col = (0, 220, 0) if mhr.right_detected else (80, 80, 200)
    n_col = (0, 220, 0) if n > 0              else (80, 80, 200)
    strip_y = 100
    ov = frame.copy()
    cv2.rectangle(ov, (w - 220, strip_y - 14), (w - 2, strip_y + 6), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    cv2.putText(frame, f"Hands:{n}",                 (w - 216, strip_y), font, scale, n_col, thick)
    cv2.putText(frame, "L:OK" if mhr.left_detected  else "L:--", (w - 148, strip_y), font, scale, l_col, thick)
    cv2.putText(frame, "R:OK" if mhr.right_detected else "R:--", (w - 90,  strip_y), font, scale, r_col, thick)


# =============================================================================
# Sentence builder sub-component (Word mode)
# =============================================================================

def _render_sentence_builder(tts) -> None:
    words = st.session_state["sentence"]
    sentence_str = " ".join(words) if words else \
        "<span style='color:#6b8aaa;font-style:italic;'>Waiting for gestures…</span>"
    lang = st.session_state["target_language"]
    translated_sentence = translate(" ".join(words), lang) if words else ""

    st.markdown(f"""
    <div class="section-title-dark" style="margin-bottom:8px;">Sentence Builder</div>
    <div class="sentence-box">{sentence_str}</div>
    """, unsafe_allow_html=True)

    if translated_sentence and lang != DEFAULT_LANGUAGE:
        st.markdown(f"""
        <div class="trans-card" style="margin-top:8px;">
          <div class="trans-lang">{lang}</div>
          <div class="trans-text">{translated_sentence}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🔊 Speak", key="btn_speak", use_container_width=True, type="primary"):
            text_to_speak = translated_sentence or " ".join(words)
            if text_to_speak and tts:
                tts.speak(text_to_speak, force=True)
    with b2:
        if st.button("🗑 Clear", key="btn_clear", use_container_width=True):
            st.session_state["sentence"] = []
            st.session_state["subtitle_text"] = ""
            if tts:
                tts.reset_last_spoken()
            st.rerun()
    with b3:
        if st.button("⬅ Undo", key="btn_undo", use_container_width=True):
            if st.session_state["sentence"]:
                st.session_state["sentence"].pop()
                st.rerun()

    # Word chips
    if words:
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
        chips = "".join(
            f'<span style="display:inline-block;background:rgba(131,166,206,0.12);'
            f'border:1px solid rgba(131,166,206,0.3);border-radius:50px;'
            f'padding:4px 14px;margin:3px;font-size:0.82rem;color:#284155;'
            f'font-weight:600;">{w}</span>'
            for w in reversed(words[-20:])
        )
        st.markdown(
            f'<div style="line-height:2.4;margin-top:4px;">{chips}</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# Word-mode webcam loop
# =============================================================================

def _webcam_loop(frame_ph, gesture_ph, interp, in_idx, out_idx, label_map,
                 scaler, detector, tts) -> None:
    cam_idx = st.session_state.get("camera_index", CAMERA_INDEX)
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        st.error(f"Cannot open camera at index {cam_idx}.")
        st.session_state["running"] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    smoother      = PredictionSmoother()
    last_stable   = None
    stable_frames = 0
    t_prev        = time.perf_counter()

    stop_ph    = st.empty()
    stop_clicked = stop_ph.button("⏹ Stop Camera", key="stop_btn",
                                   type="secondary", use_container_width=True)

    history:   GestureHistory   = st.session_state["history_obj"]
    analytics: SessionAnalytics = st.session_state["analytics_obj"]
    det_thresh = st.session_state.get("detection_threshold", CONFIDENCE_THRESHOLD)

    try:
        while st.session_state["running"] and not stop_clicked:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            annotated, gesture, confidence, has_hand, all_probs, mhr = _process_frame(
                frame, interp, in_idx, out_idx, label_map, scaler, detector, smoother,
            )

            t_now  = time.perf_counter()
            fps    = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            analytics.record_fps(fps)

            cv2.putText(annotated, f"FPS: {fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (131, 166, 206), 2)
            if gesture:
                cv2.putText(annotated, f"{gesture} ({confidence:.0%})",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (196, 140, 179), 2)
            _draw_hand_debug_strip(annotated, mhr)

            if gesture and confidence >= det_thresh:
                if gesture == last_stable:
                    stable_frames += 1
                else:
                    stable_frames = 1
                    last_stable   = gesture

                if stable_frames == STABLE_FRAME_COUNT:
                    words = st.session_state["sentence"]
                    if not words or words[-1] != gesture:
                        lang       = st.session_state["target_language"]
                        translated = translate(gesture, lang)
                        st.session_state["sentence"].append(gesture)
                        st.session_state["subtitle_text"] = translated or gesture
                        history.add(gesture, translated or gesture, confidence, lang)
                        analytics.record_gesture(gesture, confidence)
                        if st.session_state["auto_speak"] and tts:
                            tts.speak(translated or gesture)
                            st.session_state["tts_speaking"] = True
            else:
                if stable_frames > 0:
                    st.session_state["tts_speaking"] = False
                stable_frames = 0
                last_stable   = None

            st.session_state["current_gesture"] = gesture or "—"
            st.session_state["confidence"]      = confidence
            st.session_state["stable_frames"]   = stable_frames
            st.session_state["fps"]             = fps
            st.session_state["all_probs"]       = all_probs

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            with frame_ph.container():
                st.markdown('<div class="camera-wrapper">', unsafe_allow_html=True)
                st.image(rgb, channels="RGB", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                render_detection_status(has_hand, gesture, confidence)
                if st.session_state["subtitles_on"] and st.session_state["subtitle_text"]:
                    render_subtitle_strip(st.session_state["subtitle_text"])

            current = st.session_state["current_gesture"]
            conf    = st.session_state["confidence"]
            meaning = GESTURE_MEANINGS.get(current, "—") if current != "—" else "—"

            if not has_hand:
                status = "🖐 No hand detected"
            elif stable_frames >= STABLE_FRAME_COUNT:
                status = "✅ Recognized!"
            elif gesture:
                status = f"🔍 Recognizing… ({stable_frames}/{STABLE_FRAME_COUNT})"
            else:
                status = "Scanning…"

            with gesture_ph.container():
                render_ai_panel(current, meaning, status, conf,
                                st.session_state["tts_speaking"])
                render_confidence_meter(st.session_state["all_probs"])
                if has_hand and gesture and conf < LOW_CONFIDENCE_THRESHOLD:
                    top_labels = sorted(
                        st.session_state["all_probs"].items(),
                        key=lambda x: x[1], reverse=True,
                    )
                    render_suggestions([lbl for lbl, _ in top_labels[:4]])

    finally:
        cap.release()
        st.session_state["running"]      = False
        st.session_state["tts_speaking"] = False
        stop_ph.empty()


# =============================================================================
# Character-mode webcam loop
# =============================================================================

def _char_webcam_loop(frame_ph, info_ph, c_interp, c_in, c_out,
                      c_label_map, c_scaler, detector, tts) -> None:
    cam_idx = st.session_state.get("camera_index", CAMERA_INDEX)
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        st.error(f"Cannot open camera at index {cam_idx}.")
        st.session_state["running"] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    char_smoother  = PredictionSmoother()
    stable_frames  = 0
    last_stable    = None
    debounce_count = 0
    last_accepted  = None
    t_prev         = time.perf_counter()

    conf_thr   = st.session_state.get("char_conf_threshold",    CHARACTER_CONFIDENCE_THRESHOLD)
    stable_thr = st.session_state.get("char_stable_frames_cfg", CHARACTER_STABLE_FRAME_COUNT)

    stop_ph      = st.empty()
    stop_clicked = stop_ph.button("⏹ Stop Camera", key="stop_btn_char",
                                   type="secondary", use_container_width=True)
    _loop_iter = 0

    try:
        while st.session_state["running"] and not stop_clicked:
            _loop_iter += 1
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            mhr_char, annotated = detector.find_all_landmarks(frame, draw=True)
            has_hand   = mhr_char.dominant is not None
            char_pred  = None
            confidence = 0.0

            if has_hand:
                raw_lm   = mhr_char.dominant.landmarks
                features = normalise_landmarks(raw_lm)
                raw_char, conf = predict_character(
                    features, c_interp, c_in, c_out, c_label_map, c_scaler,
                )
                if conf >= conf_thr:
                    char_pred, confidence = char_smoother.update(raw_char, conf)
                else:
                    char_smoother.reset()
            else:
                char_smoother.reset()

            if char_pred is not None:
                if char_pred == last_stable:
                    stable_frames += 1
                else:
                    stable_frames = 1
                    last_stable   = char_pred

                if stable_frames >= stable_thr:
                    if debounce_count >= CHARACTER_DEBOUNCE_FRAMES:
                        if last_accepted != char_pred:
                            st.session_state["char_sentence"].append(char_pred)
                            last_accepted = char_pred
                        debounce_count = 0
                    else:
                        debounce_count += 1
                else:
                    debounce_count += 1
            else:
                stable_frames  = 0
                last_stable    = None
                debounce_count = min(debounce_count + 1, CHARACTER_DEBOUNCE_FRAMES + 5)

            t_now  = time.perf_counter()
            fps    = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            h, w = annotated.shape[:2]
            ov = annotated.copy()
            cv2.rectangle(ov, (0, 0), (w, 90), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, annotated, 0.45, 0, annotated)

            color = (0, 220, 0) if char_pred else (80, 80, 220)
            label = char_pred if char_pred else ("No hand" if not has_hand else "Low conf")
            cv2.putText(annotated, f"[Character] {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if char_pred:
                cv2.putText(annotated, f"{confidence:.0%}", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

            stable_pct = min(stable_frames / max(stable_thr, 1), 1.0)
            bar_w = w - 24
            cv2.rectangle(annotated, (12, 68), (12 + bar_w, 80), (40, 40, 40), -1)
            fill_w = int(bar_w * stable_pct)
            if fill_w > 0:
                cv2.rectangle(annotated, (12, 68), (12 + fill_w, 80),
                              (131, 166, 206), -1)

            dot_color = (131, 166, 206) if debounce_count >= CHARACTER_DEBOUNCE_FRAMES else (196, 140, 179)
            cv2.circle(annotated, (w - 18, 18), 7, dot_color, -1)
            cv2.putText(annotated, f"FPS: {fps:.0f}", (w - 110, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (131, 166, 206), 1)

            ov2 = annotated.copy()
            cv2.rectangle(ov2, (0, h - 40), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(ov2, 0.55, annotated, 0.45, 0, annotated)
            sentence_txt = "".join(st.session_state["char_sentence"]) or "—"
            cv2.putText(annotated, f"Text: {sentence_txt}", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (229, 201, 215), 1)

            _draw_hand_debug_strip(annotated, mhr_char)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            with frame_ph.container():
                st.image(rgb, channels="RGB", use_column_width=True)

            char_sentence = st.session_state["char_sentence"]
            full_text = "".join(char_sentence)
            with info_ph.container():
                st.markdown(f"""
                <div class="prediction-card" style="padding:22px 20px;">
                  <div class="prediction-label">Character Mode</div>
                  <div class="prediction-gesture" style="font-size:4rem;
                       letter-spacing:0.1em;">{char_pred or "—"}</div>
                  <div style="font-size:0.82rem;color:rgba(255,255,255,0.55);
                               margin-top:8px;">Confidence: {confidence:.0%}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="section-title-dark" style="margin-bottom:6px;">Built Text</div>
                <div class="sentence-box" style="font-family:monospace;letter-spacing:0.12em;
                             font-size:1.4rem;min-height:52px;">
                  {full_text if full_text else "<span style='color:#6b8aaa;font-style:italic;'>Waiting…</span>"}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    if st.button("🗑 Clear", key=f"char_clear_{_loop_iter}"):
                        st.session_state["char_sentence"] = []
                        last_accepted = None
                with b2:
                    if st.button("⌫ Back", key=f"char_back_{_loop_iter}"):
                        if st.session_state["char_sentence"]:
                            st.session_state["char_sentence"].pop()
                            last_accepted = None
                with b3:
                    if st.button("⎵ Space", key=f"char_space_{_loop_iter}"):
                        st.session_state["char_sentence"].append(" ")
                        last_accepted = None
                with b4:
                    if st.button("🔊 Speak", key=f"char_speak_{_loop_iter}"):
                        if full_text.strip() and tts:
                            tts.speak(full_text.strip(), force=True)

    finally:
        cap.release()
        st.session_state["running"] = False
        stop_ph.empty()


# =============================================================================
# Page entry point  (called by app.py)
# =============================================================================

def show_live_translator(interp, in_idx, out_idx, label_map, scaler, detector, tts,
                         c_interp, c_in, c_out, c_label_map, c_scaler) -> None:
    """Full Live Translator page layout for the local app."""

    # ── Top Bar with Return to Home ───────────────────────────────────────────
    top_col1, _ = st.columns([1, 4])
    with top_col1:
        if st.button("← Return to Home", key="live_back_home", type="secondary"):
            st.session_state["page"] = "🏠 Home"
            st.rerun()

    # ── Page Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:24px;">
      <div style="font-size:0.72rem;font-weight:700;letter-spacing:2.5px;
                  text-transform:uppercase;color:#83A6CE;margin-bottom:8px;">
        AI Recognition
      </div>
      <h1 style="font-size:2rem;font-weight:900;color:#284155;margin:0 0 6px;
                 letter-spacing:-0.3px;">Live Translator</h1>
      <p style="color:#6b8aaa;font-size:0.9rem;margin:0;">
        Hold a sign steady for a moment to add it to your sentence.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls row ───────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])

    with ctrl1:
        mode = st.radio(
            "Recognition Mode",
            ["Word", "Character"],
            index=0 if st.session_state["recognition_mode"] == "Word" else 1,
            horizontal=True,
            key="mode_radio",
        )
        if mode != st.session_state["recognition_mode"]:
            st.session_state["recognition_mode"] = mode
            st.session_state["running"] = False
            st.rerun()

    with ctrl2:
        if mode == "Word":
            lang_list = get_language_list()
            cur_idx   = lang_list.index(st.session_state["target_language"])
            new_lang  = st.selectbox("🌍 Language", lang_list, index=cur_idx, key="live_lang")
            st.session_state["target_language"] = new_lang

    with ctrl3:
        auto = st.checkbox("Auto-Speak", value=st.session_state["auto_speak"],
                           key="live_auto_speak")
        st.session_state["auto_speak"] = auto

    st.markdown('<hr style="margin:16px 0;">', unsafe_allow_html=True)

    # ── Two-column layout: Camera (left) | Info (right) ────────────────────────
    cam_col, info_col = st.columns([3, 2], gap="large")

    with cam_col:
        frame_ph = st.empty()

        if not st.session_state["running"]:
            frame_ph.markdown(camera_placeholder_html(), unsafe_allow_html=True)

            if mode == "Word" and interp is None:
                st.markdown(
                    '<div class="clay-error">⚠ Word model not loaded — run train_model.py first.</div>',
                    unsafe_allow_html=True,
                )
            elif mode == "Character" and c_interp is None:
                st.markdown(
                    '<div class="clay-error">⚠ Character model not loaded — run train_character_model.py first.</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button("▶  Start Camera", type="primary",
                             use_container_width=True, key="btn_start"):
                    st.session_state["running"] = True
                    st.rerun()

    with info_col:
        gesture_ph = st.empty()

        if not st.session_state["running"]:
            if mode == "Word":
                render_ai_panel("—", "Ready", "Camera off", 0.0, False)
            else:
                st.markdown("""
                <div class="prediction-card" style="padding:24px 20px;">
                  <div class="prediction-label">Character Mode</div>
                  <div class="prediction-gesture" style="font-size:2.4rem;
                       color:rgba(255,255,255,0.4);">A–Z  0–9</div>
                  <div style="font-size:0.86rem;color:rgba(255,255,255,0.5);margin-top:10px;">
                    Start the camera and hold a hand sign steady to spell characters.
                  </div>
                </div>
                """, unsafe_allow_html=True)

        if mode == "Word":
            st.markdown('<hr style="margin:12px 0;">', unsafe_allow_html=True)
            _render_sentence_builder(tts)

            last_gesture = (
                st.session_state["sentence"][-1]
                if st.session_state["sentence"] else None
            )
            if last_gesture:
                st.markdown('<hr style="margin:12px 0;">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-title-dark" style="margin-bottom:8px;">Translations</div>',
                    unsafe_allow_html=True,
                )
                translations = translate_all_languages(last_gesture)
                limited = dict(list(translations.items())[:6])
                from utils.ui_components import render_translation_cards
                render_translation_cards(limited)

    # ── Launch webcam loop ─────────────────────────────────────────────────────
    if st.session_state["running"]:
        if mode == "Word" and interp is not None:
            _webcam_loop(frame_ph, gesture_ph,
                         interp, in_idx, out_idx, label_map, scaler, detector, tts)
        elif mode == "Character" and c_interp is not None:
            _char_webcam_loop(frame_ph, gesture_ph,
                              c_interp, c_in, c_out, c_label_map, c_scaler, detector, tts)
