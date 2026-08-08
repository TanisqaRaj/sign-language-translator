# =============================================================================
# utils/translator.py
# Multi-Language Translation Utility
# =============================================================================
# Wraps deep-translator's GoogleTranslator to translate English gesture text
# into any of the 13 configured languages.
#
# Design decisions:
#   • LRU-style in-process cache avoids repeating network calls for the same
#     (text, target_language) pair during a session.
#   • Falls back to the original English text if translation fails, so the
#     app never crashes due to a transient network error.
#   • Thread-safe cache using a simple dict + lock.
# =============================================================================

import threading
from functools import lru_cache
from typing import Optional

try:
    from deep_translator import GoogleTranslator
    _DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    _DEEP_TRANSLATOR_AVAILABLE = False

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE
from utils.logger import get_logger

logger = get_logger(__name__)


class TranslationCache:
    """
    Thread-safe, in-memory translation cache.
    Key: (text, target_lang_code)  →  Value: translated string
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def get(self, text: str, lang_code: str) -> Optional[str]:
        with self._lock:
            return self._cache.get((text, lang_code))

    def set(self, text: str, lang_code: str, translation: str) -> None:
        with self._lock:
            self._cache[(text, lang_code)] = translation

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


# Module-level shared cache instance
_cache = TranslationCache()


def translate(
    text: str,
    target_language: str = DEFAULT_LANGUAGE,
    source_language: str = "English",
) -> str:
    """
    Translate ``text`` from ``source_language`` to ``target_language``.

    Parameters
    ----------
    text            : str  English text to translate (e.g. "Hello").
    target_language : str  Display name from SUPPORTED_LANGUAGES (e.g. "Hindi").
    source_language : str  Source language display name (default: "English").

    Returns
    -------
    str  Translated text, or the original text if translation fails / not needed.

    Examples
    --------
    >>> translate("Hello", "Hindi")
    'नमस्ते'
    >>> translate("Thank You", "Spanish")
    'Gracias'
    """
    if not text or not text.strip():
        return text

    # No translation needed for same-language pair
    if target_language == source_language or target_language == DEFAULT_LANGUAGE:
        return text

    target_code = SUPPORTED_LANGUAGES.get(target_language)
    if not target_code:
        logger.warning("Unknown target language: %s", target_language)
        return text

    source_code = SUPPORTED_LANGUAGES.get(source_language, "en")

    # Check cache first
    cached = _cache.get(text, target_code)
    if cached is not None:
        logger.debug("Translation cache hit: '%s' → %s", text, target_language)
        return cached

    if not _DEEP_TRANSLATOR_AVAILABLE:
        logger.warning("deep-translator not installed; returning original text.")
        return text

    try:
        translator = GoogleTranslator(source=source_code, target=target_code)
        result = translator.translate(text)
        if result:
            _cache.set(text, target_code, result)
            logger.info("Translated '%s' → '%s' (%s)", text, result, target_language)
            return result
        return text
    except Exception as exc:
        logger.warning(
            "Translation failed for '%s' → %s: %s. Returning original.",
            text, target_language, exc,
        )
        return text


def translate_sentence(
    words: list[str],
    target_language: str = DEFAULT_LANGUAGE,
) -> str:
    """
    Translate a list of English words (sentence) into the target language.

    Translates the joined sentence as one unit for better grammar compared
    to word-by-word translation.

    Parameters
    ----------
    words           : list[str]  English words, e.g. ["Hello", "Thank You"]
    target_language : str        Display name from SUPPORTED_LANGUAGES

    Returns
    -------
    str  Translated sentence.
    """
    if not words:
        return ""
    sentence = " ".join(words)
    return translate(sentence, target_language)


def translate_all_languages(text: str) -> dict[str, str]:
    """
    Translate ``text`` into all configured languages.

    Useful for rendering the multi-language translation cards panel.

    Parameters
    ----------
    text : str  English word/phrase.

    Returns
    -------
    dict[str, str]  Mapping of language display name → translated text.
    """
    results = {}
    for lang_name in SUPPORTED_LANGUAGES:
        results[lang_name] = translate(text, lang_name)
    return results


def get_language_list() -> list[str]:
    """Return sorted list of supported language display names."""
    return list(SUPPORTED_LANGUAGES.keys())


def clear_cache() -> None:
    """Clear the translation cache (useful between sessions)."""
    _cache.clear()
    logger.debug("Translation cache cleared.")


def cache_size() -> int:
    """Return the number of cached translations."""
    return _cache.size()
