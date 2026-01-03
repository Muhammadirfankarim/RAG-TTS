"""
Speech-to-Text Processor Module

This module handles audio transcription using faster-whisper.
Uses two-pass transcription for better language detection accuracy.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
TEMP_AUDIO_DIR = "temp_audio"

# Global model cache
_whisper_model = None
_current_model_name = None


def load_whisper_model(model_name: str = "small"):
    """
    Load Whisper model. Using 'medium' for better language detection accuracy.
    
    Args:
        model_name: Whisper model name ('tiny', 'base', 'small', 'medium', 'large')
        
    Returns:
        Loaded Whisper model
    """
    global _whisper_model, _current_model_name
    
    # Always load fresh model if name changed
    if _whisper_model is None or _current_model_name != model_name:
        from faster_whisper import WhisperModel
        
        logger.info(f"Loading Whisper model: {model_name}")
        _whisper_model = WhisperModel(
            model_name, 
            device="cpu", 
            compute_type="int8"
        )
        _current_model_name = model_name
        logger.info(f"Whisper '{model_name}' model loaded successfully")
    
    return _whisper_model


def detect_language(audio_file_path: str, model_name: str = "small") -> Tuple[str, float]:
    """
    Detect language from audio using Whisper's language detection.
    Uses only the first 30 seconds for faster detection.
    
    Args:
        audio_file_path: Path to audio file
        model_name: Whisper model to use
        
    Returns:
        Tuple of (language_code, probability)
    """
    model = load_whisper_model(model_name)
    
    # Use detect_language_multi_segment for better accuracy
    # This samples multiple parts of the audio
    segments, info = model.transcribe(
        audio_file_path,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=True,
        without_timestamps=True,
    )
    
    # Consume the generator to get info
    _ = list(segments)
    
    detected_lang = info.language if hasattr(info, 'language') else "en"
    lang_prob = info.language_probability if hasattr(info, 'language_probability') else 0.0
    
    logger.info(f"Language detection: {detected_lang} (prob: {lang_prob:.2f})")
    
    return detected_lang, lang_prob


def transcribe_audio(
    audio_file_path: str,
    model_name: str = "small",
    language: Optional[str] = None
) -> dict:
    """
    Transcribe audio file to text using Whisper.
    Uses two-pass approach for auto-detect: first detect language, then transcribe.
    
    Args:
        audio_file_path: Path to audio file (WAV, MP3, etc.)
        model_name: Whisper model to use
        language: Optional language code (e.g., 'en', 'id'). If None, auto-detects.
        
    Returns:
        Dictionary with 'text', 'language', 'language_probability', and 'segments' keys
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
    """
    path = Path(audio_file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    logger.info(f"Transcribing audio: {audio_file_path}")
    
    try:
        model = load_whisper_model(model_name)
        
        # If no language specified, use two-pass approach
        detected_language = language
        language_probability = 1.0
        
        if language is None:
            # PASS 1: Detect language first
            logger.info("Pass 1: Detecting language...")
            detected_language, language_probability = detect_language(audio_file_path, model_name)
            
            # If probability is low, try both English and Indonesian and compare
            if language_probability < 0.7:
                logger.info(f"Low confidence ({language_probability:.2f}), trying both languages...")
                
                # Try English
                segments_en, _ = model.transcribe(
                    str(path),
                    language="en",
                    beam_size=3,
                    temperature=0.0,
                    vad_filter=True,
                )
                text_en = " ".join([s.text.strip() for s in segments_en])
                
                # Try Indonesian
                segments_id, _ = model.transcribe(
                    str(path),
                    language="id",
                    beam_size=3,
                    temperature=0.0,
                    vad_filter=True,
                )
                text_id = " ".join([s.text.strip() for s in segments_id])
                
                # Simple heuristic: English text typically has more English-specific patterns
                # Check for common English words vs Indonesian words
                en_indicators = ["the", "is", "are", "what", "how", "this", "that", "in", "of", "for"]
                id_indicators = ["yang", "dan", "ini", "itu", "apa", "adalah", "dari", "untuk", "ke", "di"]
                
                en_score = sum(1 for word in en_indicators if word in text_en.lower().split())
                id_score = sum(1 for word in id_indicators if word in text_id.lower().split())
                
                logger.info(f"EN score: {en_score}, ID score: {id_score}")
                
                if en_score > id_score:
                    detected_language = "en"
                    language_probability = 0.8
                elif id_score > en_score:
                    detected_language = "id"
                    language_probability = 0.8
                # else keep the original detection
            
            logger.info(f"Final detected language: {detected_language} (prob: {language_probability:.2f})")
        
        # PASS 2: Transcribe with detected/specified language
        logger.info(f"Pass 2: Transcribing with language={detected_language}...")
        
        transcribe_options = {
            "language": detected_language,
            "beam_size": 5,
            "best_of": 5,
            "patience": 1.0,
            "temperature": 0.0,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": True,
            "vad_filter": True,
            "vad_parameters": {
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            }
        }
        
        segments, info = model.transcribe(str(path), **transcribe_options)
        
        # Collect all segments
        segment_list = []
        text_parts = []
        for segment in segments:
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            text_parts.append(segment.text.strip())
        
        transcribed_text = " ".join(text_parts).strip()
        
        logger.info(f"Transcription complete. Language: {detected_language} (prob: {language_probability:.2f})")
        logger.info(f"Transcribed text: {transcribed_text[:100]}...")
        
        return {
            "text": transcribed_text,
            "language": detected_language,
            "language_probability": language_probability,
            "segments": segment_list
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise Exception(f"Failed to transcribe audio: {str(e)}")


def save_uploaded_audio(
    audio_bytes: bytes,
    filename: str = "temp_audio.wav",
    save_dir: str = TEMP_AUDIO_DIR
) -> str:
    """
    Save uploaded audio bytes to a temporary file.
    
    Args:
        audio_bytes: Audio data as bytes
        filename: Name of the file to save
        save_dir: Directory to save the file
        
    Returns:
        Full path to the saved file
    """
    # Create directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Full file path
    file_path = save_path / filename
    
    try:
        with open(file_path, "wb") as f:
            f.write(audio_bytes)
        
        logger.info(f"Audio saved to: {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Error saving audio: {str(e)}")
        raise Exception(f"Failed to save audio file: {str(e)}")


def cleanup_temp_audio(save_dir: str = TEMP_AUDIO_DIR) -> int:
    """
    Clean up temporary audio files.
    
    Args:
        save_dir: Directory containing temp audio files
        
    Returns:
        Number of files deleted
    """
    save_path = Path(save_dir)
    
    if not save_path.exists():
        return 0
    
    deleted_count = 0
    for audio_file in save_path.glob("*.wav"):
        try:
            audio_file.unlink()
            deleted_count += 1
            logger.info(f"Deleted: {audio_file}")
        except Exception as e:
            logger.warning(f"Could not delete {audio_file}: {str(e)}")
    
    for audio_file in save_path.glob("*.mp3"):
        try:
            audio_file.unlink()
            deleted_count += 1
        except Exception as e:
            logger.warning(f"Could not delete {audio_file}: {str(e)}")
    
    return deleted_count


if __name__ == "__main__":
    # Test the module
    print("STT Processor module loaded successfully")
    print("Use transcribe_audio(audio_path) to transcribe audio")
    
    # Load model for testing
    try:
        model = load_whisper_model("medium")
        print("Whisper 'medium' model loaded successfully")
    except Exception as e:
        print(f"Could not load model: {e}")
