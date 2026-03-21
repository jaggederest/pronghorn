use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("STT error: {0}")]
    Stt(#[from] crate::stt::SttError),

    #[error("TTS error: {0}")]
    Tts(#[from] crate::tts::TtsError),

    #[error("intent error: {0}")]
    Intent(#[from] crate::intent::IntentError),
}
