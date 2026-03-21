use thiserror::Error;

#[derive(Debug, Error)]
pub enum ServerError {
    #[error(transparent)]
    Wire(#[from] pronghorn_wire::WireError),

    #[error(transparent)]
    Pipeline(#[from] pronghorn_pipeline::PipelineError),

    #[error(transparent)]
    Config(#[from] crate::config::ConfigError),

    #[error("STT backend initialization failed: {0}")]
    Stt(#[from] pronghorn_pipeline::SttError),

    #[error("TTS backend initialization failed: {0}")]
    Tts(#[from] pronghorn_pipeline::TtsError),

    #[error("intent backend initialization failed: {0}")]
    Intent(#[from] pronghorn_pipeline::IntentError),
}
