use thiserror::Error;

#[derive(Debug, Error)]
pub enum ServerError {
    #[error(transparent)]
    Wire(#[from] pronghorn_wire::WireError),

    #[error(transparent)]
    Pipeline(#[from] pronghorn_pipeline::PipelineError),

    #[error(transparent)]
    Config(#[from] crate::config::ConfigError),
}
