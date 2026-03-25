use std::future::Future;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The response from intent processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentResponse {
    /// Text to speak back to the user.
    pub reply_text: String,
    /// Optional structured action (future: HA device commands).
    pub action: Option<IntentAction>,
}

/// A structured action to execute (e.g., turn on a light).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentAction {
    pub domain: String,
    pub service: String,
    pub entity_id: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Error)]
pub enum IntentError {
    #[error("intent processing failed: {0}")]
    Processing(String),
}

/// Intent processing pipeline stage.
///
/// Takes a transcript and produces a response (text to speak + optional action).
pub trait IntentProcessor: Send + Sync {
    fn process(
        &self,
        transcript: &str,
    ) -> impl Future<Output = Result<IntentResponse, IntentError>> + Send;
}

/// Echo intent backend for development and testing.
///
/// Returns the transcript as the reply text with no action.
pub struct EchoIntent;

impl IntentProcessor for EchoIntent {
    async fn process(&self, transcript: &str) -> Result<IntentResponse, IntentError> {
        Ok(IntentResponse {
            reply_text: format!("You said: {transcript}"),
            action: None,
        })
    }
}

/// Echo intent backend that always returns a structured HA action.
///
/// Returns a `light.turn_on` action for `light.living_room` plus a brief confirmation
/// phrase. Useful for testing the orchestrator's action dispatch branch without a
/// live HA server.
pub struct EchoActionIntent;

impl IntentProcessor for EchoActionIntent {
    async fn process(&self, _transcript: &str) -> Result<IntentResponse, IntentError> {
        Ok(IntentResponse {
            reply_text: "Done.".into(),
            action: Some(IntentAction {
                domain: "light".into(),
                service: "turn_on".into(),
                entity_id: "light.living_room".into(),
                data: serde_json::Value::Null,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn echo_intent_echoes_transcript() {
        let intent = EchoIntent;
        let response = intent.process("turn on the lights").await.unwrap();
        assert_eq!(response.reply_text, "You said: turn on the lights");
        assert!(response.action.is_none());
    }

    #[tokio::test]
    async fn echo_action_intent_returns_action() {
        let intent = EchoActionIntent;
        let response = intent.process("turn on the lights").await.unwrap();
        assert_eq!(response.reply_text, "Done.");
        let action = response.action.unwrap();
        assert_eq!(action.domain, "light");
        assert_eq!(action.service, "turn_on");
        assert_eq!(action.entity_id, "light.living_room");
    }
}
