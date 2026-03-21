use tracing::info;

use crate::config::OllamaConfig;
use crate::intent::{IntentError, IntentProcessor, IntentResponse};

/// Intent processor backed by a local Ollama LLM.
///
/// Sends the transcript to Ollama's generate API and returns the
/// LLM's conversational response as reply_text for TTS.
pub struct OllamaIntent {
    url: String,
    model: String,
    system_prompt: String,
    #[cfg(feature = "ollama")]
    client: reqwest::Client,
}

impl OllamaIntent {
    pub fn new(config: &OllamaConfig) -> Result<Self, IntentError> {
        info!(
            url = %config.url,
            model = %config.model,
            "ollama intent backend initialized"
        );
        Ok(Self {
            url: config.url.clone(),
            model: config.model.clone(),
            system_prompt: config.system_prompt.clone(),
            #[cfg(feature = "ollama")]
            client: reqwest::Client::new(),
        })
    }
}

impl IntentProcessor for OllamaIntent {
    async fn process(&self, transcript: &str) -> Result<IntentResponse, IntentError> {
        info!(transcript = %transcript, model = %self.model, "querying ollama");

        #[cfg(feature = "ollama")]
        {
            let request_body = serde_json::json!({
                "model": self.model,
                "system": self.system_prompt,
                "prompt": transcript,
                "stream": false,
            });

            let response = self
                .client
                .post(format!("{}/api/generate", self.url))
                .json(&request_body)
                .send()
                .await
                .map_err(|e| IntentError::Processing(format!("ollama request failed: {e}")))?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_else(|_| "unknown".into());
                return Err(IntentError::Processing(format!(
                    "ollama returned {status}: {body}"
                )));
            }

            let body: serde_json::Value = response.json().await.map_err(|e| {
                IntentError::Processing(format!("ollama response parse error: {e}"))
            })?;

            let reply_text = body["response"]
                .as_str()
                .unwrap_or("I didn't understand that.")
                .trim()
                .to_string();

            info!(reply = %reply_text, "ollama response");

            Ok(IntentResponse {
                reply_text,
                action: None,
            })
        }

        #[cfg(not(feature = "ollama"))]
        {
            let _ = (transcript, &self.url, &self.model, &self.system_prompt);
            Err(IntentError::Processing("ollama feature not enabled".into()))
        }
    }
}
