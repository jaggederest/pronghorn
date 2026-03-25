//! Progressive intent resolver: trie fast path + fuzzy entity matching + LLM fallback.
//!
//! Enabled by the `progressive` feature (which pulls in `hassil` + `strsim`).
//!
//! ## Resolution strategy
//!
//! 1. **Fast path** — walk the compiled Hassil word-trie word-by-word.
//!    If the trie resolves to a unique intent with all slots filled, the action
//!    executes in microseconds — no network round-trip.
//! 2. **Fuzzy entity matching** — slot values (e.g., `"kitchen lights"`) are
//!    matched against the HA entity registry using Jaro-Winkler similarity to
//!    produce a real entity ID (e.g., `"light.kitchen_ceiling"`).
//! 3. **LLM fallback** — if the trie produces no match the full transcript is
//!    forwarded to the Ollama backend.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use pronghorn_pipeline::resolver::{EntityIndex, ProgressiveResolver};
//! use pronghorn_pipeline::ha_client::EntityInfo;
//! use pronghorn_pipeline::intent::IntentProcessor;
//!
//! let entities: Vec<EntityInfo> = vec![/* from ha_client.entities() */];
//! let resolver = ProgressiveResolver::from_config(&intent_config, entities)?;
//! let response = resolver.process("turn on the kitchen lights").await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::config::IntentConfig;
use crate::ha_client::EntityInfo;
use crate::hassil::{TrieWalker, WordTrie, hassil_intent_to_service, load_trie_from_config};
use crate::intent::{IntentAction, IntentError, IntentProcessor, IntentResponse};
use crate::ollama::OllamaIntent;

// ── Entity index ──────────────────────────────────────────────────────────────

/// Searchable index of Home Assistant entities.
///
/// Build from a slice of [`EntityInfo`] fetched via [`crate::ha_client::HaClient::entities`].
/// Empty index is valid — fuzzy matching returns `None`, and the raw slot value is used as-is.
pub struct EntityIndex {
    /// Normalized name → entity_id (for O(1) exact lookups).
    exact: HashMap<String, String>,
    /// All (normalized_name, entity_id) pairs for fuzzy search.
    corpus: Vec<(String, String)>,
}

impl EntityIndex {
    /// Create an empty index (no fuzzy matching, raw slot values pass through).
    pub fn empty() -> Self {
        EntityIndex {
            exact: HashMap::new(),
            corpus: Vec::new(),
        }
    }

    /// Build an index from a slice of HA entities.
    ///
    /// Indexes each entity's friendly name, entity_id (underscore-split), and aliases.
    pub fn from_entities(entities: &[EntityInfo]) -> Self {
        let mut index = EntityIndex::empty();
        for entity in entities {
            let id = &entity.entity_id;

            // Friendly name (e.g., "Kitchen Ceiling Light")
            if let Some(name) = &entity.name {
                index.add(name, id);
            }

            // Entity ID domain-stripped (e.g., "light.kitchen_ceiling" → "kitchen ceiling")
            if let Some(local) = id.split('.').nth(1) {
                let human = local.replace('_', " ");
                index.add(&human, id);
            }

            // User-defined aliases
            for alias in &entity.aliases {
                index.add(alias, id);
            }
        }
        index
    }

    fn add(&mut self, name: &str, entity_id: &str) {
        let normalized = Self::normalize(name);
        if normalized.is_empty() {
            return;
        }
        self.exact
            .entry(normalized.clone())
            .or_insert_with(|| entity_id.to_string());
        self.corpus.push((normalized, entity_id.to_string()));
    }

    /// Resolve a query to an entity_id.
    ///
    /// Returns the entity_id with the highest Jaro-Winkler similarity to `query`
    /// if that score is at or above `threshold`, otherwise `None`.
    pub fn resolve(&self, query: &str, threshold: f64) -> Option<String> {
        let normalized = Self::normalize(query);

        // Exact match (O(1))
        if let Some(id) = self.exact.get(&normalized) {
            return Some(id.clone());
        }

        // Fuzzy match over corpus
        if self.corpus.is_empty() {
            return None;
        }

        let mut best_score = 0.0_f64;
        let mut best_id: Option<String> = None;
        for (name, id) in &self.corpus {
            let score = strsim::jaro_winkler(&normalized, name);
            if score > best_score {
                best_score = score;
                best_id = Some(id.clone());
            }
        }

        if best_score >= threshold {
            best_id
        } else {
            None
        }
    }

    /// Normalize a string for matching: lowercase, strip leading articles, collapse whitespace.
    pub fn normalize(s: &str) -> String {
        s.to_ascii_lowercase()
            .split_whitespace()
            .filter(|w| !matches!(*w, "a" | "an" | "the"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

// ── Progressive resolver ──────────────────────────────────────────────────────

/// Intent processor combining trie fast path, fuzzy entity matching, and LLM fallback.
///
/// Requires the `progressive` feature flag.
pub struct ProgressiveResolver {
    trie: Arc<WordTrie>,
    entities: EntityIndex,
    ollama: OllamaIntent,
    fuzzy_threshold: f64,
}

impl ProgressiveResolver {
    /// Construct from a pre-built trie, entity index, and Ollama config.
    pub fn new(
        trie: Arc<WordTrie>,
        entities: EntityIndex,
        ollama_config: &crate::config::OllamaConfig,
        fuzzy_threshold: f64,
    ) -> Result<Self, IntentError> {
        let ollama = OllamaIntent::new(ollama_config)?;
        Ok(Self {
            trie,
            entities,
            ollama,
            fuzzy_threshold,
        })
    }

    /// Construct from an [`IntentConfig`] and a pre-fetched entity list.
    ///
    /// Loads the Hassil template YAML from `config.hassil.template_path` and
    /// compiles it into a word-trie. Pass `entities` from
    /// [`crate::ha_client::HaClient::entities`] for live fuzzy matching, or an
    /// empty `Vec` for offline / test use (raw slot values pass through).
    pub fn from_config(
        config: &IntentConfig,
        entities: Vec<EntityInfo>,
    ) -> Result<Self, IntentError> {
        let trie = load_trie_from_config(&config.hassil)?;
        let entity_index = EntityIndex::from_entities(&entities);
        let ollama = OllamaIntent::new(&config.ollama)?;
        Ok(Self {
            trie,
            entities: entity_index,
            ollama,
            fuzzy_threshold: config.progressive.fuzzy_threshold,
        })
    }
}

impl IntentProcessor for ProgressiveResolver {
    async fn process(&self, transcript: &str) -> Result<IntentResponse, IntentError> {
        // ── Fast path: word-trie ──────────────────────────────────────────────
        let mut walker = TrieWalker::new(Arc::clone(&self.trie));
        for word in transcript.split_whitespace() {
            walker.advance(word);
        }

        match walker.finalize() {
            Some(intent_match) => {
                tracing::debug!(
                    intent = %intent_match.intent_name,
                    slots = ?intent_match.slots,
                    "resolver: trie matched"
                );

                let (domain, service) = hassil_intent_to_service(&intent_match.intent_name);
                let raw_name = intent_match.slots.get("name").cloned().unwrap_or_default();

                // ── Fuzzy entity matching ─────────────────────────────────────
                let entity_id = if raw_name.is_empty() {
                    String::new()
                } else {
                    self.entities
                        .resolve(&raw_name, self.fuzzy_threshold)
                        .unwrap_or_else(|| raw_name.clone())
                };

                tracing::debug!(
                    raw_name = %raw_name,
                    entity_id = %entity_id,
                    "resolver: entity resolved"
                );

                let reply = if entity_id.is_empty() {
                    format!("OK, executing {}", intent_match.intent_name)
                } else {
                    format!("OK, {} {}", service.replace('_', " "), entity_id)
                };

                let action = if !domain.is_empty() && !entity_id.is_empty() {
                    Some(IntentAction {
                        domain,
                        service,
                        entity_id,
                        data: serde_json::json!({}),
                    })
                } else {
                    None
                };

                Ok(IntentResponse {
                    reply_text: reply,
                    action,
                })
            }

            // ── LLM fallback ──────────────────────────────────────────────────
            None => {
                tracing::debug!(
                    transcript = %transcript,
                    "resolver: trie miss, falling back to Ollama"
                );
                self.ollama.process(transcript).await
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OllamaConfig;

    fn sample_entities() -> Vec<EntityInfo> {
        vec![
            EntityInfo {
                entity_id: "light.kitchen_ceiling".into(),
                name: Some("Kitchen Ceiling".into()),
                aliases: vec!["kitchen lights".into()],
                area_id: Some("kitchen".into()),
                labels: vec![],
            },
            EntityInfo {
                entity_id: "light.living_room".into(),
                name: Some("Living Room".into()),
                aliases: vec![],
                area_id: Some("living_room".into()),
                labels: vec![],
            },
            EntityInfo {
                entity_id: "switch.porch_light".into(),
                name: Some("Porch Light".into()),
                aliases: vec!["front light".into()],
                area_id: Some("porch".into()),
                labels: vec![],
            },
        ]
    }

    fn make_trie() -> Arc<WordTrie> {
        Arc::new(WordTrie::from_templates([
            ("HassTurnOn", "(turn|switch) on [the] {name}"),
            ("HassTurnOff", "(turn|switch) off [the] {name}"),
            ("HassToggle", "(toggle|flip) [the] {name}"),
        ]))
    }

    // ── EntityIndex ───────────────────────────────────────────────────────────

    #[test]
    fn normalize_lowercases_and_strips_articles() {
        assert_eq!(
            EntityIndex::normalize("The Kitchen Lights"),
            "kitchen lights"
        );
        assert_eq!(
            EntityIndex::normalize("a Living Room Lamp"),
            "living room lamp"
        );
        assert_eq!(EntityIndex::normalize("an Outdoor Light"), "outdoor light");
        assert_eq!(EntityIndex::normalize("Porch Light"), "porch light");
    }

    #[test]
    fn entity_index_exact_match_by_name() {
        let index = EntityIndex::from_entities(&sample_entities());
        assert_eq!(
            index.resolve("Kitchen Ceiling", 0.85),
            Some("light.kitchen_ceiling".into())
        );
    }

    #[test]
    fn entity_index_alias_exact_match() {
        let index = EntityIndex::from_entities(&sample_entities());
        assert_eq!(
            index.resolve("kitchen lights", 0.85),
            Some("light.kitchen_ceiling".into())
        );
    }

    #[test]
    fn entity_index_entity_id_derived_match() {
        // "living room" comes from "light.living_room" → "living room"
        let index = EntityIndex::from_entities(&sample_entities());
        assert_eq!(
            index.resolve("living room", 0.85),
            Some("light.living_room".into())
        );
    }

    #[test]
    fn entity_index_fuzzy_match() {
        let index = EntityIndex::from_entities(&sample_entities());
        // "porch lite" is a typo — Jaro-Winkler should still match "porch light"
        let result = index.resolve("porch lite", 0.85);
        assert_eq!(result, Some("switch.porch_light".into()));
    }

    #[test]
    fn entity_index_below_threshold_returns_none() {
        let index = EntityIndex::from_entities(&sample_entities());
        assert!(index.resolve("completely unrelated", 0.85).is_none());
    }

    #[test]
    fn entity_index_empty_returns_none() {
        let index = EntityIndex::empty();
        assert!(index.resolve("kitchen lights", 0.85).is_none());
    }

    // ── ProgressiveResolver ───────────────────────────────────────────────────

    #[tokio::test]
    async fn resolver_fast_path_with_entity_match() {
        let trie = make_trie();
        let entity_index = EntityIndex::from_entities(&sample_entities());
        let resolver =
            ProgressiveResolver::new(trie, entity_index, &OllamaConfig::default(), 0.85).unwrap();

        let response = resolver.process("turn on kitchen ceiling").await.unwrap();
        let action = response.action.expect("should have an action");
        assert_eq!(action.domain, "homeassistant");
        assert_eq!(action.service, "turn_on");
        assert_eq!(action.entity_id, "light.kitchen_ceiling");
    }

    #[tokio::test]
    async fn resolver_fast_path_alias_entity_match() {
        let trie = make_trie();
        let entity_index = EntityIndex::from_entities(&sample_entities());
        let resolver =
            ProgressiveResolver::new(trie, entity_index, &OllamaConfig::default(), 0.85).unwrap();

        let response = resolver.process("turn on kitchen lights").await.unwrap();
        let action = response.action.expect("alias should resolve to entity");
        assert_eq!(action.entity_id, "light.kitchen_ceiling");
    }

    #[tokio::test]
    async fn resolver_fast_path_raw_slot_when_no_entity_index() {
        let trie = make_trie();
        let entity_index = EntityIndex::empty();
        let resolver =
            ProgressiveResolver::new(trie, entity_index, &OllamaConfig::default(), 0.85).unwrap();

        let response = resolver.process("turn on the lights").await.unwrap();
        let action = response.action.expect("should have action with raw slot");
        assert_eq!(action.entity_id, "lights");
    }

    #[tokio::test]
    async fn resolver_fast_path_turn_off() {
        let trie = make_trie();
        let entity_index = EntityIndex::from_entities(&sample_entities());
        let resolver =
            ProgressiveResolver::new(trie, entity_index, &OllamaConfig::default(), 0.85).unwrap();

        let response = resolver
            .process("switch off the living room")
            .await
            .unwrap();
        let action = response.action.expect("should resolve turn off");
        assert_eq!(action.domain, "homeassistant");
        assert_eq!(action.service, "turn_off");
        assert_eq!(action.entity_id, "light.living_room");
    }

    #[tokio::test]
    async fn resolver_falls_back_to_ollama_on_no_trie_match() {
        let trie = make_trie();
        let entity_index = EntityIndex::empty();
        let resolver =
            ProgressiveResolver::new(trie, entity_index, &OllamaConfig::default(), 0.85).unwrap();

        // Conversational query not in trie → Ollama fallback (will fail with no server)
        let result = resolver.process("what is the weather like today").await;
        // Expect an error since Ollama is not running in test environment
        assert!(result.is_err(), "should error when Ollama is unreachable");
    }
}
