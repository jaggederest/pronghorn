//! Hassil-format YAML intent template parser and word-trie compiler.
//!
//! Parses Home Assistant's intent YAML template format and compiles the
//! templates into a word-level trie for streaming word-by-word intent
//! resolution. No LLM required for simple home-automation commands.
//!
//! ## Template syntax
//!
//! ```text
//! "(turn|switch) on [the] {name}"
//! ```
//!
//! - `(a|b|c)` — exactly one of the alternatives
//! - `[words]`  — optional group (zero or one occurrence)
//! - `{name}`   — named slot (filled by subsequent words in the utterance)
//!
//! ## Streaming usage
//!
//! ```rust
//! use std::sync::Arc;
//! use pronghorn_pipeline::hassil::{WordTrie, TrieWalker, IntentMatch};
//!
//! let trie = Arc::new(WordTrie::empty());
//! let mut walker = TrieWalker::new(Arc::clone(&trie));
//! for word in ["turn", "on", "kitchen", "lights"] {
//!     walker.advance(word);
//! }
//! let _result: Option<IntentMatch> = walker.finalize();
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;
use thiserror::Error;

// ── YAML types (deserialised from Hassil intent YAML files) ──────────────────

/// A loaded Hassil intent template file.
///
/// ```yaml
/// language: "en"
/// intents:
///   HassTurnOn:
///     data:
///       - sentences:
///           - "(turn|switch) on [the] {name}"
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct HassylFile {
    /// Language code (e.g., `"en"`).
    pub language: String,
    /// Intent name → definition.
    pub intents: HashMap<String, IntentDef>,
}

/// One intent definition containing one or more data blocks.
#[derive(Debug, Clone, Deserialize)]
pub struct IntentDef {
    pub data: Vec<IntentData>,
}

/// One data block with sentence templates (and optional metadata we ignore).
#[derive(Debug, Clone, Deserialize)]
pub struct IntentData {
    pub sentences: Vec<String>,
}

// ── Template AST ──────────────────────────────────────────────────────────────

/// An element in a parsed Hassil sentence template.
#[derive(Debug, Clone, PartialEq)]
pub enum TemplatePart {
    /// A single lowercase literal word.
    Word(String),
    /// Exactly one of several branches: `(a|b|c)`.
    Alternatives(Vec<Vec<TemplatePart>>),
    /// Zero or one occurrence of the inner sequence: `[words]`.
    Optional(Vec<TemplatePart>),
    /// A named slot to be filled from the utterance: `{name}`.
    Slot(String),
}

// ── Template parser ───────────────────────────────────────────────────────────

/// Error produced by [`parse_template`].
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("unmatched '(' in template")]
    UnmatchedParen,
    #[error("unmatched '[' in template")]
    UnmatchedBracket,
    #[error("unmatched '{{' in template")]
    UnmatchedBrace,
}

/// Parse a Hassil template string into a sequence of [`TemplatePart`]s.
pub fn parse_template(input: &str) -> Result<Vec<TemplatePart>, ParseError> {
    let chars: Vec<char> = input.chars().collect();
    let mut p = Parser {
        chars: &chars,
        pos: 0,
    };
    Ok(p.parse_sequence(&[]))
}

struct Parser<'a> {
    chars: &'a [char],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) {
        if self.pos < self.chars.len() {
            self.pos += 1;
        }
    }

    /// Parse a sequence of parts, stopping at any char in `stop_at`.
    fn parse_sequence(&mut self, stop_at: &[char]) -> Vec<TemplatePart> {
        let mut parts = Vec::new();
        loop {
            // skip whitespace
            while matches!(self.peek(), Some(' ') | Some('\t')) {
                self.advance();
            }
            match self.peek() {
                None => break,
                Some(c) if stop_at.contains(&c) => break,
                Some('(') => {
                    self.advance(); // consume '('
                    let alts = self.parse_alternatives();
                    parts.push(TemplatePart::Alternatives(alts));
                }
                Some('[') => {
                    self.advance(); // consume '['
                    let inner = self.parse_sequence(&[']']);
                    if self.peek() == Some(']') {
                        self.advance();
                    }
                    parts.push(TemplatePart::Optional(inner));
                }
                Some('{') => {
                    self.advance(); // consume '{'
                    let mut name = String::new();
                    while let Some(c) = self.peek() {
                        if c == '}' {
                            break;
                        }
                        name.push(c);
                        self.advance();
                    }
                    if self.peek() == Some('}') {
                        self.advance();
                    }
                    let name = name.trim().to_string();
                    if !name.is_empty() {
                        parts.push(TemplatePart::Slot(name));
                    }
                }
                Some('|') | Some(')') | Some(']') => break,
                Some(_) => {
                    let mut word = String::new();
                    while let Some(c) = self.peek() {
                        if c.is_whitespace() || "()[]{|}".contains(c) {
                            break;
                        }
                        word.push(c.to_ascii_lowercase());
                        self.advance();
                    }
                    if !word.is_empty() {
                        parts.push(TemplatePart::Word(word));
                    }
                }
            }
        }
        parts
    }

    /// Parse `branch ('|' branch)* ')'`, consuming the closing `)`.
    fn parse_alternatives(&mut self) -> Vec<Vec<TemplatePart>> {
        let mut branches = Vec::new();
        loop {
            let branch = self.parse_sequence(&['|', ')']);
            branches.push(branch);
            match self.peek() {
                Some('|') => {
                    self.advance(); // consume '|', parse next branch
                }
                Some(')') => {
                    self.advance(); // consume ')'
                    break;
                }
                _ => break,
            }
        }
        branches
    }
}

// ── Token sequence expansion ──────────────────────────────────────────────────

/// A token in a fully-expanded (no alternatives/optionals) template sequence.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Word(String),
    Slot(String),
}

/// Expand a `TemplatePart` slice into all concrete token sequences.
///
/// Alternatives and optionals introduce branching; the result is the
/// Cartesian product of all branches.
fn expand(parts: &[TemplatePart]) -> Vec<Vec<Token>> {
    let mut result: Vec<Vec<Token>> = vec![vec![]];
    for part in parts {
        let mut next = Vec::new();
        for suffix in expand_one(part) {
            for prefix in &result {
                let mut seq = prefix.clone();
                seq.extend_from_slice(&suffix);
                next.push(seq);
            }
        }
        result = next;
    }
    result
}

fn expand_one(part: &TemplatePart) -> Vec<Vec<Token>> {
    match part {
        TemplatePart::Word(w) => vec![vec![Token::Word(w.clone())]],
        TemplatePart::Slot(s) => vec![vec![Token::Slot(s.clone())]],
        TemplatePart::Optional(inner) => {
            let mut branches = vec![vec![]]; // the "absent" branch
            branches.extend(expand(inner)); // the "present" branch(es)
            branches
        }
        TemplatePart::Alternatives(branches) => branches.iter().flat_map(|b| expand(b)).collect(),
    }
}

// ── Word trie ─────────────────────────────────────────────────────────────────

type NodeId = usize;

/// A completed intent annotated with the slot names it expects.
#[derive(Debug, Clone)]
pub struct Completion {
    /// Intent name (e.g., `"HassTurnOn"`).
    pub intent_name: String,
    /// Ordered slot names that must be filled for this intent.
    pub slot_names: Vec<String>,
}

/// An edge from a trie node into slot-filling mode.
#[derive(Debug, Clone)]
struct SlotEdge {
    slot_name: String,
    /// The node to resume from after the slot ends.
    after_slot: NodeId,
    /// Intents completed if the utterance ends while this slot is being filled.
    terminal_intents: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct TrieNode {
    /// Literal-word transitions.
    children: HashMap<String, NodeId>,
    /// Slot-entry transitions.
    slot_edges: Vec<SlotEdge>,
    /// Intents completed by reaching this node (non-slot path termination).
    completions: Vec<Completion>,
}

/// Word-level trie compiled from Hassil intent templates.
///
/// Build with [`WordTrie::from_hassyl_file`] (requires `hassil` feature) or
/// [`WordTrie::empty`] for testing.
#[derive(Debug, Clone)]
pub struct WordTrie {
    nodes: Vec<TrieNode>,
    root: NodeId,
}

impl WordTrie {
    /// Empty trie (matches nothing).
    pub fn empty() -> Self {
        WordTrie {
            nodes: vec![TrieNode::default()],
            root: 0,
        }
    }

    /// Build a trie from a loaded Hassil file.
    #[cfg(feature = "hassil")]
    pub fn from_hassyl_file(file: &HassylFile) -> Self {
        let mut trie = WordTrie::empty();
        for (intent_name, def) in &file.intents {
            for data in &def.data {
                for sentence in &data.sentences {
                    match parse_template(sentence) {
                        Ok(parts) => {
                            for seq in expand(&parts) {
                                trie.insert(intent_name, &seq);
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                "hassil: skipping unparseable template '{}': {}",
                                sentence,
                                e
                            );
                        }
                    }
                }
            }
        }
        trie
    }

    /// Build a trie directly from `(intent_name, template_string)` pairs.
    ///
    /// Useful in tests and for programmatic construction without YAML.
    pub fn from_templates<'a>(templates: impl IntoIterator<Item = (&'a str, &'a str)>) -> Self {
        let mut trie = WordTrie::empty();
        for (intent_name, template) in templates {
            match parse_template(template) {
                Ok(parts) => {
                    for seq in expand(&parts) {
                        trie.insert(intent_name, &seq);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "hassil: skipping unparseable template '{}': {}",
                        template,
                        e
                    );
                }
            }
        }
        trie
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn new_node(&mut self) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(TrieNode::default());
        id
    }

    fn insert(&mut self, intent_name: &str, seq: &[Token]) {
        let slot_names: Vec<String> = seq
            .iter()
            .filter_map(|t| {
                if let Token::Slot(s) = t {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect();
        self.insert_from(self.root, intent_name, seq, &slot_names);
    }

    fn insert_from(
        &mut self,
        start: NodeId,
        intent_name: &str,
        seq: &[Token],
        slot_names: &[String],
    ) {
        let mut current = start;
        let mut i = 0;
        while i < seq.len() {
            match &seq[i] {
                Token::Word(w) => {
                    let w = w.clone();
                    if !self.nodes[current].children.contains_key(&w) {
                        let new_id = self.new_node();
                        self.nodes[current].children.insert(w.clone(), new_id);
                    }
                    current = *self.nodes[current].children.get(&w).unwrap();
                    i += 1;
                }
                Token::Slot(s) => {
                    let s = s.clone();
                    let remaining = &seq[i + 1..];
                    let terminal = if remaining.is_empty() {
                        vec![intent_name.to_string()]
                    } else {
                        vec![]
                    };
                    let after = self.new_node();
                    self.nodes[current].slot_edges.push(SlotEdge {
                        slot_name: s,
                        after_slot: after,
                        terminal_intents: terminal,
                    });
                    // Insert the post-slot continuation into the new after node
                    self.insert_from(after, intent_name, remaining, slot_names);
                    return; // continuation is handled recursively
                }
            }
        }
        // End of sequence: record the completion at this node
        self.nodes[current].completions.push(Completion {
            intent_name: intent_name.to_string(),
            slot_names: slot_names.to_vec(),
        });
    }
}

// ── Trie walker ───────────────────────────────────────────────────────────────

/// A resolved intent with slot values filled from the utterance.
#[derive(Debug, Clone, PartialEq)]
pub struct IntentMatch {
    /// The Hassil intent name (e.g., `"HassTurnOn"`).
    pub intent_name: String,
    /// Slot values keyed by slot name (e.g., `{"name" → "kitchen lights"}`).
    pub slots: HashMap<String, String>,
}

/// One active hypothesis in the streaming trie walk.
#[derive(Debug, Clone)]
struct WalkerState {
    node: NodeId,
    slots: HashMap<String, String>,
    /// Active slot fill: (slot_name, word_buffer, after_slot_node).
    filling: Option<(String, Vec<String>, NodeId)>,
    /// Intent names that complete if the utterance ends in this slot fill.
    terminal_intents: Vec<String>,
}

/// Words skipped when they appear outside the trie and outside a slot.
const STOP_WORDS: &[&str] = &["a", "an", "the"];

/// Streaming word-by-word walker over a [`WordTrie`].
///
/// Feed words with [`advance`]; call [`finalize`] at utterance end.
pub struct TrieWalker {
    trie: Arc<WordTrie>,
    active: Vec<WalkerState>,
}

impl TrieWalker {
    /// Create a new walker for the given trie.
    pub fn new(trie: Arc<WordTrie>) -> Self {
        let root = trie.root;
        Self {
            trie,
            active: vec![WalkerState {
                node: root,
                slots: HashMap::new(),
                filling: None,
                terminal_intents: Vec::new(),
            }],
        }
    }

    /// Reset for a new utterance, reusing the same trie.
    pub fn reset(&mut self) {
        let root = self.trie.root;
        self.active = vec![WalkerState {
            node: root,
            slots: HashMap::new(),
            filling: None,
            terminal_intents: Vec::new(),
        }];
    }

    /// Feed the next word from streaming STT.
    pub fn advance(&mut self, word: &str) {
        let word = word.to_ascii_lowercase();
        let word = word.trim();
        if word.is_empty() {
            return;
        }

        let mut next: Vec<WalkerState> = Vec::new();
        for state in std::mem::take(&mut self.active) {
            if let Some((slot_name, buf, after)) = state.filling.clone() {
                self.advance_filling(&state, &slot_name, buf, after, word, &mut next);
            } else {
                self.advance_normal(&state, word, &mut next);
            }
        }
        self.active = next;
    }

    fn advance_filling(
        &self,
        state: &WalkerState,
        slot_name: &str,
        buf: Vec<String>,
        after: NodeId,
        word: &str,
        next: &mut Vec<WalkerState>,
    ) {
        // Branch 1: word matches a child of the after-slot node → slot ends.
        if let Some(&child) = self.trie.nodes[after].children.get(word) {
            let mut slots = state.slots.clone();
            if !buf.is_empty() {
                slots.insert(slot_name.to_string(), buf.join(" "));
            }
            next.push(WalkerState {
                node: child,
                slots,
                filling: None,
                terminal_intents: Vec::new(),
            });
        }

        // Branch 2: accumulate word into slot buffer (continue filling).
        if STOP_WORDS.contains(&word) {
            // Stop word: keep state unchanged (don't add to slot)
            next.push(WalkerState {
                node: state.node,
                slots: state.slots.clone(),
                filling: Some((slot_name.to_string(), buf, after)),
                terminal_intents: state.terminal_intents.clone(),
            });
        } else {
            let mut new_buf = buf;
            new_buf.push(word.to_string());
            next.push(WalkerState {
                node: state.node,
                slots: state.slots.clone(),
                filling: Some((slot_name.to_string(), new_buf, after)),
                terminal_intents: state.terminal_intents.clone(),
            });
        }
    }

    fn advance_normal(&self, state: &WalkerState, word: &str, next: &mut Vec<WalkerState>) {
        let node = &self.trie.nodes[state.node];

        // Literal word match
        let literal_matched = if let Some(&child) = node.children.get(word) {
            next.push(WalkerState {
                node: child,
                slots: state.slots.clone(),
                filling: None,
                terminal_intents: Vec::new(),
            });
            true
        } else {
            false
        };

        // Enter slot-filling for each slot edge
        let slot_edges = node.slot_edges.to_vec();
        for edge in slot_edges {
            let buf = if STOP_WORDS.contains(&word) {
                vec![]
            } else {
                vec![word.to_string()]
            };
            next.push(WalkerState {
                node: state.node,
                slots: state.slots.clone(),
                filling: Some((edge.slot_name, buf, edge.after_slot)),
                terminal_intents: edge.terminal_intents,
            });
        }

        // Stop word with no literal match: keep state to allow future matches
        if STOP_WORDS.contains(&word) && !literal_matched {
            next.push(WalkerState {
                node: state.node,
                slots: state.slots.clone(),
                filling: None,
                terminal_intents: state.terminal_intents.clone(),
            });
        }
    }

    /// Resolve the utterance. Call after the last word has been fed.
    ///
    /// Returns the best match, or `None` if no intent resolved.
    pub fn finalize(&self) -> Option<IntentMatch> {
        let mut matches: Vec<IntentMatch> = Vec::new();

        for state in &self.active {
            // Completions at the current trie node (reached via literal words)
            for comp in &self.trie.nodes[state.node].completions {
                if comp.slot_names.iter().all(|s| state.slots.contains_key(s)) {
                    matches.push(IntentMatch {
                        intent_name: comp.intent_name.clone(),
                        slots: state.slots.clone(),
                    });
                }
            }

            // Terminal completions for utterances that end inside a slot fill
            if let Some((slot_name, buf, _after)) = &state.filling
                && !buf.is_empty()
            {
                for intent_name in &state.terminal_intents {
                    let mut slots = state.slots.clone();
                    slots.insert(slot_name.clone(), buf.join(" "));
                    matches.push(IntentMatch {
                        intent_name: intent_name.clone(),
                        slots,
                    });
                }
            }
        }

        // Deduplicate
        matches.dedup_by(|a, b| a.intent_name == b.intent_name && a.slots == b.slots);

        // Prefer matches with more slots filled (more specific)
        matches.sort_by(|a, b| b.slots.len().cmp(&a.slots.len()));
        matches.into_iter().next()
    }

    /// Returns true if no hypotheses remain (dead end — no intent will match).
    pub fn is_dead(&self) -> bool {
        self.active.is_empty()
    }

    /// Returns the count of active hypotheses.
    pub fn num_candidates(&self) -> usize {
        self.active.len()
    }
}

// ── HassylIntent: IntentProcessor backed by a compiled trie ──────────────────

/// Load a Hassil YAML template file and compile it into an `Arc<WordTrie>`.
///
/// Shared by [`HassylIntent`] and the progressive resolver.
#[cfg(feature = "hassil")]
pub(crate) fn load_trie_from_config(
    config: &crate::config::HassylConfig,
) -> Result<Arc<WordTrie>, crate::intent::IntentError> {
    let src = std::fs::read_to_string(&config.template_path).map_err(|e| {
        crate::intent::IntentError::Processing(format!(
            "failed to read Hassil template '{}': {}",
            config.template_path.display(),
            e
        ))
    })?;
    let file: HassylFile = serde_yaml::from_str(&src).map_err(|e| {
        crate::intent::IntentError::Processing(format!(
            "failed to parse Hassil YAML '{}': {}",
            config.template_path.display(),
            e
        ))
    })?;
    let trie = WordTrie::from_hassyl_file(&file);
    tracing::info!(
        intents = file.intents.len(),
        path = %config.template_path.display(),
        "hassil: compiled {} intent(s)",
        file.intents.len(),
    );
    Ok(Arc::new(trie))
}

/// Intent processor that resolves utterances against Hassil YAML templates.
///
/// Requires the `hassil` feature (enables `serde_yaml` parsing).
/// Slot values from the trie are mapped to [`IntentAction`] domains/services.
#[cfg(feature = "hassil")]
pub struct HassylIntent {
    trie: Arc<WordTrie>,
}

#[cfg(feature = "hassil")]
impl HassylIntent {
    /// Load a Hassil YAML template file and compile it into a `WordTrie`.
    pub fn new(config: &crate::config::HassylConfig) -> Result<Self, crate::intent::IntentError> {
        let trie = load_trie_from_config(config)?;
        Ok(Self { trie })
    }
}

#[cfg(feature = "hassil")]
impl crate::intent::IntentProcessor for HassylIntent {
    async fn process(
        &self,
        transcript: &str,
    ) -> Result<crate::intent::IntentResponse, crate::intent::IntentError> {
        let mut walker = TrieWalker::new(Arc::clone(&self.trie));
        for word in transcript.split_whitespace() {
            walker.advance(word);
        }
        match walker.finalize() {
            Some(m) => {
                let (domain, service) = hassil_intent_to_service(&m.intent_name);
                let entity_id = m.slots.get("name").cloned().unwrap_or_default();
                let reply = format!("OK, executing {} {}", m.intent_name, entity_id);
                let action = if !domain.is_empty() && !entity_id.is_empty() {
                    Some(crate::intent::IntentAction {
                        domain,
                        service,
                        entity_id,
                        data: serde_json::json!({}),
                    })
                } else {
                    None
                };
                Ok(crate::intent::IntentResponse {
                    reply_text: reply,
                    action,
                })
            }
            None => Err(crate::intent::IntentError::Processing(format!(
                "no intent matched: '{}'",
                transcript
            ))),
        }
    }
}

/// Map a Hassil intent name to `(domain, service)` for HA `call_service`.
#[cfg(feature = "hassil")]
pub(crate) fn hassil_intent_to_service(intent: &str) -> (String, String) {
    match intent {
        "HassTurnOn" => ("homeassistant".into(), "turn_on".into()),
        "HassTurnOff" => ("homeassistant".into(), "turn_off".into()),
        "HassToggle" => ("homeassistant".into(), "toggle".into()),
        "HassLightSet" => ("light".into(), "turn_on".into()),
        "HassLightGet" => ("light".into(), "get_state".into()),
        "HassClimateSetTemperature" => ("climate".into(), "set_temperature".into()),
        "HassClimateGetTemperature" => ("climate".into(), "get_state".into()),
        "HassOpenCover" => ("cover".into(), "open_cover".into()),
        "HassCloseCover" => ("cover".into(), "close_cover".into()),
        "HassLockLock" => ("lock".into(), "lock".into()),
        "HassLockUnlock" => ("lock".into(), "unlock".into()),
        _ => (String::new(), String::new()),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_template ────────────────────────────────────────────────────────

    #[test]
    fn parse_literal_words() {
        let parts = parse_template("turn on the lights").unwrap();
        assert_eq!(
            parts,
            vec![
                TemplatePart::Word("turn".into()),
                TemplatePart::Word("on".into()),
                TemplatePart::Word("the".into()),
                TemplatePart::Word("lights".into()),
            ]
        );
    }

    #[test]
    fn parse_alternatives() {
        let parts = parse_template("(turn|switch) on").unwrap();
        assert_eq!(
            parts,
            vec![
                TemplatePart::Alternatives(vec![
                    vec![TemplatePart::Word("turn".into())],
                    vec![TemplatePart::Word("switch".into())],
                ]),
                TemplatePart::Word("on".into()),
            ]
        );
    }

    #[test]
    fn parse_optional() {
        let parts = parse_template("turn on [the] {name}").unwrap();
        assert_eq!(
            parts,
            vec![
                TemplatePart::Word("turn".into()),
                TemplatePart::Word("on".into()),
                TemplatePart::Optional(vec![TemplatePart::Word("the".into())]),
                TemplatePart::Slot("name".into()),
            ]
        );
    }

    #[test]
    fn parse_slot_only() {
        let parts = parse_template("{name}").unwrap();
        assert_eq!(parts, vec![TemplatePart::Slot("name".into())]);
    }

    #[test]
    fn parse_nested_optional_in_alternatives() {
        let parts = parse_template("(turn [on]|switch)").unwrap();
        assert_eq!(
            parts,
            vec![TemplatePart::Alternatives(vec![
                vec![
                    TemplatePart::Word("turn".into()),
                    TemplatePart::Optional(vec![TemplatePart::Word("on".into())]),
                ],
                vec![TemplatePart::Word("switch".into())],
            ])]
        );
    }

    #[test]
    fn parse_uppercased_words_are_lowercased() {
        let parts = parse_template("Turn ON").unwrap();
        assert_eq!(
            parts,
            vec![
                TemplatePart::Word("turn".into()),
                TemplatePart::Word("on".into()),
            ]
        );
    }

    // ── expand ────────────────────────────────────────────────────────────────

    #[test]
    fn expand_no_branching() {
        let parts = parse_template("turn on {name}").unwrap();
        let seqs = expand(&parts);
        assert_eq!(seqs.len(), 1);
        assert_eq!(
            seqs[0],
            vec![
                Token::Word("turn".into()),
                Token::Word("on".into()),
                Token::Slot("name".into()),
            ]
        );
    }

    #[test]
    fn expand_alternatives_produces_one_seq_per_branch() {
        let parts = parse_template("(turn|switch) on").unwrap();
        let seqs = expand(&parts);
        assert_eq!(seqs.len(), 2);
        assert!(seqs.contains(&vec![Token::Word("turn".into()), Token::Word("on".into()),]));
        assert!(seqs.contains(&vec![
            Token::Word("switch".into()),
            Token::Word("on".into()),
        ]));
    }

    #[test]
    fn expand_optional_produces_two_seqs() {
        let parts = parse_template("turn on [the] {name}").unwrap();
        let seqs = expand(&parts);
        assert_eq!(seqs.len(), 2);
        // with "the"
        assert!(seqs.contains(&vec![
            Token::Word("turn".into()),
            Token::Word("on".into()),
            Token::Word("the".into()),
            Token::Slot("name".into()),
        ]));
        // without "the"
        assert!(seqs.contains(&vec![
            Token::Word("turn".into()),
            Token::Word("on".into()),
            Token::Slot("name".into()),
        ]));
    }

    #[test]
    fn expand_combined_alternatives_and_optional() {
        let parts = parse_template("(turn|switch) on [the] {name}").unwrap();
        let seqs = expand(&parts);
        // 2 alternatives × 2 optional branches = 4
        assert_eq!(seqs.len(), 4);
    }

    // ── WordTrie ──────────────────────────────────────────────────────────────

    #[test]
    fn trie_from_templates_builds_without_panic() {
        let trie = WordTrie::from_templates([
            ("HassTurnOn", "(turn|switch) on [the] {name}"),
            ("HassTurnOff", "(turn|switch) off [the] {name}"),
        ]);
        assert!(trie.nodes.len() > 1);
    }

    // ── TrieWalker ────────────────────────────────────────────────────────────

    fn make_trie() -> Arc<WordTrie> {
        Arc::new(WordTrie::from_templates([
            ("HassTurnOn", "(turn|switch) on [the] {name}"),
            ("HassTurnOff", "(turn|switch) off [the] {name}"),
            ("HassToggle", "(toggle|flip) [the] {name}"),
        ]))
    }

    #[test]
    fn walker_resolves_turn_on_kitchen_lights() {
        let trie = make_trie();
        let mut walker = TrieWalker::new(trie);
        for word in ["turn", "on", "kitchen", "lights"] {
            walker.advance(word);
        }
        let m = walker.finalize().expect("should resolve");
        assert_eq!(m.intent_name, "HassTurnOn");
        assert_eq!(m.slots["name"], "kitchen lights");
    }

    #[test]
    fn walker_resolves_with_optional_the() {
        let trie = make_trie();
        let mut walker = TrieWalker::new(trie);
        for word in ["turn", "on", "the", "bedroom", "fan"] {
            walker.advance(word);
        }
        let m = walker.finalize().expect("should resolve");
        assert_eq!(m.intent_name, "HassTurnOn");
        assert_eq!(m.slots["name"], "bedroom fan");
    }

    #[test]
    fn walker_resolves_turn_off() {
        let trie = make_trie();
        let mut walker = TrieWalker::new(trie);
        for word in ["switch", "off", "living", "room", "lights"] {
            walker.advance(word);
        }
        let m = walker.finalize().expect("should resolve");
        assert_eq!(m.intent_name, "HassTurnOff");
        assert_eq!(m.slots["name"], "living room lights");
    }

    #[test]
    fn walker_resolves_toggle() {
        let trie = make_trie();
        let mut walker = TrieWalker::new(trie);
        for word in ["toggle", "the", "porch", "light"] {
            walker.advance(word);
        }
        let m = walker.finalize().expect("should resolve");
        assert_eq!(m.intent_name, "HassToggle");
        assert_eq!(m.slots["name"], "porch light");
    }

    #[test]
    fn walker_no_match_returns_none() {
        let trie = make_trie();
        let mut walker = TrieWalker::new(trie);
        for word in ["what", "is", "the", "weather"] {
            walker.advance(word);
        }
        assert!(walker.finalize().is_none());
    }

    #[test]
    fn walker_reset_clears_state() {
        let trie = make_trie();
        let mut walker = TrieWalker::new(Arc::clone(&trie));
        walker.advance("turn");
        walker.advance("on");
        walker.reset();
        // after reset, "kitchen lights" alone shouldn't match
        walker.advance("kitchen");
        walker.advance("lights");
        assert!(walker.finalize().is_none());
    }

    #[test]
    fn walker_single_word_slot() {
        let trie = Arc::new(WordTrie::from_templates([("HassTurnOn", "turn on {name}")]));
        let mut walker = TrieWalker::new(trie);
        for word in ["turn", "on", "lamp"] {
            walker.advance(word);
        }
        let m = walker.finalize().expect("should resolve");
        assert_eq!(m.intent_name, "HassTurnOn");
        assert_eq!(m.slots["name"], "lamp");
    }

    #[test]
    fn walker_multiple_templates_same_intent_both_work() {
        let trie = Arc::new(WordTrie::from_templates([
            ("HassTurnOn", "(turn|switch) on [the] {name}"),
            ("HassTurnOn", "(turn|switch) [the] {name} on"),
        ]));

        // First form: "turn on the kitchen lights"
        let mut w = TrieWalker::new(Arc::clone(&trie));
        for word in ["turn", "on", "kitchen", "lights"] {
            w.advance(word);
        }
        let m = w.finalize().expect("first form should resolve");
        assert_eq!(m.intent_name, "HassTurnOn");

        // Second form: "switch the kitchen lights on"
        let mut w = TrieWalker::new(Arc::clone(&trie));
        for word in ["switch", "the", "kitchen", "lights", "on"] {
            w.advance(word);
        }
        let m = w.finalize().expect("second form should resolve");
        assert_eq!(m.intent_name, "HassTurnOn");
        assert_eq!(m.slots["name"], "kitchen lights");
    }

    // ── hassil_intent_to_service ──────────────────────────────────────────────

    #[test]
    fn intent_to_service_mapping() {
        assert_eq!(
            hassil_intent_to_service("HassTurnOn"),
            ("homeassistant".to_string(), "turn_on".to_string())
        );
        assert_eq!(
            hassil_intent_to_service("HassTurnOff"),
            ("homeassistant".to_string(), "turn_off".to_string())
        );
        assert_eq!(
            hassil_intent_to_service("Unknown"),
            (String::new(), String::new())
        );
    }
}
