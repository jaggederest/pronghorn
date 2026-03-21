# Pronghorn STT and intent pipeline: a technical research brief

**Sherpa-ONNX's streaming Zipformer transducer is the strongest path to sub-second voice command execution on Raspberry Pi 5.** A 20M-parameter streaming model at just 27MB on disk delivers true frame-by-frame transcription with real-time factors well under 0.1 on ARM Cortex-A76, directly compatible with your `ort` v2 stack. Combined with a word-by-word deterministic intent resolver built on Home Assistant's own sentence templates, simple home commands can resolve in under 200ms from end of speech—no LLM round-trip required. The current 5-second Whisper-base bottleneck is an architectural mismatch, not a hardware limitation: Whisper's fixed 30-second encoder window wastes ~90% of compute on silence padding for 2.5-second utterances.

---

## Part 1: Streaming STT candidates ranked for Pi 5

Seven candidates were evaluated against five hard requirements: true streaming (partial transcripts as audio arrives), ONNX model format, ARM aarch64 compatibility, a viable Rust integration path, and reasonable accuracy for short voice commands. Three tiers emerged.

### Tier 1: Production-ready streaming on Pi 5

**Sherpa-ONNX (k2-fsa/sherpa-onnx)** is the clear frontrunner. This Apache-2.0 project from the k2/icefall team is built entirely around ONNX Runtime, with **~10,000 GitHub stars**, releases every few weeks (v1.12.29 as of March 12, 2026), and a model zoo exceeding 50 pre-built ONNX models. Its "Online" recognizer API provides genuine frame-by-frame streaming: audio samples feed in incrementally via `AcceptWaveform()`, the engine processes them in ~320ms chunks, and partial transcripts emerge via `GetOnlineStreamResult()`. Built-in endpoint detection segments utterances automatically.

The key English streaming models and their profiles:

| Model | Params | Disk (int8) | WER (LS clean) | RAM | Pi 5 RTF |
|-------|--------|------------|----------------|-----|----------|
| `streaming-zipformer-en-20M` | 20M | ~27MB | ~4.8% | ~45MB | <0.05 |
| `streaming-zipformer-en-2023-06-26` | ~65M | ~75MB | ~3.5% | ~100MB | <0.1 |
| `streaming-zipformer-bilingual-zh-en` | ~50M | ~90MB | ~5% (EN) | ~80MB | <0.1 |

The 20M model is explicitly documented as "suitable for Cortex-A7 CPU"—the A76 cores in Pi 5 are roughly **4× faster per core**. Real-time streaming is comfortable at 1–2 threads. INT8 quantized models are provided and essential for ARM performance.

**Rust integration is first-class.** Sherpa-ONNX added an official Rust API in early 2026 (PRs #3205, #3207, #3209, #3352, #3372) with examples in the `rust-api-examples/` directory. The community `sherpa-rs` crate (by thewh1teagle) wraps the C API and is available via `cargo add sherpa-rs`. Alternatively, the comprehensive C API (`c-api.h`) with functions like `SherpaOnnxCreateOnlineRecognizer`, `SherpaOnnxOnlineStreamAcceptWaveform`, and `SherpaOnnxDecodeOnlineStream` can be called via raw FFI bindings. A third path exists: load the encoder/decoder/joiner ONNX files directly with `ort` v2 and implement the streaming transducer decoding loop in Rust—this gives maximum control but requires more implementation effort.

The architecture is a **Zipformer encoder + stateless RNN-T (transducer) decoder**. The Zipformer uses causal chunked attention, processing audio in fixed-size chunks (~320ms for chunk-16). The transducer decoder runs a lightweight joiner network after each encoder chunk, producing tokens without the autoregressive bottleneck of Whisper's attention decoder. This is fundamentally different from Whisper's architecture and is why it achieves true streaming.

**Moonshine by Useful Sensors** is the strong second choice, especially once v2 streaming ONNX exports become publicly available. Moonshine v1 (MIT license, 27M params for Tiny, 61.5M for Base) already solves half the problem: its variable-length encoder processes only the actual audio duration, eliminating Whisper's 30-second padding waste. For a 2.5-second command, Moonshine v1 Base processes ~5× less data than Whisper-base. The `transcribe-rs` crate provides direct Rust/ONNX support via `MoonshineModel` with `ort` backend.

Moonshine v2 (released 2025) adds true sliding-window streaming with incremental caching—the encoder does work as audio arrives, and the decoder only runs at phrase boundaries. On Apple M3, v2 Tiny achieves **50ms latency** (5.8× faster than Whisper Tiny). The catch: **v2 streaming ONNX model exports are not yet available as standalone downloads**. The Moonshine Voice C++ library uses them internally, but you cannot yet load them independently with `ort`. When these exports drop, Moonshine v2 Tiny (34M params) becomes extremely competitive. The Pi 5 is explicitly listed as a supported platform with dedicated examples.

### Tier 2: Viable with caveats

**Vosk** provides genuine Kaldi-based streaming with good Rust bindings (`vosk-rs` v0.3.1 on crates.io), but it fails the ONNX requirement—it uses Kaldi's native nnet3 format and brings its own inference engine. The small English model (`vosk-model-small-en-us-0.22`, ~50MB) runs on Pi but achieves **15–25% WER**, significantly worse than modern transformer models. Most critically, Alphacep (Vosk's creator) appears to be shifting focus to sherpa-onnx, with Vosk in maintenance mode. Building Vosk from source requires the full Kaldi dependency tree, making cross-compilation painful.

**NeMo Parakeet-TDT-CTC-110M** achieves exceptional accuracy (**2.4% WER** on LibriSpeech clean) and has ONNX exports via sherpa-onnx (~126MB int8). However, it is not natively streaming—the FastConformer encoder uses full attention. It can work in a VAD-segmented offline pipeline (sherpa-onnx supports this), which would improve over Whisper by avoiding the 30-second padding. At 114M parameters, real-time inference on Pi 5 is marginal. Best suited as a high-accuracy offline fallback, not the primary streaming engine.

**Silero STT** offers small CTC-based ONNX models (down to ~20MB quantized for the xxsmall variant), and CTC architecture theoretically enables streaming. But the models are designed for batch inference with no official streaming API. The **AGPL v3 license** is a serious concern for an open-source project that may not want strong copyleft obligations. English STT models appear less actively maintained than Silero's VAD and TTS products. Standard WER benchmarks are not publicly available.

### Tier 3: Not recommended for this use case

**Distil-Whisper** retains Whisper's fixed 30-second encoder window—the fundamental bottleneck. The decoder layer reduction (2 layers instead of 32) saves ~20–40% of total inference time, bringing your 5-second latency down to maybe 3–4 seconds. That is insufficient for real-time voice commands. Model sizes start at 166M params for `distil-small.en`, larger than necessary.

**wav2vec2** has CTC output heads but uses full self-attention in the encoder, preventing true streaming without architectural changes. The base model at 95M params produces ~360MB fp32 ONNX files. A distilled streaming variant (DistillW2V2) exists in research but has no public ONNX release.

**Whisper Streaming** (LocalAgreement by Macháček) is pseudo-streaming: it repeatedly re-runs the full encoder on a growing audio buffer. Average latency is ~3.3 seconds on an NVIDIA A40 GPU. On Pi 5, it would be slower than your current batch approach.

### Recommended STT architecture for Pronghorn

The optimal configuration uses sherpa-onnx as the streaming engine:

1. **Primary ASR**: `sherpa-onnx-streaming-zipformer-en-20M` (int8) via `sherpa-rs` or direct C FFI. Delivers partial transcripts every ~320ms chunk. Total ONNX footprint: ~27MB, ~45MB RAM.
2. **VAD**: Keep your existing energy-based VAD for wake-word-to-STT handoff. Sherpa-onnx also bundles Silero VAD (2MB ONNX) for endpoint detection during streaming if you want belt-and-suspenders.
3. **Watch list**: Monitor Moonshine v2 streaming ONNX exports. When available, benchmark against Zipformer-20M—Moonshine's variable-length architecture and MIT license are attractive.

If accuracy proves insufficient with the 20M model for certain commands, step up to `streaming-zipformer-en-2023-06-26` (~65M, ~75MB) which drops WER from ~4.8% to ~3.5% on clean speech while remaining comfortably real-time on Pi 5.

---

## Part 2: Progressive intent resolution without LLM latency

The second half of the pipeline transforms streaming STT output into executed Home Assistant actions. The core insight: **~70–80% of home automation commands follow predictable grammar patterns** ("turn on the kitchen lights," "set thermostat to 72," "lock the front door"). These should resolve deterministically in microseconds, reserving LLM inference for genuinely ambiguous or conversational queries.

### How Rhasspy and Home Assistant already solve half this problem

Home Assistant's **Hassil** library (github.com/home-assistant/hassil) and its companion **intents repository** (github.com/home-assistant/intents) define sentence templates in YAML for 40+ languages covering all built-in intents: `HassTurnOn`, `HassTurnOff`, `HassLightSet`, `HassClimateSetTemperature`, `HassOpenCover`, and dozens more. Templates use a compact syntax: `"(turn|switch) [the] {name} (on|off)"` where `()` denotes alternatives, `[]` optional words, and `{name}` slots filled from Home Assistant's entity registry at runtime.

Rhasspy's **fsticuffs** intent recognizer takes a similar approach but compiles templates into a **Finite State Transducer (FST)** graph. Edges carry input/output word pairs with meta-annotations for intent names and slot boundaries. Recognition walks the graph word-by-word, collecting slot values as it goes. It processes millions of candidate sentences in milliseconds. The key limitation: both Hassil and fsticuffs operate on complete transcripts. Neither supports progressive word-by-word resolution—but their template compilation approach is the right foundation.

Commercial assistants validate this architecture. Amazon's Alexa used an on-device **Transformer Transducer** with context-aware biasing for local command processing before moving everything cloud-side for Alexa+ in 2025. Google's Assistant runs **8 on-device ML models** simultaneously on Pixel phones for low-latency intent matching. Picovoice's Rhino engine fuses ASR and NLU into a single model achieving ~100ms speech-to-intent latency. The pattern is consistent: **simple structured commands resolve locally and fast; complex queries go to cloud/LLM**.

### A word-trie state machine for streaming intent resolution

No existing open-source implementation of progressive streaming intent resolution was found—this is greenfield work. The architecture below synthesizes the best ideas from Rhasspy's FST approach, Home Assistant's template system, and incremental NLU research.

**At startup**, the system loads Home Assistant's intent YAML templates and compiles them into a **word-level trie** (prefix tree). Each path through the trie represents a valid command sequence. Leaf nodes and intermediate nodes annotated with slot markers enable progressive matching. Simultaneously, the system fetches all entities, areas, and floors from Home Assistant's WebSocket API (`config/entity_registry/list_for_display`, `config/area_registry/list`) and builds a fuzzy-searchable entity index.

**At runtime**, as each word arrives from streaming STT, the resolver advances through the trie:

- **"turn"** → Matches trie root children. Active candidates: {HassTurnOn, HassTurnOff}. Confidence: low.
- **"on"** → Narrows to {HassTurnOn}. Single intent identified. Transition to slot-filling mode for `{name}` and `{area}`.
- **"the"** → Recognized as stop word, skipped.
- **"kitchen"** → Begins entity buffer. Fuzzy-matched against entity names and area names. Area "kitchen" matches exactly (score 1.0).
- **"lights"** → Entity buffer becomes "kitchen lights." Fuzzy match against entity registry. If `light.kitchen_ceiling` has friendly name "Kitchen Lights," Jaro-Winkler score exceeds 0.85 threshold. Slot filled.
- **STT endpoint detected** → All slots filled. Execute `HassTurnOn` immediately via HA's `call_service` WebSocket command.

Total deterministic resolution time for this 5-word command: **~200 microseconds** for trie traversal and fuzzy matching, well within the 500ms budget even accounting for STT chunk delivery delays.

**Fallback triggers** route to the LLM when the deterministic path fails: no action word match after 2–3 words, multiple ambiguous intents remaining after all words consumed, entity fuzzy match score below 0.7, detected question words ("what," "why," "how"), or a configurable timeout (~2 seconds) without resolution. The accumulated transcript passes to Ollama/gemma3:1b with entity context injected into the prompt.

### Rust crate stack for building the resolver

The Rust ecosystem provides excellent building blocks. For the core state machine, a **manual enum-based pattern** (the "typestate" pattern) is most flexible—each state carries its own data, and `match` handles transitions. The `statig` crate offers a richer hierarchical state machine if the complexity grows. For the word trie, `ptrie` provides prefix/postfix search with `find_prefixes()` and `find_longest_prefix()`. For fuzzy entity matching, **`strsim`** (maintained by the rapidfuzz team) implements Jaro-Winkler, Levenshtein, and Sørensen-Dice distances; **`fuzzt`** adds `get_top_n()` for finding the best N matches above a cutoff score; and **`nucleo`** (from the Helix editor) offers ~6× faster fuzzy matching for large entity sets. For parsing Home Assistant's intent YAML templates at startup, `serde_yaml` handles deserialization, and `nom` parser combinators can tokenize the template syntax into `WordMatcher` nodes (literals, alternatives, optionals, slot references).

Home Assistant integration requires `tokio-tungstenite` for WebSocket connectivity to fetch and subscribe to entity registry updates. Action execution uses either the WebSocket `call_service` command or the REST API (`POST /api/services/{domain}/{service}`) via `reqwest`.

One particularly useful resource: the `snips-nlu-rs` crate (github.com/snipsco/snips-nlu-rs) is a full Rust NLU inference engine from the defunct Snips project. While unmaintained since Sonos's acquisition in 2019, its architecture—CRF slot filling plus logistic regression intent classification—could serve as reference code for a lightweight ML fallback layer between the deterministic trie and the full LLM.

### Entity discovery and fuzzy matching strategy

Home Assistant exposes entities through its WebSocket API with rich metadata: `entity_id`, `name` (friendly name), `original_name`, `aliases` (user-defined alternative names), `area_id`, `device_id`, `domain`, and `labels`. The resolver should build three lookup structures at startup:

- **Exact HashMap**: normalized friendly name → EntityInfo (O(1) lookup)
- **Alias expansion**: each entity's aliases indexed identically to friendly names
- **Fuzzy corpus**: all names and aliases stored for Jaro-Winkler scoring via `fuzzt::get_top_n`

Normalization strips articles ("the"), lowercases, and collapses whitespace. Common STT transcription errors should be handled with phonetic normalization—"livingroom" vs. "living room," "bed room" vs. "bedroom." A subscribe to `config/entity_registry/list` keeps the index current as entities change.

For multi-word entity names, the resolver accumulates words in an entity buffer during slot-filling mode, attempting fuzzy matches after each new word. It accepts the match when the score exceeds 0.85 and the next word either doesn't improve the score or belongs to a different syntactic position in the template. This handles both "kitchen lights" (2 words) and "master bedroom ceiling fan" (4 words) gracefully.

---

## Putting it all together: the full streaming pipeline

The end-to-end architecture replaces the current batch pipeline with a fully streaming one:

1. **Wake word** (rustpotter) triggers recording start.
2. **Audio streams** via UDP to the STT engine.
3. **Sherpa-ONNX streaming Zipformer** (via `sherpa-rs`) processes audio in ~320ms chunks, emitting partial transcripts.
4. **Progressive intent resolver** consumes words as they arrive, walking the intent trie and fuzzy-matching entities.
5. **Fast path**: If the trie resolves to a single intent with all slots filled, execute immediately via HA WebSocket. Target: **<500ms from end of speech**.
6. **Slow path**: If resolution fails or detects conversational input, accumulate the full transcript and forward to Ollama/gemma3:1b. Response feeds back through Kokoro TTS.

Expected latency budget for a simple "turn on the kitchen lights" command:

| Stage | Latency |
|-------|---------|
| Last audio chunk to STT | ~160ms (half a 320ms chunk, average) |
| STT endpoint detection | ~100ms |
| Trie resolution + entity match | <1ms |
| HA WebSocket call_service | ~50ms (local network) |
| **Total from end of speech** | **~310ms** |

This is a **16× improvement** over the current ~5-second batch pipeline, achieved through architectural changes rather than hardware upgrades.

## Conclusion

The 5-second Whisper bottleneck stems from two architectural mismatches: a batch encoder that pads all input to 30 seconds, and a pipeline that waits for complete transcription before starting intent resolution. Both are solvable without changing hardware. **Sherpa-ONNX's streaming Zipformer-20M** eliminates the first problem with a 27MB model purpose-built for real-time ARM inference. A **word-trie progressive resolver** eliminates the second by matching intents incrementally as words arrive, using Home Assistant's own sentence templates and entity registry for zero-configuration command coverage. The LLM remains available for everything the deterministic path can't handle—but for the ~80% of commands that follow predictable patterns, the response will feel instantaneous. Moonshine v2's streaming ONNX exports, when they ship, deserve immediate benchmarking as an alternative STT engine given their MIT license and variable-length architecture.