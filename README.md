# FireRedVADKit

Swift package for real-time Voice Activity Detection (VAD) on Apple Neural Engine.

Ports [FireRedVAD](https://huggingface.co/FireRedTeam/FireRedVAD) Stream-VAD (DFSMN architecture) to CoreML with Kaldi-compatible feature extraction — zero Python dependencies, pure Swift + Accelerate.

## Features

- **ANE-optimized**: CoreML model runs on Apple Neural Engine for minimal CPU usage
- **Streaming**: Processes 100ms audio chunks with stateful DFSMN lookback caches
- **Kaldi-compatible**: Fbank extraction matches `kaldi_native_fbank` (max diff < 2.1e-05)
- **Near-exact parity**: Full pipeline max diff vs Python reference: **5.96e-08**
- **No dependencies**: Uses only Accelerate and CoreML frameworks

## Requirements

- macOS 14+ / iOS 17+
- Swift 5.9+
- Xcode 15+

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/beshkenadze/FireRedVADKit.git", from: "1.0.0"),
]
```

Then add `"FireRedVADKit"` to your target's dependencies.

## Usage

### Basic Streaming VAD

```swift
import FireRedVADKit

// Initialize with ANE acceleration (default)
let vad = try FireRedStreamVAD()

// Feed 100ms chunks (1600 samples at 16kHz, normalized [-1, 1])
let segments = try vad.processChunk(audio: audioChunk)

for segment in segments {
    print("Speech: \(segment.startTimeMs)ms - \(segment.endTimeMs)ms")
}

// Check current state
if vad.isSpeech {
    print("Currently in speech")
}

// Reset for new audio stream
try vad.reset()
```

### Custom Configuration

```swift
// Custom postprocessor settings
var config = StreamVADPostprocessor.Config()
config.speechThreshold = 0.3     // Speech probability threshold
config.minSpeechFrame = 8        // Min frames to confirm speech (80ms)
config.minSilenceFrame = 20      // Min frames to confirm silence (200ms)
config.maxSpeechFrame = 2_000    // Force-end after 20s
config.smoothWindowSize = 5      // Smoothing window
config.padStartFrame = 5         // Pad speech start by 50ms

let vad = try FireRedStreamVAD(
    computeUnits: .all,           // .all = ANE, .cpuOnly = CPU
    postprocessorConfig: config
)
```

### Using a Custom Model

```swift
// From a compiled .mlmodelc
let vad = try FireRedStreamVAD(
    modelURL: URL(fileURLWithPath: "/path/to/model.mlmodelc"),
    computeUnits: .all
)
```

### Lower-Level API

```swift
// Individual components for custom pipelines
let fbank = KaldiFbank()
let (features, numFrames) = fbank.compute(audio: audioInt16Scale)

CMVN.apply(features: &features, numFrames: numFrames)

let inference = try FireRedVADInference(computeUnits: .all)
let probs = try inference.predict(features: features, numFrames: numFrames)

let postprocessor = StreamVADPostprocessor()
let segments = postprocessor.process(probs: probs)
```

## Architecture

```
Audio (16kHz) → Fbank (80-mel) → CMVN → CoreML DFSMN → Postprocessor → Segments
     [-1,1]      Accelerate/vDSP   Global stats    ANE inference    4-state FSM
```

**DFSMN Model**: 8 blocks, hidden=256, projection=128, lookback=20 frames.

**Input**: 1600 samples (100ms) → 8 frames of 80-dim log-fbank features.

**Output**: 8 per-frame speech probabilities → state machine emits `(startMs, endMs)` segments.

## Benchmarks

Measured on Apple Silicon (M-series), 8 frames per inference (80ms audio):

| Compute Units | Avg Latency | P50 | P99 | RTF |
|---|---|---|---|---|
| CPU Only | 0.294 ms | 0.269 ms | 0.858 ms | 0.0037x |
| CPU + GPU | 0.394 ms | 0.387 ms | 0.639 ms | 0.0049x |
| **All (ANE)** | **0.260 ms** | **0.243 ms** | **0.522 ms** | **0.0032x** |
| CPU + ANE | 0.266 ms | 0.244 ms | 0.495 ms | 0.0033x |

**RTF** (Real-Time Factor): ratio of processing time to audio duration. Values < 1.0 mean faster than real-time. At 0.0032x RTF, the model processes audio **312x faster than real-time**.

## Pipeline Parity

Verified against Python reference (`kaldi_native_fbank` + PyTorch):

| Stage | Max Absolute Diff |
|---|---|
| Fbank (80-mel log filterbank) | 2.10e-05 |
| CMVN (global normalization) | 1.19e-07 |
| CoreML Inference | 5.96e-08 |
| Full Pipeline | 5.96e-08 |

## Audio Input Format

- **Sample rate**: 16kHz
- **Format**: Float32 normalized to [-1, 1] (standard `AVAudioEngine` format)
- **Chunk size**: 1600 samples (100ms) recommended; smaller chunks are buffered

Internally, audio is converted to int16 scale to match Kaldi conventions.

## License

This project is a derivative work of [FireRedVAD](https://huggingface.co/FireRedTeam/FireRedVAD) (Apache-2.0).

Licensed under the [Apache License, Version 2.0](LICENSE).

## Credits

- [FireRedVAD](https://huggingface.co/FireRedTeam/FireRedVAD) by FireRedTeam — original DFSMN VAD model
- [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank) — reference fbank implementation
