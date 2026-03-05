import CoreML
import Foundation

/// Streaming Voice Activity Detection using FireRedVAD on Apple Neural Engine.
///
/// Usage:
/// ```swift
/// let vad = try FireRedStreamVAD()
///
/// // Feed 100ms chunks (1600 samples at 16kHz)
/// let segments = try vad.processChunk(audio: chunk1600samples)
/// for seg in segments {
///     print("Speech: \(seg.startTimeMs)ms - \(seg.endTimeMs)ms")
/// }
///
/// // Reset for new stream
/// vad.reset()
/// ```
///
/// Pipeline: Audio → Fbank(80) → CMVN → CoreML(ANE) → PostProcessor → Segments
public final class FireRedStreamVAD {
    private let fbank: KaldiFbank
    private let inference: FireRedVADInference
    private let postprocessor: StreamVADPostprocessor

    /// Number of audio samples per chunk (100ms at 16kHz = 1600 samples).
    public let chunkSize: Int = 1_600

    /// Number of fbank frames per 1600-sample chunk (snip_edges=true: 1+(1600-400)/160 = 8).
    public let framesPerChunk: Int = 8

    /// Audio buffer for accumulating partial chunks
    private var audioBuffer: [Float] = []

    /// Create streaming VAD from bundled CoreML model.
    ///
    /// - Parameters:
    ///   - computeUnits: CoreML compute units (.all for ANE, .cpuOnly for CPU)
    ///   - postprocessorConfig: VAD postprocessor configuration
    public init(
        computeUnits: MLComputeUnits = .all,
        postprocessorConfig: StreamVADPostprocessor.Config = .init()
    ) throws {
        fbank = KaldiFbank()
        inference = try FireRedVADInference(computeUnits: computeUnits)
        postprocessor = StreamVADPostprocessor(config: postprocessorConfig)
    }

    /// Create streaming VAD from a custom model URL.
    ///
    /// - Parameters:
    ///   - modelURL: URL to compiled .mlmodelc directory
    ///   - computeUnits: CoreML compute units
    ///   - postprocessorConfig: VAD postprocessor configuration
    public init(
        modelURL: URL,
        computeUnits: MLComputeUnits = .all,
        postprocessorConfig: StreamVADPostprocessor.Config = .init()
    ) throws {
        fbank = KaldiFbank()
        inference = try FireRedVADInference(modelURL: modelURL, computeUnits: computeUnits)
        postprocessor = StreamVADPostprocessor(config: postprocessorConfig)
    }

    /// Process a chunk of 16kHz audio and return completed speech segments.
    ///
    /// Audio should be in normalized [-1, 1] range (standard AVAudioEngine format).
    /// It will be converted to int16 scale internally to match kaldi_native_fbank.
    ///
    /// The chunk should be exactly 1600 samples (100ms at 16kHz) for optimal
    /// operation. Smaller chunks are buffered; larger chunks are processed
    /// in 1600-sample increments.
    ///
    /// - Parameter audio: Raw 16kHz Float32 audio in [-1, 1] range
    /// - Returns: Completed speech segments (may be empty if speech is ongoing)
    public func processChunk(audio: [Float]) throws -> [StreamVADPostprocessor.Segment] {
        audioBuffer.append(contentsOf: audio)

        var allSegments: [StreamVADPostprocessor.Segment] = []

        while audioBuffer.count >= chunkSize {
            let chunk = Array(audioBuffer.prefix(chunkSize))
            audioBuffer.removeFirst(chunkSize)

            let segments = try processExactChunk(chunk)
            allSegments.append(contentsOf: segments)
        }

        return allSegments
    }

    /// Whether the VAD is currently detecting speech.
    public var isSpeech: Bool {
        postprocessor.isSpeech
    }

    /// Current postprocessor state.
    public var state: StreamVADPostprocessor.State {
        postprocessor.state
    }

    /// Reset all state for a new audio stream.
    public func reset() throws {
        audioBuffer.removeAll()
        try inference.resetCaches()
        postprocessor.reset()
    }

    // MARK: - Private

    private func processExactChunk(_ audio: [Float]) throws -> [StreamVADPostprocessor.Segment] {
        // 0. Convert float [-1,1] to int16 scale (kaldi_native_fbank expects int16 values)
        //    Truncate to int16 range to match Python: (audio * 32768).astype(np.int16).astype(np.float32)
        let audioInt16Scale = audio.map { Float(Int16(clamping: Int32($0 * 32_768.0))) }

        // 1. Extract fbank features
        var (features, numFrames) = fbank.compute(audio: audioInt16Scale)

        guard numFrames == framesPerChunk else {
            throw FireRedVADError.audioTooShort
        }

        // 2. Apply CMVN normalization
        CMVN.apply(features: &features, numFrames: numFrames)

        // 3. Run inference on ANE
        let probs = try inference.predict(features: features, numFrames: numFrames)

        // 4. Postprocess into speech segments
        return postprocessor.process(probs: probs)
    }
}
