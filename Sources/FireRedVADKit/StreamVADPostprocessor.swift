import Foundation

/// 4-state VAD postprocessor matching FireRedVAD's StreamVadPostprocessor.
///
/// State machine:
/// ```
/// SILENCE → POSSIBLE_SPEECH → SPEECH → POSSIBLE_SILENCE → SILENCE
///                ↓                         ↓
///            (back to SILENCE          (back to SPEECH
///             if too short)             if speech resumes)
/// ```
///
/// Produces speech segments as (startFrame, endFrame) pairs.
public final class StreamVADPostprocessor {
    public enum State: Int, Sendable {
        case silence = 0
        case possibleSpeech = 1
        case speech = 2
        case possibleSilence = 3
    }

    public struct Config: Sendable {
        public var smoothWindowSize: Int = 5
        public var speechThreshold: Float = 0.3
        public var padStartFrame: Int = 5
        public var minSpeechFrame: Int = 8
        public var maxSpeechFrame: Int = 2_000
        public var minSilenceFrame: Int = 20

        public init() {}
    }

    public struct Segment: Sendable {
        public let startFrame: Int
        public let endFrame: Int

        public var startTimeMs: Int {
            startFrame * 10
        }

        public var endTimeMs: Int {
            endFrame * 10
        }
    }

    private let config: Config
    public private(set) var state: State = .silence

    /// Smoothing buffer
    private var smoothBuffer: [Float] = []

    // Frame counters
    private var totalFrames: Int = 0
    private var speechStartFrame: Int = 0
    private var speechFrameCount: Int = 0
    private var silenceFrameCount: Int = 0

    /// Pending segments
    private var pendingSegments: [Segment] = []

    public init(config: Config = Config()) {
        self.config = config
    }

    /// Process a batch of speech probabilities and return completed segments.
    ///
    /// - Parameter probs: Array of per-frame speech probabilities [0, 1]
    /// - Returns: Newly completed speech segments (may be empty)
    public func process(probs: [Float]) -> [Segment] {
        pendingSegments.removeAll()

        for prob in probs {
            processFrame(prob: prob)
            totalFrames += 1
        }

        return pendingSegments
    }

    /// Reset to initial state.
    public func reset() {
        state = .silence
        smoothBuffer.removeAll()
        totalFrames = 0
        speechStartFrame = 0
        speechFrameCount = 0
        silenceFrameCount = 0
        pendingSegments.removeAll()
    }

    /// Whether currently in speech (state is SPEECH or POSSIBLE_SILENCE).
    public var isSpeech: Bool {
        state == .speech || state == .possibleSilence
    }

    // MARK: - Private

    private func processFrame(prob: Float) {
        // Smoothing: running average over last N frames
        smoothBuffer.append(prob)
        if smoothBuffer.count > config.smoothWindowSize {
            smoothBuffer.removeFirst()
        }
        let smoothedProb = smoothBuffer.reduce(0, +) / Float(smoothBuffer.count)
        let isSpeechFrame = smoothedProb >= config.speechThreshold

        switch state {
        case .silence:
            handleSilence(isSpeechFrame: isSpeechFrame)
        case .possibleSpeech:
            handlePossibleSpeech(isSpeechFrame: isSpeechFrame)
        case .speech:
            handleSpeech(isSpeechFrame: isSpeechFrame)
        case .possibleSilence:
            handlePossibleSilence(isSpeechFrame: isSpeechFrame)
        }
    }

    private func handleSilence(isSpeechFrame: Bool) {
        if isSpeechFrame {
            state = .possibleSpeech
            speechStartFrame = max(0, totalFrames - config.padStartFrame)
            speechFrameCount = 1
        }
    }

    private func handlePossibleSpeech(isSpeechFrame: Bool) {
        if isSpeechFrame {
            speechFrameCount += 1
            if speechFrameCount >= config.minSpeechFrame {
                state = .speech
            }
        } else {
            state = .silence
            speechFrameCount = 0
        }
    }

    private func handleSpeech(isSpeechFrame: Bool) {
        if isSpeechFrame {
            speechFrameCount += 1
            if speechFrameCount >= config.maxSpeechFrame {
                emitSegment(endFrame: totalFrames)
                state = .silence
                speechFrameCount = 0
            }
        } else {
            state = .possibleSilence
            silenceFrameCount = 1
        }
    }

    private func handlePossibleSilence(isSpeechFrame: Bool) {
        if isSpeechFrame {
            state = .speech
            speechFrameCount += silenceFrameCount + 1
            silenceFrameCount = 0
        } else {
            silenceFrameCount += 1
            if silenceFrameCount >= config.minSilenceFrame {
                emitSegment(endFrame: totalFrames - config.minSilenceFrame)
                state = .silence
                speechFrameCount = 0
                silenceFrameCount = 0
            }
        }
    }

    private func emitSegment(endFrame: Int) {
        pendingSegments.append(Segment(startFrame: speechStartFrame, endFrame: endFrame))
    }
}
