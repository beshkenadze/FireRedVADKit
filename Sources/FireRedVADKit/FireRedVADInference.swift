import CoreML
import Foundation

/// CoreML inference wrapper for FireRedVAD DFSMN model.
///
/// Uses neuralNetwork format (espresso backend) with Float32 I/O.
/// The .mlmodel source is compiled at runtime to ensure compatibility
/// with the current macOS/iOS version.
///
/// Model interface:
/// - Input "feat": (1, T, 80) Float32 — T fbank frames
/// - Input "caches": (8, 128, 19) Float32 — DFSMN lookback buffers
/// - Output "probs": (1, T, 1) Float32 — speech probabilities
/// - Output "caches_out": (8, 128, 19) Float32 — updated caches
public final class FireRedVADInference {
    private let model: MLModel
    private var caches: MLMultiArray

    // Model constants
    private let numBlocks = 8
    private let projDim = 128
    private let lookbackPadding = 19
    private let featureDim = 80

    /// Initialize with a pre-compiled .mlmodelc URL.
    public init(modelURL: URL, computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: modelURL, configuration: config)
        caches = try Self.makeZeroCaches(numBlocks: numBlocks, projDim: projDim, lookbackPadding: lookbackPadding)
    }

    /// Initialize from a .mlmodel source file by compiling at runtime.
    ///
    /// Runtime compilation ensures the model is compatible with the current OS version.
    /// The compiled model is stored in a temporary directory.
    public init(modelSourceURL: URL, computeUnits: MLComputeUnits = .all) throws {
        let compiledURL = try MLModel.compileModel(at: modelSourceURL)
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: compiledURL, configuration: config)
        caches = try Self.makeZeroCaches(numBlocks: numBlocks, projDim: projDim, lookbackPadding: lookbackPadding)
    }

    /// Initialize from bundled model resource.
    ///
    /// Looks for `FireRedVAD_Stream_N8.mlmodel` in the bundle and compiles at runtime,
    /// or uses a pre-compiled `FireRedVAD_Stream_N8.mlmodelc` if available.
    public convenience init(computeUnits: MLComputeUnits = .all) throws {
        if let sourceURL = Bundle.module.url(forResource: "FireRedVAD_Stream_N8", withExtension: "mlmodel") {
            try self.init(modelSourceURL: sourceURL, computeUnits: computeUnits)
            return
        }
        if let compiledURL = Bundle.module.url(forResource: "FireRedVAD_Stream_N8", withExtension: "mlmodelc") {
            try self.init(modelURL: compiledURL, computeUnits: computeUnits)
            return
        }
        throw FireRedVADError.modelNotFound
    }

    /// Run inference on normalized fbank features.
    ///
    /// - Parameters:
    ///   - features: Flat array of [numFrames, 80] CMVN-normalized fbank features
    ///   - numFrames: Number of frames
    /// - Returns: Array of speech probabilities, one per frame
    public func predict(features: [Float], numFrames: Int) throws -> [Float] {
        precondition(features.count == numFrames * featureDim)

        let featArray = try MLMultiArray(
            shape: [1, NSNumber(value: numFrames), NSNumber(value: featureDim)],
            dataType: .float32
        )

        let fptr = featArray.dataPointer.bindMemory(to: Float.self, capacity: numFrames * featureDim)
        for i in 0 ..< (numFrames * featureDim) {
            fptr[i] = features[i]
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "feat": MLFeatureValue(multiArray: featArray),
            "caches": MLFeatureValue(multiArray: caches),
        ])

        let output = try model.prediction(from: input)

        guard let cachesOut = output.featureValue(for: "caches_out")?.multiArrayValue else {
            throw FireRedVADError.invalidOutput("missing caches_out")
        }
        caches = cachesOut

        guard let probsArray = output.featureValue(for: "probs")?.multiArrayValue else {
            throw FireRedVADError.invalidOutput("missing probs")
        }

        var probs = [Float](repeating: 0, count: numFrames)
        let pptr = probsArray.dataPointer.bindMemory(to: Float.self, capacity: numFrames)
        for i in 0 ..< numFrames {
            probs[i] = pptr[i]
        }

        return probs
    }

    /// Reset caches to zero state (call when starting a new audio stream).
    public func resetCaches() throws {
        caches = try Self.makeZeroCaches(numBlocks: numBlocks, projDim: projDim, lookbackPadding: lookbackPadding)
    }

    /// Create a zero-initialized caches MLMultiArray.
    /// MLMultiArray does NOT zero-initialize memory — we must do it explicitly.
    private static func makeZeroCaches(numBlocks: Int, projDim: Int, lookbackPadding: Int) throws -> MLMultiArray {
        let caches = try MLMultiArray(
            shape: [NSNumber(value: numBlocks), NSNumber(value: projDim), NSNumber(value: lookbackPadding)],
            dataType: .float32
        )
        memset(caches.dataPointer, 0, numBlocks * projDim * lookbackPadding * MemoryLayout<Float>.size)
        return caches
    }
}

public enum FireRedVADError: Error, LocalizedError, Sendable {
    case modelNotFound
    case invalidOutput(String)
    case audioTooShort

    public var errorDescription: String? {
        switch self {
        case .modelNotFound:
            "FireRedVAD model not found in bundle (tried .mlmodel and .mlmodelc)"
        case let .invalidOutput(detail):
            "Invalid model output: \(detail)"
        case .audioTooShort:
            "Audio buffer too short for feature extraction"
        }
    }
}
