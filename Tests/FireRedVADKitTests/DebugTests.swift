import CoreML
import XCTest
@testable import FireRedVADKit

final class DebugTests: XCTestCase {
    /// Directly test CoreML model in xctest environment
    func testDirectCoreMLInXCTest() throws {
        let (refCmvn, _) = try ParityTests.loadBin("cmvn.bin")
        let (refProbs, _) = try ParityTests.loadBin("probs.bin")

        // Runtime compile from .mlmodel source for fresh compilation
        let sourceURL = URL(fileURLWithPath: "/Volumes/DATA/ane-research/FireRedVAD/FireRedVAD_Stream_N8_nn.mlmodel")
        let compiledURL = try MLModel.compileModel(at: sourceURL)
        defer { try? FileManager.default.removeItem(at: compiledURL) }

        let config = MLModelConfiguration()
        config.computeUnits = .all
        let model = try MLModel(contentsOf: compiledURL, configuration: config)

        let feat = try MLMultiArray(shape: [1, 8, 80], dataType: .float32)
        let caches = try MLMultiArray(shape: [8, 128, 19], dataType: .float32)

        // Write feat data via pointer
        let fptr = feat.dataPointer.bindMemory(to: Float.self, capacity: 640)
        for i in 0 ..< 640 {
            fptr[i] = refCmvn[i]
        }

        // MLMultiArray does NOT zero-initialize — must zero caches explicitly
        memset(caches.dataPointer, 0, 8 * 128 * 19 * MemoryLayout<Float>.size)

        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "feat": MLFeatureValue(multiArray: feat),
            "caches": MLFeatureValue(multiArray: caches),
        ]))

        let probs = try XCTUnwrap(output.featureValue(for: "probs")?.multiArrayValue)
        print("DIRECT COREML IN XCTEST:")
        print("  probs dtype: \(probs.dataType.rawValue)")
        let pptr = probs.dataPointer.bindMemory(to: Float.self, capacity: 8)
        for i in 0 ..< 8 {
            let diff = abs(pptr[i] - refProbs[i])
            print("  probs[\(i)] = \(pptr[i])  ref=\(refProbs[i])  diff=\(diff)")
        }

        let maxDiff = ParityTests.maxAbsDiff(Array(0 ..< 8).map { pptr[$0] }, refProbs)
        print("  Max diff: \(maxDiff)")
        XCTAssertLessThan(maxDiff, 0.001, "CoreML inference max diff too large")
    }
}
