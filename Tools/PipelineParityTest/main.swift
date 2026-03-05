import CoreML
import FireRedVADKit
import Foundation

let testDataDir = "/Volumes/DATA/ane-research/FireRedVAD/test_data"

func loadBin(_ name: String) throws -> (data: [Float], shape: [Int]) {
    let url = URL(fileURLWithPath: "\(testDataDir)/\(name)")
    let raw = try Data(contentsOf: url)
    var offset = 0
    func readUInt32() -> UInt32 {
        let val = raw.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
        offset += 4
        return val
    }
    let ndim = Int(readUInt32())
    var shape = [Int]()
    for _ in 0 ..< ndim {
        shape.append(Int(readUInt32()))
    }
    let count = shape.reduce(1, *)
    var data = [Float](repeating: 0, count: count)
    raw.withUnsafeBytes { ptr in
        let src = ptr.baseAddress!.advanced(by: offset).assumingMemoryBound(to: Float.self)
        for i in 0 ..< count {
            data[i] = src[i]
        }
    }
    return (data, shape)
}

func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
    precondition(a.count == b.count)
    var d: Float = 0
    for i in 0 ..< a.count {
        d = max(d, abs(a[i] - b[i]))
    }
    return d
}

let (refCmvn, _) = try loadBin("cmvn.bin")
let (refProbs, _) = try loadBin("probs.bin")
let numFrames = 8
let featureDim = 80

func testModel(name: String, path: String, units: MLComputeUnits) {
    do {
        let sourceURL = URL(fileURLWithPath: path)
        let compiledURL = try MLModel.compileModel(at: sourceURL)
        defer { try? FileManager.default.removeItem(at: compiledURL) }

        let config = MLModelConfiguration()
        config.computeUnits = units
        let model = try MLModel(contentsOf: compiledURL, configuration: config)

        // Check types
        for (iname, fd) in model.modelDescription.inputDescriptionsByName {
            if let mc = fd.multiArrayConstraint {
                print("  Input '\(iname)': type=\(mc.dataType.rawValue) shape=\(mc.shape)")
            }
        }

        // Determine the input type the model expects
        let featDesc = model.modelDescription.inputDescriptionsByName["feat"]!
        let expectedType = featDesc.multiArrayConstraint!.dataType

        let feat = try MLMultiArray(
            shape: [1, NSNumber(value: numFrames), NSNumber(value: featureDim)],
            dataType: expectedType
        )

        // Fill based on type
        if expectedType == .float32 || expectedType.rawValue == 65_568 {
            let fp = feat.dataPointer.bindMemory(to: Float.self, capacity: numFrames * featureDim)
            for i in 0 ..< (numFrames * featureDim) {
                fp[i] = refCmvn[i]
            }
        } else if expectedType == .float16 || expectedType.rawValue == 65_552 {
            // Float16 — convert through Int16 representation
            let fp = feat.dataPointer.bindMemory(to: UInt16.self, capacity: numFrames * featureDim)
            for i in 0 ..< (numFrames * featureDim) {
                var f = refCmvn[i]
                var h: UInt16 = 0
                // Use vImageConvert for float16
                withUnsafePointer(to: &f) { src in
                    withUnsafeMutablePointer(to: &h) { dst in
                        var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: 1, rowBytes: 4)
                        var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(dst), height: 1, width: 1, rowBytes: 2)
                        vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
                    }
                }
                fp[i] = h
            }
        } else {
            let fp = feat.dataPointer.bindMemory(to: Double.self, capacity: numFrames * featureDim)
            for i in 0 ..< (numFrames * featureDim) {
                fp[i] = Double(refCmvn[i])
            }
        }

        let caches = try MLMultiArray(shape: [8, 128, 19], dataType: expectedType)
        memset(caches.dataPointer, 0, 8 * 128 * 19 * (expectedType == .float32 ? 4 : expectedType == .double ? 8 : 2))

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "feat": MLFeatureValue(multiArray: feat),
            "caches": MLFeatureValue(multiArray: caches),
        ])
        let output = try model.prediction(from: input)
        let pa = output.featureValue(for: "probs")!.multiArrayValue!
        print("  Output type: \(pa.dataType.rawValue), shape: \(pa.shape)")

        var probs = [Float]()
        if pa.dataType == .float32 || pa.dataType.rawValue == 65_568 {
            let pp = pa.dataPointer.bindMemory(to: Float.self, capacity: numFrames)
            for i in 0 ..< numFrames {
                probs.append(pp[i])
            }
        } else if pa.dataType.rawValue == 65_552 { // float16
            for i in 0 ..< numFrames {
                probs.append(pa[[0, i, 0] as [NSNumber]].floatValue)
            }
        } else {
            let pp = pa.dataPointer.bindMemory(to: Double.self, capacity: numFrames)
            for i in 0 ..< numFrames {
                probs.append(Float(pp[i]))
            }
        }
        let diff = maxAbsDiff(probs, refProbs)
        print("  \(diff < 0.01 ? "✅" : "❌") \(name) (\(units)): max_diff=\(diff)")
        print("  Probs: \(probs.map { String(format: "%.6f", $0) }.joined(separator: ", "))")
    } catch {
        print("  ❌ \(name): \(error)")
    }
}

import Accelerate

/// Test all available model formats
let models: [(String, String)] = [
    ("neuralNetwork", "/Volumes/DATA/ane-research/FireRedVAD/FireRedVAD_Stream_N8_nn.mlmodel"),
    ("mlprogram N8", "/Volumes/DATA/ane-research/FireRedVAD/FireRedVAD_Stream_N8.mlpackage"),
    ("mlprogram v14", "/Volumes/DATA/ane-research/FireRedVAD/FireRedVAD_Stream_N8_v14.mlpackage"),
]

for (name, path) in models {
    print("\n=== \(name) ===")
    for (ulabel, units) in [("cpuOnly", MLComputeUnits.cpuOnly), ("all", MLComputeUnits.all)] {
        print("--- \(ulabel) ---")
        testModel(name: name, path: path, units: units)
    }
}

print("\nRef: \(refProbs.map { String(format: "%.6f", $0) }.joined(separator: ", "))")
