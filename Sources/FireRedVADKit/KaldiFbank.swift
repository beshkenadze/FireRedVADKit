import Accelerate
import Foundation

/// Kaldi-compatible 80-mel log-fbank feature extractor using Accelerate/vDSP.
///
/// Matches kaldi_native_fbank (knf) defaults used by FireRedVAD:
/// - sample_rate: 16000
/// - frame_length: 25ms (400 samples), frame_shift: 10ms (160 samples)
/// - nFFT: 512 (round_to_power_of_two)
/// - nMels: 80, low_freq: 20.0, high_freq: 8000.0 (Nyquist)
/// - window: Povey (hanning^0.85), preemph: 0.97
/// - remove_dc_offset: true, snip_edges: true
/// - dither: 0.0, use_power: true, use_log_fbank: true
/// - mel_scale: HTK = 1127 * ln(1 + f/700)
///
/// Processing order per frame (matches Kaldi ProcessWindow):
///   extract frame → DC removal → preemphasis → window → FFT → power → mel → log
///
/// - Warning: NOT thread-safe. Each thread needs its own instance.
public final class KaldiFbank {
    private let sampleRate: Int = 16_000
    private let nFFT: Int = 512
    private let hopLength: Int = 160
    private let winLength: Int = 400
    private let nMels: Int = 80
    private let preemph: Float = 0.97

    private let poveyWindow: [Float]
    private let melFilterbankFlat: [Float]
    private let fftSetup: vDSP_DFT_Setup

    private var rawFrame: [Float]
    private var paddedFrame: [Float]
    private var realIn: [Float]
    private var imagIn: [Float]
    private var realOut: [Float]
    private var imagOut: [Float]
    private var powerSpec: [Float]
    private var imagSq: [Float]
    private var melFrame: [Float]

    public init() {
        let numFreqBins = nFFT / 2 + 1

        // Povey window = hanning(N)^0.85 (NOT hamming!)
        // Kaldi: pow(0.5 - 0.5 * cos(2*pi*i/(N-1)), 0.85)
        var win = [Float](repeating: 0, count: winLength)
        let divisor = Float(winLength - 1)
        for i in 0 ..< winLength {
            let hanning: Float = 0.5 - 0.5 * cos(2.0 * Float.pi * Float(i) / divisor)
            win[i] = pow(hanning, 0.85)
        }
        poveyWindow = win

        // Mel filterbank: triangles in mel-space (not Hz-space!)
        melFilterbankFlat = Self.createMelFilterbank(
            nFFT: nFFT, nMels: 80, sampleRate: 16_000, fMin: 20.0, fMax: 8_000.0
        )

        guard let setup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nFFT), .FORWARD) else {
            fatalError("Failed to create vDSP DFT setup")
        }
        fftSetup = setup

        rawFrame = [Float](repeating: 0, count: winLength)
        paddedFrame = [Float](repeating: 0, count: nFFT)
        realIn = [Float](repeating: 0, count: nFFT)
        imagIn = [Float](repeating: 0, count: nFFT)
        realOut = [Float](repeating: 0, count: nFFT)
        imagOut = [Float](repeating: 0, count: nFFT)
        powerSpec = [Float](repeating: 0, count: numFreqBins)
        imagSq = [Float](repeating: 0, count: numFreqBins)
        melFrame = [Float](repeating: 0, count: 80)
    }

    deinit {
        vDSP_DFT_DestroySetup(fftSetup)
    }

    /// Compute log-fbank features from raw 16kHz audio in int16 scale.
    ///
    /// - Parameter audio: Audio samples at 16kHz in int16 scale ([-32768, 32767]).
    ///   kaldi_native_fbank operates on int16 values internally.
    /// - Returns: Flat [numFrames * 80] features in row-major, and frame count.
    public func compute(audio: [Float]) -> (features: [Float], numFrames: Int) {
        guard audio.count >= winLength else {
            return (features: [], numFrames: 0)
        }

        let numFrames = 1 + (audio.count - winLength) / hopLength
        guard numFrames > 0 else {
            return (features: [], numFrames: 0)
        }

        let numFreqBins = nFFT / 2 + 1
        var output = [Float](repeating: 0, count: numFrames * nMels)

        for frameIdx in 0 ..< numFrames {
            let startIdx = frameIdx * hopLength

            // 1. Extract frame
            for i in 0 ..< winLength {
                rawFrame[i] = audio[startIdx + i]
            }

            // 2. Remove DC offset (subtract frame mean)
            var frameMean: Float = 0
            vDSP_meanv(rawFrame, 1, &frameMean, vDSP_Length(winLength))
            var negMean = -frameMean
            vDSP_vsadd(rawFrame, 1, &negMean, &rawFrame, 1, vDSP_Length(winLength))

            // 3. Preemphasis (backward, Kaldi style)
            //    frame[i] -= coeff * frame[i-1] for i=N-1..1
            //    frame[0] *= (1 - coeff)
            for i in stride(from: winLength - 1, through: 1, by: -1) {
                rawFrame[i] -= preemph * rawFrame[i - 1]
            }
            rawFrame[0] *= (1.0 - preemph)

            // 4. Apply Povey window, zero-pad to nFFT
            vDSP_vclr(&paddedFrame, 1, vDSP_Length(nFFT))
            vDSP_vmul(rawFrame, 1, poveyWindow, 1, &paddedFrame, 1, vDSP_Length(winLength))

            // 5. FFT → power spectrum
            for i in 0 ..< nFFT {
                realIn[i] = paddedFrame[i]
            }
            vDSP_vclr(&imagIn, 1, vDSP_Length(nFFT))
            vDSP_DFT_Execute(fftSetup, realIn, imagIn, &realOut, &imagOut)

            vDSP_vsq(realOut, 1, &powerSpec, 1, vDSP_Length(numFreqBins))
            vDSP_vsq(imagOut, 1, &imagSq, 1, vDSP_Length(numFreqBins))
            vDSP_vadd(powerSpec, 1, imagSq, 1, &powerSpec, 1, vDSP_Length(numFreqBins))

            // 6. Mel filterbank
            melFilterbankFlat.withUnsafeBufferPointer { filterPtr in
                powerSpec.withUnsafeBufferPointer { specPtr in
                    melFrame.withUnsafeMutableBufferPointer { outPtr in
                        guard let filterBase = filterPtr.baseAddress,
                              let specBase = specPtr.baseAddress,
                              let outBase = outPtr.baseAddress else { return }
                        vDSP_mmul(
                            filterBase, 1,
                            specBase, 1,
                            outBase, 1,
                            vDSP_Length(nMels),
                            vDSP_Length(1),
                            vDSP_Length(numFreqBins)
                        )
                    }
                }
            }

            // 7. Log with FLT_EPSILON floor (matches Kaldi)
            let offset = frameIdx * nMels
            for m in 0 ..< nMels {
                output[offset + m] = log(max(melFrame[m], Float.ulpOfOne))
            }
        }

        return (features: output, numFrames: numFrames)
    }

    // MARK: - Mel Filterbank (HTK scale, mel-space triangles)

    /// Create mel filterbank matching Kaldi's MelBanks constructor.
    /// Key: triangle weights are computed in mel-space, not Hz-space.
    private static func createMelFilterbank(
        nFFT: Int, nMels: Int, sampleRate: Int, fMin: Float, fMax: Float
    ) -> [Float] {
        let numFreqBins = nFFT / 2 + 1
        let fftBinWidth = Float(sampleRate) / Float(nFFT)

        /// HTK mel scale (equivalent forms):
        ///   mel = 1127 * ln(1 + f/700)     (Kaldi internal)
        ///   mel = 2595 * log10(1 + f/700)   (HTK convention)
        func hzToMel(_ hz: Float) -> Float {
            1_127.0 * log(1.0 + hz / 700.0)
        }
        func melToHz(_ mel: Float) -> Float {
            700.0 * (exp(mel / 1_127.0) - 1.0)
        }

        let melLow = hzToMel(fMin)
        let melHigh = hzToMel(fMax)
        let melDelta = (melHigh - melLow) / Float(nMels + 1)

        var flat = [Float](repeating: 0, count: nMels * numFreqBins)

        for m in 0 ..< nMels {
            let leftMel = melLow + Float(m) * melDelta
            let centerMel = melLow + Float(m + 1) * melDelta
            let rightMel = melLow + Float(m + 2) * melDelta

            // Kaldi iterates i = 0..<numFftBins (= nFFT/2, excludes Nyquist)
            // but for our matrix multiply we include numFreqBins columns anyway;
            // the Nyquist bin gets weight 0 naturally since mel >= rightMel boundary.
            for f in 0 ..< numFreqBins {
                let freq = fftBinWidth * Float(f)
                let mel = hzToMel(freq)

                // Strict inequality: mel > leftMel && mel < rightMel
                if mel > leftMel, mel < rightMel {
                    let weight: Float = if mel <= centerMel {
                        (mel - leftMel) / (centerMel - leftMel)
                    } else {
                        (rightMel - mel) / (rightMel - centerMel)
                    }
                    flat[m * numFreqBins + f] = weight
                }
            }
        }

        return flat
    }
}
