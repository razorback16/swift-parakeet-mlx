import Accelerate
import Foundation
import MLX

// MARK: - Audio Processing Functions

/// Apply pre-emphasis filter to audio signal
/// - Parameters:
///   - audio: Input audio signal as MLXArray
///   - coefficient: Pre-emphasis coefficient (typically 0.97)
/// - Returns: Pre-emphasized audio signal
public func preEmphasis(_ audio: MLXArray, coefficient: Float = 0.97) -> MLXArray {
    return applyPreEmphasis(audio, coefficient: coefficient)
}

/// Compute log mel spectrogram from audio data
/// - Parameters:
///   - audio: Input audio signal as MLXArray
///   - config: Preprocessing configuration including window parameters, FFT size, etc.
/// - Returns: Log mel spectrogram with shape [1, n_mels, n_frames]
/// - Throws: ParakeetError if audio processing fails
public func getLogMel(_ audio: MLXArray, config: PreprocessConfig) throws -> MLXArray {
    let originalDType = audio.dtype
    var x = audio

    // Pad audio if needed
    if config.padTo > 0 && x.shape.last! < config.padTo {
        let padLength = config.padTo - x.shape.last!
        let padArray = Array(repeating: (0, 0), count: x.ndim)
        var padArray2 = padArray
        padArray2[padArray2.count - 1] = (0, padLength)
        x = MLX.padded(
            x, widths: padArray2.map { IntOrPair($0) }, mode: .constant,
            value: MLXArray(config.padValue))
    }

    // Apply pre-emphasis filter if configured
    // Pre-emphasis: y[n] = x[n] - α * x[n-1], where α is typically 0.97
    if let preemph = config.preemph, preemph > 0 {
        x = applyPreEmphasis(x, coefficient: preemph)
    }

    // Get window function
    let window = try getWindow(config.window, length: config.winLength, dtype: x.dtype)

    // Compute STFT
    x = try stft(
        x,
        nFFT: config.nFFT,
        hopLength: config.hopLength,
        winLength: config.winLength,
        window: window
    )

    // Compute magnitude spectrum
    let magnitude = abs(x)
    var powerSpectrum = magnitude

    if config.magPower != 1.0 {
        powerSpectrum = pow(magnitude, config.magPower)
    }

    // Apply mel filterbank
    let melFilters = try createMelFilterbank(
        sampleRate: config.sampleRate,
        nFFT: config.nFFT,
        nMels: config.features
    )

    let melSpectrum = matmul(
        melFilters.asType(powerSpectrum.dtype), powerSpectrum.transposed(axes: [1, 0]))
    let logMelSpectrum = log(melSpectrum + 1e-5)

    // Normalize
    let normalizedMel: MLXArray
    if config.normalize == "per_feature" {
        let mean = logMelSpectrum.mean(axes: [1], keepDims: true)
        let std = logMelSpectrum.std(axes: [1], keepDims: true)
        normalizedMel = (logMelSpectrum - mean) / (std + 1e-5)
    } else {
        let mean = logMelSpectrum.mean()
        let std = logMelSpectrum.std()
        normalizedMel = (logMelSpectrum - mean) / (std + 1e-5)
    }

    // Transpose and add batch dimension
    let output = normalizedMel.transposed(axes: [1, 0]).expandedDimensions(axis: 0)

    return output.asType(originalDType)
}

// MARK: - Pre-emphasis Filter

/// Apply pre-emphasis filter to audio signal
/// Pre-emphasis: y[n] = x[n] - α * x[n-1]
private func applyPreEmphasis(_ signal: MLXArray, coefficient: Float) -> MLXArray {
    guard signal.shape[0] > 1 else { return signal }
    
    // Keep the first sample unchanged
    let firstSample = signal[0..<1]
    
    // Apply pre-emphasis filter: x[n] - α * x[n-1]
    let shifted = signal[0..<(signal.shape[0] - 1)]
    let filtered = signal[1...] - coefficient * shifted
    
    // Concatenate the first sample with the filtered signal
    return MLX.concatenated([firstSample, filtered], axis: 0)
}

// MARK: - Window Functions

private func getWindow(_ windowType: String, length: Int, dtype: DType) throws -> MLXArray {
    switch windowType.lowercased() {
    case "hanning", "hann":
        return hanningWindow(length: length, dtype: dtype)
    case "hamming":
        return hammingWindow(length: length, dtype: dtype)
    case "blackman":
        return blackmanWindow(length: length, dtype: dtype)
    case "bartlett":
        return bartlettWindow(length: length, dtype: dtype)
    default:
        throw ParakeetError.audioProcessingError("Unsupported window type: \(windowType)")
    }
}

private func hanningWindow(length: Int, dtype: DType) -> MLXArray {
    // NumPy's hanning window formula: 0.5 - 0.5 * cos(2*pi*n/(N-1))
    // This is equivalent to: 0.5 * (1 - cos(2*pi*n/(N-1)))
    guard length > 1 else {
        return MLXArray([1.0]).asType(dtype)
    }
    
    let n = Float(length)
    let indices = MLXArray(0..<length).asType(.float32)
    
    // Use the standard Hann window formula
    // Note: For length=1, this would cause division by zero, hence the guard above
    let window = 0.5 - 0.5 * cos(2.0 * Float.pi * indices / (n - 1))
    
    return window.asType(dtype)
}

private func hammingWindow(length: Int, dtype: DType) -> MLXArray {
    let n = Float(length)
    let indices = MLXArray(0..<length).asType(.float32)
    let window = 0.54 - 0.46 * cos(2.0 * Float.pi * indices / (n - 1))
    return window.asType(dtype)
}

private func blackmanWindow(length: Int, dtype: DType) -> MLXArray {
    let n = Float(length)
    let indices = MLXArray(0..<length).asType(.float32)
    let a0: Float = 0.42
    let a1: Float = 0.5
    let a2: Float = 0.08
    let window =
        a0 - a1 * cos(2.0 * Float.pi * indices / (n - 1)) + a2
        * cos(4.0 * Float.pi * indices / (n - 1))
    return window.asType(dtype)
}

private func bartlettWindow(length: Int, dtype: DType) -> MLXArray {
    let n = Float(length)
    let indices = MLXArray(0..<length).asType(.float32)
    let window = 1.0 - abs((indices - (n - 1) / 2.0) / ((n - 1) / 2.0))
    return window.asType(dtype)
}

// MARK: - STFT Implementation

private func stft(
    _ x: MLXArray,
    nFFT: Int,
    hopLength: Int,
    winLength: Int,
    window: MLXArray
) throws -> MLXArray {

    // Ensure the window is properly shaped as a 1D array
    var actualWindow = window
    
    // Pad or truncate the window to nFFT length if needed
    if winLength != nFFT {
        if winLength > nFFT {
            // Truncate window if it's longer than nFFT
            actualWindow = window[0..<nFFT]
        } else {
            // Pad window with zeros to match nFFT length
            let padding = nFFT - winLength
            // Center the window by padding on both sides
            let leftPad = padding / 2
            let rightPad = padding - leftPad
            let padArray = [(leftPad, rightPad)]
            actualWindow = MLX.padded(
                window, widths: padArray.map { IntOrPair($0) }, mode: .constant,
                value: MLXArray(0.0))
        }
    }

    // Pad the signal using reflection padding (matching librosa's default)
    let padding = nFFT / 2
    var paddedX = x

    // Reflect padding to minimize edge artifacts
    // This mirrors the signal at boundaries for better spectral analysis
    if padding > 0 && x.shape[0] > padding {
        let prefix = x[1..<min(padding + 1, x.shape[0])].reversed(axes: [0])
        let suffixStart = max(0, x.shape[0] - padding - 1)
        let suffixEnd = max(suffixStart, x.shape[0] - 1)
        let suffix = x[suffixStart..<suffixEnd].reversed(axes: [0])
        paddedX = MLX.concatenated([prefix, x, suffix], axis: 0)
    } else if padding > 0 {
        // If signal is too short for reflection, use zero padding
        let padArray = [(padding, padding)]
        paddedX = MLX.padded(
            x, widths: padArray.map { IntOrPair($0) }, mode: .constant,
            value: MLXArray(0.0))
    }

    // Create frames using sliding window approach
    let numFrames = max(1, (paddedX.shape[0] - nFFT + hopLength) / hopLength)
    var frames: [MLXArray] = []

    for i in 0..<numFrames {
        let start = i * hopLength
        let end = start + nFFT
        if end <= paddedX.shape[0] {
            // Apply window to each frame
            let frame = paddedX[start..<end] * actualWindow
            frames.append(frame)
        }
    }

    if frames.isEmpty {
        throw ParakeetError.audioProcessingError("No frames could be extracted from audio signal")
    }

    // Stack frames into a 2D matrix [num_frames, nFFT]
    let frameMatrix = MLX.stacked(frames, axis: 0)

    // Apply real-valued FFT along the last axis
    // This returns complex values with shape [num_frames, nFFT/2 + 1]
    let fftResult = MLX.rfft(frameMatrix, axis: -1)

    return fftResult
}

// MARK: - Mel Filterbank

private func createMelFilterbank(
    sampleRate: Int,
    nFFT: Int,
    nMels: Int
) throws -> MLXArray {

    let nyquist = Float(sampleRate) / 2.0
    let nFreqs = nFFT / 2 + 1

    // Create mel scale points
    let melMin = hzToMel(0.0)
    let melMax = hzToMel(nyquist)
    let melPoints = MLXArray.linspace(melMin, melMax, count: nMels + 2)

    // Convert back to Hz
    let hzPoints = melToHz(melPoints)

    // Convert to FFT bin indices
    let binIndices = hzPoints * Float(nFFT) / Float(sampleRate)

    // Create filterbank
    let filterbank = MLXArray.zeros([nMels, nFreqs])

    for m in 0..<nMels {
        let leftBin = binIndices[m].item(Float.self)
        let centerBin = binIndices[m + 1].item(Float.self)
        let rightBin = binIndices[m + 2].item(Float.self)

        // Create triangular filter with continuous values (not just integer bins)
        for f in 0..<nFreqs {
            let freq = Float(f)

            if freq >= leftBin && freq <= centerBin && centerBin > leftBin {
                let weight = (freq - leftBin) / (centerBin - leftBin)
                filterbank[m, f] = MLXArray(weight)
            } else if freq > centerBin && freq <= rightBin && rightBin > centerBin {
                let weight = (rightBin - freq) / (rightBin - centerBin)
                filterbank[m, f] = MLXArray(weight)
            }
        }

        // Apply exact "slaney" normalization to match librosa
        // Slaney normalization: 2.0 / (mel_f[i+2] - mel_f[i])
        let melRange = melPoints[m + 2].item(Float.self) - melPoints[m].item(Float.self)
        if melRange > 0 {
            let slaneynorm = 2.0 / melRange
            filterbank[m] = filterbank[m] * slaneynorm
        }
    }

    return filterbank
}

// MARK: - Mel Scale Conversion

private func hzToMel(_ hz: Float) -> Float {
    return 2595.0 * log10(1.0 + hz / 700.0)
}

private func hzToMel(_ hz: MLXArray) -> MLXArray {
    return 2595.0 * log10(1.0 + hz / 700.0)
}

private func melToHz(_ mel: MLXArray) -> MLXArray {
    return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
}

// MARK: - Utility Functions

private func concatenate(_ arrays: [MLXArray], axis: Int) -> MLXArray {
    return MLX.concatenated(arrays, axis: axis)
}

private func abs(_ x: MLXArray) -> MLXArray {
    return MLX.abs(x)
}

private func pow(_ x: MLXArray, _ exp: Float) -> MLXArray {
    return MLX.pow(x, exp)
}

private func pow(_ base: Float, _ exp: MLXArray) -> MLXArray {
    return MLX.pow(base, exp)
}

private func log(_ x: MLXArray) -> MLXArray {
    return MLX.log(x)
}

private func log10(_ x: Float) -> Float {
    return Foundation.log10(x)
}

private func log10(_ x: MLXArray) -> MLXArray {
    return MLX.log(x) / MLX.log(MLXArray(10.0))
}

private func cos(_ x: MLXArray) -> MLXArray {
    return MLX.cos(x)
}

private func matmul(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
    return MLX.matmul(a, b)
}

// MARK: - MLXArray Extensions

extension MLXArray {
    func std(axes: [Int]? = nil, keepDims: Bool = false) -> MLXArray {
        let meanVal =
            axes != nil ? self.mean(axes: axes!, keepDims: true) : self.mean(keepDims: true)
        let variance =
            axes != nil
            ? ((self - meanVal) * (self - meanVal)).mean(axes: axes!, keepDims: keepDims)
            : ((self - meanVal) * (self - meanVal)).mean(keepDims: keepDims)
        return MLX.sqrt(variance)
    }

    func reversed(axes: [Int]) -> MLXArray {
        // For 1D reversal on axis 0
        let indices = MLXArray((0..<self.shape[0]).reversed())
        return self[indices]
    }

    static func linspace(_ start: Float, _ end: Float, count: Int) -> MLXArray {
        let step = (end - start) / Float(count - 1)
        let values = (0..<count).map { start + Float($0) * step }
        return MLXArray(values)
    }

    static func stacked(_ arrays: [MLXArray], axis: Int) -> MLXArray {
        return MLX.stacked(arrays, axis: axis)
    }
}
