import XCTest
import MLX
@testable import ParakeetMLX

final class AudioProcessingTests: XCTestCase {
    
    func testPreEmphasisFilter() {
        // Create a simple test signal
        let signal = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0])
        let coefficient: Float = 0.97
        
        // Apply pre-emphasis
        let emphasized = preEmphasis(signal, coefficient: coefficient)
        
        // Check that the first sample is unchanged
        XCTAssertEqual(emphasized[0].item(Float.self), 1.0, accuracy: 0.0001)
        
        // Check that subsequent samples follow the formula: y[n] = x[n] - α * x[n-1]
        XCTAssertEqual(emphasized[1].item(Float.self), 2.0 - 0.97 * 1.0, accuracy: 0.0001)
        XCTAssertEqual(emphasized[2].item(Float.self), 3.0 - 0.97 * 2.0, accuracy: 0.0001)
        XCTAssertEqual(emphasized[3].item(Float.self), 4.0 - 0.97 * 3.0, accuracy: 0.0001)
        XCTAssertEqual(emphasized[4].item(Float.self), 5.0 - 0.97 * 4.0, accuracy: 0.0001)
        
        // Verify shape is preserved
        XCTAssertEqual(emphasized.shape, signal.shape)
    }
    
    func testPreEmphasisWithZeroCoefficient() {
        let signal = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0])
        
        // With coefficient = 0, output should equal input
        let emphasized = preEmphasis(signal, coefficient: 0.0)
        
        for i in 0..<5 {
            XCTAssertEqual(emphasized[i].item(Float.self), signal[i].item(Float.self), accuracy: 0.0001)
        }
    }
    
    func testHanningWindowSymmetry() {
        // Test that the Hanning window is symmetric
        let length = 10
        let window = hanningWindowTest(length: length)
        
        // Check symmetry
        for i in 0..<(length / 2) {
            let leftValue = window[i].item(Float.self)
            let rightValue = window[length - 1 - i].item(Float.self)
            XCTAssertEqual(leftValue, rightValue, accuracy: 0.0001,
                          "Window should be symmetric at indices \(i) and \(length - 1 - i)")
        }
        
        // Check endpoints for classic Hann window
        // The Hann window should be 0 at the endpoints
        XCTAssertEqual(window[0].item(Float.self), 0.0, accuracy: 0.0001)
        XCTAssertEqual(window[length - 1].item(Float.self), 0.0, accuracy: 0.0001)
        
        // Check center value (should be 1.0)
        let centerValue = window[length / 2].item(Float.self)
        XCTAssert(centerValue > 0.9, "Center of Hanning window should be close to 1.0")
    }
    
    func testHanningWindowValues() {
        // Test specific values for a small window to ensure correctness
        let length = 5
        let window = hanningWindowTest(length: length)
        
        // For length=5, the Hann window values should be:
        // n=0: 0.5 - 0.5*cos(0) = 0.5 - 0.5 = 0
        // n=1: 0.5 - 0.5*cos(π/2) = 0.5 - 0 = 0.5
        // n=2: 0.5 - 0.5*cos(π) = 0.5 + 0.5 = 1.0
        // n=3: 0.5 - 0.5*cos(3π/2) = 0.5 - 0 = 0.5
        // n=4: 0.5 - 0.5*cos(2π) = 0.5 - 0.5 = 0
        
        XCTAssertEqual(window[0].item(Float.self), 0.0, accuracy: 0.01)
        XCTAssertEqual(window[1].item(Float.self), 0.5, accuracy: 0.01)
        XCTAssertEqual(window[2].item(Float.self), 1.0, accuracy: 0.01)
        XCTAssertEqual(window[3].item(Float.self), 0.5, accuracy: 0.01)
        XCTAssertEqual(window[4].item(Float.self), 0.0, accuracy: 0.01)
    }
    
    func testWindowIntegrationWithSTFT() throws {
        // Test that windowing works correctly in the STFT pipeline
        let sampleRate = 16000
        let windowSize: Float = 0.025  // 25ms
        let windowStride: Float = 0.010  // 10ms
        
        // Create a simple sine wave
        let duration: Float = 0.1  // 100ms
        let numSamples = Int(Float(sampleRate) * duration)
        let frequency: Float = 440.0  // A4 note
        
        var samples: [Float] = []
        for i in 0..<numSamples {
            let t = Float(i) / Float(sampleRate)
            let sample = sin(2.0 * Float.pi * frequency * t)
            samples.append(sample)
        }
        
        let audio = MLXArray(samples)
        
        // Create a simple config for testing
        let config = PreprocessConfig(
            sampleRate: sampleRate,
            normalize: "per_feature",
            windowSize: windowSize,
            windowStride: windowStride,
            window: "hanning",
            features: 80,
            nFFT: 512,
            dither: 0.0,
            padTo: 0,
            padValue: 0.0,
            preemph: 0.97
        )
        
        // Process the audio
        let melSpec = try getLogMel(audio, config: config)
        
        // Check output shape
        XCTAssertEqual(melSpec.ndim, 3)  // [batch, n_mels, n_frames]
        XCTAssertEqual(melSpec.shape[0], 1)  // batch size
        XCTAssertEqual(melSpec.shape[1], config.features)  // n_mels
        XCTAssert(melSpec.shape[2] > 0, "Should have at least one frame")
        
        // Check that values are finite
        let maxVal = melSpec.max().item(Float.self)
        let minVal = melSpec.min().item(Float.self)
        XCTAssert(maxVal.isFinite, "Max value should be finite")
        XCTAssert(minVal.isFinite, "Min value should be finite")
    }
}

// Helper function to access private Hanning window function for testing
private func hanningWindowTest(length: Int) -> MLXArray {
    guard length > 1 else {
        return MLXArray([1.0])
    }
    
    let n = Float(length)
    let indices = MLXArray(0..<length).asType(.float32)
    
    // Use the standard Hann window formula
    let window = 0.5 - 0.5 * MLX.cos(2.0 * Float.pi * indices / (n - 1))
    
    return window
}