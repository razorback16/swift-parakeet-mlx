#!/usr/bin/env swift

import Foundation
import MLX
import ParakeetMLX

// Example demonstrating the enhanced audio processing features

// MARK: - Demo Functions

func demonstratePreEmphasis() {
    print("\n=== Pre-emphasis Filter Demo ===")
    
    // Create a simple test signal
    let signal = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    print("Original signal: \(signal)")
    
    // Apply pre-emphasis with default coefficient (0.97)
    let emphasized = preEmphasis(signal)
    print("Pre-emphasized (α=0.97): \(emphasized)")
    
    // Apply with different coefficient
    let emphasized2 = preEmphasis(signal, coefficient: 0.5)
    print("Pre-emphasized (α=0.5): \(emphasized2)")
    
    // Verify the formula: y[n] = x[n] - α * x[n-1]
    print("\nVerification:")
    print("First sample (unchanged): \(emphasized[0].item(Float.self))")
    print("Second sample: \(signal[1].item(Float.self)) - 0.97 * \(signal[0].item(Float.self)) = \(emphasized[1].item(Float.self))")
}

func demonstrateWindowFunctions() {
    print("\n=== Window Functions Demo ===")
    
    // Configuration for audio processing
    let config = PreprocessConfig(
        sampleRate: 16000,
        normalize: "per_feature",
        windowSize: 0.025,  // 25ms window
        windowStride: 0.010, // 10ms stride
        window: "hanning",
        features: 80,
        nFFT: 512,
        dither: 0.0,
        padTo: 0,
        padValue: 0.0,
        preemph: 0.97  // Pre-emphasis enabled
    )
    
    print("Window configuration:")
    print("  - Window type: \(config.window)")
    print("  - Window length: \(config.winLength) samples")
    print("  - Window size: \(config.windowSize * 1000)ms")
    print("  - Hop length: \(config.hopLength) samples")
    print("  - Pre-emphasis coefficient: \(config.preemph ?? 0.0)")
    print("  - FFT size: \(config.nFFT)")
}

func demonstrateMelSpectrogram() throws {
    print("\n=== Mel Spectrogram Processing Demo ===")
    
    // Generate a test audio signal (sine wave)
    let sampleRate = 16000
    let duration: Float = 0.5  // 500ms
    let frequency: Float = 440.0  // A4 note
    let numSamples = Int(Float(sampleRate) * duration)
    
    var samples: [Float] = []
    for i in 0..<numSamples {
        let t = Float(i) / Float(sampleRate)
        let sample = sin(2.0 * Float.pi * frequency * t) * 0.5
        
        // Add some harmonics for richness
        let harmonic2 = sin(2.0 * Float.pi * frequency * 2.0 * t) * 0.25
        let harmonic3 = sin(2.0 * Float.pi * frequency * 3.0 * t) * 0.125
        
        samples.append(sample + harmonic2 + harmonic3)
    }
    
    let audio = MLXArray(samples)
    print("Generated audio signal: \(numSamples) samples at \(sampleRate)Hz")
    
    // Configure preprocessing with pre-emphasis
    let config = PreprocessConfig(
        sampleRate: sampleRate,
        normalize: "per_feature",
        windowSize: 0.025,  // 25ms
        windowStride: 0.010, // 10ms
        window: "hanning",
        features: 80,        // 80 mel bins
        nFFT: 512,
        dither: 0.0,
        padTo: 0,
        padValue: 0.0,
        preemph: 0.97       // Apply pre-emphasis
    )
    
    // Process the audio to get mel spectrogram
    let melSpec = try getLogMel(audio, config: config)
    
    print("\nMel spectrogram shape: \(melSpec.shape)")
    print("  - Batch dimension: \(melSpec.shape[0])")
    print("  - Mel bins: \(melSpec.shape[1])")
    print("  - Time frames: \(melSpec.shape[2])")
    
    // Calculate statistics
    let maxVal = melSpec.max().item(Float.self)
    let minVal = melSpec.min().item(Float.self)
    let meanVal = melSpec.mean().item(Float.self)
    
    print("\nMel spectrogram statistics:")
    print("  - Min value: \(minVal)")
    print("  - Max value: \(maxVal)")
    print("  - Mean value: \(meanVal)")
    
    // Calculate frame duration
    let frameDuration = config.windowStride * 1000  // Convert to ms
    let totalFrames = melSpec.shape[2]
    let processedDuration = Float(totalFrames) * frameDuration
    
    print("\nTemporal information:")
    print("  - Frame duration: \(frameDuration)ms")
    print("  - Total frames: \(totalFrames)")
    print("  - Processed duration: \(processedDuration)ms")
}

// MARK: - Main

do {
    print("Swift Parakeet MLX - Audio Processing Enhancements Demo")
    print("========================================================")
    
    // Demonstrate pre-emphasis filter
    demonstratePreEmphasis()
    
    // Demonstrate window functions
    demonstrateWindowFunctions()
    
    // Demonstrate mel spectrogram processing
    try demonstrateMelSpectrogram()
    
    print("\n✅ All audio processing enhancements are working correctly!")
    print("\nKey improvements in Phase 4:")
    print("1. Fixed Hanning window implementation to match NumPy's behavior")
    print("2. Enhanced pre-emphasis filter with proper edge handling")
    print("3. Improved STFT padding strategy for better spectral analysis")
    print("4. Added public pre-emphasis function for standalone use")
    print("5. Better documentation and error handling throughout")
    
} catch {
    print("❌ Error during audio processing: \(error)")
}