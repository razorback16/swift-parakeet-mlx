import Foundation
import ParakeetMLX
@preconcurrency import MLX
import AVFoundation

/// Example demonstrating real-time streaming transcription with Parakeet
class StreamingTranscriptionExample {
    
    let model: ParakeetTDT
    let streamingContext: StreamingContext
    
    init() async throws {
        // Load the model
        print("Loading Parakeet model...")
        self.model = try await ParakeetTDT.fromPretrained(
            repoId: "nvidia/parakeet-tdt-1.1b"
        )
        
        // Create streaming context with optimized parameters
        self.streamingContext = model.createStreamingContext(
            contextWindow: 256,      // Context window size
            lookbackFrames: 32,      // Frames to look back for context
            depth: 1                 // Depth for layer preservation
        )
        
        print("Model loaded and streaming context created.")
    }
    
    /// Demonstrate streaming transcription with simulated audio chunks
    func demonstrateStreaming() async throws {
        print("\n--- Starting Streaming Transcription Demo ---\n")
        
        // Start the streaming session
        try await streamingContext.start()
        print("Streaming session started.")
        
        // Simulate processing audio chunks
        // In a real application, these would come from a microphone or audio file
        let sampleRate = model.preprocessConfig.sampleRate
        let chunkDuration = 1.0  // Process 1 second chunks
        let chunkSamples = Int(Float(sampleRate) * chunkDuration)
        
        for i in 1...5 {
            print("\nProcessing chunk \(i)...")
            
            // Create a simulated audio chunk (replace with real audio in production)
            let audioChunk = createSimulatedAudioChunk(samples: chunkSamples)
            
            // Process the audio chunk
            let startTime = Date()
            let result = try await streamingContext.process(audioChunk)
            let processingTime = Date().timeIntervalSince(startTime)
            
            // Display results
            displayStreamingResult(result, chunkNumber: i, processingTime: processingTime)
            
            // Simulate real-time delay
            try await Task.sleep(nanoseconds: UInt64(0.5 * 1_000_000_000))
        }
        
        // Finalize the streaming session
        print("\n--- Finalizing Transcription ---")
        let finalResult = try await streamingContext.end()
        displayFinalResult(finalResult)
    }
    
    /// Demonstrate async stream processing
    func demonstrateAsyncStream() async throws {
        print("\n--- Starting Async Stream Demo ---\n")
        
        // Create a streaming transcriber
        let transcriber = model.createStreamingTranscriber(
            contextWindow: 256,
            lookbackFrames: 32,
            depth: 1,
            windowDuration: 10.0,   // 10 second processing window
            overlapDuration: 2.0     // 2 second overlap
        )
        
        // Create an async stream of audio chunks
        let audioStream = createAudioStream()
        
        // Process the stream and get results
        let resultStream = transcriber.transcriptionStream(audioStream: audioStream)
        
        var chunkNumber = 0
        for await result in resultStream {
            chunkNumber += 1
            print("\nStream Result \(chunkNumber):")
            print("  Position: \(String(format: "%.2f", result.audioPosition))s")
            print("  Finalized: \(result.finalizedText)")
            if !result.draftText.isEmpty {
                print("  Draft: \(result.draftText)")
            }
        }
        
        print("\nStreaming completed.")
    }
    
    /// Create a simulated audio chunk for testing
    private func createSimulatedAudioChunk(samples: Int) -> MLXArray {
        // In a real application, this would be actual audio data
        // For testing, create a sine wave with some noise
        let frequency = 440.0  // A4 note
        let sampleRate = Float(model.preprocessConfig.sampleRate)
        let amplitude = 0.3
        
        var audioData: [Float] = []
        for i in 0..<samples {
            let t = Float(i) / sampleRate
            let sample = amplitude * sin(2 * Float.pi * Float(frequency) * t)
            let noise = Float.random(in: -0.01...0.01)
            audioData.append(sample + noise)
        }
        
        return MLXArray(audioData)
    }
    
    /// Create an async stream of audio chunks
    private func createAudioStream() -> AsyncStream<MLXArray> {
        AsyncStream { continuation in
            Task {
                let sampleRate = model.preprocessConfig.sampleRate
                let chunkSamples = sampleRate  // 1 second chunks
                
                for i in 1...10 {
                    let chunk = createSimulatedAudioChunk(samples: chunkSamples)
                    continuation.yield(chunk)
                    
                    // Simulate real-time audio capture
                    try? await Task.sleep(nanoseconds: UInt64(0.8 * 1_000_000_000))
                    
                    if i % 3 == 0 {
                        print("  [Audio Stream: Sent chunk \(i)]")
                    }
                }
                
                continuation.finish()
            }
        }
    }
    
    /// Display streaming result
    private func displayStreamingResult(
        _ result: StreamingResult,
        chunkNumber: Int,
        processingTime: TimeInterval
    ) {
        print("  Audio Position: \(String(format: "%.2f", result.audioPosition)) seconds")
        print("  Processing Time: \(String(format: "%.3f", processingTime)) seconds")
        print("  Finalized Tokens: \(result.finalizedTokens.count)")
        print("  Draft Tokens: \(result.draftTokens.count)")
        
        if !result.finalizedText.isEmpty {
            print("  Finalized Text: \"\(result.finalizedText)\"")
        }
        
        if !result.draftText.isEmpty {
            print("  Draft Text: \"\(result.draftText)\"")
        }
    }
    
    /// Display final result
    private func displayFinalResult(_ result: AlignedResult) {
        print("\nFinal Transcription:")
        print("  Total Sentences: \(result.sentences.count)")
        print("  Full Text: \"\(result.text)\"")
        
        if !result.sentences.isEmpty {
            print("\nSentence Details:")
            for (i, sentence) in result.sentences.enumerated() {
                print("  \(i + 1). [\(String(format: "%.2f", sentence.start))s - \(String(format: "%.2f", sentence.end))s]: \"\(sentence.text)\"")
            }
        }
    }
}

// MARK: - Main Execution

@main
struct StreamingExampleApp {
    static func main() async {
        do {
            let example = try await StreamingTranscriptionExample()
            
            // Run the basic streaming demo
            try await example.demonstrateStreaming()
            
            // Run the async stream demo
            try await example.demonstrateAsyncStream()
            
            print("\n--- All Demos Completed Successfully ---")
            
        } catch {
            print("Error: \(error)")
            if let localizedError = error as? LocalizedError {
                if let description = localizedError.errorDescription {
                    print("Description: \(description)")
                }
            }
        }
    }
}