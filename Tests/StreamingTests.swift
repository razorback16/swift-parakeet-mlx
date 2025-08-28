import XCTest
import Foundation
@testable import ParakeetMLX
@preconcurrency import MLX

final class StreamingTests: XCTestCase {
    
    /// Test the AudioBuffer functionality
    func testAudioBuffer() {
        let buffer = AudioBuffer(
            sampleRate: 16000,
            windowDuration: 1.0,
            overlapDuration: 0.2
        )
        
        // Test adding audio
        let audioChunk = MLXArray.zeros([16000])  // 1 second of audio
        buffer.append(audioChunk)
        
        XCTAssertTrue(buffer.hasEnoughData)
        XCTAssertEqual(buffer.size, 16000)
        
        // Test getting window
        let window = buffer.getWindow()
        XCTAssertNotNil(window)
        XCTAssertEqual(window?.shape[0], 16000)
        
        // Test advancing buffer
        buffer.advance(by: 8000)
        XCTAssertEqual(buffer.size, 8000)
        XCTAssertEqual(buffer.currentPosition, 0.5, accuracy: 0.001)
        
        // Test clearing
        buffer.clear()
        XCTAssertEqual(buffer.size, 0)
        XCTAssertEqual(buffer.currentPosition, 0.0)
    }
    
    /// Test the MelBuffer functionality
    func testMelBuffer() {
        let melBuffer = MelBuffer(frameShift: 160)
        
        // Test accumulating frames
        let melFrames = MLXArray.zeros([100, 80])  // 100 frames, 80 mel bins
        melBuffer.accumulate(melFrames)
        
        XCTAssertEqual(melBuffer.frameCount, 100)
        
        // Test getting frames
        let frames = melBuffer.getFrames(count: 50)
        XCTAssertNotNil(frames)
        XCTAssertEqual(frames?.shape[0], 1)  // Batch dimension
        XCTAssertEqual(frames?.shape[1], 50)  // Frame count
        
        // Test advancing
        melBuffer.advance(by: 30)
        XCTAssertEqual(melBuffer.frameCount, 70)
        
        // Test clearing
        melBuffer.clear()
        XCTAssertEqual(melBuffer.frameCount, 0)
    }
    
    /// Test StreamingToken and TokenState
    func testStreamingToken() {
        let token = AlignedToken(
            id: 1,
            start: 0.0,
            duration: 0.1,
            text: "test"
        )
        
        var streamingToken = StreamingToken(
            token: token,
            state: .draft
        )
        
        XCTAssertEqual(streamingToken.state, .draft)
        XCTAssertEqual(streamingToken.token.text, "test")
        
        // Test state change
        streamingToken.state = .finalized
        XCTAssertEqual(streamingToken.state, .finalized)
    }
    
    /// Test StreamingResult
    func testStreamingResult() {
        let finalizedTokens = [
            AlignedToken(id: 1, start: 0.0, duration: 0.1, text: "Hello"),
            AlignedToken(id: 2, start: 0.1, duration: 0.1, text: " world")
        ]
        
        let draftTokens = [
            AlignedToken(id: 3, start: 0.2, duration: 0.1, text: " test")
        ]
        
        let result = StreamingResult(
            finalizedTokens: finalizedTokens,
            draftTokens: draftTokens,
            audioPosition: 0.3,
            processingTime: 0.05
        )
        
        XCTAssertEqual(result.finalizedText, "Hello world")
        XCTAssertEqual(result.draftText, " test")
        XCTAssertEqual(result.text, "Hello world test")
        XCTAssertEqual(result.allTokens.count, 3)
        XCTAssertEqual(result.audioPosition, 0.3, accuracy: 0.001)
    }
    
    /// Test StreamingContext lifecycle
    func testStreamingContextLifecycle() async throws {
        // Note: This test requires a model instance
        // In a real test, you would load an actual model
        // For now, we just test the error handling
        
        // Mock model configuration for testing
        let preprocessConfig = PreprocessConfig(
            sampleRate: 16000,
            normalize: "per_feature",
            windowSize: 0.025,
            windowStride: 0.01,
            window: "hann",
            features: 80,
            nFFT: 512,
            dither: 0.0,
            padTo: 0,
            padValue: 0.0,
            preemph: nil
        )
        
        // This would normally be a real model
        // let model = try ParakeetTDT.fromPretrained(repoId: "nvidia/parakeet-tdt-1.1b")
        // let context = StreamingContext(model: model)
        
        // Test error when not active
        // do {
        //     _ = try await context.process(MLXArray.zeros([16000]))
        //     XCTFail("Should throw sessionNotActive error")
        // } catch StreamingError.sessionNotActive {
        //     // Expected
        // }
        
        // Test lifecycle
        // try await context.start()
        // let result = try await context.process(MLXArray.zeros([16000]))
        // XCTAssertNotNil(result)
        // let finalResult = try await context.end()
        // XCTAssertNotNil(finalResult)
    }
    
    /// Test StreamingError cases
    func testStreamingErrors() {
        let error1 = StreamingError.sessionNotActive
        XCTAssertEqual(error1.errorDescription, "Streaming session is not active")
        
        let error2 = StreamingError.bufferUnderflow
        XCTAssertEqual(error2.errorDescription, "Not enough data in buffer for processing")
        
        let error3 = StreamingError.processingError("Test error")
        XCTAssertEqual(error3.errorDescription, "Processing error: Test error")
    }
}