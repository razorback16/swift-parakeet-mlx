import XCTest
import MLX
import MLXNN
@testable import ParakeetMLX

final class APITests: XCTestCase {
    
    func testDecodingConfiguration() {
        // Test default configurations
        let ctcConfig = DecodingConfiguration.ctcDefault()
        XCTAssertEqual(ctcConfig.strategy, .greedy)
        XCTAssertEqual(ctcConfig.blankId, 0)
        XCTAssertTrue(ctcConfig.logProbs)
        
        let rnntConfig = DecodingConfiguration.rnntDefault()
        XCTAssertEqual(rnntConfig.strategy, .greedy)
        XCTAssertEqual(rnntConfig.maxSymbols, 10)
        
        let tdtConfig = DecodingConfiguration.tdtDefault(maxSymbols: 15)
        XCTAssertEqual(tdtConfig.strategy, .greedy)
        XCTAssertEqual(tdtConfig.maxSymbols, 15)
        
        // Test custom configuration
        let customConfig = DecodingConfiguration(
            strategy: .beamSearch,
            beamSize: 10,
            lmWeight: 0.5,
            temperature: 0.8
        )
        XCTAssertEqual(customConfig.strategy, .beamSearch)
        XCTAssertEqual(customConfig.beamSize, 10)
        XCTAssertEqual(customConfig.lmWeight, 0.5)
        XCTAssertEqual(customConfig.temperature, 0.8)
    }
    
    func testModelTypeEnum() {
        XCTAssertEqual(ParakeetModelType.ctc.rawValue, "ctc")
        XCTAssertEqual(ParakeetModelType.rnnt.rawValue, "rnnt")
        XCTAssertEqual(ParakeetModelType.tdtctc.rawValue, "tdt")
        
        XCTAssertEqual(ParakeetModelType.ctc.description, "Connectionist Temporal Classification (CTC)")
        XCTAssertEqual(ParakeetModelType.rnnt.description, "Recurrent Neural Network Transducer (RNN-T)")
        XCTAssertEqual(ParakeetModelType.tdtctc.description, "Token-and-Duration Transducer CTC (TDT-CTC)")
    }
    
    func testTranscriptionResult() {
        let tokens = [
            AlignedToken(id: 1, start: 0.0, duration: 0.5, text: "Hello"),
            AlignedToken(id: 2, start: 0.5, duration: 0.3, text: " world")
        ]
        
        let sentences = [AlignedSentence(tokens: tokens)]
        
        let result = TranscriptionResult(
            text: "Hello world",
            tokens: tokens,
            sentences: sentences,
            confidence: 0.95,
            processingTime: 2.5
        )
        
        XCTAssertEqual(result.text, "Hello world")
        XCTAssertEqual(result.tokens.count, 2)
        XCTAssertEqual(result.sentences.count, 1)
        XCTAssertEqual(result.confidence, 0.95)
        XCTAssertEqual(result.processingTime, 2.5)
    }
    
    func testStreamingConfig() {
        let defaultConfig = StreamingConfig()
        XCTAssertEqual(defaultConfig.chunkDuration, 10.0)
        XCTAssertEqual(defaultConfig.overlapDuration, 1.0)
        XCTAssertEqual(defaultConfig.minChunkSize, 1600)
        XCTAssertFalse(defaultConfig.useVAD)
        
        let customConfig = StreamingConfig(
            chunkDuration: 5.0,
            overlapDuration: 0.5,
            minChunkSize: 800,
            useVAD: true
        )
        XCTAssertEqual(customConfig.chunkDuration, 5.0)
        XCTAssertEqual(customConfig.overlapDuration, 0.5)
        XCTAssertEqual(customConfig.minChunkSize, 800)
        XCTAssertTrue(customConfig.useVAD)
    }
    
    func testParakeetAudioVAD() {
        // Create a simple test audio signal
        let sampleRate = 16000
        let duration: Float = 3.0
        let numSamples = Int(Float(sampleRate) * duration)
        
        // Create audio with silence at beginning and end
        var audioData = [Float](repeating: 0.0, count: numSamples)
        
        // Add speech-like signal in the middle (1-2 seconds)
        let startSample = sampleRate
        let endSample = 2 * sampleRate
        for i in startSample..<endSample {
            audioData[i] = Float.random(in: -0.8...0.8)
        }
        
        let audio = MLXArray(audioData)
        
        // Test VAD
        let segments = ParakeetAudio.detectVoiceActivity(
            in: audio,
            sampleRate: sampleRate,
            frameDuration: 0.03,
            threshold: 0.3
        )
        
        // Should detect at least one segment with speech
        XCTAssertGreaterThan(segments.count, 0)
        
        // The detected segment should be roughly in the middle
        if let firstSegment = segments.first {
            XCTAssertGreaterThan(firstSegment.start, sampleRate / 2)
            XCTAssertLessThan(firstSegment.start, Int(1.5 * Float(sampleRate)))
        }
    }
    
    func testParakeetAudioNoiseReduction() {
        // Create noisy audio
        let audioData = [Float](repeating: 0.5, count: 1000)
        let audio = MLXArray(audioData)
        
        // Apply noise reduction
        let denoised = ParakeetAudio.reduceNoise(
            in: audio,
            reductionStrength: 0.5
        )
        
        // Check that output has same shape
        XCTAssertEqual(denoised.shape, audio.shape)
        
        // Check that values are reduced
        let maxValue = denoised.max().item(Float.self)
        XCTAssertLessThanOrEqual(maxValue, 0.5)
    }
    
    func testModelDetection() throws {
        // Create test configurations
        let tdtConfig: [String: Any] = [
            "decoding": ["model_type": "tdt"],
            "preprocessor": ["sample_rate": 16000]
        ]
        
        let ctcConfig: [String: Any] = [
            "decoder": ["decoder_type": "ctc"],
            "preprocessor": ["sample_rate": 16000]
        ]
        
        // Write test configs
        let tempDir = FileManager.default.temporaryDirectory
        let tdtConfigPath = tempDir.appendingPathComponent("tdt_config.json")
        let ctcConfigPath = tempDir.appendingPathComponent("ctc_config.json")
        
        try JSONSerialization.data(withJSONObject: tdtConfig)
            .write(to: tdtConfigPath)
        try JSONSerialization.data(withJSONObject: ctcConfig)
            .write(to: ctcConfigPath)
        
        // Test detection
        XCTAssertEqual(try ParakeetMLX.detectModelType(from: tdtConfigPath), .tdtctc)
        XCTAssertEqual(try ParakeetMLX.detectModelType(from: ctcConfigPath), .ctc)
        
        // Clean up
        try FileManager.default.removeItem(at: tdtConfigPath)
        try FileManager.default.removeItem(at: ctcConfigPath)
    }
    
    func testErrorTypes() {
        let configError = ParakeetError.configurationError("Test config error")
        XCTAssertEqual(configError.errorDescription, "Configuration error: Test config error")
        
        let streamingError = ParakeetError.streamingError("Test streaming error")
        XCTAssertEqual(streamingError.errorDescription, "Streaming error: Test streaming error")
        
        let decodingError = ParakeetError.decodingError("Test decoding error")
        XCTAssertEqual(decodingError.errorDescription, "Decoding error: Test decoding error")
    }
}