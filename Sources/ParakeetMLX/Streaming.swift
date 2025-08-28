import Foundation
@preconcurrency import MLX
import Combine

// MARK: - Streaming Types

/// Token state for streaming transcription
public enum TokenState: Sendable {
    case draft      // Preliminary token that may change
    case finalized  // Confirmed token that won't change
}

/// Extended token for streaming with state information
public struct StreamingToken: Sendable {
    public let token: AlignedToken
    public var state: TokenState
    public let timestamp: Date
    
    public init(token: AlignedToken, state: TokenState = .draft) {
        self.token = token
        self.state = state
        self.timestamp = Date()
    }
}

/// Streaming transcription result with draft and finalized tokens
public struct StreamingResult: Sendable {
    public let finalizedTokens: [AlignedToken]
    public let draftTokens: [AlignedToken]
    public let audioPosition: Float  // Position in seconds
    public let processingTime: TimeInterval
    
    public var allTokens: [AlignedToken] {
        finalizedTokens + draftTokens
    }
    
    public var text: String {
        allTokens.map { $0.text }.joined()
    }
    
    public var finalizedText: String {
        finalizedTokens.map { $0.text }.joined()
    }
    
    public var draftText: String {
        draftTokens.map { $0.text }.joined()
    }
}

// MARK: - Audio Buffer Management

/// Manages audio buffering for streaming transcription
public class AudioBuffer {
    private var buffer: MLXArray
    private let sampleRate: Int
    private let windowSize: Int  // Size of processing window in samples
    private let overlapSize: Int // Size of overlap between windows
    private var totalSamplesProcessed: Int = 0
    
    public init(
        sampleRate: Int,
        windowDuration: Float = 10.0,
        overlapDuration: Float = 2.0
    ) {
        self.sampleRate = sampleRate
        self.windowSize = Int(windowDuration * Float(sampleRate))
        self.overlapSize = Int(overlapDuration * Float(sampleRate))
        self.buffer = MLXArray([])
    }
    
    /// Add audio samples to buffer
    public func append(_ audio: MLXArray) {
        if buffer.shape[0] == 0 {
            buffer = audio
        } else {
            buffer = MLX.concatenated([buffer, audio], axis: 0)
        }
    }
    
    /// Check if buffer has enough data for processing
    public var hasEnoughData: Bool {
        buffer.shape[0] >= windowSize
    }
    
    /// Get current window for processing
    public func getWindow() -> MLXArray? {
        guard hasEnoughData else { return nil }
        return buffer[0..<min(windowSize, buffer.shape[0])]
    }
    
    /// Advance buffer by removing processed samples
    public func advance(by samples: Int) {
        let advanceAmount = min(samples, buffer.shape[0])
        totalSamplesProcessed += advanceAmount
        
        if advanceAmount < buffer.shape[0] {
            buffer = buffer[advanceAmount...]
        } else {
            buffer = MLXArray([])
        }
    }
    
    /// Get current position in seconds
    public var currentPosition: Float {
        Float(totalSamplesProcessed) / Float(sampleRate)
    }
    
    /// Clear buffer
    public func clear() {
        buffer = MLXArray([])
        totalSamplesProcessed = 0
    }
    
    /// Get buffer size in samples
    public var size: Int {
        buffer.shape[0]
    }
}

// MARK: - Mel Buffer for Streaming

/// Manages mel spectrogram buffering for continuous processing
public class MelBuffer {
    private var melFrames: MLXArray?
    private let frameShift: Int
    private var totalFramesProcessed: Int = 0
    
    public init(frameShift: Int) {
        self.frameShift = frameShift
    }
    
    /// Accumulate new mel frames
    public func accumulate(_ mel: MLXArray) {
        // Remove batch dimension if present
        let frames = mel.ndim == 3 ? mel.squeezed(axis: 0) : mel
        
        if melFrames == nil {
            melFrames = frames
        } else {
            melFrames = MLX.concatenated([melFrames!, frames], axis: 0)
        }
    }
    
    /// Get frames for processing
    public func getFrames(count: Int? = nil) -> MLXArray? {
        guard let frames = melFrames else { return nil }
        
        if let count = count {
            let availableCount = min(count, frames.shape[0])
            return frames[0..<availableCount].expandedDimensions(axis: 0)
        }
        
        return frames.expandedDimensions(axis: 0)
    }
    
    /// Advance buffer by removing processed frames
    public func advance(by frameCount: Int) {
        guard let frames = melFrames else { return }
        
        totalFramesProcessed += frameCount
        
        if frameCount < frames.shape[0] {
            melFrames = frames[frameCount...]
        } else {
            melFrames = nil
        }
    }
    
    /// Clear buffer
    public func clear() {
        melFrames = nil
        totalFramesProcessed = 0
    }
    
    public var frameCount: Int {
        melFrames?.shape[0] ?? 0
    }
}

// MARK: - Streaming Parakeet Implementation

/// Real-time streaming transcription with Parakeet
public actor StreamingParakeet {
    private let model: ParakeetTDT
    private let contextWindow: Int
    private let lookbackFrames: Int
    private let depth: Int
    
    // Buffers
    private let audioBuffer: AudioBuffer
    private let melBuffer: MelBuffer
    
    // Cache management
    private var conformerCaches: [RotatingConformerCache]
    
    // Decoder state
    private var decoderHidden: (MLXArray, MLXArray)?
    private var lastTokenId: Int?
    
    // Token management
    private var finalizedTokens: [AlignedToken] = []
    private var draftTokens: [AlignedToken] = []
    private var lastProcessedPosition: Float = 0.0
    
    // Configuration
    private let minProcessingSize: Int  // Minimum audio samples for processing
    private let maxDraftTokens: Int = 10  // Maximum number of draft tokens to keep
    
    public init(
        model: ParakeetTDT,
        contextWindow: Int = 256,
        lookbackFrames: Int = 32,
        depth: Int = 1,
        windowDuration: Float = 10.0,
        overlapDuration: Float = 2.0
    ) {
        self.model = model
        self.contextWindow = contextWindow
        self.lookbackFrames = lookbackFrames
        self.depth = depth
        
        // Initialize buffers
        self.audioBuffer = AudioBuffer(
            sampleRate: model.preprocessConfig.sampleRate,
            windowDuration: windowDuration,
            overlapDuration: overlapDuration
        )
        
        self.melBuffer = MelBuffer(
            frameShift: model.preprocessConfig.hopLength
        )
        
        // Initialize caches for each conformer layer
        self.conformerCaches = (0..<model.encoderConfig.nLayers).map { _ in
            RotatingConformerCache(
                capacity: contextWindow,
                cacheDropSize: lookbackFrames * depth
            )
        }
        
        // Calculate minimum processing size (1 second of audio)
        self.minProcessingSize = model.preprocessConfig.sampleRate
    }
    
    // MARK: - Public API
    
    /// Process audio chunk and return streaming result
    public func processAudioChunk(_ audioChunk: MLXArray) async throws -> StreamingResult {
        let startTime = Date()
        
        // Add audio to buffer
        audioBuffer.append(audioChunk)
        
        // Check if we have enough data to process
        guard audioBuffer.size >= minProcessingSize else {
            return StreamingResult(
                finalizedTokens: finalizedTokens,
                draftTokens: draftTokens,
                audioPosition: audioBuffer.currentPosition,
                processingTime: Date().timeIntervalSince(startTime)
            )
        }
        
        // Get processing window
        guard let audioWindow = audioBuffer.getWindow() else {
            return StreamingResult(
                finalizedTokens: finalizedTokens,
                draftTokens: draftTokens,
                audioPosition: audioBuffer.currentPosition,
                processingTime: Date().timeIntervalSince(startTime)
            )
        }
        
        // Compute mel spectrogram
        let mel = try getLogMel(audioWindow, config: model.preprocessConfig)
        melBuffer.accumulate(mel)
        
        // Get frames for processing
        guard let melFrames = melBuffer.getFrames() else {
            return StreamingResult(
                finalizedTokens: finalizedTokens,
                draftTokens: draftTokens,
                audioPosition: audioBuffer.currentPosition,
                processingTime: Date().timeIntervalSince(startTime)
            )
        }
        
        // Process through encoder with caching
        let (features, lengths) = model.encode(melFrames, cache: conformerCaches)
        
        // Decode tokens
        let tokens = try decodeWithStreaming(features: features, lengths: lengths)
        
        // Update token states
        updateTokenStates(newTokens: tokens)
        
        // Advance buffers
        let processedSamples = model.preprocessConfig.hopLength * melBuffer.frameCount / 2
        audioBuffer.advance(by: processedSamples)
        melBuffer.advance(by: melBuffer.frameCount / 2)
        
        return StreamingResult(
            finalizedTokens: finalizedTokens,
            draftTokens: draftTokens,
            audioPosition: audioBuffer.currentPosition,
            processingTime: Date().timeIntervalSince(startTime)
        )
    }
    
    /// Process multiple audio chunks as a stream
    public func processStream(_ audioStream: AsyncStream<MLXArray>) async throws {
        for await audioChunk in audioStream {
            _ = try await processAudioChunk(audioChunk)
        }
    }
    
    /// Get current transcription result
    public func getCurrentResult() async -> StreamingResult {
        StreamingResult(
            finalizedTokens: finalizedTokens,
            draftTokens: draftTokens,
            audioPosition: audioBuffer.currentPosition,
            processingTime: 0
        )
    }
    
    /// Finalize all remaining draft tokens
    public func finalize() async -> AlignedResult {
        // Move all draft tokens to finalized
        finalizedTokens.append(contentsOf: draftTokens)
        draftTokens.removeAll()
        
        // Create sentences and result
        let sentences = tokensToSentences(finalizedTokens)
        return AlignedResult(sentences: sentences)
    }
    
    /// Reset streaming state
    public func reset() async {
        audioBuffer.clear()
        melBuffer.clear()
        
        conformerCaches = (0..<model.encoderConfig.nLayers).map { _ in
            RotatingConformerCache(
                capacity: contextWindow,
                cacheDropSize: lookbackFrames * depth
            )
        }
        
        decoderHidden = nil
        lastTokenId = nil
        finalizedTokens.removeAll()
        draftTokens.removeAll()
        lastProcessedPosition = 0.0
    }
    
    // MARK: - Private Methods
    
    private func decodeWithStreaming(
        features: MLXArray,
        lengths: MLXArray
    ) throws -> [AlignedToken] {
        // Determine clean and dirty regions
        let totalLength = Int(lengths[0].item(Int32.self))
        let dropSize = lookbackFrames * depth
        let cleanLength = max(0, totalLength - dropSize)
        
        var allTokens: [AlignedToken] = []
        
        // Process clean region (won't change)
        if cleanLength > 0 {
            let cleanFeatures = features[0..<1, 0..<cleanLength, 0...]
            let cleanLengths = MLXArray([cleanLength])
            
            let (cleanResult, newHiddenState) = try model.decode(
                features: cleanFeatures,
                lengths: cleanLengths,
                lastToken: lastTokenId.map { [$0] },
                hiddenState: decoderHidden.map { [$0] },
                config: DecodingConfig()
            )
            
            if let firstBatch = cleanResult.first {
                allTokens.append(contentsOf: adjustTokenTimestamps(firstBatch))
            }
            
            // Update decoder state
            if let hidden = newHiddenState.first {
                decoderHidden = hidden
            }
            
            if let lastToken = cleanResult.first?.last {
                lastTokenId = lastToken.id
            }
        }
        
        // Process dirty region (may change in next iteration)
        if totalLength > cleanLength {
            let dirtyFeatures = features[0..<1, cleanLength..<totalLength, 0...]
            let dirtyLengths = MLXArray([totalLength - cleanLength])
            
            let (dirtyResult, _) = try model.decode(
                features: dirtyFeatures,
                lengths: dirtyLengths,
                lastToken: lastTokenId.map { [$0] },
                hiddenState: decoderHidden.map { [$0] },
                config: DecodingConfig()
            )
            
            if let firstBatch = dirtyResult.first {
                allTokens.append(contentsOf: adjustTokenTimestamps(firstBatch))
            }
        }
        
        return allTokens
    }
    
    private func adjustTokenTimestamps(_ tokens: [AlignedToken]) -> [AlignedToken] {
        // Adjust token timestamps based on current position
        return tokens.map { token in
            var adjustedToken = token
            adjustedToken.start += lastProcessedPosition
            return adjustedToken
        }
    }
    
    private func updateTokenStates(newTokens: [AlignedToken]) {
        guard !newTokens.isEmpty else { return }
        
        // Find boundary between finalized and draft tokens
        let finalizeThreshold = newTokens.count - maxDraftTokens
        
        if finalizeThreshold > 0 {
            // Move tokens from draft to finalized
            let toFinalize = Array(newTokens.prefix(finalizeThreshold))
            finalizedTokens.append(contentsOf: toFinalize)
            
            // Update draft tokens
            draftTokens = Array(newTokens.suffix(maxDraftTokens))
        } else {
            // All tokens are draft
            draftTokens = newTokens
        }
        
        // Update last processed position
        if let lastToken = newTokens.last {
            lastProcessedPosition = lastToken.end
        }
    }
    
    private func tokensToSentences(_ tokens: [AlignedToken]) -> [AlignedSentence] {
        guard !tokens.isEmpty else { return [] }
        
        var sentences: [AlignedSentence] = []
        var currentTokens: [AlignedToken] = []
        
        for token in tokens {
            currentTokens.append(token)
            
            // Sentence boundary detection
            if token.text.contains(".") || token.text.contains("!") || token.text.contains("?") {
                sentences.append(AlignedSentence(tokens: currentTokens))
                currentTokens = []
            }
        }
        
        // Add remaining tokens as final sentence
        if !currentTokens.isEmpty {
            sentences.append(AlignedSentence(tokens: currentTokens))
        }
        
        return sentences
    }
}

// MARK: - Async Stream Extensions

public extension StreamingParakeet {
    /// Create an async stream for continuous transcription
    func transcriptionStream(
        audioStream: AsyncStream<MLXArray>
    ) -> AsyncStream<StreamingResult> {
        AsyncStream { continuation in
            Task {
                do {
                    for await audioChunk in audioStream {
                        let result = try await self.processAudioChunk(audioChunk)
                        continuation.yield(result)
                    }
                    
                    // Finalize and send last result
                    let finalResult = await self.finalize()
                    let streamingResult = StreamingResult(
                        finalizedTokens: finalResult.sentences.flatMap { $0.tokens },
                        draftTokens: [],
                        audioPosition: self.audioBuffer.currentPosition,
                        processingTime: 0
                    )
                    continuation.yield(streamingResult)
                    continuation.finish()
                } catch {
                    continuation.finish()
                }
            }
        }
    }
}

// MARK: - Swift Async/Await Context Manager Pattern

/// Context manager for streaming transcription session
public class StreamingContext {
    private let streamingParakeet: StreamingParakeet
    private var isActive = false
    
    public init(
        model: ParakeetTDT,
        contextWindow: Int = 256,
        lookbackFrames: Int = 32,
        depth: Int = 1
    ) {
        self.streamingParakeet = StreamingParakeet(
            model: model,
            contextWindow: contextWindow,
            lookbackFrames: lookbackFrames,
            depth: depth
        )
    }
    
    /// Start streaming session
    public func start() async throws {
        guard !isActive else { return }
        isActive = true
        await streamingParakeet.reset()
    }
    
    /// Process audio chunk
    public func process(_ audio: MLXArray) async throws -> StreamingResult {
        guard isActive else {
            throw StreamingError.sessionNotActive
        }
        return try await streamingParakeet.processAudioChunk(audio)
    }
    
    /// End streaming session and get final result
    public func end() async throws -> AlignedResult {
        guard isActive else {
            throw StreamingError.sessionNotActive
        }
        isActive = false
        return await streamingParakeet.finalize()
    }
    
    /// Get current result without ending session
    public func getCurrentResult() async -> StreamingResult {
        await streamingParakeet.getCurrentResult()
    }
}

// MARK: - Streaming Errors

public enum StreamingError: Error, LocalizedError {
    case sessionNotActive
    case bufferUnderflow
    case processingError(String)
    
    public var errorDescription: String? {
        switch self {
        case .sessionNotActive:
            return "Streaming session is not active"
        case .bufferUnderflow:
            return "Not enough data in buffer for processing"
        case .processingError(let message):
            return "Processing error: \(message)"
        }
    }
}

// MARK: - Convenience Extensions

public extension ParakeetTDT {
    /// Create a streaming context for this model
    func createStreamingContext(
        contextWindow: Int = 256,
        lookbackFrames: Int = 32,
        depth: Int = 1
    ) -> StreamingContext {
        StreamingContext(
            model: self,
            contextWindow: contextWindow,
            lookbackFrames: lookbackFrames,
            depth: depth
        )
    }
    
    /// Create a streaming transcriber
    func createStreamingTranscriber(
        contextWindow: Int = 256,
        lookbackFrames: Int = 32,
        depth: Int = 1,
        windowDuration: Float = 10.0,
        overlapDuration: Float = 2.0
    ) -> StreamingParakeet {
        StreamingParakeet(
            model: self,
            contextWindow: contextWindow,
            lookbackFrames: lookbackFrames,
            depth: depth,
            windowDuration: windowDuration,
            overlapDuration: overlapDuration
        )
    }
}