import Foundation
import MLX
import MLXNN

// MARK: - Helper Functions

/// Log softmax function
fileprivate func logSoftmax(_ x: MLXArray, axis: Int = -1) -> MLXArray {
    let maxVals = x.max(axis: axis, keepDims: true)
    let shifted = x - maxVals
    let expShifted = exp(shifted)
    let sumExp = expShifted.sum(axis: axis, keepDims: true)
    return shifted - log(sumExp)
}

// MARK: - CTC Model Configuration

public struct CTCModelConfig: Codable {
    public let encoder: ConformerConfig
    public let preprocessor: PreprocessConfig
    public let numClasses: Int
    public let vocabulary: [String]
    public let blankToken: Int
    
    enum CodingKeys: String, CodingKey {
        case encoder
        case preprocessor
        case numClasses = "num_classes"
        case vocabulary
        case blankToken = "blank_token"
    }
    
    public init(
        encoder: ConformerConfig,
        preprocessor: PreprocessConfig,
        numClasses: Int,
        vocabulary: [String],
        blankToken: Int = 0
    ) {
        self.encoder = encoder
        self.preprocessor = preprocessor
        self.numClasses = numClasses
        self.vocabulary = vocabulary
        self.blankToken = blankToken
    }
}

// MARK: - CTC Output Head

/// CTC output head for classification
public class CTCHead: Module {
    private let linear: Linear
    private let dropout: Dropout?
    
    /// Initialize CTC head
    /// - Parameters:
    ///   - inputDim: Input dimension from encoder
    ///   - numClasses: Number of output classes (including blank)
    ///   - dropoutRate: Dropout rate (optional)
    public init(inputDim: Int, numClasses: Int, dropoutRate: Float = 0.0) {
        self.linear = Linear(inputDim, numClasses)
        self.dropout = dropoutRate > 0 ? Dropout(p: dropoutRate) : nil
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, training: Bool = false) -> MLXArray {
        var output = x
        
        // Apply dropout if in training mode
        if let dropout = dropout, training {
            output = dropout(output)
        }
        
        // Apply linear projection
        output = linear(output)
        
        return output
    }
}

// MARK: - Parakeet CTC Model

/// Parakeet model with CTC (Connectionist Temporal Classification) head
public class ParakeetCTC: Module {
    
    // MARK: - Properties
    
    public let config: CTCModelConfig
    private let encoder: Conformer
    private let ctcHead: CTCHead
    private let decoder: CTCDecoder
    
    // MARK: - Initialization
    
    /// Initialize Parakeet CTC model
    /// - Parameter config: CTC model configuration
    public init(config: CTCModelConfig) {
        self.config = config
        
        // Initialize encoder
        self.encoder = Conformer(config: config.encoder)
        
        // Initialize CTC head
        // Add 1 for blank token if not already included
        let numClasses = config.blankToken >= config.numClasses 
            ? config.numClasses + 1 
            : config.numClasses
        self.ctcHead = CTCHead(
            inputDim: config.encoder.dModel,
            numClasses: numClasses
        )
        
        // Initialize decoder
        self.decoder = CTCDecoder(
            vocabulary: config.vocabulary,
            blankToken: config.blankToken
        )
        
        super.init()
    }
    
    // MARK: - Forward Pass
    
    /// Forward pass through the model
    /// - Parameters:
    ///   - audioFeatures: Input audio features [batch, time, features]
    ///   - lengths: Valid lengths for each batch item
    ///   - cache: Optional cache for streaming
    /// - Returns: CTC logits [batch, time, num_classes]
    public func callAsFunction(
        _ audioFeatures: MLXArray,
        lengths: [Int]? = nil,
        cache: [ConformerCache?]? = nil
    ) -> MLXArray {
        // Encode audio features
        let (encoderOutput, _) = encoder(audioFeatures, cache: cache)
        
        // Apply CTC head
        let logits = ctcHead(encoderOutput)
        
        return logits
    }
    
    // MARK: - Inference Methods
    
    /// Process raw audio and return CTC logits
    /// - Parameters:
    ///   - audio: Raw audio samples
    ///   - sampleRate: Audio sample rate
    /// - Returns: CTC logits
    public func processAudio(
        _ audio: MLXArray,
        sampleRate: Int? = nil
    ) -> MLXArray {
        // Extract features
        let features = try! getLogMel(audio, config: config.preprocessor)
        
        // Add batch dimension if needed
        let batchedFeatures = features.ndim == 2 
            ? features.expandedDimensions(axis: 0) 
            : features
        
        // Forward pass
        return callAsFunction(batchedFeatures)
    }
    
    /// Transcribe audio using greedy decoding
    /// - Parameters:
    ///   - audio: Raw audio samples
    ///   - sampleRate: Audio sample rate
    /// - Returns: Transcribed text
    public func transcribe(
        _ audio: MLXArray,
        sampleRate: Int? = nil
    ) -> String {
        // Get logits
        let logits = processAudio(audio, sampleRate: sampleRate)
        
        // Greedy decode
        let tokens = decoder.greedyDecode(logits: logits)
        
        // Convert to string
        return decoder.tokensToString(tokens[0])
    }
    
    /// Transcribe audio using beam search
    /// - Parameters:
    ///   - audio: Raw audio samples
    ///   - beamSize: Beam size for decoding
    ///   - sampleRate: Audio sample rate
    /// - Returns: Transcribed text
    public func transcribeWithBeamSearch(
        _ audio: MLXArray,
        beamSize: Int = 10,
        sampleRate: Int? = nil
    ) -> String {
        // Get logits
        let logits = processAudio(audio, sampleRate: sampleRate)
        
        // Beam search decode
        let tokens = decoder.beamSearchDecode(
            logits: logits,
            beamSize: beamSize
        )
        
        // Convert to string
        return decoder.tokensToString(tokens[0])
    }
    
    // MARK: - Streaming Support
    
    /// Streaming state for CTC model
    public class CTCStreamingState {
        var encoderCache: [ConformerCache]?
        var decoderState: CTCDecoder.PrefixBeamSearchState
        var audioBuffer: MLXArray?
        let chunkSize: Int
        let contextSize: Int
        
        init(chunkSize: Int, contextSize: Int) {
            self.chunkSize = chunkSize
            self.contextSize = contextSize
            self.decoderState = CTCDecoder.PrefixBeamSearchState()
        }
    }
    
    /// Initialize streaming state
    /// - Parameters:
    ///   - chunkSize: Size of audio chunks in samples
    ///   - contextSize: Context size for encoder
    /// - Returns: Initialized streaming state
    public func initStreamingState(
        chunkSize: Int = 16000,  // 1 second at 16kHz
        contextSize: Int = 64
    ) -> CTCStreamingState {
        let state = CTCStreamingState(
            chunkSize: chunkSize,
            contextSize: contextSize
        )
        
        // Initialize encoder cache if using local attention
        if config.encoder.selfAttentionModel == "rel_local_selfattn" {
            // Create cache array for all layers
            var caches: [ConformerCache] = []
            for _ in 0..<config.encoder.nLayers {
                caches.append(ConformerCache())
            }
            state.encoderCache = caches
        }
        
        return state
    }
    
    /// Process streaming audio chunk
    /// - Parameters:
    ///   - audioChunk: Audio chunk to process
    ///   - state: Current streaming state
    ///   - beamSize: Beam size for decoding
    /// - Returns: Current partial transcription
    public func streamingStep(
        audioChunk: MLXArray,
        state: CTCStreamingState,
        beamSize: Int = 10
    ) -> String {
        // Buffer audio if needed
        let audio: MLXArray
        if let buffer = state.audioBuffer {
            audio = MLX.concatenated([buffer, audioChunk], axis: 0)
        } else {
            audio = audioChunk
        }
        
        // Check if we have enough samples
        guard audio.shape[0] >= state.chunkSize else {
            state.audioBuffer = audio
            return decoder.tokensToString(state.decoderState.beams.first?.prefix ?? [])
        }
        
        // Process chunk
        let chunk = audio[..<state.chunkSize]
        state.audioBuffer = audio.shape[0] > state.chunkSize 
            ? audio[state.chunkSize...] 
            : nil
        
        // Extract features for chunk
        let features = try! getLogMel(chunk, config: config.preprocessor)
        
        // Add batch dimension
        let batchedFeatures = features.expandedDimensions(axis: 0)
        
        // Encode with cache
        let (encoderOutput, _) = encoder(batchedFeatures, cache: state.encoderCache)
        
        // Apply CTC head
        let logits = ctcHead(encoderOutput)
        
        // Get log probabilities for current frames
        let logProbs = logSoftmax(logits[0], axis: -1)
        
        // Process each frame through beam search
        for t in 0..<logProbs.shape[0] {
            let frameLogProbs = logProbs[t]
            _ = decoder.streamingBeamSearchStep(
                logProbs: frameLogProbs,
                state: &state.decoderState,
                beamSize: beamSize
            )
        }
        
        // Return current best hypothesis
        return decoder.tokensToString(state.decoderState.beams.first?.prefix ?? [])
    }
    
    // MARK: - Training Support
    
    /// Compute CTC loss for training
    /// - Parameters:
    ///   - audioFeatures: Input audio features [batch, time, features]
    ///   - targets: Target token sequences [batch, target_length]
    ///   - inputLengths: Valid input lengths
    ///   - targetLengths: Valid target lengths
    /// - Returns: CTC loss value
    public func computeLoss(
        audioFeatures: MLXArray,
        targets: MLXArray,
        inputLengths: [Int],
        targetLengths: [Int]
    ) -> MLXArray {
        // Forward pass
        let logits = callAsFunction(audioFeatures)
        
        // Compute CTC loss
        let ctcLoss = CTCLoss(blankToken: config.blankToken)
        return ctcLoss(
            logits: logits,
            targets: targets,
            inputLengths: inputLengths,
            targetLengths: targetLengths
        )
    }
    
    // MARK: - Model Loading
    
    /// Load model weights from file
    /// - Parameter path: Path to weights file
    public func loadWeights(from path: URL) throws {
        // Load weights dictionary
        let weights = try MLX.loadArrays(url: path)
        
        // Update model parameters
        // Note: This would need proper weight mapping
        // based on the saved format
        // Convert to ModuleParameters format
        var moduleParams = ModuleParameters()
        for (key, value) in weights {
            moduleParams[key] = .value(value)
        }
        update(parameters: moduleParams)
    }
    
    /// Save model weights to file
    /// - Parameter path: Path to save weights
    public func saveWeights(to path: URL) throws {
        // Get model parameters
        let weights = parameters()
        
        // Save weights
        // Convert flattened parameters to dictionary
        var dict: [String: MLXArray] = [:]
        for (key, value) in weights.flattened() {
            dict[key] = value
        }
        try MLX.save(arrays: dict, url: path)
    }
}