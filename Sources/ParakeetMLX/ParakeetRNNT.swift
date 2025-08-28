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

/// Top-k function to get top k values and indices
fileprivate func topK(_ x: MLXArray, k: Int) -> (values: MLXArray, indices: MLXArray) {
    // Sort in descending order and take top k
    let sorted = MLX.argSort(x, axis: -1)
    let topIndices = sorted[(sorted.shape[0] - k)...]
    let topValues = MLX.take(x, topIndices, axis: -1)
    return (topValues, topIndices)
}

// MARK: - RNNT Model Configuration

public struct RNNTModelConfig: Codable {
    public let encoder: ConformerConfig
    public let preprocessor: PreprocessConfig
    public let prediction: PredictConfig
    public let joint: JointConfig
    public let maxSymbolsPerStep: Int
    
    enum CodingKeys: String, CodingKey {
        case encoder
        case preprocessor
        case prediction
        case joint
        case maxSymbolsPerStep = "max_symbols_per_step"
    }
    
    public init(
        encoder: ConformerConfig,
        preprocessor: PreprocessConfig,
        prediction: PredictConfig,
        joint: JointConfig,
        maxSymbolsPerStep: Int = 5
    ) {
        self.encoder = encoder
        self.preprocessor = preprocessor
        self.prediction = prediction
        self.joint = joint
        self.maxSymbolsPerStep = maxSymbolsPerStep
    }
}

// MARK: - RNNT Decoder

/// RNNT decoder implementation with greedy and beam search
public class RNNTDecoder {
    
    // MARK: - Properties
    
    private let blankToken: Int
    private let vocabulary: [String]
    private let maxSymbolsPerStep: Int
    
    // MARK: - Initialization
    
    public init(
        vocabulary: [String],
        blankToken: Int,
        maxSymbolsPerStep: Int = 5
    ) {
        self.vocabulary = vocabulary
        self.blankToken = blankToken
        self.maxSymbolsPerStep = maxSymbolsPerStep
    }
    
    // MARK: - Greedy Decoding
    
    /// Performs greedy RNNT decoding
    /// - Parameters:
    ///   - encoder: Encoder output [batch, time, hidden]
    ///   - predictor: Prediction network
    ///   - joiner: Joint network
    /// - Returns: Decoded token sequences
    public func greedyDecode(
        encoderOutput: MLXArray,
        predictor: PredictNetwork,
        joiner: JointNetwork
    ) -> [[Int]] {
        let batchSize = encoderOutput.shape[0]
        let maxTime = encoderOutput.shape[1]
        
        var decodedBatch: [[Int]] = []
        
        // Process each batch item
        for b in 0..<batchSize {
            var decoded: [Int] = []
            var hiddenState: (MLXArray, MLXArray)? = nil
            var timeIdx = 0
            
            // Get encoder output for this batch
            let encBatch = encoderOutput[b].expandedDimensions(axis: 0)
            
            while timeIdx < maxTime {
                let encFrame = encBatch[0..., timeIdx].expandedDimensions(axis: 1)
                
                // Symbols emitted at this time step
                var symbolsThisStep = 0
                
                while symbolsThisStep < maxSymbolsPerStep {
                    // Get prediction
                    let lastToken = decoded.isEmpty ? nil : MLXArray(decoded.last!)
                    let inputToken = lastToken?.expandedDimensions(axis: 0)
                                               .expandedDimensions(axis: 0)
                    
                    let (predOutput, newHidden) = predictor(inputToken, hiddenState)
                    hiddenState = newHidden
                    
                    // Joint network
                    let jointOutput = joiner(encFrame, predOutput)
                    
                    // Get probabilities
                    let probs = MLX.softmax(jointOutput[0, 0, 0], axis: -1)
                    let nextToken = MLX.argMax(probs).item(Int.self)
                    
                    if nextToken == blankToken {
                        // Move to next time step
                        break
                    } else {
                        // Emit token
                        decoded.append(nextToken)
                        symbolsThisStep += 1
                    }
                }
                
                timeIdx += 1
            }
            
            decodedBatch.append(decoded)
        }
        
        return decodedBatch
    }
    
    // MARK: - Beam Search
    
    private struct RNNTBeamState {
        let tokens: [Int]
        let score: Float
        let hiddenState: (MLXArray, MLXArray)?
        let timeIdx: Int
        
        init(
            tokens: [Int] = [],
            score: Float = 0.0,
            hiddenState: (MLXArray, MLXArray)? = nil,
            timeIdx: Int = 0
        ) {
            self.tokens = tokens
            self.score = score
            self.hiddenState = hiddenState
            self.timeIdx = timeIdx
        }
    }
    
    /// Performs beam search RNNT decoding
    /// - Parameters:
    ///   - encoderOutput: Encoder output [batch, time, hidden]
    ///   - predictor: Prediction network
    ///   - joiner: Joint network
    ///   - beamSize: Beam size
    /// - Returns: Decoded token sequences
    public func beamSearchDecode(
        encoderOutput: MLXArray,
        predictor: PredictNetwork,
        joiner: JointNetwork,
        beamSize: Int = 10
    ) -> [[Int]] {
        let batchSize = encoderOutput.shape[0]
        let maxTime = encoderOutput.shape[1]
        
        var decodedBatch: [[Int]] = []
        
        // Process each batch item
        for b in 0..<batchSize {
            // Initialize beam with empty hypothesis
            var beams = [RNNTBeamState()]
            
            // Get encoder output for this batch
            let encBatch = encoderOutput[b].expandedDimensions(axis: 0)
            
            // Process all time steps
            for t in 0..<maxTime {
                var nextBeams: [RNNTBeamState] = []
                let encFrame = encBatch[0..., t].expandedDimensions(axis: 1)
                
                for beam in beams {
                    // Skip if beam already processed all time steps
                    guard beam.timeIdx <= t else {
                        nextBeams.append(beam)
                        continue
                    }
                    
                    // Expand beam with possible tokens
                    // var expansions: [(token: Int, score: Float)] = []
                    
                    // Get prediction for current hypothesis
                    let lastToken = beam.tokens.isEmpty ? nil : MLXArray(beam.tokens.last!)
                    let inputToken = lastToken?.expandedDimensions(axis: 0)
                                               .expandedDimensions(axis: 0)
                    
                    let (predOutput, newHidden) = predictor(inputToken, beam.hiddenState)
                    
                    // Joint network
                    let jointOutput = joiner(encFrame, predOutput)
                    
                    // Get log probabilities
                    let logProbs = logSoftmax(jointOutput[0, 0, 0], axis: -1)
                    
                    // Get top-k tokens
                    let topKCount = min(beamSize, vocabulary.count)
                    let (topValues, topIndices) = topK(logProbs, k: topKCount)
                    
                    // Process top tokens
                    for k in 0..<topKCount {
                        let token = topIndices[k].item(Int.self)
                        let logProb = topValues[k].item(Float.self)
                        let newScore = beam.score + logProb
                        
                        if token == blankToken {
                            // Blank token - move to next time step
                            nextBeams.append(RNNTBeamState(
                                tokens: beam.tokens,
                                score: newScore,
                                hiddenState: beam.hiddenState,
                                timeIdx: t + 1
                            ))
                        } else {
                            // Non-blank token - emit and stay at current time
                            nextBeams.append(RNNTBeamState(
                                tokens: beam.tokens + [token],
                                score: newScore,
                                hiddenState: newHidden,
                                timeIdx: t
                            ))
                        }
                    }
                }
                
                // Prune beams
                beams = nextBeams.sorted { $0.score > $1.score }
                    .prefix(beamSize)
                    .map { $0 }
            }
            
            // Get best hypothesis
            let bestBeam = beams.max { $0.score < $1.score } ?? RNNTBeamState()
            decodedBatch.append(bestBeam.tokens)
        }
        
        return decodedBatch
    }
    
    /// Decode tokens to string
    public func tokensToString(_ tokens: [Int]) -> String {
        return tokens.compactMap { idx in
            guard idx >= 0 && idx < vocabulary.count else { return nil }
            return vocabulary[idx]
        }.joined()
    }
}

// MARK: - Parakeet RNNT Model

/// Parakeet RNN-Transducer model implementation
public class ParakeetRNNT: Module {
    
    // MARK: - Properties
    
    public let config: RNNTModelConfig
    private let encoder: Conformer
    private let predictor: PredictNetwork
    private let joiner: JointNetwork
    private let decoder: RNNTDecoder
    
    // MARK: - Initialization
    
    /// Initialize Parakeet RNNT model
    /// - Parameter config: RNNT model configuration
    public init(config: RNNTModelConfig) {
        self.config = config
        
        // Initialize components
        self.encoder = Conformer(config: config.encoder)
        self.predictor = PredictNetwork(config: config.prediction)
        self.joiner = JointNetwork(config: config.joint)
        
        // Initialize decoder
        self.decoder = RNNTDecoder(
            vocabulary: config.joint.vocabulary,
            blankToken: config.joint.numClasses,
            maxSymbolsPerStep: config.maxSymbolsPerStep
        )
        
        
        super.init()
    }
    
    // MARK: - Forward Pass
    
    /// Forward pass through encoder
    /// - Parameters:
    ///   - audioFeatures: Input audio features [batch, time, features]
    ///   - cache: Optional cache for streaming
    /// - Returns: Encoder output [batch, time, hidden]
    public func encode(
        _ audioFeatures: MLXArray,
        cache: [ConformerCache?]? = nil
    ) -> MLXArray {
        let (encoderOutput, _) = encoder(audioFeatures, cache: cache)
        return encoderOutput
    }
    
    /// Compute joint network output
    /// - Parameters:
    ///   - encoderOutput: Encoder output [batch, enc_time, hidden]
    ///   - predictionOutput: Prediction output [batch, pred_time, hidden]
    /// - Returns: Joint output [batch, enc_time, pred_time, vocab]
    public func computeJoint(
        encoderOutput: MLXArray,
        predictionOutput: MLXArray
    ) -> MLXArray {
        return joiner(encoderOutput, predictionOutput)
    }
    
    // MARK: - Inference Methods
    
    /// Process raw audio and transcribe using greedy decoding
    /// - Parameters:
    ///   - audio: Raw audio samples
    ///   - sampleRate: Audio sample rate
    /// - Returns: Transcribed text
    public func transcribe(
        _ audio: MLXArray,
        sampleRate: Int? = nil
    ) -> String {
        // Extract features
        let features = try! getLogMel(audio, config: config.preprocessor)
        
        // Add batch dimension if needed
        let batchedFeatures = features.ndim == 2 
            ? features.expandedDimensions(axis: 0) 
            : features
        
        // Encode
        let encoderOutput = encode(batchedFeatures)
        
        // Decode
        let tokens = decoder.greedyDecode(
            encoderOutput: encoderOutput,
            predictor: predictor,
            joiner: joiner
        )
        
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
        // Extract features
        let features = try! getLogMel(audio, config: config.preprocessor)
        
        // Add batch dimension
        let batchedFeatures = features.ndim == 2 
            ? features.expandedDimensions(axis: 0) 
            : features
        
        // Encode
        let encoderOutput = encode(batchedFeatures)
        
        // Beam search decode
        let tokens = decoder.beamSearchDecode(
            encoderOutput: encoderOutput,
            predictor: predictor,
            joiner: joiner,
            beamSize: beamSize
        )
        
        // Convert to string
        return decoder.tokensToString(tokens[0])
    }
    
    // MARK: - Streaming Support
    
    /// Streaming state for RNNT model
    public class RNNTStreamingState {
        var encoderCache: [ConformerCache]?
        var predictorHidden: (MLXArray, MLXArray)?
        var tokens: [Int] = []
        var audioBuffer: MLXArray?
        let chunkSize: Int
        
        init(chunkSize: Int) {
            self.chunkSize = chunkSize
        }
    }
    
    /// Initialize streaming state
    /// - Parameters:
    ///   - chunkSize: Size of audio chunks in samples
    ///   - contextSize: Context size for encoder
    /// - Returns: Initialized streaming state
    public func initStreamingState(
        chunkSize: Int = 16000,
        contextSize: Int = 64
    ) -> RNNTStreamingState {
        let state = RNNTStreamingState(chunkSize: chunkSize)
        
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
    /// - Returns: Current partial transcription
    public func streamingStep(
        audioChunk: MLXArray,
        state: RNNTStreamingState
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
            return decoder.tokensToString(state.tokens)
        }
        
        // Process chunk
        let chunk = audio[..<state.chunkSize]
        state.audioBuffer = audio.shape[0] > state.chunkSize 
            ? audio[state.chunkSize...] 
            : nil
        
        // Extract features
        let features = try! getLogMel(chunk, config: config.preprocessor)
        
        // Add batch dimension
        let batchedFeatures = features.expandedDimensions(axis: 0)
        
        // Encode with cache
        let encoderOutput = encode(batchedFeatures, cache: state.encoderCache)
        
        // Decode frames
        for t in 0..<encoderOutput.shape[1] {
            let encFrame = encoderOutput[0..., t]
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 1)
            
            var symbolsThisStep = 0
            
            while symbolsThisStep < config.maxSymbolsPerStep {
                // Get prediction
                let lastToken = state.tokens.isEmpty ? nil : MLXArray(state.tokens.last!)
                let inputToken = lastToken?.expandedDimensions(axis: 0)
                                           .expandedDimensions(axis: 0)
                
                let (predOutput, newHidden) = predictor(inputToken, state.predictorHidden)
                state.predictorHidden = newHidden
                
                // Joint network
                let jointOutput = joiner(encFrame, predOutput)
                
                // Get probabilities
                let probs = MLX.softmax(jointOutput[0, 0, 0], axis: -1)
                let nextToken = MLX.argMax(probs).item(Int.self)
                
                if nextToken == config.joint.numClasses {
                    // Blank token - move to next frame
                    break
                } else {
                    // Emit token
                    state.tokens.append(nextToken)
                    symbolsThisStep += 1
                }
            }
        }
        
        return decoder.tokensToString(state.tokens)
    }
    
    // MARK: - Training Support
    
    /// Compute RNNT loss for training
    /// - Note: Full RNNT loss implementation requires complex forward-backward algorithm
    /// This is a placeholder that should be replaced with proper implementation
    public func computeLoss(
        audioFeatures: MLXArray,
        targets: MLXArray,
        inputLengths: [Int],
        targetLengths: [Int]
    ) -> MLXArray {
        // Encode audio
        let _ = encode(audioFeatures)
        
        // This is a simplified placeholder
        // Full RNNT loss would require implementing the
        // forward-backward algorithm over the lattice
        
        return MLXArray(0.0)
    }
    
    // MARK: - Model Loading
    
    /// Load model weights from file
    /// - Parameter path: Path to weights file
    public func loadWeights(from path: URL) throws {
        let weights = try MLX.loadArrays(url: path)
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
        let weights = parameters()
        // Convert flattened parameters to dictionary
        var dict: [String: MLXArray] = [:]
        for (key, value) in weights.flattened() {
            dict[key] = value
        }
        try MLX.save(arrays: dict, url: path)
    }
}