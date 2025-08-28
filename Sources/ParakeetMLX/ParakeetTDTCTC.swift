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

// MARK: - TDNN Layer

/// Time Delay Neural Network (TDNN) layer implementation
public class TDNNLayer: Module {
    private let conv: Conv1d
    private let activation: ReLU
    private let batchNorm: BatchNorm?
    private let dropout: Dropout?
    
    /// Initialize TDNN layer
    /// - Parameters:
    ///   - inputDim: Input dimension
    ///   - outputDim: Output dimension
    ///   - context: Context window as array of offsets (e.g., [-2, -1, 0, 1, 2])
    ///   - dilation: Dilation factor for convolution
    ///   - useBatchNorm: Whether to use batch normalization
    ///   - dropoutRate: Dropout rate (0 to disable)
    public init(
        inputDim: Int,
        outputDim: Int,
        context: [Int],
        dilation: Int = 1,
        useBatchNorm: Bool = true,
        dropoutRate: Float = 0.0
    ) {
        // Calculate kernel size from context
        let kernelSize = context.count
        
        // Create 1D convolution for temporal modeling
        self.conv = Conv1d(
            inputChannels: inputDim,
            outputChannels: outputDim,
            kernelSize: kernelSize,
            stride: 1,
            padding: 0,
            dilation: dilation
        )
        
        self.activation = ReLU()
        self.batchNorm = useBatchNorm ? BatchNorm(featureCount: outputDim) : nil
        self.dropout = dropoutRate > 0 ? Dropout(p: dropoutRate) : nil
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, training: Bool = false) -> MLXArray {
        var output = x
        
        // Apply convolution
        output = conv(output)
        
        // Apply batch norm if enabled
        if let batchNorm = batchNorm {
            output = batchNorm(output)
        }
        
        // Apply activation
        output = activation(output)
        
        // Apply dropout if training
        if let dropout = dropout, training {
            output = dropout(output)
        }
        
        return output
    }
}

// MARK: - TDNN Block

/// TDNN block with multiple layers and residual connections
public class TDNNBlock: Module {
    private let layers: [TDNNLayer]
    private let residual: Linear?
    private let finalActivation: ReLU
    
    /// Initialize TDNN block
    /// - Parameters:
    ///   - inputDim: Input dimension
    ///   - outputDim: Output dimension per layer
    ///   - contexts: Array of contexts for each layer
    ///   - useResidual: Whether to use residual connection
    public init(
        inputDim: Int,
        outputDim: Int,
        contexts: [[Int]],
        useResidual: Bool = true
    ) {
        var tdnnLayers: [TDNNLayer] = []
        var currentDim = inputDim
        
        // Create TDNN layers
        for (i, context) in contexts.enumerated() {
            let layerOutputDim = (i == contexts.count - 1) ? outputDim : outputDim
            tdnnLayers.append(
                TDNNLayer(
                    inputDim: currentDim,
                    outputDim: layerOutputDim,
                    context: context,
                    useBatchNorm: true,
                    dropoutRate: 0.1
                )
            )
            currentDim = layerOutputDim
        }
        
        self.layers = tdnnLayers
        
        // Residual connection if dimensions match
        self.residual = (useResidual && inputDim != outputDim) 
            ? Linear(inputDim, outputDim) 
            : nil
        
        self.finalActivation = ReLU()
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, training: Bool = false) -> MLXArray {
        var output = x
        
        // Process through TDNN layers
        for layer in layers {
            output = layer(output, training: training)
        }
        
        // Add residual connection
        if let residual = residual {
            output = output + residual(x)
        } else if x.shape.last == output.shape.last {
            output = output + x
        }
        
        // Final activation
        output = finalActivation(output)
        
        return output
    }
}

// MARK: - TDT-CTC Model Configuration

public struct TDTCTCModelConfig: Codable {
    public let preprocessor: PreprocessConfig
    public let tdnnHiddenDim: Int
    public let tdnnNumBlocks: Int
    public let tdnnContexts: [[Int]]
    public let numClasses: Int
    public let vocabulary: [String]
    public let blankToken: Int
    public let dropoutRate: Float
    
    enum CodingKeys: String, CodingKey {
        case preprocessor
        case tdnnHiddenDim = "tdnn_hidden_dim"
        case tdnnNumBlocks = "tdnn_num_blocks"
        case tdnnContexts = "tdnn_contexts"
        case numClasses = "num_classes"
        case vocabulary
        case blankToken = "blank_token"
        case dropoutRate = "dropout_rate"
    }
    
    public init(
        preprocessor: PreprocessConfig,
        tdnnHiddenDim: Int = 512,
        tdnnNumBlocks: Int = 5,
        tdnnContexts: [[Int]]? = nil,
        numClasses: Int,
        vocabulary: [String],
        blankToken: Int = 0,
        dropoutRate: Float = 0.1
    ) {
        self.preprocessor = preprocessor
        self.tdnnHiddenDim = tdnnHiddenDim
        self.tdnnNumBlocks = tdnnNumBlocks
        
        // Default contexts if not provided
        self.tdnnContexts = tdnnContexts ?? [
            [-2, -1, 0, 1, 2],  // Context of 5 frames
            [-1, 0, 1],         // Context of 3 frames
            [-3, 0, 3],         // Dilated context
            [0],                // Current frame only
            [-2, 0, 2]          // Dilated context
        ]
        
        self.numClasses = numClasses
        self.vocabulary = vocabulary
        self.blankToken = blankToken
        self.dropoutRate = dropoutRate
    }
}

// MARK: - Parakeet TDT-CTC Model

/// Parakeet model with Time Delay Temporal CTC architecture
public class ParakeetTDTCTC: Module {
    
    // MARK: - Properties
    
    public let config: TDTCTCModelConfig
    private let inputProjection: Linear
    private let tdnnBlocks: [TDNNBlock]
    private let outputProjection: Linear
    private let ctcHead: CTCHead
    private let decoder: CTCDecoder
    private let dropout: Dropout
    
    // MARK: - Initialization
    
    /// Initialize Parakeet TDT-CTC model
    /// - Parameter config: TDT-CTC model configuration
    public init(config: TDTCTCModelConfig) {
        self.config = config
        
        // Input projection from features to hidden dimension
        self.inputProjection = Linear(
            config.preprocessor.features,
            config.tdnnHiddenDim
        )
        
        // Create TDNN blocks
        var blocks: [TDNNBlock] = []
        for i in 0..<config.tdnnNumBlocks {
            let contexts = i < config.tdnnContexts.count 
                ? [config.tdnnContexts[i]]
                : [[-1, 0, 1]]  // Default context
            
            blocks.append(
                TDNNBlock(
                    inputDim: config.tdnnHiddenDim,
                    outputDim: config.tdnnHiddenDim,
                    contexts: contexts,
                    useResidual: true
                )
            )
        }
        self.tdnnBlocks = blocks
        
        // Output projection
        self.outputProjection = Linear(
            config.tdnnHiddenDim,
            config.tdnnHiddenDim
        )
        
        // CTC head
        let numClasses = config.blankToken >= config.numClasses 
            ? config.numClasses + 1 
            : config.numClasses
        self.ctcHead = CTCHead(
            inputDim: config.tdnnHiddenDim,
            numClasses: numClasses,
            dropoutRate: config.dropoutRate
        )
        
        // Decoder
        self.decoder = CTCDecoder(
            vocabulary: config.vocabulary,
            blankToken: config.blankToken
        )
        
        // Dropout layer
        self.dropout = Dropout(p: config.dropoutRate)
        
        super.init()
    }
    
    // MARK: - Forward Pass
    
    /// Forward pass through the model
    /// - Parameters:
    ///   - audioFeatures: Input audio features [batch, time, features]
    ///   - training: Whether in training mode
    /// - Returns: CTC logits [batch, time, num_classes]
    public func callAsFunction(
        _ audioFeatures: MLXArray,
        training: Bool = false
    ) -> MLXArray {
        var output = audioFeatures
        
        // Project input features to hidden dimension
        output = inputProjection(output)
        
        // Apply dropout
        if training {
            output = dropout(output)
        }
        
        // Process through TDNN blocks
        for block in tdnnBlocks {
            output = block(output, training: training)
        }
        
        // Output projection
        output = outputProjection(output)
        
        // Apply CTC head
        output = ctcHead(output, training: training)
        
        return output
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
    
    /// Streaming state for TDT-CTC model
    public class TDTCTCStreamingState {
        var audioBuffer: MLXArray?
        var contextBuffer: MLXArray?
        var decoderState: CTCDecoder.PrefixBeamSearchState
        let chunkSize: Int
        let contextFrames: Int
        
        init(chunkSize: Int, contextFrames: Int) {
            self.chunkSize = chunkSize
            self.contextFrames = contextFrames
            self.decoderState = CTCDecoder.PrefixBeamSearchState()
        }
    }
    
    /// Initialize streaming state
    /// - Parameters:
    ///   - chunkSize: Size of audio chunks in samples
    ///   - contextFrames: Number of context frames to maintain
    /// - Returns: Initialized streaming state
    public func initStreamingState(
        chunkSize: Int = 16000,
        contextFrames: Int = 10
    ) -> TDTCTCStreamingState {
        return TDTCTCStreamingState(
            chunkSize: chunkSize,
            contextFrames: contextFrames
        )
    }
    
    /// Process streaming audio chunk
    /// - Parameters:
    ///   - audioChunk: Audio chunk to process
    ///   - state: Current streaming state
    ///   - beamSize: Beam size for decoding
    /// - Returns: Current partial transcription
    public func streamingStep(
        audioChunk: MLXArray,
        state: TDTCTCStreamingState,
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
        
        // Add context from previous chunk if available
        let inputFeatures: MLXArray
        if let context = state.contextBuffer {
            inputFeatures = MLX.concatenated([context, features], axis: 0)
        } else {
            inputFeatures = features
        }
        
        // Save context for next chunk
        if features.shape[0] > state.contextFrames {
            let startIdx = features.shape[0] - state.contextFrames
            state.contextBuffer = features[startIdx...]
        }
        
        // Add batch dimension
        let batchedFeatures = inputFeatures.expandedDimensions(axis: 0)
        
        // Forward pass
        let logits = callAsFunction(batchedFeatures)
        
        // Get log probabilities
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
        // Forward pass in training mode
        let logits = callAsFunction(audioFeatures, training: true)
        
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
    
    // MARK: - Model Information
    
    /// Get model parameter count
    public func parameterCount() -> Int {
        let params = parameters()
        return params.flattened().reduce(0) { count, item in
            count + item.1.size
        }
    }
    
    /// Get model memory footprint in MB
    public func memoryFootprint() -> Float {
        let paramCount = parameterCount()
        // Assuming Float32 parameters (4 bytes each)
        return Float(paramCount * 4) / (1024 * 1024)
    }
}