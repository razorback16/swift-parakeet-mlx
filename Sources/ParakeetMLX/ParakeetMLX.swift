import Foundation
import Hub
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

// MARK: - Configuration Structures

/// Comprehensive decoding configuration for all model types
public struct DecodingConfiguration {
    /// Decoding strategy: "greedy", "beam_search", "modified_beam_search"
    public let strategy: DecodingStrategy
    
    /// Beam size for beam search (default: 5)
    public let beamSize: Int
    
    /// Language model weight for beam search scoring
    public let lmWeight: Float
    
    /// Word insertion penalty for beam search
    public let wordScore: Float
    
    /// Maximum symbols to prevent infinite loops (TDT specific)
    public let maxSymbols: Int
    
    /// Blank token ID (CTC specific)
    public let blankId: Int
    
    /// Whether to use log probabilities
    public let logProbs: Bool
    
    /// Temperature for sampling (1.0 = no scaling)
    public let temperature: Float
    
    public enum DecodingStrategy: String {
        case greedy = "greedy"
        case beamSearch = "beam_search"
        case modifiedBeamSearch = "modified_beam_search"
    }
    
    public init(
        strategy: DecodingStrategy = .greedy,
        beamSize: Int = 5,
        lmWeight: Float = 0.0,
        wordScore: Float = 0.0,
        maxSymbols: Int = 10,
        blankId: Int = 0,
        logProbs: Bool = false,
        temperature: Float = 1.0
    ) {
        self.strategy = strategy
        self.beamSize = beamSize
        self.lmWeight = lmWeight
        self.wordScore = wordScore
        self.maxSymbols = maxSymbols
        self.blankId = blankId
        self.logProbs = logProbs
        self.temperature = temperature
    }
    
    /// Convenience initializer for CTC models
    public static func ctcDefault(blankId: Int = 0) -> DecodingConfiguration {
        DecodingConfiguration(
            strategy: .greedy,
            blankId: blankId,
            logProbs: true
        )
    }
    
    /// Convenience initializer for RNNT models
    public static func rnntDefault() -> DecodingConfiguration {
        DecodingConfiguration(
            strategy: .greedy,
            maxSymbols: 10
        )
    }
    
    /// Convenience initializer for TDT models
    public static func tdtDefault(maxSymbols: Int = 10) -> DecodingConfiguration {
        DecodingConfiguration(
            strategy: .greedy,
            maxSymbols: maxSymbols
        )
    }
}

public struct PreprocessConfig: Codable {
    public let sampleRate: Int
    public let normalize: String
    public let windowSize: Float
    public let windowStride: Float
    public let window: String
    public let features: Int
    public let nFFT: Int
    public let dither: Float
    public let padTo: Int
    public let padValue: Float
    public let preemph: Float?
    public let magPower: Float = 2.0

    public var winLength: Int {
        Int(windowSize * Float(sampleRate))
    }

    public var hopLength: Int {
        Int(windowStride * Float(sampleRate))
    }

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case normalize
        case windowSize = "window_size"
        case windowStride = "window_stride"
        case window
        case features
        case nFFT = "n_fft"
        case dither
        case padTo = "pad_to"
        case padValue = "pad_value"
        case preemph
        case magPower = "mag_power"
    }
}

public struct ConformerConfig: Codable {
    public let featIn: Int
    public let nLayers: Int
    public let dModel: Int
    public let nHeads: Int
    public let ffExpansionFactor: Int
    public let subsamplingFactor: Int
    public let selfAttentionModel: String
    public let subsampling: String
    public let convKernelSize: Int
    public let subsamplingConvChannels: Int
    public let posEmbMaxLen: Int
    public let causalDownsampling: Bool = false
    public let useBias: Bool = true
    public let xscaling: Bool = false
    public let subsamplingConvChunkingFactor: Int = 1
    public let attContextSize: [Int]?
    public let posBiasU: [Float]?
    public let posBiasV: [Float]?

    public func posBiasUArray() -> MLXArray? {
        posBiasU.map { MLXArray($0) }
    }

    public func posBiasVArray() -> MLXArray? {
        posBiasV.map { MLXArray($0) }
    }

    enum CodingKeys: String, CodingKey {
        case featIn = "feat_in"
        case nLayers = "n_layers"
        case dModel = "d_model"
        case nHeads = "n_heads"
        case ffExpansionFactor = "ff_expansion_factor"
        case subsamplingFactor = "subsampling_factor"
        case selfAttentionModel = "self_attention_model"
        case subsampling
        case convKernelSize = "conv_kernel_size"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case posEmbMaxLen = "pos_emb_max_len"
        case causalDownsampling = "causal_downsampling"
        case useBias = "use_bias"
        case xscaling
        case subsamplingConvChunkingFactor = "subsampling_conv_chunking_factor"
        case attContextSize = "att_context_size"
        case posBiasU = "pos_bias_u"
        case posBiasV = "pos_bias_v"
    }
}

public struct PredictNetworkConfig: Codable {
    public let predHidden: Int
    public let predRNNLayers: Int
    public let rnnHiddenSize: Int?

    enum CodingKeys: String, CodingKey {
        case predHidden = "pred_hidden"
        case predRNNLayers = "pred_rnn_layers"
        case rnnHiddenSize = "rnn_hidden_size"
    }
}

public struct JointNetworkConfig: Codable {
    public let jointHidden: Int
    public let activation: String
    public let encoderHidden: Int
    public let predHidden: Int

    enum CodingKeys: String, CodingKey {
        case jointHidden = "joint_hidden"
        case activation
        case encoderHidden = "encoder_hidden"
        case predHidden = "pred_hidden"
    }
}

public struct PredictConfig: Codable {
    public let blankAsPad: Bool
    public let vocabSize: Int
    public let prednet: PredictNetworkConfig

    enum CodingKeys: String, CodingKey {
        case blankAsPad = "blank_as_pad"
        case vocabSize = "vocab_size"
        case prednet
    }
}

public struct JointConfig: Codable {
    public let numClasses: Int
    public let vocabulary: [String]
    public let jointnet: JointNetworkConfig
    public let numExtraOutputs: Int

    enum CodingKeys: String, CodingKey {
        case numClasses = "num_classes"
        case vocabulary
        case jointnet
        case numExtraOutputs = "num_extra_outputs"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        numClasses = try container.decode(Int.self, forKey: .numClasses)
        vocabulary = try container.decode([String].self, forKey: .vocabulary)
        jointnet = try container.decode(JointNetworkConfig.self, forKey: .jointnet)
        numExtraOutputs = try container.decodeIfPresent(Int.self, forKey: .numExtraOutputs) ?? 0
    }
}

public struct TDTDecodingConfig: Codable {
    public let modelType: String
    public let durations: [Int]
    public let greedy: [String: Any]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case durations
        case greedy
    }

    public init(modelType: String, durations: [Int], greedy: [String: Any]? = nil) {
        self.modelType = modelType
        self.durations = durations
        self.greedy = greedy
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        durations = try container.decode([Int].self, forKey: .durations)

        if container.contains(.greedy) {
            let greedyData = try container.decode([String: Int].self, forKey: .greedy)
            greedy = greedyData
        } else {
            greedy = nil
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(durations, forKey: .durations)
        if let greedy = greedy {
            try container.encode(greedy as? [String: Int], forKey: .greedy)
        }
    }
}

// MARK: - Model Configurations

public struct ParakeetCTCConfig: Codable {
    public let preprocessor: PreprocessConfig
    public let encoder: ConformerConfig
    public let decoder: CTCDecoderConfig
    
    public struct CTCDecoderConfig: Codable {
        public let numClasses: Int
        public let vocabulary: [String]
        public let blankId: Int
        
        enum CodingKeys: String, CodingKey {
            case numClasses = "num_classes"
            case vocabulary
            case blankId = "blank_id"
        }
    }
}

public struct ParakeetRNNTConfig: Codable {
    public let preprocessor: PreprocessConfig
    public let encoder: ConformerConfig
    public let decoder: PredictConfig
    public let joint: JointConfig
}

public struct ParakeetTDTConfig: Codable {
    public let preprocessor: PreprocessConfig
    public let encoder: ConformerConfig
    public let decoder: PredictConfig
    public let joint: JointConfig
    public let decoding: TDTDecodingConfig
}

// MARK: - Aligned Token Structures

public struct AlignedToken: Sendable {
    public let id: Int
    public var start: Float
    public var duration: Float
    public let text: String

    public var end: Float {
        get { start + duration }
        set { duration = newValue - start }
    }

    public init(id: Int, start: Float, duration: Float, text: String) {
        self.id = id
        self.start = start
        self.duration = duration
        self.text = text
    }
}

public struct AlignedSentence: Sendable {
    public let tokens: [AlignedToken]
    public let start: Float
    public let end: Float

    public var text: String {
        tokens.map { $0.text }.joined()
    }

    public init(tokens: [AlignedToken]) {
        self.tokens = tokens
        self.start = tokens.first?.start ?? 0.0
        self.end = tokens.last?.end ?? 0.0
    }
}

public struct AlignedResult: Sendable {
    public let sentences: [AlignedSentence]

    public var text: String {
        sentences.map { $0.text }.joined(separator: " ")
    }

    public init(sentences: [AlignedSentence]) {
        self.sentences = sentences
    }
}

// MARK: - Model Type Configuration

/// Enum representing different Parakeet model architectures
public enum ParakeetModelType: String {
    case ctc = "ctc"
    case rnnt = "rnnt"
    case tdtctc = "tdt"
    
    public var description: String {
        switch self {
        case .ctc:
            return "Connectionist Temporal Classification (CTC)"
        case .rnnt:
            return "Recurrent Neural Network Transducer (RNN-T)"
        case .tdtctc:
            return "Token-and-Duration Transducer CTC (TDT-CTC)"
        }
    }
}

// Legacy compatibility
public struct DecodingConfig {
    public let decoding: String

    public init(decoding: String = "greedy") {
        self.decoding = decoding
    }
}

// MARK: - Unified Parakeet Model Protocol

/// Protocol defining common interface for all Parakeet models
public protocol ParakeetModel: Module {
    var vocabulary: [String] { get }
    var preprocessConfig: PreprocessConfig { get }
    
    func transcribe(
        audioData: MLXArray,
        dtype: DType,
        decodingConfig: DecodingConfiguration?,
        streamingConfig: StreamingConfig?
    ) async throws -> TranscriptionResult
    
    func encode(_ input: MLXArray, cache: [ConformerCache?]?) -> (MLXArray, MLXArray)
}

/// Unified transcription result
public struct TranscriptionResult {
    public let text: String
    public let tokens: [AlignedToken]
    public let sentences: [AlignedSentence]
    public let confidence: Float?
    public let processingTime: TimeInterval?
    
    public init(
        text: String,
        tokens: [AlignedToken] = [],
        sentences: [AlignedSentence] = [],
        confidence: Float? = nil,
        processingTime: TimeInterval? = nil
    ) {
        self.text = text
        self.tokens = tokens
        self.sentences = sentences
        self.confidence = confidence
        self.processingTime = processingTime
    }
    
    /// Convert from AlignedResult
    public init(from aligned: AlignedResult, confidence: Float? = nil, processingTime: TimeInterval? = nil) {
        self.text = aligned.text
        self.tokens = aligned.sentences.flatMap { $0.tokens }
        self.sentences = aligned.sentences
        self.confidence = confidence
        self.processingTime = processingTime
    }
}

// MARK: - Streaming Configuration

public struct StreamingConfig {
    public let chunkDuration: Float
    public let overlapDuration: Float
    public let minChunkSize: Int
    public let useVAD: Bool
    
    public init(
        chunkDuration: Float = 10.0,
        overlapDuration: Float = 1.0,
        minChunkSize: Int = 1600,
        useVAD: Bool = false
    ) {
        self.chunkDuration = chunkDuration
        self.overlapDuration = overlapDuration
        self.minChunkSize = minChunkSize
        self.useVAD = useVAD
    }
}

// MARK: - Main Parakeet Model

public class ParakeetTDT: Module, @unchecked Sendable {
    public let preprocessConfig: PreprocessConfig
    public let encoderConfig: ConformerConfig
    public let vocabulary: [String]
    public let durations: [Int]
    public let maxSymbols: Int

    private let encoder: Conformer
    private let decoder: PredictNetwork
    private let joint: JointNetwork

    public init(config: ParakeetTDTConfig) throws {
        guard config.decoding.modelType == "tdt" else {
            throw ParakeetError.invalidModelType("Model must be a TDT model")
        }

        self.preprocessConfig = config.preprocessor
        self.encoderConfig = config.encoder
        self.vocabulary = config.joint.vocabulary
        self.durations = config.decoding.durations
        // Set a default maxSymbols value if not provided in config to prevent infinite loops
        self.maxSymbols = config.decoding.greedy?["max_symbols"] as? Int ?? 10

        self.encoder = Conformer(config: config.encoder)
        self.decoder = PredictNetwork(config: config.decoder)
        self.joint = JointNetwork(config: config.joint)

        super.init()
    }

    /// Main transcription interface
    public func transcribe(
        audioData: MLXArray,
        dtype: DType = .float32,
        chunkDuration: Float? = nil,
        overlapDuration: Float = 15.0,
        chunkCallback: ((Float, Float) -> Void)? = nil
    ) throws -> AlignedResult {

        let processedAudio = audioData.dtype == dtype ? audioData : audioData.asType(dtype)

        // If no chunking requested or audio is short enough
        if let chunkDuration = chunkDuration {
            let audioLengthSeconds = Float(audioData.shape[0]) / Float(preprocessConfig.sampleRate)

            if audioLengthSeconds <= chunkDuration {
                let mel = try getLogMel(processedAudio, config: preprocessConfig)
                return try generate(mel: mel)[0]
            }

            // Process in chunks
            return try transcribeChunked(
                audio: processedAudio,
                chunkDuration: chunkDuration,
                overlapDuration: overlapDuration,
                chunkCallback: chunkCallback
            )
        } else {
            let mel = try getLogMel(processedAudio, config: preprocessConfig)
            return try generate(mel: mel)[0]
        }
    }

    private func transcribeChunked(
        audio: MLXArray,
        chunkDuration: Float,
        overlapDuration: Float,
        chunkCallback: ((Float, Float) -> Void)?
    ) throws -> AlignedResult {

        let chunkSamples = Int(chunkDuration * Float(preprocessConfig.sampleRate))
        let overlapSamples = Int(overlapDuration * Float(preprocessConfig.sampleRate))
        let audioLength = audio.shape[0]

        var allTokens: [AlignedToken] = []
        var start = 0

        while start < audioLength {
            let end = min(start + chunkSamples, audioLength)

            chunkCallback?(Float(end), Float(audioLength))

            if end - start < preprocessConfig.hopLength {
                break  // Prevent zero-length log mel
            }

            let chunkAudio = audio[start..<end]
            let chunkMel = try getLogMel(chunkAudio, config: preprocessConfig)
            let chunkResult = try generate(mel: chunkMel)[0]

            let chunkOffset = Float(start) / Float(preprocessConfig.sampleRate)
            var chunkTokens: [AlignedToken] = []

            for sentence in chunkResult.sentences {
                for var token in sentence.tokens {
                    token.start += chunkOffset
                    chunkTokens.append(token)
                }
            }

            if !allTokens.isEmpty {
                // Merge with overlap handling
                allTokens = try mergeLongestContiguous(
                    allTokens,
                    chunkTokens,
                    overlapDuration: overlapDuration
                )
            } else {
                allTokens = chunkTokens
            }

            start += chunkSamples - overlapSamples
        }

        return sentencesToResult(tokensToSentences(allTokens))
    }

    public func generate(mel: MLXArray) throws -> [AlignedResult] {
        let inputMel = mel.ndim == 2 ? mel.expandedDimensions(axis: 0) : mel

        let (features, lengths) = encoder(inputMel)

        let (results, _) = try decode(
            features: features,
            lengths: lengths,
            config: DecodingConfig()
        )

        return results.map { tokens in
            sentencesToResult(tokensToSentences(tokens))
        }
    }

    public func decode(
        features: MLXArray,
        lengths: MLXArray? = nil,
        lastToken: [Int?]? = nil,
        hiddenState: [(MLXArray, MLXArray)?]? = nil,
        config: DecodingConfig = DecodingConfig()
    ) throws -> ([[AlignedToken]], [(MLXArray, MLXArray)?]) {

        guard config.decoding == "greedy" else {
            throw ParakeetError.unsupportedDecoding(
                "Only greedy decoding is supported for TDT decoder")
        }

        let (B, S) = (features.shape[0], features.shape[1])
        let actualLengths = lengths ?? MLXArray(Array(repeating: S, count: B))
        let actualLastToken = lastToken ?? Array(repeating: nil, count: B)
        var actualHiddenState = hiddenState ?? Array(repeating: nil, count: B)

        var results: [[AlignedToken]] = []

        for batch in 0..<B {
            var hypothesis: [AlignedToken] = []
            let feature = features[batch].expandedDimensions(axis: 0)
            let length = Int(actualLengths[batch].item(Int32.self))

            var step = 0
            var newSymbols = 0
            var currentLastToken = actualLastToken[batch]
            var debugCounter = 0

            while step < length {
                // Decoder pass
                let decoderInput = currentLastToken.map { token in
                    MLXArray([token]).expandedDimensions(axis: 0)  // Shape: [1, 1] (batch_size, seq_len)
                }

                let (decoderOut, newHidden) = decoder(decoderInput, actualHiddenState[batch])

                let decoderOutput = decoderOut.asType(feature.dtype)
                let decoderHidden = (
                    newHidden.0.asType(feature.dtype), newHidden.1.asType(feature.dtype)
                )

                // Joint pass
                let jointOut = joint(
                    feature[0..., step..<(step + 1)],
                    decoderOutput
                )

                // Ensure we're in inference mode
                MLX.eval(jointOut)

                // Check for NaN by comparing with itself (NaN != NaN is true)
                let jointOutNaNCheck = jointOut.max().item(Float.self)
                if jointOutNaNCheck.isNaN {
                    break
                }

                // Sampling - match Python implementation exactly
                let vocabSize = vocabulary.count

                // Check if we have enough dimensions and size
                guard jointOut.shape.count >= 4 else {
                    throw ParakeetError.audioProcessingError(
                        "Joint output has insufficient dimensions: \(jointOut.shape)")
                }

                let lastDim = jointOut.shape[jointOut.shape.count - 1]  // Always get the last dimension

                guard lastDim > vocabSize else {
                    throw ParakeetError.audioProcessingError(
                        "Joint output last dimension (\(lastDim)) is not larger than vocab size (\(vocabSize))"
                    )
                }

                // Match Python exactly: joint_out[0, 0, :, : len(self.vocabulary) + 1]
                // and joint_out[0, 0, :, len(self.vocabulary) + 1 :]
                let vocabSlice = jointOut[0, 0, 0..., 0..<(vocabSize + 1)]
                let decisionSlice = jointOut[0, 0, 0..., (vocabSize + 1)..<lastDim]

                guard vocabSlice.shape[0] > 0 && decisionSlice.shape[0] > 0 else {
                    throw ParakeetError.audioProcessingError(
                        "Empty slices: vocab=\(vocabSlice.shape), decision=\(decisionSlice.shape)")
                }

                // The joint output should be [batch, enc_time, pred_time, num_classes]
                // We want to argmax over the last dimension after slicing
                let predToken = Int(vocabSlice.argMax(axis: -1).item(Int32.self))
                let decision = Int(decisionSlice.argMax(axis: -1).item(Int32.self))

                // TDT decoding rule
                if predToken != vocabSize {
                    let tokenText = Tokenizer.decode([predToken], vocabulary)
                    let startTime =
                        Float(step * encoderConfig.subsamplingFactor)
                        / Float(preprocessConfig.sampleRate) * Float(preprocessConfig.hopLength)
                    let duration =
                        Float(durations[decision] * encoderConfig.subsamplingFactor)
                        / Float(preprocessConfig.sampleRate) * Float(preprocessConfig.hopLength)

                    hypothesis.append(
                        AlignedToken(
                            id: predToken,
                            start: startTime,
                            duration: duration,
                            text: tokenText
                        ))

                    currentLastToken = predToken
                    actualHiddenState[batch] = decoderHidden
                } else {
                }

                step += durations[decision]

                // Prevent stucking rule
                newSymbols += 1

                if durations[decision] != 0 {
                    newSymbols = 0
                } else {
                    if newSymbols >= maxSymbols {
                        step += 1
                        newSymbols = 0
                    }
                }

                // Safety break to prevent infinite loops
                if newSymbols > 100 {
                    break
                }

                debugCounter += 1
            }

            results.append(hypothesis)
        }

        return (results, actualHiddenState)
    }

    /// Load weights from a safetensors file
    public func loadWeights(from url: URL) throws {
        // Load weights using MLX
        let weights = try MLX.loadArrays(url: url)

        let transformedWeights = weights

        // Map safetensors keys to Swift model parameter paths
        var mappedWeights: [String: MLXArray] = [:]

        for (safetensorsKey, weight) in transformedWeights {
            // Convert safetensors key to Swift model parameter path
            if let swiftKey = mapSafetensorsKeyToSwiftPath(safetensorsKey) {
                mappedWeights[swiftKey] = weight
            }
        }

        let flatWeights = ModuleParameters.unflattened(mappedWeights)
        self.update(parameters: flatWeights)
    }

    /// Maps safetensors parameter keys to Swift model parameter paths
    private func mapSafetensorsKeyToSwiftPath(_ safetensorsKey: String) -> String? {
        // Handle encoder parameters
        if safetensorsKey.hasPrefix("encoder.") {
            let encoderKey = String(safetensorsKey.dropFirst("encoder.".count))

            // Handle pre_encode parameters
            if encoderKey.hasPrefix("pre_encode.") {
                let preEncodeKey = String(encoderKey.dropFirst("pre_encode.".count))

                // Handle conv layers: "conv.0.weight" -> "conv.0.weight"
                if preEncodeKey.hasPrefix("conv.") {
                    // The conv layers are already in the right format for Swift
                    return "encoder.preEncode.\(preEncodeKey)"
                }

                // Handle out layer: "out.weight" -> "out.weight"
                if preEncodeKey.hasPrefix("out.") {
                    return "encoder.preEncode.\(preEncodeKey)"
                }
            }

            // Handle conformer layers: "layers.0.norm_self_att.weight" -> "layers.0.normSelfAtt.weight"
            if encoderKey.hasPrefix("layers.") {
                var layerKey = encoderKey

                // Convert snake_case to camelCase for Swift property names
                layerKey = layerKey.replacingOccurrences(of: "norm_self_att", with: "normSelfAtt")
                layerKey = layerKey.replacingOccurrences(
                    of: "self_attn.linear_q", with: "selfAttn.wq")
                layerKey = layerKey.replacingOccurrences(
                    of: "self_attn.linear_k", with: "selfAttn.wk")
                layerKey = layerKey.replacingOccurrences(
                    of: "self_attn.linear_v", with: "selfAttn.wv")
                layerKey = layerKey.replacingOccurrences(
                    of: "self_attn.linear_out", with: "selfAttn.wo")
                layerKey = layerKey.replacingOccurrences(
                    of: "self_attn.linear_pos", with: "selfAttn.linearPos")
                layerKey = layerKey.replacingOccurrences(
                    of: "self_attn.pos_bias_u", with: "selfAttn.posBiasU")
                layerKey = layerKey.replacingOccurrences(
                    of: "self_attn.pos_bias_v", with: "selfAttn.posBiasV")
                layerKey = layerKey.replacingOccurrences(of: "norm_conv", with: "normConv")
                layerKey = layerKey.replacingOccurrences(
                    of: "conv.pointwise_conv1", with: "conv.pointwiseConv1")
                layerKey = layerKey.replacingOccurrences(
                    of: "conv.depthwise_conv", with: "conv.depthwiseConv")
                layerKey = layerKey.replacingOccurrences(
                    of: "conv.batch_norm", with: "conv.batchNorm")
                layerKey = layerKey.replacingOccurrences(
                    of: "conv.pointwise_conv2", with: "conv.pointwiseConv2")
                layerKey = layerKey.replacingOccurrences(
                    of: "norm_feed_forward1", with: "normFeedForward1")
                layerKey = layerKey.replacingOccurrences(of: "feed_forward1", with: "feedForward1")
                layerKey = layerKey.replacingOccurrences(
                    of: "norm_feed_forward2", with: "normFeedForward2")
                layerKey = layerKey.replacingOccurrences(of: "feed_forward2", with: "feedForward2")
                layerKey = layerKey.replacingOccurrences(of: "norm_out", with: "normOut")

                return "encoder.\(layerKey)"
            }
        }

        // Handle decoder parameters
        if safetensorsKey.hasPrefix("decoder.") {
            var decoderKey = String(safetensorsKey.dropFirst("decoder.".count))

            // Convert snake_case to camelCase
            decoderKey = decoderKey.replacingOccurrences(of: "prediction.embed", with: "embed")
            decoderKey = decoderKey.replacingOccurrences(
                of: "prediction.dec_rnn.lstm", with: "decRNN.lstmLayers")

            return "decoder.\(decoderKey)"
        }

        // Handle joint parameters
        if safetensorsKey.hasPrefix("joint.") {
            var jointKey = String(safetensorsKey.dropFirst("joint.".count))

            // Convert snake_case to camelCase
            jointKey = jointKey.replacingOccurrences(of: "enc_proj", with: "encLinear")
            jointKey = jointKey.replacingOccurrences(of: "pred_proj", with: "predLinear")
            jointKey = jointKey.replacingOccurrences(of: "joint_proj", with: "jointLinear")

            // Handle direct enc/pred mappings
            jointKey = jointKey.replacingOccurrences(of: "enc.", with: "encLinear.")
            jointKey = jointKey.replacingOccurrences(of: "pred.", with: "predLinear.")

            // Handle joint_net layers - map to jointLinear since that's the final linear layer
            if jointKey.hasPrefix("joint_net.") {
                let jointNetKey = String(jointKey.dropFirst("joint_net.".count))
                // joint_net.2 is the final linear layer in Python, map to jointLinear
                if jointNetKey.hasPrefix("2.") {
                    let layerParam = String(jointNetKey.dropFirst("2.".count))
                    jointKey = "jointLinear.\(layerParam)"
                } else {
                    // Other joint_net layers (0 is activation, 1 is identity) - skip for now
                    return nil
                }
            }

            return "joint.\(jointKey)"
        }

        return nil
    }

    /// Public access to encoder for streaming
    public func encode(_ input: MLXArray, cache: [ConformerCache?]? = nil) -> (MLXArray, MLXArray) {
        return encoder(input, cache: cache)
    }

}

// MARK: - Unified Model Factory

/// Main factory class for creating Parakeet models
public class ParakeetMLX {
    
    /// Create a model from configuration and model type
    public static func createModel(
        type: ParakeetModelType,
        configPath: URL,
        weightsPath: URL? = nil,
        dtype: DType = .float32
    ) async throws -> any ParakeetModel {
        
        // Load configuration
        let configData = try Data(contentsOf: configPath)
        
        switch type {
        case .ctc:
            // For CTC, we need to handle the configuration differently
            // Since ParakeetCTC isn't implemented yet, we'll use TDT as fallback
            throw ParakeetError.configurationError("CTC model support is not yet fully implemented. Please use TDT or RNNT models.")
            
        case .rnnt:
            // For RNNT, we need to handle the configuration differently
            // Since ParakeetRNNT isn't implemented yet, we'll use TDT as fallback
            throw ParakeetError.configurationError("RNNT model support is not yet fully implemented. Please use TDT models.")
            
        case .tdtctc:
            let config = try JSONDecoder().decode(ParakeetTDTConfig.self, from: configData)
            let model = try ParakeetTDT(config: config)
            if let weightsPath = weightsPath {
                try model.loadWeights(from: weightsPath)
                castModelWeights(model, to: dtype)
            }
            return model
        }
    }
    
    /// Auto-detect model type from configuration
    public static func detectModelType(from configPath: URL) throws -> ParakeetModelType {
        let configData = try Data(contentsOf: configPath)
        let json = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
        
        // Check for model-specific configuration keys
        if let decoder = json?["decoder"] as? [String: Any] {
            if decoder["decoder_type"] as? String == "ctc" {
                return .ctc
            }
            if decoder["decoder_type"] as? String == "rnnt" {
                return .rnnt
            }
        }
        
        if let decoding = json?["decoding"] as? [String: Any],
           let modelType = decoding["model_type"] as? String {
            switch modelType {
            case "tdt":
                return .tdtctc
            case "ctc":
                return .ctc
            case "rnnt":
                return .rnnt
            default:
                break
            }
        }
        
        // Default to TDT for backward compatibility
        return .tdtctc
    }
    
    /// Load model from Hugging Face Hub or local path
    public static func load(
        from modelPath: String,
        modelType: ParakeetModelType? = nil,
        dtype: DType = .float32,
        cacheDirectory: URL? = nil,
        progressHandler: ((Progress) -> Void)? = nil
    ) async throws -> any ParakeetModel {
        
        let configURL: URL
        let weightsURL: URL
        
        // Determine if local or HuggingFace
        if FileManager.default.fileExists(atPath: modelPath) {
            configURL = URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")
            weightsURL = URL(fileURLWithPath: modelPath).appendingPathComponent("model.safetensors")
        } else {
            // Download from HuggingFace
            let (config, weights) = try await downloadFromHub(
                modelId: modelPath,
                cacheDirectory: cacheDirectory,
                progressHandler: progressHandler
            )
            configURL = config
            weightsURL = weights
        }
        
        // Auto-detect model type if not specified
        let detectedType = try modelType ?? detectModelType(from: configURL)
        
        // Create and return model
        return try await createModel(
            type: detectedType,
            configPath: configURL,
            weightsPath: weightsURL,
            dtype: dtype
        )
    }
    
    /// Create a streaming-capable model
    public static func createStreamingModel(
        from model: ParakeetTDT,
        contextWindow: Int = 256,
        lookbackFrames: Int = 5
    ) -> StreamingParakeet {
        return StreamingParakeet(
            model: model,
            contextWindow: contextWindow,
            lookbackFrames: lookbackFrames
        )
    }
    
    // MARK: - Helper Methods
    
    private static func loadWeights(
        into model: Module,
        from url: URL,
        dtype: DType
    ) async throws {
        let weights = try MLX.loadArrays(url: url)
        let castedWeights = weights.mapValues { $0.asType(dtype) }
        let flatWeights = ModuleParameters.unflattened(castedWeights)
        model.update(parameters: flatWeights)
    }
    
    private static func castModelWeights(_ model: Module, to dtype: DType) {
        let parameters = model.parameters()
        let castedParameters = parameters.mapValues { $0.asType(dtype) }
        model.update(parameters: castedParameters)
    }
    
    private static func downloadFromHub(
        modelId: String,
        cacheDirectory: URL?,
        progressHandler: ((Progress) -> Void)?
    ) async throws -> (config: URL, weights: URL) {
        
        let downloadDirectory = try getSandboxSafeModelDirectory(
            cacheDirectory: cacheDirectory,
            modelId: modelId
        )
        
        let hubApi = HubApi(downloadBase: downloadDirectory, useOfflineMode: false)
        let repo = Hub.Repo(id: modelId)
        let filesToDownload = ["config.json", "*.safetensors"]
        
        let snapshot = try await hubApi.snapshot(
            from: repo,
            matching: filesToDownload,
            progressHandler: progressHandler ?? { progress in
                print("Download: \(String(format: "%.1f", progress.fractionCompleted * 100))%")
            }
        )
        
        return (
            config: snapshot.appendingPathComponent("config.json"),
            weights: snapshot.appendingPathComponent("model.safetensors")
        )
    }
}

// MARK: - Model Extensions for Protocol Conformance

extension ParakeetTDT: ParakeetModel {
    public func transcribe(
        audioData: MLXArray,
        dtype: DType = .float32,
        decodingConfig: DecodingConfiguration? = nil,
        streamingConfig: StreamingConfig? = nil
    ) async throws -> TranscriptionResult {
        
        let startTime = Date()
        
        let result = try transcribe(
            audioData: audioData,
            dtype: dtype,
            chunkDuration: streamingConfig?.chunkDuration,
            overlapDuration: streamingConfig?.overlapDuration ?? 15.0
        )
        
        let processingTime = Date().timeIntervalSince(startTime)
        
        return TranscriptionResult(
            from: result,
            processingTime: processingTime
        )
    }
}

// MARK: - Model Loading

/// Legacy function for backward compatibility
@available(*, deprecated, message: "Use ParakeetMLX.load() instead")
public func loadParakeetModel(
    from modelPath: String,
    dtype: DType = .float32,
    cacheDirectory: URL? = nil,
    progressHandler: ((Progress) -> Void)? = nil
) async throws -> ParakeetTDT {

    let configURL: URL
    let weightsURL: URL

    // Try loading from local path first, then Hugging Face Hub
    if FileManager.default.fileExists(atPath: modelPath) {
        configURL = URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")
        weightsURL = URL(fileURLWithPath: modelPath).appendingPathComponent("model.safetensors")
    } else {
        // Assume it's a Hugging Face model ID and download it
        print("Downloading model from Hugging Face Hub: \(modelPath)")

        // Use provided cache directory or create a sandboxed-safe default
        let downloadDirectory = try getSandboxSafeModelDirectory(
            cacheDirectory: cacheDirectory, modelId: modelPath)

        // Create a custom HubApi instance with our custom download directory
        // Force online mode for CLI usage - we want to download even on constrained networks
        let hubApi = HubApi(downloadBase: downloadDirectory, useOfflineMode: false)
        let repo = Hub.Repo(id: modelPath)
        let filesToDownload = ["config.json", "*.safetensors"]

        let snapshot = try await hubApi.snapshot(
            from: repo,
            matching: filesToDownload,
            progressHandler: { progress in
                if let handler = progressHandler {
                    handler(progress)
                } else {
                    print(
                        "Download progress: \(String(format: "%.1f", progress.fractionCompleted * 100))%"
                    )
                }
            }
        )

        configURL = snapshot.appendingPathComponent("config.json")
        weightsURL = snapshot.appendingPathComponent("model.safetensors")
        print("Model downloaded to: \(snapshot.path)")
    }

    // Load configuration
    let configData = try Data(contentsOf: configURL)
    let config = try JSONDecoder().decode(ParakeetTDTConfig.self, from: configData)

    // Create model
    let model = try ParakeetTDT(config: config)

    // Load weights from safetensors
    try model.loadWeights(from: weightsURL)

    // Cast to desired dtype
    let parameters = model.parameters()
    let castedParameters = parameters.mapValues { $0.asType(dtype) }
    model.update(parameters: castedParameters)

    return model
}

/// Helper function to get a sandboxed-safe directory for model downloads
private func getSandboxSafeModelDirectory(cacheDirectory: URL?, modelId: String) throws -> URL {
    let baseDirectory: URL

    if let cacheDirectory = cacheDirectory {
        // Use provided cache directory
        baseDirectory = cacheDirectory
    } else {
        // Create a sandboxed-safe directory
        let fileManager = FileManager.default

        // Try Application Support directory first (persistent, app-specific)
        if let appSupportDir = fileManager.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first {
            baseDirectory = appSupportDir.appendingPathComponent("ParakeetMLX")
        }
        // Fallback to Caches directory (might be cleaned by system)
        else if let cachesDir = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first {
            baseDirectory = cachesDir.appendingPathComponent("ParakeetMLX")
        }
        // Final fallback to temporary directory
        else {
            baseDirectory = fileManager.temporaryDirectory.appendingPathComponent(
                "ParakeetMLX")
        }
    }

    // Create the base directory if it doesn't exist
    if !FileManager.default.fileExists(atPath: baseDirectory.path) {
        try FileManager.default.createDirectory(
            at: baseDirectory, withIntermediateDirectories: true, attributes: nil)
    }

    // HubApi will create the model-specific subdirectory (models/modelId)
    return baseDirectory
}

/// Public helper function to get appropriate cache directories for sandboxed environments
public func getParakeetCacheDirectory() -> URL? {
    let fileManager = FileManager.default

    // Try Application Support directory first (best for app-specific data)
    if let appSupportDir = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)
        .first
    {
        return appSupportDir.appendingPathComponent("ParakeetMLX")
    }
    // Fallback to Caches directory
    else if let cachesDir = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first {
        return cachesDir.appendingPathComponent("ParakeetMLX")
    }

    return nil
}

// MARK: - Error Types

public enum ParakeetError: Error, LocalizedError {
    case invalidModelType(String)
    case unsupportedDecoding(String)
    case audioProcessingError(String)
    case modelLoadingError(String)
    case configurationError(String)
    case streamingError(String)
    case decodingError(String)

    public var errorDescription: String? {
        switch self {
        case .invalidModelType(let message):
            return "Invalid model type: \(message)"
        case .unsupportedDecoding(let message):
            return "Unsupported decoding: \(message)"
        case .audioProcessingError(let message):
            return "Audio processing error: \(message)"
        case .modelLoadingError(let message):
            return "Model loading error: \(message)"
        case .configurationError(let message):
            return "Configuration error: \(message)"
        case .streamingError(let message):
            return "Streaming error: \(message)"
        case .decodingError(let message):
            return "Decoding error: \(message)"
        }
    }
}

// MARK: - Public API Utilities

/// Audio utilities for preprocessing
public struct ParakeetAudio {
    
    /// Load audio file and convert to appropriate format
    public static func loadAudio(
        from url: URL,
        targetSampleRate: Int = 16000
    ) throws -> MLXArray {
        // For now, use basic audio loading
        // In a real implementation, you'd use AVFoundation or similar
        throw ParakeetError.audioProcessingError("Audio loading from file not yet implemented. Please provide MLXArray audio data directly.")
    }
    
    /// Apply preprocessing to raw audio
    public static func preprocess(
        audio: MLXArray,
        config: PreprocessConfig
    ) throws -> MLXArray {
        return try getLogMel(audio, config: config)
    }
    
    /// Apply Voice Activity Detection
    public static func detectVoiceActivity(
        in audio: MLXArray,
        sampleRate: Int = 16000,
        frameDuration: Float = 0.03,
        threshold: Float = 0.5
    ) -> [(start: Int, end: Int)] {
        // Basic VAD implementation
        let frameSize = Int(Float(sampleRate) * frameDuration)
        let numFrames = audio.shape[0] / frameSize
        
        var segments: [(start: Int, end: Int)] = []
        var inSpeech = false
        var segmentStart = 0
        
        for i in 0..<numFrames {
            let start = i * frameSize
            let end = min((i + 1) * frameSize, audio.shape[0])
            let frame = audio[start..<end]
            
            // Simple energy-based VAD
            let energy = MLX.mean(MLX.abs(frame)).item(Float.self)
            
            if energy > threshold {
                if !inSpeech {
                    segmentStart = start
                    inSpeech = true
                }
            } else if inSpeech {
                segments.append((start: segmentStart, end: end))
                inSpeech = false
            }
        }
        
        if inSpeech {
            segments.append((start: segmentStart, end: audio.shape[0]))
        }
        
        return segments
    }
    
    /// Apply noise reduction
    public static func reduceNoise(
        in audio: MLXArray,
        noiseProfile: MLXArray? = nil,
        reductionStrength: Float = 0.5
    ) -> MLXArray {
        // Simple spectral subtraction noise reduction
        guard reductionStrength > 0 && reductionStrength <= 1 else {
            return audio
        }
        
        // If no noise profile provided, estimate from first 0.5 seconds
        let noiseEstimate: MLXArray
        if let profile = noiseProfile {
            noiseEstimate = profile
        } else {
            // Estimate noise from beginning of audio
            let sampleRate = 16000
            let noiseFrames = min(sampleRate / 2, audio.shape[0])
            let noiseSegment = audio[0..<noiseFrames]
            noiseEstimate = MLX.mean(MLX.abs(noiseSegment))
        }
        
        // Apply simple spectral subtraction
        let reducedNoise = audio - (noiseEstimate * reductionStrength)
        
        // Ensure non-negative values
        return MLX.maximum(reducedNoise, MLXArray(0.0))
    }
}

/// Model information utilities
public struct ParakeetInfo {
    
    /// Get model information from configuration file
    public static func getModelInfo(from configPath: URL) throws -> ModelInfo {
        let configData = try Data(contentsOf: configPath)
        let json = try JSONSerialization.jsonObject(with: configData) as? [String: Any] ?? [:]
        
        let modelType = try ParakeetMLX.detectModelType(from: configPath)
        
        return ModelInfo(
            type: modelType,
            vocabulary: extractVocabulary(from: json),
            sampleRate: extractSampleRate(from: json),
            features: extractFeatures(from: json)
        )
    }
    
    public struct ModelInfo {
        public let type: ParakeetModelType
        public let vocabulary: [String]
        public let sampleRate: Int
        public let features: Int
    }
    
    private static func extractVocabulary(from json: [String: Any]) -> [String] {
        if let joint = json["joint"] as? [String: Any],
           let vocab = joint["vocabulary"] as? [String] {
            return vocab
        }
        if let decoder = json["decoder"] as? [String: Any],
           let vocab = decoder["vocabulary"] as? [String] {
            return vocab
        }
        return []
    }
    
    private static func extractSampleRate(from json: [String: Any]) -> Int {
        if let preprocessor = json["preprocessor"] as? [String: Any],
           let sampleRate = preprocessor["sample_rate"] as? Int {
            return sampleRate
        }
        return 16000
    }
    
    private static func extractFeatures(from json: [String: Any]) -> Int {
        if let preprocessor = json["preprocessor"] as? [String: Any],
           let features = preprocessor["features"] as? Int {
            return features
        }
        return 80
    }
}

// MARK: - Utility Functions

private func tokensToSentences(_ tokens: [AlignedToken]) -> [AlignedSentence] {
    guard !tokens.isEmpty else { return [] }

    var sentences: [AlignedSentence] = []
    var currentTokens: [AlignedToken] = []

    for token in tokens {
        currentTokens.append(token)

        // Simple sentence boundary detection (you might want to improve this)
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

private func sentencesToResult(_ sentences: [AlignedSentence]) -> AlignedResult {
    return AlignedResult(sentences: sentences)
}

private func mergeLongestContiguous(
    _ tokens1: [AlignedToken],
    _ tokens2: [AlignedToken],
    overlapDuration: Float
) throws -> [AlignedToken] {
    // Simplified merge - you might want to implement a more sophisticated algorithm
    let cutoffTime = tokens1.last?.end ?? 0.0 - overlapDuration
    let filteredTokens1 = tokens1.filter { $0.end <= cutoffTime }
    let filteredTokens2 = tokens2.filter { $0.start >= cutoffTime }

    return filteredTokens1 + filteredTokens2
}

// MARK: - Quick Start Examples

/// Example usage:
/// ```swift
/// // Load a model from HuggingFace Hub
/// let model = try await ParakeetMLX.load(
///     from: "nvidia/parakeet-tdt_ctc-110m",
///     dtype: .float16
/// )
/// 
/// // Load and preprocess audio
/// let audio = try ParakeetAudio.loadAudio(from: audioURL)
/// 
/// // Transcribe with default settings
/// let result = try await model.transcribe(
///     audioData: audio,
///     dtype: .float16
/// )
/// print(result.text)
/// 
/// // Transcribe with streaming
/// let streamingModel = ParakeetMLX.createStreamingModel(from: model)
/// for chunk in audioChunks {
///     let partial = try await streamingModel.processChunk(chunk)
///     print("Partial: \(partial.text)")
/// }
/// let final = try await streamingModel.finalize()
/// print("Final: \(final.text)")
/// ```
