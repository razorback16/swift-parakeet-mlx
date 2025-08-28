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

// MARK: - CTC Decoder

/// Shared CTC decoder implementation supporting greedy and beam search decoding
public class CTCDecoder {
    
    // MARK: - Properties
    
    private let blankToken: Int
    private let vocabulary: [String]
    private let vocabSize: Int
    
    // MARK: - Initialization
    
    /// Initialize CTC decoder with vocabulary configuration
    /// - Parameters:
    ///   - vocabulary: Array of vocabulary tokens
    ///   - blankToken: Index of the blank token (default: 0)
    public init(vocabulary: [String], blankToken: Int = 0) {
        self.vocabulary = vocabulary
        self.blankToken = blankToken
        self.vocabSize = vocabulary.count
    }
    
    // MARK: - Greedy Decoding
    
    /// Performs greedy CTC decoding on logits
    /// - Parameters:
    ///   - logits: Model output logits of shape [batch, time, vocab]
    ///   - logitsLengths: Length of valid logits for each batch item
    /// - Returns: Array of decoded token sequences for each batch item
    public func greedyDecode(
        logits: MLXArray,
        logitsLengths: [Int]? = nil
    ) -> [[Int]] {
        // Get dimensions
        let batchSize = logits.shape[0]
        let timeSteps = logits.shape[1]
        
        // Apply softmax to get probabilities
        let probs = MLX.softmax(logits, axis: -1)
        
        // Get argmax predictions
        let predictions = MLX.argMax(probs, axis: -1)
        
        var decodedBatch: [[Int]] = []
        
        // Process each batch item
        for b in 0..<batchSize {
            let length = logitsLengths?[b] ?? timeSteps
            var decoded: [Int] = []
            var prevToken = blankToken
            
            // Process time steps for this batch item
            for t in 0..<length {
                let token = predictions[b, t].item(Int.self)
                
                // Skip blank tokens and repeated tokens
                if token != blankToken && token != prevToken {
                    decoded.append(token)
                }
                prevToken = token
            }
            
            decodedBatch.append(decoded)
        }
        
        return decodedBatch
    }
    
    // MARK: - Beam Search Decoding
    
    /// Beam search state for tracking hypotheses
    public struct BeamState {
        let prefix: [Int]
        let score: Float
        let blankScore: Float
        let nonBlankScore: Float
        
        init(prefix: [Int] = [], score: Float = 0.0, 
             blankScore: Float = Float.leastNormalMagnitude,
             nonBlankScore: Float = Float.leastNormalMagnitude) {
            self.prefix = prefix
            self.score = score
            self.blankScore = blankScore
            self.nonBlankScore = nonBlankScore
        }
    }
    
    /// Performs beam search CTC decoding
    /// - Parameters:
    ///   - logits: Model output logits of shape [batch, time, vocab]
    ///   - beamSize: Number of beams to maintain (default: 10)
    ///   - logitsLengths: Length of valid logits for each batch item
    ///   - lmWeight: Language model weight (optional)
    /// - Returns: Array of top hypothesis sequences for each batch item
    public func beamSearchDecode(
        logits: MLXArray,
        beamSize: Int = 10,
        logitsLengths: [Int]? = nil,
        lmWeight: Float = 0.0
    ) -> [[Int]] {
        // Get dimensions
        let batchSize = logits.shape[0]
        let timeSteps = logits.shape[1]
        
        // Apply log softmax for numerical stability
        let logProbs = logSoftmax(logits, axis: -1)
        
        var decodedBatch: [[Int]] = []
        
        // Process each batch item
        for b in 0..<batchSize {
            let length = logitsLengths?[b] ?? timeSteps
            
            // Initialize beam with empty prefix
            var beams = [BeamState(score: 0.0, blankScore: 0.0, nonBlankScore: Float.leastNormalMagnitude)]
            
            // Process each time step
            for t in 0..<length {
                var nextBeams: [String: BeamState] = [:]
                
                // Get log probabilities for this time step
                let frameLogProbs = logProbs[b, t]
                
                // Expand each beam
                for beam in beams {
                    // Score for blank token
                    let blankLogProb = frameLogProbs[blankToken].item(Float.self)
                    let prefixKey = beam.prefix.map { String($0) }.joined(separator: ",")
                    
                    // Update existing beam with blank
                    if let existing = nextBeams[prefixKey] {
                        nextBeams[prefixKey] = BeamState(
                            prefix: beam.prefix,
                            score: logAdd(existing.score, beam.score + blankLogProb),
                            blankScore: logAdd(existing.blankScore, beam.score + blankLogProb),
                            nonBlankScore: existing.nonBlankScore
                        )
                    } else {
                        nextBeams[prefixKey] = BeamState(
                            prefix: beam.prefix,
                            score: beam.score + blankLogProb,
                            blankScore: beam.score + blankLogProb,
                            nonBlankScore: Float.leastNormalMagnitude
                        )
                    }
                    
                    // Score for non-blank tokens
                    for token in 0..<vocabSize {
                        guard token != blankToken else { continue }
                        
                        let tokenLogProb = frameLogProbs[token].item(Float.self)
                        
                        // Check if extending or repeating
                        if !beam.prefix.isEmpty && beam.prefix.last == token {
                            // Repeating last token
                            let score = beam.blankScore + tokenLogProb
                            let newPrefixKey = prefixKey
                            
                            if let existing = nextBeams[newPrefixKey] {
                                nextBeams[newPrefixKey] = BeamState(
                                    prefix: beam.prefix,
                                    score: logAdd(existing.score, score),
                                    blankScore: existing.blankScore,
                                    nonBlankScore: logAdd(existing.nonBlankScore, score)
                                )
                            } else {
                                nextBeams[newPrefixKey] = BeamState(
                                    prefix: beam.prefix,
                                    score: score,
                                    blankScore: Float.leastNormalMagnitude,
                                    nonBlankScore: score
                                )
                            }
                        }
                        
                        // Extending with new token
                        let newPrefix = beam.prefix + [token]
                        let newPrefixKey = newPrefix.map { String($0) }.joined(separator: ",")
                        let score = beam.score + tokenLogProb
                        
                        if let existing = nextBeams[newPrefixKey] {
                            nextBeams[newPrefixKey] = BeamState(
                                prefix: newPrefix,
                                score: logAdd(existing.score, score),
                                blankScore: existing.blankScore,
                                nonBlankScore: logAdd(existing.nonBlankScore, score)
                            )
                        } else {
                            nextBeams[newPrefixKey] = BeamState(
                                prefix: newPrefix,
                                score: score,
                                blankScore: Float.leastNormalMagnitude,
                                nonBlankScore: score
                            )
                        }
                    }
                }
                
                // Prune beams to keep top-k
                beams = Array(nextBeams.values)
                    .sorted { $0.score > $1.score }
                    .prefix(beamSize)
                    .map { $0 }
            }
            
            // Get best hypothesis
            let bestBeam = beams.max { $0.score < $1.score } ?? BeamState()
            decodedBatch.append(bestBeam.prefix)
        }
        
        return decodedBatch
    }
    
    // MARK: - Helper Methods
    
    /// Log-sum-exp for numerical stability
    private func logAdd(_ a: Float, _ b: Float) -> Float {
        if a == Float.leastNormalMagnitude { return b }
        if b == Float.leastNormalMagnitude { return a }
        let maxVal = max(a, b)
        return maxVal + log(exp(a - maxVal) + exp(b - maxVal))
    }
    
    /// Decode token indices to string
    /// - Parameter tokens: Array of token indices
    /// - Returns: Decoded string
    public func tokensToString(_ tokens: [Int]) -> String {
        return tokens.compactMap { idx in
            guard idx >= 0 && idx < vocabulary.count else { return nil }
            return vocabulary[idx]
        }.joined()
    }
    
    /// Batch decode token sequences to strings
    /// - Parameter tokenBatch: Array of token sequences
    /// - Returns: Array of decoded strings
    public func batchTokensToStrings(_ tokenBatch: [[Int]]) -> [String] {
        return tokenBatch.map { tokensToString($0) }
    }
    
    // MARK: - Prefix Beam Search (for streaming)
    
    /// State for prefix beam search used in streaming scenarios
    public struct PrefixBeamSearchState {
        var beams: [BeamState]
        var frameIndex: Int
        
        public init() {
            self.beams = [BeamState(score: 0.0, blankScore: 0.0)]
            self.frameIndex = 0
        }
    }
    
    /// Performs incremental beam search for streaming
    /// - Parameters:
    ///   - logProbs: Log probabilities for current frame [vocab_size]
    ///   - state: Current beam search state
    ///   - beamSize: Number of beams to maintain
    /// - Returns: Updated state and current best hypothesis
    public func streamingBeamSearchStep(
        logProbs: MLXArray,
        state: inout PrefixBeamSearchState,
        beamSize: Int = 10
    ) -> [Int] {
        var nextBeams: [String: BeamState] = [:]
        
        // Process each beam
        for beam in state.beams {
            // Score for blank token
            let blankLogProb = logProbs[blankToken].item(Float.self)
            let prefixKey = beam.prefix.map { String($0) }.joined(separator: ",")
            
            // Update with blank
            if let existing = nextBeams[prefixKey] {
                nextBeams[prefixKey] = BeamState(
                    prefix: beam.prefix,
                    score: logAdd(existing.score, beam.score + blankLogProb),
                    blankScore: logAdd(existing.blankScore, beam.score + blankLogProb),
                    nonBlankScore: existing.nonBlankScore
                )
            } else {
                nextBeams[prefixKey] = BeamState(
                    prefix: beam.prefix,
                    score: beam.score + blankLogProb,
                    blankScore: beam.score + blankLogProb,
                    nonBlankScore: Float.leastNormalMagnitude
                )
            }
            
            // Score for non-blank tokens
            for token in 0..<vocabSize {
                guard token != blankToken else { continue }
                
                let tokenLogProb = logProbs[token].item(Float.self)
                
                // Check if extending or repeating
                if !beam.prefix.isEmpty && beam.prefix.last == token {
                    // Repeating last token
                    let score = beam.blankScore + tokenLogProb
                    
                    if let existing = nextBeams[prefixKey] {
                        nextBeams[prefixKey] = BeamState(
                            prefix: beam.prefix,
                            score: logAdd(existing.score, score),
                            blankScore: existing.blankScore,
                            nonBlankScore: logAdd(existing.nonBlankScore, score)
                        )
                    }
                }
                
                // Extending with new token
                let newPrefix = beam.prefix + [token]
                let newPrefixKey = newPrefix.map { String($0) }.joined(separator: ",")
                let score = beam.score + tokenLogProb
                
                if let existing = nextBeams[newPrefixKey] {
                    nextBeams[newPrefixKey] = BeamState(
                        prefix: newPrefix,
                        score: logAdd(existing.score, score),
                        blankScore: existing.blankScore,
                        nonBlankScore: logAdd(existing.nonBlankScore, score)
                    )
                } else {
                    nextBeams[newPrefixKey] = BeamState(
                        prefix: newPrefix,
                        score: score,
                        blankScore: Float.leastNormalMagnitude,
                        nonBlankScore: score
                    )
                }
            }
        }
        
        // Prune beams
        state.beams = Array(nextBeams.values)
            .sorted { $0.score > $1.score }
            .prefix(beamSize)
            .map { $0 }
        
        state.frameIndex += 1
        
        // Return current best hypothesis
        return state.beams.first?.prefix ?? []
    }
}

// MARK: - CTC Loss

/// CTC Loss computation for training
public class CTCLoss: Module {
    private let blankToken: Int
    private let reduction: String
    
    /// Initialize CTC loss
    /// - Parameters:
    ///   - blankToken: Index of blank token (default: 0)
    ///   - reduction: Reduction method: "none", "mean", "sum" (default: "mean")
    public init(blankToken: Int = 0, reduction: String = "mean") {
        self.blankToken = blankToken
        self.reduction = reduction
        super.init()
    }
    
    /// Compute CTC loss
    /// - Parameters:
    ///   - logits: Model predictions [batch, time, vocab]
    ///   - targets: Target sequences [batch, target_length]
    ///   - inputLengths: Valid input lengths per batch
    ///   - targetLengths: Valid target lengths per batch
    /// - Returns: CTC loss value
    public func callAsFunction(
        logits: MLXArray,
        targets: MLXArray,
        inputLengths: [Int],
        targetLengths: [Int]
    ) -> MLXArray {
        // Apply log softmax for numerical stability
        let logProbs = logSoftmax(logits, axis: -1)
        
        // Note: Full CTC loss implementation would require
        // forward-backward algorithm which is complex.
        // This is a simplified placeholder that should be
        // replaced with proper CTC loss computation
        
        // For now, return a placeholder loss
        // In production, use MLX's CTC loss when available
        // or implement the forward-backward algorithm
        
        let batchSize = logits.shape[0]
        var losses: [Float] = []
        
        for b in 0..<batchSize {
            // Simplified negative log likelihood
            var loss: Float = 0.0
            let inputLen = inputLengths[b]
            let targetLen = targetLengths[b]
            
            // Basic alignment cost (placeholder)
            for t in 0..<min(inputLen, targetLen) {
                let target = targets[b, t].item(Int.self)
                let logProb = logProbs[b, t, target].item(Float.self)
                loss -= logProb
            }
            
            losses.append(loss / Float(targetLen))
        }
        
        let lossArray = MLXArray(losses)
        
        // Apply reduction
        switch reduction {
        case "mean":
            return lossArray.mean()
        case "sum":
            return lossArray.sum()
        default:
            return lossArray
        }
    }
}