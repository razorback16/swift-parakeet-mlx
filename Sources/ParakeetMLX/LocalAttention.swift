import Foundation
import MLX
import MLXNN

// MARK: - Local Attention Support

// This file provides additional utilities and optimizations for local attention mechanisms.
// The main implementations of LocalRelPositionalEncoding and RelPositionMultiHeadLocalAttention
// are already in Conformer.swift and Attention.swift respectively.

// MARK: - Local Attention Utilities

/// Utility functions for local attention computation
public struct LocalAttentionUtils {
    
    /// Create a local attention mask for windowed attention.
    /// - Parameters:
    ///   - sequenceLength: Length of the sequence
    ///   - contextSize: Tuple of (left_context, right_context) for attention window
    /// - Returns: Attention mask as MLXArray
    public static func createLocalAttentionMask(
        sequenceLength: Int,
        contextSize: (Int, Int)
    ) -> MLXArray {
        let leftContext = contextSize.0
        let rightContext = contextSize.1
        
        // Create a mask that allows attention only within the specified window
        let mask = MLXArray.zeros([sequenceLength, sequenceLength], dtype: .bool)
        
        for i in 0..<sequenceLength {
            let start = max(0, i - leftContext)
            let end = min(sequenceLength, i + rightContext + 1)
            
            for j in start..<end {
                mask[i, j] = MLXArray(true)
            }
        }
        
        return mask
    }
    
    /// Calculate the effective receptive field for local attention.
    /// - Parameters:
    ///   - layerDepth: Number of transformer layers
    ///   - contextSize: Base context size per layer
    /// - Returns: Total receptive field size
    public static func effectiveReceptiveField(
        layerDepth: Int,
        contextSize: (Int, Int)
    ) -> Int {
        // Each layer expands the receptive field
        return (contextSize.0 + contextSize.1) * layerDepth
    }
    
    /// Optimize context size based on sequence length.
    /// - Parameters:
    ///   - sequenceLength: Input sequence length
    ///   - targetContext: Desired context size
    /// - Returns: Optimized context size tuple
    public static func optimizeContextSize(
        sequenceLength: Int,
        targetContext: (Int, Int)
    ) -> (Int, Int) {
        // For short sequences, use full attention
        if sequenceLength <= targetContext.0 + targetContext.1 {
            return (sequenceLength, sequenceLength)
        }
        
        // Otherwise use the target context
        return targetContext
    }
}

// MARK: - Chunked Local Attention

/// Implements chunked local attention for very long sequences.
/// This is useful for processing sequences that don't fit in memory with full local attention.
public class ChunkedLocalAttention: Module {
    let chunkSize: Int
    let overlap: Int
    let baseAttention: Module
    
    /// Initialize chunked local attention.
    /// - Parameters:
    ///   - chunkSize: Size of each chunk to process
    ///   - overlap: Number of overlapping tokens between chunks
    ///   - baseAttention: The base attention module to use for each chunk
    public init(
        chunkSize: Int,
        overlap: Int,
        baseAttention: Module
    ) {
        self.chunkSize = chunkSize
        self.overlap = overlap
        self.baseAttention = baseAttention
        super.init()
    }
    
    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: ConformerCache? = nil
    ) -> MLXArray {
        let batch = x.shape[0]
        let seqLen = x.shape[1]
        let hiddenDim = x.shape[2]
        
        // If sequence is short enough, process normally
        if seqLen <= chunkSize {
            if let attn = baseAttention as? RelPositionMultiHeadLocalAttention {
                return attn(x, x, x, mask: mask, cache: cache)
            }
            return x  // Fallback
        }
        
        // Process in chunks with overlap
        var outputs: [MLXArray] = []
        let stepSize = chunkSize - overlap
        
        for start in Swift.stride(from: 0, to: seqLen, by: stepSize) {
            let end = min(start + chunkSize, seqLen)
            let chunk = x[0..., start..<end, 0...]
            
            // Apply attention to chunk
            let chunkOutput: MLXArray
            if let attn = baseAttention as? RelPositionMultiHeadLocalAttention {
                chunkOutput = attn(chunk, chunk, chunk, mask: mask, cache: cache)
            } else {
                chunkOutput = chunk  // Fallback
            }
            
            // For overlapping regions, we'll average the outputs
            if start > 0 && overlap > 0 {
                // Handle overlap with previous chunk
                let overlapStart = 0
                let overlapEnd = min(overlap, chunkOutput.shape[1])
                let prevOverlapStart = outputs.last!.shape[1] - overlap
                
                // Average the overlapping region
                let avgOverlap = (outputs.last![0..., prevOverlapStart..., 0...] + 
                                  chunkOutput[0..., overlapStart..<overlapEnd, 0...]) / 2.0
                
                // Update the last output's overlap region
                outputs[outputs.count - 1] = concatenated([
                    outputs.last![0..., 0..<prevOverlapStart, 0...],
                    avgOverlap
                ], axis: 1)
                
                // Add the non-overlapping part of current chunk
                outputs.append(chunkOutput[0..., overlapEnd..., 0...])
            } else {
                outputs.append(chunkOutput)
            }
        }
        
        // Concatenate all chunks
        return concatenated(outputs, axis: 1)
    }
}

// MARK: - Helper Functions

private func concatenated(_ arrays: [MLXArray], axis: Int) -> MLXArray {
    return MLX.concatenated(arrays, axis: axis)
}

private func sqrt(_ x: Float) -> Float {
    return Foundation.sqrt(x)
}