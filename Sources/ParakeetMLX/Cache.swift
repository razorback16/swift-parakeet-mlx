import Foundation
import MLX

// MARK: - ConformerCache

/// Cache implementation for efficient inference in Conformer models.
/// Stores key-value pairs and convolution states to avoid redundant computation.
public class ConformerCache {
    /// Cached keys for attention mechanism
    public var keys: MLXArray?
    
    /// Cached values for attention mechanism
    public var values: MLXArray?
    
    /// Cached convolution states
    public var conv: MLXArray?
    
    /// Current offset in the cache
    public var offset: Int = 0
    
    /// Step size for cache expansion
    public let step: Int = 256
    
    public init() {}
    
    /// Update and fetch key-value pairs for attention.
    /// - Parameters:
    ///   - keys: New keys to add to cache [batch, head, seq, dim]
    ///   - values: New values to add to cache [batch, head, seq, dim]
    /// - Returns: Updated keys and values from cache
    public func updateAndFetchKV(
        _ keys: MLXArray,
        _ values: MLXArray
    ) -> (MLXArray, MLXArray) {
        let prev = offset
        
        // Check if we need to expand the cache
        if self.keys == nil || 
           self.values == nil || 
           (prev + keys.shape[2]) > self.keys!.shape[2] {
            
            let B = keys.shape[0]
            let H = keys.shape[1]
            let S = keys.shape[2]
            let DKeys = keys.shape[3]
            let DValues = values.shape[3]
            
            // Calculate cache size aligned to step
            let SCache = ((step + S - 1) / step) * step
            
            let newK = MLXArray.zeros([B, H, SCache, DKeys], dtype: keys.dtype)
            let newV = MLXArray.zeros([B, H, SCache, DValues], dtype: values.dtype)
            
            if self.keys == nil || self.values == nil {
                self.keys = newK
                self.values = newV
            } else {
                // Trim existing cache if not aligned to step
                if prev % step != 0 {
                    self.keys = self.keys![0..., 0..., 0..<prev, 0...]
                    self.values = self.values![0..., 0..., 0..<prev, 0...]
                }
                
                // Concatenate old and new cache
                self.keys = concatenated([self.keys!, newK], axis: 2)
                self.values = concatenated([self.values!, newV], axis: 2)
            }
        }
        
        // Update cache with new keys and values
        offset += keys.shape[2]
        self.keys![0..., 0..., prev..<offset, 0...] = keys
        self.values![0..., 0..., prev..<offset, 0...] = values
        
        // Return the active portion of the cache
        return (
            self.keys![0..., 0..., 0..<offset, 0...],
            self.values![0..., 0..., 0..<offset, 0...]
        )
    }
    
    /// Update and fetch convolution cache.
    /// - Parameters:
    ///   - x: Input tensor [batch, seq, dim]
    ///   - padding: Padding size for convolution
    /// - Returns: Padded input with cached context
    public func updateAndFetchConv(_ x: MLXArray, padding: Int = 0) -> MLXArray {
        if padding == 0 {
            return x
        }
        
        let B = x.shape[0]
        let S = x.shape[1]
        let D = x.shape[2]
        
        // Initialize conv cache if needed
        if conv == nil {
            conv = MLXArray.zeros([B, padding, D], dtype: x.dtype)
        }
        
        // Determine how many tokens to cache
        let tokensToCache = min(padding, S)
        
        // Extract the tokens to cache from the end of the sequence
        let cacheUpdate = x[0..., (S - tokensToCache)..<S, 0...]
        
        // Update the cache with a sliding window
        if tokensToCache < padding {
            // Slide the window and append new tokens
            conv = concatenated([
                conv![0..., tokensToCache..., 0...],
                cacheUpdate
            ], axis: 1)
        } else {
            // Replace entire cache
            conv = cacheUpdate
        }
        
        // Concatenate cache with input and add padding
        let result = concatenated([conv!, x], axis: 1)
        
        // Add padding at the end to match expected output size
        return MLX.padded(
            result,
            widths: [(0, 0), (0, padding), (0, 0)].map { IntOrPair($0) },
            mode: .constant,
            value: MLXArray(0.0)
        )
    }
}

// MARK: - RotatingConformerCache

/// Rotating cache implementation for streaming scenarios.
/// Uses a ring buffer to efficiently manage memory in long-running inference.
public class RotatingConformerCache: ConformerCache {
    /// Maximum capacity of the cache
    public let capacity: Int
    
    /// Number of tokens to drop from cache (for streaming)
    public let cacheDropSize: Int
    
    /// Initialize rotating cache with specified capacity.
    /// - Parameters:
    ///   - capacity: Maximum number of tokens to cache
    ///   - cacheDropSize: Number of tokens to drop from the end (for streaming)
    public init(capacity: Int, cacheDropSize: Int = 0) {
        self.capacity = capacity
        self.cacheDropSize = cacheDropSize
        super.init()
    }
    
    /// Append new data to ring buffer.
    private func ringAppend(_ buf: MLXArray, _ new: MLXArray) {
        let C = capacity
        let pos = offset % C
        let T = new.shape[2]
        let first = min(T, C - pos)
        
        // Write first part up to buffer end
        buf[0..., 0..., pos..<(pos + first), 0...] = new[0..., 0..., 0..<first, 0...]
        
        // Write remaining part from buffer beginning
        if T > first {
            buf[0..., 0..., 0..<(T - first), 0...] = new[0..., 0..., first..., 0...]
        }
    }
    
    /// Update and fetch key-value pairs with rotating cache.
    public override func updateAndFetchKV(
        _ keys: MLXArray,
        _ values: MLXArray
    ) -> (MLXArray, MLXArray) {
        let B = keys.shape[0]
        let H = keys.shape[1]
        let S = keys.shape[2]
        let D = keys.shape[3]
        
        // Initialize cache if needed
        if self.keys == nil || self.values == nil {
            self.keys = MLXArray.zeros([B, H, capacity, D], dtype: keys.dtype)
            self.values = MLXArray.zeros([B, H, capacity, D], dtype: values.dtype)
        }
        
        // Prepare historical keys and values
        let histK: MLXArray
        let histV: MLXArray
        
        if offset < capacity {
            // Cache not full yet, use only filled portion
            histK = self.keys![0..., 0..., 0..<offset, 0...]
            histV = self.values![0..., 0..., 0..<offset, 0...]
        } else {
            // Cache is full, roll to align history
            let shift = -(offset % capacity)
            histK = MLX.roll(self.keys!, shift: shift, axis: 2)
            histV = MLX.roll(self.values!, shift: shift, axis: 2)
        }
        
        // Concatenate history with new keys/values
        let kOut = concatenated([histK, keys], axis: 2)
        let vOut = concatenated([histV, values], axis: 2)
        
        // Determine how many tokens to cache
        let drop = cacheDropSize
        let toCache = min(max(0, S - drop), capacity)
        
        if toCache > 0 {
            // Extract chunk to cache (excluding dropped tokens)
            let startIdx = S - cacheDropSize - toCache
            let endIdx = S - cacheDropSize
            
            let kChunk = keys[0..., 0..., startIdx..<endIdx, 0...]
            let vChunk = values[0..., 0..., startIdx..<endIdx, 0...]
            
            // Append to ring buffer
            ringAppend(self.keys!, kChunk)
            ringAppend(self.values!, vChunk)
            
            offset += toCache
        }
        
        return (kOut, vOut)
    }
    
    /// Update and fetch convolution cache with rotation.
    public override func updateAndFetchConv(_ x: MLXArray, padding: Int = 0) -> MLXArray {
        if padding == 0 {
            return x
        }
        
        let B = x.shape[0]
        let S = x.shape[1]
        let D = x.shape[2]
        
        // Initialize conv cache if needed
        if conv == nil {
            conv = MLXArray.zeros([B, padding, D], dtype: x.dtype)
        }
        
        // Only cache tokens that aren't being dropped
        if S > cacheDropSize {
            let tokensToCache = min(padding, S - cacheDropSize)
            let cacheUpdate = x[0..., (S - tokensToCache)..<S, 0...]
            
            if tokensToCache < padding {
                // Slide the window and append new tokens
                conv = concatenated([
                    conv![0..., tokensToCache..., 0...],
                    cacheUpdate
                ], axis: 1)
            } else {
                // Replace entire cache
                conv = cacheUpdate
            }
        }
        
        // Concatenate cache with input and add padding
        let result = concatenated([conv!, x], axis: 1)
        
        return MLX.padded(
            result,
            widths: [(0, 0), (0, padding), (0, 0)].map { IntOrPair($0) },
            mode: .constant,
            value: MLXArray(0.0)
        )
    }
}

// MARK: - Helper Functions

private func concatenated(_ arrays: [MLXArray], axis: Int) -> MLXArray {
    return MLX.concatenated(arrays, axis: axis)
}