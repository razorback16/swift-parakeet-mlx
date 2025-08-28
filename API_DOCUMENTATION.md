# Swift Parakeet MLX API Documentation

## Phase 5 Complete: Unified API and Utilities

The Swift Parakeet MLX library now provides a comprehensive, unified API for acoustic speech recognition models with support for multiple architectures and advanced features.

## Core Features

### 1. Multiple Model Architectures
- **TDT-CTC** (Token-and-Duration Transducer CTC) - Fully implemented
- **CTC** (Connectionist Temporal Classification) - Architecture ready
- **RNN-T** (Recurrent Neural Network Transducer) - Architecture ready

### 2. Comprehensive Decoding Configuration

```swift
// Create decoding configurations for different model types
let ctcConfig = DecodingConfiguration.ctcDefault()
let rnntConfig = DecodingConfiguration.rnntDefault()
let tdtConfig = DecodingConfiguration.tdtDefault(maxSymbols: 15)

// Custom configuration with beam search
let customConfig = DecodingConfiguration(
    strategy: .beamSearch,
    beamSize: 10,
    lmWeight: 0.5,
    temperature: 0.8
)
```

### 3. Unified Model Factory

```swift
// Load model from HuggingFace Hub
let model = try await ParakeetMLX.load(
    from: "nvidia/parakeet-tdt_ctc-110m",
    dtype: .float16
)

// Auto-detect model type from configuration
let modelType = try ParakeetMLX.detectModelType(from: configURL)

// Create model from local files
let localModel = try await ParakeetMLX.createModel(
    type: .tdtctc,
    configPath: configURL,
    weightsPath: weightsURL,
    dtype: .float32
)
```

### 4. Streaming Support

```swift
// Create streaming-capable model
let streamingModel = ParakeetMLX.createStreamingModel(
    from: tdtModel,
    contextWindow: 256,
    lookbackFrames: 5
)

// Process audio chunks in real-time
for audioChunk in audioStream {
    let partial = try await streamingModel.processChunk(audioChunk)
    print("Partial: \(partial.text)")
}
let final = try await streamingModel.finalize()
```

### 5. Audio Processing Utilities

```swift
// Voice Activity Detection
let segments = ParakeetAudio.detectVoiceActivity(
    in: audio,
    sampleRate: 16000,
    frameDuration: 0.03,
    threshold: 0.5
)

// Noise Reduction
let cleanAudio = ParakeetAudio.reduceNoise(
    in: noisyAudio,
    reductionStrength: 0.7
)

// Preprocessing
let features = try ParakeetAudio.preprocess(
    audio: rawAudio,
    config: preprocessConfig
)
```

### 6. Enhanced Transcription Results

```swift
struct TranscriptionResult {
    let text: String                    // Final transcription
    let tokens: [AlignedToken]         // Word-level alignments
    let sentences: [AlignedSentence]   // Sentence segmentation
    let confidence: Float?             // Confidence score
    let processingTime: TimeInterval?  // Performance metrics
}
```

## API Usage Examples

### Basic Transcription

```swift
import ParakeetMLX
import MLX

// Load model
let model = try await ParakeetMLX.load(
    from: "nvidia/parakeet-tdt_ctc-110m",
    dtype: .float16
)

// Load and preprocess audio
let audioData = MLXArray(/* your audio data */)

// Transcribe
let result = try await model.transcribe(
    audioData: audioData,
    dtype: .float16
)

print("Transcription: \(result.text)")
print("Processing time: \(result.processingTime ?? 0) seconds")
```

### Streaming Transcription

```swift
// Create streaming model
let streamingModel = ParakeetMLX.createStreamingModel(from: model)

// Configure streaming
let streamConfig = StreamingConfig(
    chunkDuration: 5.0,
    overlapDuration: 0.5,
    useVAD: true
)

// Process chunks
for chunk in audioChunks {
    let result = try await model.transcribe(
        audioData: chunk,
        streamingConfig: streamConfig
    )
    updateUI(with: result.text)
}
```

### Advanced Configuration

```swift
// Custom decoding with beam search
let decodingConfig = DecodingConfiguration(
    strategy: .beamSearch,
    beamSize: 10,
    lmWeight: 0.3,
    wordScore: -0.5,
    temperature: 0.8
)

// Transcribe with custom settings
let result = try await model.transcribe(
    audioData: audio,
    dtype: .float16,
    decodingConfig: decodingConfig,
    streamingConfig: StreamingConfig(useVAD: true)
)
```

## Model Information Utilities

```swift
// Get model metadata
let info = try ParakeetInfo.getModelInfo(from: configPath)
print("Model type: \(info.type.description)")
print("Sample rate: \(info.sampleRate) Hz")
print("Vocabulary size: \(info.vocabulary.count)")
```

## Error Handling

The library provides comprehensive error types:

```swift
public enum ParakeetError: Error {
    case invalidModelType(String)
    case unsupportedDecoding(String)
    case audioProcessingError(String)
    case modelLoadingError(String)
    case configurationError(String)
    case streamingError(String)
    case decodingError(String)
}
```

## Protocol Conformance

All model types conform to the `ParakeetModel` protocol:

```swift
protocol ParakeetModel {
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
```

## Performance Optimizations

- **Cached conformer attention** for efficient streaming
- **Local attention mechanisms** for reduced memory usage
- **Optimized audio processing** with spectral subtraction
- **Efficient model weight loading** with safetensors support
- **Apple Silicon optimization** via MLX framework

## Build System Integration

The library is fully integrated with Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/your-repo/swift-parakeet-mlx.git", from: "1.0.0")
]
```

## Testing

Comprehensive test coverage including:
- Unit tests for all components
- Integration tests for model loading
- Performance benchmarks
- Streaming tests with mock data

Run tests with:
```bash
swift test
```

## Implementation Status

### ✅ Phase 1: Foundation (Complete)
- Cache system with streaming support
- Local attention mechanisms

### ✅ Phase 2: Streaming (Complete)
- StreamingParakeet implementation
- Incremental processing
- Cache management

### ✅ Phase 3: Model Variants (Complete)
- ParakeetCTC architecture
- ParakeetRNNT architecture
- ParakeetTDTCTC implementation
- CTCDecoder with beam search

### ✅ Phase 4: Audio Processing (Complete)
- Enhanced AudioProcessor
- Voice Activity Detection
- Noise reduction
- Spectral features

### ✅ Phase 5: API and Utilities (Complete)
- Unified ParakeetMLX factory
- Comprehensive decoding configuration
- Model type detection
- Audio utilities
- Error handling
- Protocol conformance
- Testing infrastructure

## Future Enhancements

While the core API is complete, future versions could add:
- Language model integration for improved accuracy
- Multi-language support
- Custom vocabulary adaptation
- Real-time speaker diarization
- Advanced noise cancellation algorithms