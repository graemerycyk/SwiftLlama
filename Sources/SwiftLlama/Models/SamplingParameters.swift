import Foundation

/// Parameters controlling the sampling behavior during text generation.
/// These parameters help produce coherent, focused output and prevent degenerate repetition.
public struct SamplingParameters: Sendable {
    /// Controls randomness in token selection. Lower values make output more focused/deterministic.
    /// - Range: 0.0 to 1.0+ (0.0 = greedy/deterministic, higher = more random)
    /// - Typical: 0.3-0.7 for focused output, 0.8-1.0 for creative output
    public let temperature: Float
    
    /// Penalizes repeated tokens to prevent degenerate loops.
    /// - Range: 1.0+ (1.0 = no penalty, 1.1-1.2 typical)
    /// - Higher values more aggressively discourage repetition
    public let repeatPenalty: Float
    
    /// Nucleus sampling: only considers tokens whose cumulative probability exceeds this threshold.
    /// - Range: 0.0 to 1.0 (0.9 typical)
    /// - Lower values make output more focused
    public let topP: Float
    
    /// Limits vocabulary to the top K most probable tokens.
    /// - Range: 1+ (40 typical, 0 = disabled)
    /// - Lower values make output more focused
    public let topK: Int32
    
    /// Number of recent tokens to consider for repeat penalty.
    /// - Range: 0+ (64 typical, 0 = disabled, -1 = context size)
    public let penaltyLastN: Int32
    
    /// Seed for random sampling (for reproducibility).
    public let seed: UInt32
    
    /// Default parameters optimized for coherent, focused output.
    public static let `default` = SamplingParameters()
    
    /// More creative parameters with higher randomness.
    public static let creative = SamplingParameters(
        temperature: 0.8,
        repeatPenalty: 1.05,
        topP: 0.95,
        topK: 50
    )
    
    /// Very focused/deterministic parameters.
    public static let focused = SamplingParameters(
        temperature: 0.1,
        repeatPenalty: 1.15,
        topP: 0.8,
        topK: 30
    )
    
    public init(
        temperature: Float = 0.3,
        repeatPenalty: Float = 1.1,
        topP: Float = 0.9,
        topK: Int32 = 40,
        penaltyLastN: Int32 = 64,
        seed: UInt32 = 1234
    ) {
        self.temperature = temperature
        self.repeatPenalty = repeatPenalty
        self.topP = topP
        self.topK = topK
        self.penaltyLastN = penaltyLastN
        self.seed = seed
    }
}

