import Foundation
import llama
import Combine

public class SwiftLlama {
    private let model: LlamaModel
    private let configuration: Configuration
    private var contentStarted = false
    private var sessionSupport = false {
        didSet {
            if !sessionSupport {
                session = nil
            }
        }
    }

    private var session: Session?
    private lazy var resultSubject: CurrentValueSubject<String, Error> = {
        .init("")
    }()
    private var generatedTokenCache = ""

    var maxLengthOfStopToken: Int {
        configuration.stopTokens.map { $0.count }.max() ?? 0
    }

    public init(modelPath: String,
                 modelConfiguration: Configuration = .init()) throws {
        self.model = try LlamaModel(path: modelPath, configuration: modelConfiguration)
        self.configuration = modelConfiguration
    }

    private func prepare(sessionSupport: Bool, for prompt: Prompt) -> Prompt {
        contentStarted = false
        generatedTokenCache = ""
        self.sessionSupport = sessionSupport
        if sessionSupport {
            if session == nil {
                session = Session(lastPrompt: prompt)
            } else {
                session?.lastPrompt = prompt
            }
            return session?.sessionPrompt ?? prompt
        } else {
            return prompt
        }
    }

    private func isStopToken() -> Bool {
        configuration.stopTokens.reduce(false) { partialResult, stopToken in
            generatedTokenCache.hasSuffix(stopToken)
        }
    }

    private func response(for prompt: Prompt, samplingParams: SamplingParameters?, output: (String) -> Void, finish: () -> Void) {
        func finaliseOutput() {
            configuration.stopTokens.forEach {
                generatedTokenCache = generatedTokenCache.replacingOccurrences(of: $0, with: "")
            }
            output(generatedTokenCache)
            finish()
            generatedTokenCache = ""
        }
        defer { model.clear() }
        do {
            try model.start(for: prompt, samplingParams: samplingParams)
            while model.shouldContinue {
                var delta = try model.continue()
                if contentStarted { // remove the prefix empty spaces
                    if needToStop(after: delta, output: output) {
                        finish()
                        break
                    }
                } else {
                    delta = delta.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !delta.isEmpty {
                        contentStarted = true
                        if needToStop(after: delta, output: output) {
                            finish()
                            break
                        }
                    }
                }
            }
            finaliseOutput()
        } catch {
            finaliseOutput()
        }
    }

    /// Handling logic of StopToken
    private func needToStop(after delta: String, output: (String) -> Void) -> Bool {
        // If no stop tokens, just stream through
        guard maxLengthOfStopToken > 0 else {
            output(delta)
            return false
        }

        generatedTokenCache += delta

        // 1) If any stop token appears, cut output before it and stop
        if let stopRange = configuration.stopTokens
            .compactMap({ generatedTokenCache.range(of: $0) })
            .min(by: { $0.lowerBound < $1.lowerBound }) // earliest occurrence
        {
            let before = String(generatedTokenCache[..<stopRange.lowerBound])
            if !before.isEmpty { output(before) }
            generatedTokenCache.removeAll(keepingCapacity: false)
            return true
        }

        // 2) Stream everything except a small tail so split stop tokens are caught next time
        let tail = max(maxLengthOfStopToken - 1, 0)
        if generatedTokenCache.count > tail {
            let cut = generatedTokenCache.index(generatedTokenCache.endIndex, offsetBy: -tail)
            let safe = String(generatedTokenCache[..<cut])
            if !safe.isEmpty { output(safe) }
            generatedTokenCache.removeFirst(safe.count)
        }

        return false
    }

    @SwiftLlamaActor
    public func start(
        for prompt: Prompt,
        sessionSupport: Bool = false,
        samplingParams: SamplingParameters? = nil
    ) -> AsyncThrowingStream<String, Error> {
        let sessionPrompt = prepare(sessionSupport: sessionSupport, for: prompt)
        return .init { continuation in
            Task {
                response(for: sessionPrompt, samplingParams: samplingParams) { [weak self] delta in
                    continuation.yield(delta)
                    self?.session?.response(delta: delta)
                } finish: { [weak self] in
                    continuation.finish()
                    self?.session?.endResponse()
                }
            }
        }
    }

    @SwiftLlamaActor
    public func start(
        for prompt: Prompt,
        sessionSupport: Bool = false,
        samplingParams: SamplingParameters? = nil
    ) -> AnyPublisher<String, Error> {
        let sessionPrompt = prepare(sessionSupport: sessionSupport, for: prompt)
        Task {
            response(for: sessionPrompt, samplingParams: samplingParams) { delta in
                resultSubject.send(delta)
                session?.response(delta: delta)
            } finish: {
                resultSubject.send(completion: .finished)
                session?.endResponse()
            }
        }
        return resultSubject.eraseToAnyPublisher()
    }

    @SwiftLlamaActor
    public func start(
        for prompt: Prompt,
        sessionSupport: Bool = false,
        samplingParams: SamplingParameters? = nil
    ) async throws -> String {
        var result = ""
        for try await value in start(for: prompt, sessionSupport: sessionSupport, samplingParams: samplingParams) as AsyncThrowingStream<String, Error> {
            result += value
        }
        return result
    }
    
    // MARK: - Convenience methods with individual parameters
    
    /// Start inference with individual sampling parameters for convenience.
    /// - Parameters:
    ///   - prompt: The prompt to generate from
    ///   - sessionSupport: Whether to maintain session context
    ///   - temperature: Controls randomness (0.0-1.0+, lower = more focused). Default: 0.3
    ///   - repeatPenalty: Penalizes repeated tokens (1.0 = no penalty, 1.1-1.2 typical). Default: 1.1
    ///   - topP: Nucleus sampling threshold (0.0-1.0). Default: 0.9
    ///   - topK: Limits vocabulary to top K tokens. Default: 40
    /// - Returns: The generated text
    @SwiftLlamaActor
    public func start(
        for prompt: Prompt,
        sessionSupport: Bool = false,
        temperature: Float = 0.3,
        repeatPenalty: Float = 1.1,
        topP: Float = 0.9,
        topK: Int32 = 40
    ) async throws -> String {
        let params = SamplingParameters(
            temperature: temperature,
            repeatPenalty: repeatPenalty,
            topP: topP,
            topK: topK
        )
        return try await start(for: prompt, sessionSupport: sessionSupport, samplingParams: params)
    }
    
    /// Start streaming inference with individual sampling parameters.
    @SwiftLlamaActor
    public func start(
        for prompt: Prompt,
        sessionSupport: Bool = false,
        temperature: Float,
        repeatPenalty: Float,
        topP: Float,
        topK: Int32
    ) -> AsyncThrowingStream<String, Error> {
        let params = SamplingParameters(
            temperature: temperature,
            repeatPenalty: repeatPenalty,
            topP: topP,
            topK: topK
        )
        return start(for: prompt, sessionSupport: sessionSupport, samplingParams: params)
    }
    
    // MARK: - Embedding Extraction
    
    /// Extract embedding vector from text using the loaded model
    /// - Parameter text: The input text to embed
    /// - Returns: Normalized embedding vector as [Float] with magnitude ~1.0
    /// - Throws: SwiftLlamaError if embedding extraction fails
    @SwiftLlamaActor
    public func extractEmbedding(for text: String) async throws -> [Float] {
        return try model.extractEmbedding(for: text)
    }
    
    /// Convenience method to embed text - alias for extractEmbedding
    /// - Parameter text: The input text to embed
    /// - Returns: Normalized embedding vector as [Float] with magnitude ~1.0
    /// - Throws: SwiftLlamaError if embedding extraction fails
    @SwiftLlamaActor
    public func embed(text: String) async throws -> [Float] {
        return try model.extractEmbedding(for: text)
    }
    
    /// Get the embedding dimension of the loaded model
    /// - Returns: The number of dimensions in the model's embeddings (e.g., 768 for nomic-embed-text)
    @SwiftLlamaActor
    public func embeddingDimension() -> Int32 {
        return model.embeddingDimension()
    }
}
