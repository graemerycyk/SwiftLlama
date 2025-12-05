import Foundation
import llama

class LlamaModel {
    private let model: Model
    private let configuration: Configuration
    private let context: OpaquePointer
    private let vocab: OpaquePointer
    private var sampler: UnsafeMutablePointer<llama_sampler>
    private var batch: Batch
    private var tokens: [Token]
    private var generatedTokenAccount: Int32 = 0
    private var ended = false
    private let n_len: Int32 = 1024
    private var currentSamplingParams: SamplingParameters

    var shouldContinue: Bool {
        generatedTokenAccount < configuration.maxTokenCount && !ended
    }

    init(path: String, configuration: Configuration = .init()) throws {
        self.configuration = configuration
        llama_backend_init()
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED)

        var model_params = llama_model_default_params()
        #if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        #endif

        guard let model = llama_model_load_from_file(path, model_params) else {
            throw SwiftLlamaError.others("Cannot load model at path \(path)")
        }
        self.model = model
        
        guard let vocab = llama_model_get_vocab(model) else {
             throw SwiftLlamaError.others("Cannot load model vocabulary")
        }
        self.vocab = vocab

        guard let context = llama_init_from_model(model, configuration.contextParameters) else {
            throw SwiftLlamaError.others("Cannot load model context")
        }
        self.context = context

        self.tokens = []
        self.batch = llama_batch_init(Int32(configuration.batchSize * Configuration.historySize * 2), 0, 1)

        // Initialize with default sampling parameters
        self.currentSamplingParams = .default
        self.sampler = LlamaModel.createSampler(with: currentSamplingParams)

        try checkContextLength(context: context, model: model)
    }

    private func checkContextLength(context: Context, model: Model) throws {
        let n_ctx = llama_n_ctx(context)
        let n_ctx_train = llama_model_n_ctx_train(model)
        if n_ctx > n_ctx_train {
            throw SwiftLlamaError.others("Model was trained on \(n_ctx_train) context but tokens \(n_ctx) specified")
        }
    }
    
    /// Creates a new sampler chain with the specified sampling parameters.
    /// The sampler chain applies: penalties → top-k → top-p → temperature → softmax → distribution sampling
    private static func createSampler(with params: SamplingParameters) -> UnsafeMutablePointer<llama_sampler> {
        let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params())!
        
        // Add repeat penalty sampler first (operates on logits before other samplers)
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
            params.penaltyLastN,    // penalty_last_n: number of tokens to consider
            params.repeatPenalty,   // penalty_repeat: repeat penalty (1.0 = disabled)
            0.0,                    // penalty_freq: frequency penalty (0.0 = disabled)
            0.0                     // penalty_present: presence penalty (0.0 = disabled)
        ))
        
        // Top-K sampling: limit to top K tokens
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.topK))
        
        // Top-P (nucleus) sampling: limit to tokens covering top P probability mass
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.topP, 1))
        
        // Temperature: scale logits to control randomness
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature))
        
        // Final distribution sampling with seed (includes softmax internally)
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed))
        
        return sampler
    }
    
    /// Updates the sampler with new sampling parameters.
    /// This should be called before starting inference if you want to change parameters.
    func updateSamplingParameters(_ params: SamplingParameters) {
        // Free the old sampler
        llama_sampler_free(sampler)
        
        // Create new sampler with updated parameters
        currentSamplingParams = params
        sampler = LlamaModel.createSampler(with: params)
    }

    func start(for prompt: Prompt, samplingParams: SamplingParameters? = nil) throws {
        // Update sampling parameters if provided
        if let params = samplingParams {
            updateSamplingParameters(params)
        }
        
        ended = false
        tokens = tokenize(text: prompt.prompt, addBos: true)

        batch.clear()
        tokens.enumerated().forEach { index, token in
            batch.add(token: token, position: Int32(index), seqIDs: [0], logit: false)
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1 // true

        if llama_decode(context, batch) != 0 {
            throw SwiftLlamaError.decodeError
        }
        generatedTokenAccount = batch.n_tokens
    }

    func `continue`() throws -> String {
        let newToken = llama_sampler_sample(sampler, context, batch.n_tokens - 1)

        if llama_vocab_is_eog(vocab, newToken) || generatedTokenAccount == n_len {
            ended = true
            return ""
        }

        let piece = tokenToString(token: newToken)

        batch.clear()
        batch.add(token: newToken, position: generatedTokenAccount, seqIDs: [0], logit: true)
        generatedTokenAccount += 1

        if llama_decode(context, batch) != 0 {
            throw SwiftLlamaError.decodeError
        }
        return piece
    }

    // MARK: - Helpers

    /// Convert a sampled token to a Swift String (valid UTF-8, no interleaved \0 bytes).
    private func tokenToString(token: llama_token) -> String {
        var cap: Int32 = 32
        var buf = [CChar](repeating: 0, count: Int(cap))

        // First attempt
        var written = buf.withUnsafeMutableBufferPointer { p -> Int32 in
            guard let base = p.baseAddress else { return 0 }
            // Use the signature your llama module exposes (this matches your previous use: 6 args)
            return llama_token_to_piece(vocab, token, base, cap, 0, false)
        }

        // If negative, allocate required size and retry
        if written < 0 {
            cap = -written
            buf = [CChar](repeating: 0, count: Int(cap))
            written = buf.withUnsafeMutableBufferPointer { p -> Int32 in
                guard let base = p.baseAddress else { return 0 }
                return llama_token_to_piece(vocab, token, base, cap, 0, false)
            }
        }

        let count = Int(max(0, written))
        if count == 0 { return "" }

        // Decode exact byte count (no trailing NUL included)
        let bytes: [UInt8] = buf.prefix(count).map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }

    private func tokenize(text: String, addBos: Bool) -> [Token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (addBos ? 1 : 0) + 1

        return Array(unsafeUninitializedCapacity: n_tokens) { buffer, initializedCount in
            initializedCount = Int(
                llama_tokenize(vocab, text, Int32(utf8Count), buffer.baseAddress, Int32(n_tokens), addBos, false)
            )
        }
    }

    func clear() {
        tokens.removeAll()
        let memory = llama_get_memory(context)
        llama_memory_clear(memory, true)  // Clear both metadata and data buffers
    }
    
    // MARK: - Embedding Extraction
    
    /// Extract embedding vector from text
    /// - Parameter text: The input text to embed
    /// - Returns: Normalized embedding vector as [Float]
    /// - Throws: SwiftLlamaError if embedding extraction fails
    func extractEmbedding(for text: String) throws -> [Float] {
        // Get embedding dimension from model
        let embeddingDim = llama_model_n_embd(model)
        
        guard embeddingDim > 0 else {
            throw SwiftLlamaError.invalidEmbeddingDimension
        }
        
        // Tokenize the input text
        let tokens = tokenize(text: text, addBos: true)
        
        guard !tokens.isEmpty else {
            throw SwiftLlamaError.tokenizationFailed
        }
        
        // Create batch for embeddings
        var embeddingBatch = llama_batch_init(Int32(tokens.count), 0, 1)
        defer { llama_batch_free(embeddingBatch) }
        
        // Add tokens to batch
        for (i, token) in tokens.enumerated() {
            embeddingBatch.add(token: token, position: Int32(i), seqIDs: [0], logit: false)
        }
        
        // Enable embeddings mode
        llama_set_embeddings(context, true)
        defer {
            // Always restore to generation mode
            llama_set_embeddings(context, false)
        }
        
        // Decode to generate embeddings
        guard llama_decode(context, embeddingBatch) == 0 else {
            throw SwiftLlamaError.embeddingExtractionFailed("Failed to decode for embeddings")
        }
        
        // Get the embedding pointer
        guard let embeddingPtr = llama_get_embeddings(context) else {
            throw SwiftLlamaError.embeddingExtractionFailed("Failed to get embeddings from context")
        }
        
        // Copy embeddings to Float array
        var embedding = [Float](repeating: 0, count: Int(embeddingDim))
        for i in 0..<Int(embeddingDim) {
            embedding[i] = embeddingPtr[i]
        }
        
        // Normalize the embedding vector (L2 normalization)
        return normalize(embedding)
    }
    
    /// Normalize a vector using L2 normalization
    /// - Parameter vector: Input vector
    /// - Returns: Normalized vector with magnitude ~1.0
    private func normalize(_ vector: [Float]) -> [Float] {
        let magnitude = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        
        guard magnitude > 0 else {
            return vector
        }
        
        return vector.map { $0 / magnitude }
    }

    deinit {
        llama_batch_free(batch)
        llama_free(context)
        llama_model_free(model)
        llama_backend_free()
    }
}
