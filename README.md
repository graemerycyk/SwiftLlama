# SwiftLlama

SwiftLlama is a Swift-first wrapper around [llama.cpp](https://github.com/ggerganov/llama.cpp.git) that brings fast local LLM text generation *and* embedding extraction to Apple platforms with a familiar async/await API.

## Features
- 🔤 **Text generation**: streaming or non-streaming responses using AsyncSequence, Combine, or plain async calls.
- 🧠 **Embedding extraction**: normalized sentence embeddings from GGUF models (nomic-embed, bert-style encoders, etc.) with automatic pooling and context management.
- 🧰 **Example projects**: command-line, iOS, and macOS sample apps in `TestProjects/`.

## Installation

Add SwiftLlama to your `Package.swift`:

```swift
.package(url: "https://github.com/graemerycyk/SwiftLlama.git", from: "0.6.0")
```

Then add `SwiftLlama` as a dependency of your target. (Use a tagged version if you’re tracking releases.)

## Text Generation

```swift
import SwiftLlama

let swiftLlama = try SwiftLlama(modelPath: "/path/to/llama-model.gguf")
let prompt = Prompt(prompt: "Write a haiku about sunsets.")

// Non-streaming
let response = try await swiftLlama.start(for: prompt)

// Async stream
for try await delta in swiftLlama.start(for: prompt) {
    print(delta, terminator: "")
}

// Combine publisher
swiftLlama.start(for: prompt)
    .sink(receiveCompletion: { print($0) },
          receiveValue: { print($0, terminator: "") })
    .store(in: &cancellables)
```

## Embedding Quick Start

### 1. Download an embedding model

```bash
mkdir -p ~/models && cd ~/models
wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf
ls -lh nomic-embed-text-v1.5.Q8_0.gguf # ~450MB
```

### 2. Extract your first embedding

```swift
import SwiftLlama

let modelPath = "\(NSHomeDirectory())/models/nomic-embed-text-v1.5.Q8_0.gguf"
let swiftLlama = try SwiftLlama(modelPath: modelPath)
let embedding = try await swiftLlama.extractEmbedding(for: "Hello, world!")

print("Dimension:", embedding.count)          // 384
print("Magnitude:", sqrt(embedding.reduce(0) { $0 + $1 * $1 })) // ≈ 1.0
```

### 3. Compute similarity

```swift
let emb1 = try await swiftLlama.extractEmbedding(for: "The cat sat on the mat")
let emb2 = try await swiftLlama.extractEmbedding(for: "A feline rested on the rug")

let similarity = zip(emb1, emb2).reduce(0) { $0 + $1.0 * $1.1 } // cosine similarity
print("Similarity:", similarity) // ~0.7-0.9 for similar texts
```

### Recommended models
- **nomic-embed-text-v1.5** (384 dims, max 2048 tokens) – great all-rounder.
- **all-MiniLM-L6-v2** (384 dims, lightweight) – lower latency / memory.
- Use Q8_0 for quality, Q6_K for balance, Q4_K_M for mobile-friendly builds.

All embeddings returned by SwiftLlama are L2-normalized and the library automatically mean-pools encoder-only architectures (BERT/nomic) while using last-token embeddings for decoder models.

## Common Embedding Workflows

### Semantic search

```swift
let docs = ["Reset your password", "Create an account", "Update profile"]
let docEmbeddings = try await docs.asyncMap { try await swiftLlama.extractEmbedding(for: $0) }

let query = "I forgot my password"
let queryEmbedding = try await swiftLlama.extractEmbedding(for: query)

let bestMatch = zip(docs.indices, docEmbeddings)
    .map { (idx, emb) in (idx, cosineSimilarity(queryEmbedding, emb)) }
    .max { $0.1 < $1.1 }!

print("Best match:", docs[bestMatch.0])
```

### Retrieval-Augmented Generation (RAG)

```swift
let kbEmbeddings = try await knowledgeBase.asyncMap { article in
    (article, try await swiftLlama.extractEmbedding(for: article.content))
}

let questionEmbedding = try await swiftLlama.extractEmbedding(for: userQuestion)
let context = kbEmbeddings
    .map { ($0.0, cosineSimilarity(questionEmbedding, $0.1)) }
    .sorted { $0.1 > $1.1 }
    .prefix(3)
    .map(\.0.content)
    .joined(separator: "\n\n")

let prompt = Prompt(prompt: "Context:\n\(context)\n\nQuestion: \(userQuestion)")
let answer = try await swiftLlama.start(for: prompt)
```

### Duplicate detection

```swift
let similarity = cosineSimilarity(
    try await swiftLlama.extractEmbedding(for: textA),
    try await swiftLlama.extractEmbedding(for: textB)
)

if similarity > 0.9 {
    print("Potential duplicate")
}
```

### Helper utilities

```swift
func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count else { return 0 }
    return zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
}

extension SwiftLlama {
    func extractEmbeddings(for texts: [String]) async throws -> [[Float]] {
        try await texts.asyncMap { try await extractEmbedding(for: $0) }
    }
}

func findTopK(query: [Float], in embeddings: [[Float]], k: Int = 5) -> [(Int, Float)] {
    embeddings.enumerated()
        .map { ($0.offset, cosineSimilarity(query, $0.element)) }
        .sorted { $0.1 > $1.1 }
        .prefix(k)
}
```

## Performance & Usage Tips
- Process texts sequentially; each `extractEmbedding` call spins up a dedicated llama.cpp context and cleans it up automatically.
- Cache embeddings for frequently requested strings to avoid recomputation.
- Preprocess text (trim whitespace, collapse repeated spaces, optional lowercase) for better similarity.
- Pick the smallest quantization that meets your quality needs (`Q4_K_M` is great for mobile, `Q8_0` for desktops).
- Break very large corpora into chunks (e.g. batches of 50–100 texts) to keep memory predictable.

## Troubleshooting
- **“Invalid embedding dimension”** → You loaded a chat/generative model; switch to an embedding GGUF such as `nomic-embed-text`.
- **`embeddingExtractionFailed("Failed to get embeddings from context")`** → Upgrade to this build; embeddings now create their own context and no longer depend on `start()`.
- **Low similarity for similar texts** → Normalize or lowercase text, or try a domain-specific model (BGE, MiniLM, etc.).
- **Slow extraction** → Use a lighter quantization, smaller model, or ensure you’re not running multiple SwiftLlama instances simultaneously.
- **Different dimension than expected** → Print `embedding.count` to verify the actual model dimension (384, 512, 768, …).

## Testing & Verification

1. **Set up an embedding model path**
   ```bash
   export EMBEDDING_MODEL_PATH="$HOME/models/nomic-embed-text-v1.5.Q8_0.gguf"
   ```
2. **Run unit tests**
   ```bash
   cd /Users/grae/Developer/SwiftLlama
   swift test
   ```
   Tests verify normalization, similarity thresholds, and edge cases (skipped automatically if `EMBEDDING_MODEL_PATH` is unset).
3. **Manual smoke test**
   ```bash
   swift run test-embedding "$EMBEDDING_MODEL_PATH"
   ```
   Expect three checks: single embedding (dimension/magnitude), similar-text similarity (~0.7–0.9), and different-text similarity (<0.3).
4. **Custom script**
   ```bash
   cat > test_basic.swift <<'EOF'
   import SwiftLlama
   let path = ProcessInfo.processInfo.environment["EMBEDDING_MODEL_PATH"]!
   let model = try SwiftLlama(modelPath: path)
   let emb = try await model.extractEmbedding(for: "Hello, world!")
   print("Dim:", emb.count, "Mag:", sqrt(emb.reduce(0) { $0 + $1 * $1 }))
   EOF
   swift test_basic.swift
   ```

## Example / Test Projects
- `TestProjects/TestApp-Commandline` – CLI demo (see `test-embedding.swift`).
- `TestProjects/TestApp-iOS` & `TestApp-macOS` – sample apps showing integration with SwiftUI.
- Video walkthrough of the CLI app running Llama 3: [YouTube link](https://youtu.be/w1VEM00cJWo).

## Contributing

Issues and pull requests are welcome! If you find a bug or have feature ideas, open an issue on GitHub. When contributing code, please run `swift test` (with `EMBEDDING_MODEL_PATH` set) and include any relevant docs updates.



