// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "SwiftLlama",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        .watchOS(.v11),
        .tvOS(.v18),
        .visionOS(.v2)
    ],
    products: [
        .library(name: "SwiftLlama", targets: ["SwiftLlama"]),
    ],
    targets: [
        .target(
            name: "llama",
            dependencies: [],
            path: "Sources/llama",
            exclude: [], // We rely on the provided sources
            sources: ["src"],
            resources: [
                .process("Resources")
            ],
            publicHeadersPath: "include",
            cSettings: [
                .define("GGML_USE_ACCELERATE"),
                .define("ACCELERATE_NEW_LAPACK"),
                .define("ACCELERATE_LAPACK_ILP64"),
                .define("GGML_USE_METAL"),
                .define("GGML_USE_CPU"),
                .define("GGML_VERSION", to: "\"b6906\""),
                .define("GGML_COMMIT", to: "\"0de0a015\""),
                .headerSearchPath("src"),
                .headerSearchPath("src/ggml-cpu"),
                .headerSearchPath("src/ggml-metal"),
                .unsafeFlags(["-O3", "-fno-objc-arc"])
            ],
            cxxSettings: [
                .define("GGML_USE_ACCELERATE"),
                .define("ACCELERATE_NEW_LAPACK"),
                .define("ACCELERATE_LAPACK_ILP64"),
                .define("GGML_USE_METAL"),
                .define("GGML_USE_CPU"),
                .headerSearchPath("src"),
                .headerSearchPath("src/ggml-cpu"),
                .headerSearchPath("src/ggml-metal"),
                .unsafeFlags(["-O3"])
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("Foundation")
            ]
        ),
        .target(
            name: "SwiftLlama",
            dependencies: ["llama"]
        ),
        .testTarget(
            name: "SwiftLlamaTests",
            dependencies: ["SwiftLlama"]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
