// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "ParakeetMLX",
    platforms: [
        .macOS(.v14),
        .iOS(.v16),
    ],
    products: [
        .library(
            name: "ParakeetMLX",
            targets: ["ParakeetMLX"]
        ),
        .executable(
            name: "TranscriptionApp",
            targets: ["TranscriptionApp"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.17"),
    ],
    targets: [
        .target(
            name: "ParakeetMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/ParakeetMLX"
        ),
        .executableTarget(
            name: "TranscriptionApp",
            dependencies: [
                "ParakeetMLX"
            ],
            path: "Sources/TranscriptionApp"
        ),
        .testTarget(
            name: "ParakeetMLXTests",
            dependencies: ["ParakeetMLX"],
            path: "Tests"
        ),
    ]
)
