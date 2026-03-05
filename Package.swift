// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FireRedVADKit",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "FireRedVADKit", targets: ["FireRedVADKit"]),
    ],
    targets: [
        .target(
            name: "FireRedVADKit",
            resources: [
                .copy("Resources/FireRedVAD_Stream_N8.mlmodel"),
            ]
        ),
        .executableTarget(
            name: "PipelineParityTest",
            dependencies: ["FireRedVADKit"],
            path: "Tools/PipelineParityTest"
        ),
        .testTarget(
            name: "FireRedVADKitTests",
            dependencies: ["FireRedVADKit"]
        ),
    ]
)
