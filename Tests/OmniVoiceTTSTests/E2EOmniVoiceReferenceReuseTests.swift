import AudioCommon
import Foundation
import MLX
import XCTest

@testable import OmniVoiceTTS

/// A reference encoded once must give the same audio as encoding it from the
/// file on every call, for every sentence it is reused for. Uses the cached
/// default bundle without downloads.
final class E2EOmniVoiceReferenceReuseTests: XCTestCase {
    func testReusedReferenceMatchesPerCallEncode() async throws {
        let dir = try HuggingFaceDownloader.getCacheDirectory(for: OmniVoiceTTSModel.defaultModelId)
        for file in ["model.safetensors", "audio_tokenizer/model.safetensors", "tokenizer.json"]
        where !FileManager.default.fileExists(atPath: dir.appendingPathComponent(file).path) {
            throw XCTSkip("missing cached \(OmniVoiceTTSModel.defaultModelId)/\(file)")
        }
        let tts = try await OmniVoiceTTSModel.fromBundle(dir)
        let referenceAudio = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("MagpieTTSCoreMLTests/Resources/magpie-hello-world.wav")

        let reference = try tts.encodeReference(referenceAudio)
        XCTAssertGreaterThan(reference.tokenCount, 0)

        for (text, duration) in [("Hello there.", 1.2), ("This is a second sentence.", 2.0)] {
            let reused = try tts.generate(
                text: text, reference: reference, referenceText: "Hello world.",
                duration: duration, numSteps: 16)
            let perCall = try tts.generate(
                text: text, referenceAudio: referenceAudio, referenceText: "Hello world.",
                duration: duration, numSteps: 16)
            XCTAssertGreaterThan(reused.map { abs($0) }.max() ?? 0, 0.001, "\(text): silence")
            XCTAssertTrue(
                reused.elementsEqual(perCall, by: { $0.bitPattern == $1.bitPattern }), text)
        }
    }
}
