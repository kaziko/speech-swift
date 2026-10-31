import Foundation
import MLX
import MLXNN
import XCTest

@testable import OmniVoiceTTS

/// `loadWeights` keeps the bundle's float dtype (no fp32 upcast), and a decode
/// on fp16 weights still runs its guidance math in fp32 and fills every target
/// position. Tiny random model saved to a temp safetensors file: no download.
final class OmniVoiceWeightDTypeTests: XCTestCase {
    private func tinyConfig() -> OmniVoiceConfig {
        var cfg = OmniVoiceConfig()
        cfg.numLayers = 2
        cfg.hiddenSize = 64
        cfg.numAttentionHeads = 4
        cfg.numKeyValueHeads = 2
        cfg.headDim = 16
        cfg.intermediateSize = 128
        cfg.textVocabSize = 64
        cfg.audioVocabSize = 33
        cfg.audioMaskId = 32
        return cfg
    }

    /// Saves a random tiny model's weights as `dtype` and loads them into a fresh model.
    private func roundTrip(_ dtype: DType) throws -> OmniVoiceModel {
        MLXRandom.seed(5)
        let cfg = tinyConfig()
        let source = OmniVoiceModel(cfg)
        let arrays = Dictionary(uniqueKeysWithValues: source.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) })
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("omnivoice-dtype-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try MLX.save(arrays: arrays, url: url)
        let model = OmniVoiceModel(cfg)
        try model.loadWeights(from: url)
        return model
    }

    func testFp16BundleStaysFp16() throws {
        let model = try roundTrip(.float16)
        for (key, value) in model.parameters().flattened() {
            XCTAssertEqual(value.dtype, .float16, key)
        }
    }

    func testFp32BundleStaysFp32() throws {
        let model = try roundTrip(.float32)
        for (key, value) in model.parameters().flattened() {
            XCTAssertEqual(value.dtype, .float32, key)
        }
    }

    func testFp16DecodeFillsEveryPosition() throws {
        let model = try roundTrip(.float16)
        let cfg = model.cfg
        let (condLen, t) = (16, 5)
        var cond = MLXRandom.randInt(0 ..< cfg.audioMaskId, [1, cfg.numAudioCodebook, condLen]).asType(.int32)
        cond[0..., 0..., (condLen - t)...] = MLXArray.full(
            [1, cfg.numAudioCodebook, t], values: MLXArray(Int32(cfg.audioMaskId)))
        let mask = MLXArray((0 ..< condLen).map { Int32($0 < 6 ? 0 : 1) }).reshaped([1, condLen])

        let tokens = model.generateTokens(condInputIds: cond, audioMask: mask, targetLen: t, numSteps: 4)
        let values = tokens.asArray(Int32.self)
        XCTAssertEqual(values.count, cfg.numAudioCodebook * t)
        XCTAssertFalse(values.contains(Int32(cfg.audioMaskId)), "a target position was left masked")
        XCTAssertTrue(values.allSatisfy { $0 >= 0 && $0 < Int32(cfg.audioMaskId) })
    }
}
