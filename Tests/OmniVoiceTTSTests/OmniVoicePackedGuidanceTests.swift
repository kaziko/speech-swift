import Foundation
import MLX
import MLXNN
import XCTest

@testable import OmniVoiceTTS

/// The packed guidance forward (cond + uncond in one backbone pass, each
/// attending only to itself) must match the two separate forwards it replaces.
/// Tiny random model: no weights, no download.
final class OmniVoicePackedGuidanceTests: XCTestCase {
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

    /// Cond `[1, C, condLen]` with text at the first `textLen` positions, the
    /// target tokens `[1, C, T]` (some masked), and the audio mask.
    private func inputs(_ cfg: OmniVoiceConfig, condLen: Int, textLen: Int, t: Int)
        -> (cond: MLXArray, mask: MLXArray, tokens: MLXArray)
    {
        let c = cfg.numAudioCodebook
        let tokens = MLXRandom.randInt(0 ..< cfg.audioVocabSize, [1, c, t]).asType(.int32)
        var cond = MLXRandom.randInt(0 ..< cfg.audioMaskId, [1, c, condLen]).asType(.int32)
        cond[0..., 0..., (condLen - t)...] = tokens
        let mask = MLXArray((0 ..< condLen).map { Int32($0 < textLen ? 0 : 1) }).reshaped([1, condLen])
        return (cond, mask, tokens)
    }

    func testPackedLogitsMatchSeparateForwards() {
        MLXRandom.seed(7)
        let cfg = tinyConfig()
        let model = OmniVoiceModel(cfg)
        eval(model.parameters())
        let (condLen, t) = (24, 7)
        let (cond, mask, tokens) = inputs(cfg, condLen: condLen, textLen: 9, t: t)

        let condRef = model(inputIds: cond, audioMask: mask)[0..., 0..., (condLen - t) ..< condLen, 0...]
        let uncondRef = model(inputIds: tokens, audioMask: MLXArray.ones([1, t]).asType(.int32))
        let (condGot, uncondGot) = model.guidanceLogits(condInputIds: cond, audioMask: mask, tokens: tokens)

        XCTAssertEqual(condGot.shape, condRef.shape)
        XCTAssertEqual(uncondGot.shape, uncondRef.shape)
        XCTAssertTrue(allClose(condGot, condRef, rtol: 1e-4, atol: 1e-5).item(Bool.self), "cond logits differ")
        XCTAssertTrue(allClose(uncondGot, uncondRef, rtol: 1e-4, atol: 1e-5).item(Bool.self), "uncond logits differ")
    }

    /// Segments must not see each other: changing the uncond tokens leaves the
    /// cond logits of the packed pass unchanged.
    func testSegmentsDoNotAttendAcross() {
        MLXRandom.seed(11)
        let cfg = tinyConfig()
        let model = OmniVoiceModel(cfg)
        eval(model.parameters())
        let (cond, mask, tokens) = inputs(cfg, condLen: 20, textLen: 6, t: 5)
        let other = (tokens + 1) % Int32(cfg.audioVocabSize)

        let a = model.guidanceLogits(condInputIds: cond, audioMask: mask, tokens: tokens).cond
        let b = model.guidanceLogits(condInputIds: cond, audioMask: mask, tokens: other).cond
        XCTAssertTrue(allClose(a, b, rtol: 0, atol: 0).item(Bool.self), "uncond tokens leaked into cond")
    }

    func testPackedDecodeMatchesUnpacked() {
        MLXRandom.seed(3)
        let cfg = tinyConfig()
        let model = OmniVoiceModel(cfg)
        eval(model.parameters())
        let t = 6
        var (cond, mask, _) = inputs(cfg, condLen: 18, textLen: 7, t: t)
        cond[0..., 0..., (18 - t)...] = MLXArray.full([1, cfg.numAudioCodebook, t], values: MLXArray(Int32(cfg.audioMaskId)))

        let ref = model.generateTokens(condInputIds: cond, audioMask: mask, targetLen: t, numSteps: 4, packed: false)
        let got = model.generateTokens(condInputIds: cond, audioMask: mask, targetLen: t, numSteps: 4, packed: true)
        XCTAssertEqual(got.asArray(Int32.self), ref.asArray(Int32.self))
    }
}
