import Foundation
import MLX
import MLXNN
import MLXFast

// The OmniVoice backbone is a standard Qwen3 transformer driven by input
// embeddings and run **bidirectionally** (no causal mask) — it's a NAR
// diffusion model, so every position attends to every other. Checkpoint keys
// live under `llm.*`.

/// Holder for the precomputed RoPE so it isn't registered as a loadable param.
final class OVFreqs {
    let rope: MLXNN.RoPE
    init(_ cfg: OmniVoiceConfig) {
        rope = MLXNN.RoPE(dimensions: cfg.headDim, traditional: false, base: cfg.ropeTheta)
    }
}

/// Qwen3 attention: GQA (16 query / 8 KV heads), per-head RMSNorm on Q and K
/// before RoPE, full rotary, no attention bias. Bidirectional (no mask).
final class OVAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let freqs: OVFreqs

    init(_ cfg: OmniVoiceConfig, freqs: OVFreqs) {
        nHeads = cfg.numAttentionHeads
        nKVHeads = cfg.numKeyValueHeads
        headDim = cfg.headDim
        scale = pow(Float(cfg.headDim), -0.5)
        _qProj.wrappedValue = Linear(cfg.hiddenSize, nHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(cfg.hiddenSize, nKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(cfg.hiddenSize, nKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(nHeads * headDim, cfg.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: cfg.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: cfg.rmsNormEps)
        self.freqs = freqs
    }

    /// - segments: lengths of independent sequences packed along `L` (they sum
    ///   to `L`). Each one attends only to itself, with RoPE positions from 0, so
    ///   the result equals running each sequence alone. `nil` = one sequence.
    func callAsFunction(_ x: MLXArray, segments: [Int]? = nil) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        // [B, L, H, D] → per-head RMSNorm → [B, H, L, D]
        let q = qNorm(qProj(x).reshaped([b, l, nHeads, headDim])).transposed(0, 2, 1, 3)
        let k = kNorm(kProj(x).reshaped([b, l, nKVHeads, headDim])).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([b, l, nKVHeads, headDim]).transposed(0, 2, 1, 3)
        // Bidirectional: no mask. SDPA handles GQA (q heads > kv heads) directly.
        func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
            MLXFast.scaledDotProductAttention(
                queries: freqs.rope(q), keys: freqs.rope(k), values: v, scale: scale, mask: .none)
        }
        let out: MLXArray
        if let segments {
            var parts = [MLXArray]()
            var start = 0
            for n in segments {
                let r = start ..< start + n
                parts.append(attend(q[0..., 0..., r, 0...], k[0..., 0..., r, 0...], v[0..., 0..., r, 0...]))
                start += n
            }
            out = concatenated(parts, axis: 2)
        } else {
            out = attend(q, k, v)
        }
        return oProj(out.transposed(0, 2, 1, 3).reshaped([b, l, nHeads * headDim]))
    }
}

/// SwiGLU MLP.
final class OVMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ cfg: OmniVoiceConfig) {
        _gateProj.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(cfg.intermediateSize, cfg.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class OVBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: OVAttention
    @ModuleInfo(key: "mlp") var mlp: OVMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ cfg: OmniVoiceConfig, freqs: OVFreqs) {
        _selfAttn.wrappedValue = OVAttention(cfg, freqs: freqs)
        _mlp.wrappedValue = OVMLP(cfg)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, segments: [Int]? = nil) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), segments: segments)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// The Qwen3 backbone (`llm.*`): token embeddings + N bidirectional blocks +
/// final norm. Driven by precomputed input embeddings (text + audio fused).
final class OVBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [OVBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ cfg: OmniVoiceConfig) {
        let freqs = OVFreqs(cfg)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: cfg.textVocabSize, dimensions: cfg.hiddenSize)
        _layers.wrappedValue = (0 ..< cfg.numLayers).map { _ in OVBlock(cfg, freqs: freqs) }
        _norm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
    }

    /// `embeddings`: `[B, L, hidden]` → `[B, L, hidden]` (post final norm).
    /// `segments`: see `OVAttention`; `nil` = one sequence.
    func callAsFunction(_ embeddings: MLXArray, segments: [Int]? = nil) -> MLXArray {
        var h = embeddings
        for layer in layers { h = layer(h, segments: segments) }
        return norm(h)
    }
}
