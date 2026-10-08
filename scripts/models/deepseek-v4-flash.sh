# DeepSeek-V4-Flash architecture args.
#
# Shapes and attention/MoE flags follow Megatron's DSv4-Flash recipe
# (examples/moe_recipes/deepseek_v4_flash). RoPE scaling, the compress-rope
# base, and the per-layer compress pattern are recipe values — check them
# against the checkpoint config.json before training. Megatron requires
# --tensor-model-parallel-size 1 for dsv4_hybrid.

NLAYERS="${MODEL_ARGS_NUM_LAYERS:-43}"
CSA_COMPRESS_RATIOS="${MODEL_ARGS_CSA_COMPRESS_RATIOS:-([0,0,4]+[128,4]*20+[0])}"

MODEL_ARGS=(
    --disable-bias-linear
    --num-layers "$NLAYERS"
    --hidden-size 4096
    --ffn-hidden-size 2048
    --num-attention-heads 64
    --kv-channels 512
    --normalization RMSNorm
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 129280
    --make-vocab-size-divisible-by 3232
    --attention-dropout 0.0
    --hidden-dropout 0.0

    --multi-latent-attention
    --experimental-attention-variant dsv4_hybrid
    --q-lora-rank 1024
    --qk-pos-emb-head-dim 64
    --v-head-dim 512
    --o-groups 8
    --o-lora-rank 1024
    --qk-layernorm
    --rotary-base 10000
    --rotary-scaling-factor 4
    --original-max-position-embeddings 65536
    --mscale 1.0
    --mscale-all-dim 1.0
    --no-rope-fusion

    --csa-window-size 128
    --csa-compress-ratios "$CSA_COMPRESS_RATIOS"
    --csa-compress-rotary-base 40000
    --dsa-indexer-n-heads 64
    --dsa-indexer-head-dim 128
    --dsa-indexer-topk 512
    --dsa-indexer-loss-coeff 1e-2
    --dsa-indexer-use-sparse-loss

    --num-experts 256
    --moe-layer-freq 1
    --moe-n-hash-layers 3
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-topk 6
    --moe-router-score-function sqrtsoftplus
    --moe-router-topk-scaling-factor 1.5
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type seq_aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-aux-loss-coeff 0
    --moe-router-bias-update-rate 0
    --moe-grouped-gemm
    --moe-router-dtype fp32
    --moe-permute-fusion
    --activation-func-clamp-value 10.0

    --enable-hyper-connections
    --num-residual-streams 4
    --mhc-sinkhorn-iterations 20
    --use-fused-mhc

    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1

    # dsv4_hybrid attention is only implemented for TP=1.
    --tensor-model-parallel-size 1
    --expert-tensor-parallel-size 1
)
