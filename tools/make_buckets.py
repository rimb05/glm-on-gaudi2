#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the VLLM_BUCKETING_FROM_FILE bucket set for the two-server GLM-5.3 deployment.

  make_buckets.py OUT [--max-blocks 5632] [--chunk 4096]

Every distinct (query, context) prompt shape and every decode shape is compiled the first time it is used (seconds
each, cached on disk afterwards), and a padded shape costs compute in proportion to its size. The set balances the
two:
  * short prompts (context up to 8K tokens): the fine query sizes of the stock exponential strategy, so a 2K-token
    prompt is not padded to 4K;
  * longer contexts: five query sizes only (128, 512, 2048, the chunk size, 8192). The bulk of a long prompt is
    full chunks; only its last piece and the few new tokens of a cached follow-up turn use the small sizes;
  * context sizes step by 64 blocks up to 1024 blocks (131K tokens) and by 128 above: a prefill pads its context by
    at most 16K tokens;
  * decode sizes (one row per sequence; speculative decoding doubles rows and blocks, the loader derives those)
    step by 64 blocks up to 512, 128 up to 2048 and 256 above: the sparse decode scan costs in proportion.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--max-blocks", type=int, default=5632)
    ap.add_argument("--chunk", type=int, default=4096)
    a = ap.parse_args()
    mb = a.max_blocks
    q_fine = [128, 256, 384, 512, 768, 1024, 1536, 2048, 2944, 4096, 5888, 8192]
    ctx_short = [0, 1, 2, 4, 8, 16, 32, 64]
    q_long = sorted({128, 512, 2048, a.chunk, 8192})
    ctx_long = list(range(128, 1025, 64)) + list(range(1152, mb + 1, 128))
    if ctx_long[-1] != mb:
        ctx_long.append(mb)
    dec1 = [1, 4, 8, 16, 24, 40, 64, 96, 128, 192, 256, 320, 384, 448, 512]
    dec1 += list(range(640, 2049, 128)) + list(range(2304, mb + 1, 256))
    if dec1[-1] != mb:
        dec1.append(mb)
    decn = [4, 8, 16, 24, 40, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, mb]
    lines = [
        f"# GLM-5.3 two-server bucket set (tools/make_buckets.py --max-blocks {mb} --chunk {a.chunk}): --block-size 128,",
        f"# --max-num-batched-tokens 8192, --long-prefill-token-threshold {a.chunk}, --max-num-seqs 4.",
        "# Prompt buckets (bs, query, ctx_blocks): fine query sizes for short contexts",
        f"(1, {q_fine}, {ctx_short})",
        "# ... and five query sizes for long contexts",
        f"(1, {q_long}, {ctx_long})",
        "# Decode buckets (bs, 1, num_blocks); the loader adds the two-row (MTP) shapes",
        f"(1, 1, {dec1})",
        f"(2, 1, {decn})",
        f"(3, 1, {decn})",
        f"(4, 1, {decn})",
    ]
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{a.out}: {len(q_fine) * len(ctx_short) + len(q_long) * len(ctx_long)} prompt buckets, "
          f"{len(dec1) + 3 * len(decn)} decode buckets")


if __name__ == "__main__":
    main()
