"""CPU checks of the FP8 per-channel quantizer. Run: python3 convert/tests/test_fp8.py"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fp8

def check(name, w):
    q, s = fp8.quantize_rows(w)
    g, s2 = fp8.quantize_rows_grid(w)
    assert torch.equal(s, s2)
    qb = fp8.codes_as_bytes(q)
    # +0 and -0 are different bytes with the same value; the cast keeps the sign of a negative zero
    same = torch.equal(qb, g)
    if not same:
        bad = (qb != g).nonzero()
        i, j = bad[0].tolist()
        raise AssertionError(f"{name}: {bad.shape[0]} codes differ, first at {i},{j}: cast {int(qb[i,j]):#x} grid {int(g[i,j]):#x} "
                             f"x={float(w[i,j].float()/s[i,0])!r}")
    fp8.check_codes(qb)
    d = fp8.dequantize(g, s)
    assert torch.equal(d, fp8.dequantize(q, s))
    err = fp8.rel_error(w, q, s)
    print(f"ok {name}: shape {tuple(w.shape)} rel err {err:.4f} max code {int((qb & 0x7F).max()):#x}")
    return err

torch.manual_seed(0)
t = fp8.e4m3_table()
assert t.numel() == 120 and float(t[1]) == 2.0**-9 and float(t[0x77]) == 240.0 and float(t[8]) == 2.0**-6
# every table value must round-trip through torch's dtype
codes = torch.arange(0x78, dtype=torch.uint8)
assert torch.equal(codes.view(fp8.F8).to(torch.float64), t), "table differs from torch's float8_e4m3fn decoding"

# 1. gaussian weights like an expert projection, bf16 source
check("gauss bf16", (torch.randn(64, 6144) * 0.02).to(torch.bfloat16))
check("gauss wide", (torch.randn(96, 2048) * 0.05).to(torch.bfloat16))
# 2. heavy tails / outlier rows
w = torch.randn(32, 4096) * 0.01; w[3, 7] = 5.0; w[9] *= 1e-6
check("outliers", w.to(torch.bfloat16))
# 3. exact ties: build rows whose scaled values hit every midpoint exactly (amax = 240 -> scale = 1)
mid = ((t[1:] + t[:-1]) / 2).to(torch.float32)
row = torch.cat([mid, -mid, t.to(torch.float32), -t.to(torch.float32), torch.tensor([240.0])])
assert float(row.abs().max()) == 240.0
check("ties", row.repeat(4, 1))
# 4. zero row and negative zero: scale 1.0 for the zero row, every zero is the byte 0x00
w = torch.zeros(3, 16); w[1, 0] = 1.0; w[2, 5] = -0.0; w[1, 3] = -1e-30
q, s = fp8.quantize_rows(w)
assert float(s[0, 0]) == 1.0 and torch.equal(fp8.dequantize(q, s), w.abs() * (w > 0))
assert int((fp8.codes_as_bytes(q) == 0x80).sum()) == 0
# 4b. subnormal range: values below 2^-6 after scaling go to 0 or to 2^-6, never to a subnormal code
row = torch.tensor([240.0] + [k / 8 * 2.0**-6 for k in range(0, 9)] + [-k / 8 * 2.0**-6 for k in range(1, 9)])
q, s = fp8.quantize_rows(row.repeat(2, 1)); b = fp8.codes_as_bytes(q)[0]
assert b.tolist() == [0x77] + [0, 0, 0, 0, 8, 8, 8, 8, 8] + [0, 0, 0, 0x88, 0x88, 0x88, 0x88, 0x88], b.tolist()
g, _ = fp8.quantize_rows_grid(row.repeat(2, 1)); assert torch.equal(g[0], b)
# 5. scale is exactly amax / 240 in fp32
w = (torch.randn(8, 512) * 0.3).to(torch.bfloat16)
_, s = fp8.quantize_rows(w)
assert torch.equal(s[:, 0], w.float().abs().amax(1) / torch.tensor(240.0))
# 6. error level on gaussian data (the cookbook reports 2.65% on real experts)
e = check("gauss big", (torch.randn(256, 6144) * 0.02).to(torch.bfloat16))
assert 0.01 < e < 0.05, e
print("ALL OK, torch", torch.__version__)
