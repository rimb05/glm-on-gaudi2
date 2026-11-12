# SPDX-License-Identifier: Apache-2.0
"""FP8 (E4M3) per-output-channel weight quantization for Gaudi 2.

One fp32 scale per output channel (row of a [out, in] weight): scale = amax / 240. 240 is the largest finite value
of the E4M3 format Gaudi 2's matrix engine uses (exponent field 15 is not used for finite numbers there), while the
`float8_e4m3fn` dtype used for storage goes up to 448. With amax / 240 no code ever needs the top exponent, so the
stored bytes mean the same thing in both conventions.

Definition used by the converter AND the validator (so they must agree bit for bit):

    scale[c] = float32(max_j |w[c, j]|) / float32(240)          (1.0 for an all-zero row)
    x[c, j]  = float32(w[c, j]) / scale[c]                      (IEEE float32 division)
    code     = nearest E4M3 value to x, ties to the even code   (|x| <= 240 by construction)
    then two byte-level restrictions, both conservative for Gaudi 2 (the gaudi-cookbook's FP8 codecs apply the
    same ones because the hardware's handling of these codes is not documented):
      - no subnormal codes: a code whose magnitude is 1..7 (values below 2^-6) becomes 0 if it is below 4,
        else 8 (2^-6, the smallest normal). These are weights below amax / 15360 of their row.
      - no negative zero: 0x80 is written as 0x00.

`quantize_rows` uses torch's float8_e4m3fn cast; `quantize_rows_grid` is an independent implementation that
searches the explicit table of representable values. tests/test_fp8.py checks that they agree, and validate.py
re-derives every stored code with the grid version.
"""

from __future__ import annotations

import torch

GAUDI2_FP8_MAX = 240.0
F8 = torch.float8_e4m3fn


def e4m3_table() -> torch.Tensor:
    """The 120 non-negative E4M3 values with exponent field 0..14, ascending, indexed by code (0x00..0x77)."""
    vals = []
    for code in range(0x78):
        e, m = code >> 3, code & 7
        vals.append((m / 8.0) * 2.0**-6 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7))
    t = torch.tensor(vals, dtype=torch.float64)
    assert bool((t[1:] > t[:-1]).all()) and float(t[-1]) == GAUDI2_FP8_MAX
    return t


_TABLE = e4m3_table()
# midpoints between neighbouring codes; a value exactly on midpoint i (between code i and i + 1) goes to the even one
_MID = (_TABLE[1:] + _TABLE[:-1]) / 2.0


def restrict_codes(b: torch.Tensor) -> torch.Tensor:
    """uint8 E4M3 code bytes -> the same codes without subnormals and without negative zero (see module doc)."""
    mag = b & 0x7F
    sub = (mag >= 1) & (mag <= 7)
    mag = torch.where(sub, torch.where(mag >= 4, torch.full_like(mag, 8), torch.zeros_like(mag)), mag)
    out = torch.where(mag == 0, mag, mag | (b & 0x80))
    return out


def check_codes(b: torch.Tensor) -> None:
    """Raise if a code byte is one this converter must never write."""
    mag = b & 0x7F
    if bool((b == 0x80).any()):
        raise ValueError("negative zero (0x80) present")
    if bool(((mag >= 1) & (mag <= 7)).any()):
        raise ValueError("subnormal code present")
    if bool((mag > 0x77).any()):
        raise ValueError("a code uses exponent field 15 (above 240)")


def row_scales(w: torch.Tensor) -> torch.Tensor:
    """[out, in] weight (any float dtype) -> fp32 scale per row, shape [out, 1]."""
    amax = w.to(torch.float32).abs().amax(dim=1, keepdim=True)
    scale = amax / torch.tensor(GAUDI2_FP8_MAX, dtype=torch.float32)
    return torch.where(amax > 0, scale, torch.ones_like(scale))


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[out, in] -> (codes as float8_e4m3fn [out, in], fp32 scales [out, 1]). Uses torch's cast."""
    scale = row_scales(w)
    x = w.to(torch.float32) / scale
    b = restrict_codes(x.to(F8).view(torch.uint8))
    return b.view(F8), scale


def quantize_rows_grid(w: torch.Tensor, scale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent reference: [out, in] -> (code bytes as uint8 [out, in], fp32 scales [out, 1])."""
    if scale is None:
        scale = row_scales(w)
    x = (w.to(torch.float32) / scale.to(torch.float32)).to(torch.float64)
    a = x.abs()
    # float32 rounding of amax / 240 and of w / scale can leave the row maximum one ulp above 240. Anything below
    # 248 (the midpoint to 256, the next e4m3fn value) still rounds to 240, the top code used here.
    if float(a.max()) >= 248.0:
        raise ValueError(f"|w / scale| reaches {float(a.max())}: it would not round to {GAUDI2_FP8_MAX}")
    # number of midpoints strictly below a = index of the nearest code, except on an exact tie
    idx = torch.bucketize(a, _MID, right=False)  # midpoints < a ... first index with _MID[i] >= a
    on_tie = (idx < _MID.numel()) & (a == _MID[idx.clamp(max=_MID.numel() - 1)])
    # on a tie between code idx and idx + 1 pick the even code
    idx = torch.where(on_tie & (idx % 2 == 1), idx + 1, idx)
    code = idx.to(torch.uint8)
    neg = torch.signbit(x)
    code = torch.where(neg, code | 0x80, code)
    return restrict_codes(code), scale


def codes_as_bytes(q: torch.Tensor) -> torch.Tensor:
    """float8_e4m3fn tensor -> its raw bytes as uint8 (same shape)."""
    return q.contiguous().view(torch.uint8)


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """codes (float8_e4m3fn or raw uint8) and [out, 1] scales -> fp32 weight."""
    if q.dtype == torch.uint8:
        mag = _TABLE.to(torch.float32)[(q & 0x7F).to(torch.long)]
        v = torch.where((q & 0x80) != 0, -mag, mag)
    else:
        v = q.to(torch.float32)
    return v * scale.to(torch.float32)


def rel_error(w: torch.Tensor, q: torch.Tensor, scale: torch.Tensor) -> float:
    """Frobenius relative error of the quantized weight against the source."""
    w32 = w.to(torch.float32)
    return float((dequantize(q, scale) - w32).norm() / w32.norm().clamp_min(1e-30))
