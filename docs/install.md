# Install

Requires Python ≥ 3.10 and a Rust toolchain (for source builds).

```bash
pip install maturin
maturin develop --release        # builds the extension into the active environment
```

Reading RIEGL `.rxp` files needs RiVLib's `libscanifc` (proprietary; download
from RIEGL). Point `RIVLIB_PATH` at the extracted directory or pass
`library=` to `sylva.io.read_rxp`.
