// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! NumPy's default random generator, reproduced to the bit.
//!
//! `np.random.default_rng(seed)` is a PCG64 generator (O'Neill 2014, the
//! XSL-RR 128/64 variant) seeded through NumPy's `SeedSequence` hash. Where a
//! computation moved from NumPy into the core draws random subsets, drawing
//! them from this generator keeps its results those of the NumPy code.
//! Implemented: the raw stream, bounded integers by Lemire's (2019) method
//! with NumPy's 32-bit buffering, and `Generator.choice(n, k,
//! replace=False)` with both of its paths (Floyd's algorithm with a hash set,
//! and a tail shuffle).

pub mod dist;

const MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const POOL: usize = 4;

/// `np.random.SeedSequence(seed).generate_state(n_words, np.uint32)`.
pub fn seed_sequence_state(seed: u64, n_words: usize) -> Vec<u32> {
    let mut entropy = vec![seed as u32];
    if seed >> 32 != 0 {
        entropy.push((seed >> 32) as u32);
    }
    let mut hash_const = INIT_A;
    let mut hashmix = |v: u32| -> u32 {
        let mut v = v ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_A);
        v = v.wrapping_mul(hash_const);
        v ^ (v >> 16)
    };
    let mix = |x: u32, y: u32| -> u32 {
        let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        r ^ (r >> 16)
    };
    let mut pool = [0u32; POOL];
    for (i, p) in pool.iter_mut().enumerate() {
        *p = hashmix(entropy.get(i).copied().unwrap_or(0));
    }
    for i_src in 0..POOL {
        for i_dst in 0..POOL {
            if i_src != i_dst {
                let h = hashmix(pool[i_src]);
                pool[i_dst] = mix(pool[i_dst], h);
            }
        }
    }
    for &e in entropy.iter().skip(POOL) {
        for p in pool.iter_mut() {
            let h = hashmix(e);
            *p = mix(*p, h);
        }
    }
    let mut hash_const = INIT_B;
    (0..n_words)
        .map(|i| {
            let mut v = pool[i % POOL] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            v = v.wrapping_mul(hash_const);
            v ^ (v >> 16)
        })
        .collect()
}

/// `np.random.default_rng(seed)`: PCG64 with NumPy's seeding.
#[derive(Debug, Clone)]
pub struct Generator {
    state: u128,
    inc: u128,
    buffered: Option<u32>,
}

impl Generator {
    pub fn new(seed: u64) -> Self {
        let w = seed_sequence_state(seed, 8);
        let u: Vec<u64> = (0..4).map(|i| w[2 * i] as u64 | (w[2 * i + 1] as u64) << 32).collect();
        let initstate = (u[0] as u128) << 64 | u[1] as u128;
        let initseq = (u[2] as u128) << 64 | u[3] as u128;
        let mut g = Generator { state: 0, inc: initseq << 1 | 1, buffered: None };
        g.step();
        g.state = g.state.wrapping_add(initstate);
        g.step();
        g
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(MULT).wrapping_add(self.inc);
    }

    /// The next raw 64-bit output (`bit_generator.random_raw()`).
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let x = ((self.state >> 64) as u64) ^ (self.state as u64);
        x.rotate_right((self.state >> 122) as u32)
    }

    /// The next 32 bits: the halves of one 64-bit output, low half first.
    pub fn next_u32(&mut self) -> u32 {
        if let Some(v) = self.buffered.take() {
            return v;
        }
        let v = self.next_u64();
        self.buffered = Some((v >> 32) as u32);
        v as u32
    }

    /// A uniform double in `[0, 1)` (`Generator.random()`).
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// A uniform integer in `[0, max]` (NumPy's `random_bounded_uint64`,
    /// unmasked).
    pub fn bounded(&mut self, max: u64) -> u64 {
        if max == 0 {
            0
        } else if max <= u32::MAX as u64 {
            if max == u32::MAX as u64 {
                return self.next_u32() as u64;
            }
            let excl = max as u32 + 1;
            let mut m = self.next_u32() as u64 * excl as u64;
            let mut left = m as u32;
            if left < excl {
                let threshold = (u32::MAX - max as u32) % excl;
                while left < threshold {
                    m = self.next_u32() as u64 * excl as u64;
                    left = m as u32;
                }
            }
            m >> 32
        } else {
            if max == u64::MAX {
                return self.next_u64();
            }
            let excl = max + 1;
            let mut m = self.next_u64() as u128 * excl as u128;
            let mut left = m as u64;
            if left < excl {
                let threshold = (u64::MAX - max) % excl;
                while left < threshold {
                    m = self.next_u64() as u128 * excl as u128;
                    left = m as u64;
                }
            }
            (m >> 64) as u64
        }
    }

    /// Fisher-Yates over `data[first..]`, from the end (NumPy's `_shuffle_int`).
    fn shuffle_tail(&mut self, data: &mut [usize], first: usize) {
        for i in (first..data.len()).rev() {
            let j = self.bounded(i as u64) as usize;
            data.swap(i, j);
        }
    }

    /// `Generator.choice(n, k, replace=False)`: `k` distinct indices below
    /// `n`, in NumPy's order. Panics if `k > n`.
    pub fn choice(&mut self, n: usize, k: usize) -> Vec<usize> {
        assert!(k <= n, "cannot take a larger sample than the population");
        if n > 10000 && k > n / 50 {
            let mut idx: Vec<usize> = (0..n).collect();
            self.shuffle_tail(&mut idx, (n - k).max(1));
            return idx[n - k..].to_vec();
        }
        let mut idx = vec![0usize; k];
        let set_size = (1.2 * k as f64) as u64;
        let mut mask = set_size;
        for s in [1, 2, 4, 8, 16, 32] {
            mask |= mask >> s;
        }
        let mut set = vec![u64::MAX; mask as usize + 1];
        for j in n - k..n {
            let val = self.bounded(j as u64);
            let mut loc = val & mask;
            while set[loc as usize] != u64::MAX && set[loc as usize] != val {
                loc = (loc + 1) & mask;
            }
            if set[loc as usize] == u64::MAX {
                set[loc as usize] = val;
                idx[j + k - n] = val as usize;
            } else {
                loc = j as u64 & mask;
                while set[loc as usize] != u64::MAX {
                    loc = (loc + 1) & mask;
                }
                set[loc as usize] = j as u64;
                idx[j + k - n] = j;
            }
        }
        self.shuffle_tail(&mut idx, 1);
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_stream_is_numpys() {
        // np.random.default_rng(0).bit_generator.random_raw(3)
        let mut g = Generator::new(0);
        let raw: Vec<u64> = (0..3).map(|_| g.next_u64()).collect();
        assert_eq!(raw, vec![11749869230777074271, 4976686463289251617, 755828109848996024]);
        assert_eq!(seed_sequence_state(0, 2), vec![2968811710, 3677149159]);
    }

    #[test]
    fn choice_is_numpys() {
        // np.random.default_rng(1).choice(100, 5, replace=False)
        assert_eq!(Generator::new(1).choice(100, 5), vec![49, 3, 74, 94, 45]);
        // np.random.default_rng(2).choice(20000, 4, replace=False)
        let a = Generator::new(2).choice(20000, 4);
        assert_eq!(a, vec![5969, 16748, 2185, 5231]);
        // np.random.default_rng(3).choice(12000, 1000, replace=False)[:3] (tail shuffle)
        let b = Generator::new(3).choice(12000, 1000);
        assert_eq!(&b[..3], &[8288, 7694, 7361]);
        let mut s = b.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 1000);
    }

    #[test]
    fn random_is_numpys() {
        // np.random.default_rng(4).random()
        assert_eq!(Generator::new(4).random(), 0.9430561055723676);
    }
}
