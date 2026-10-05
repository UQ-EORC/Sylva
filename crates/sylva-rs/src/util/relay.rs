// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Messages from long-running work, and work spread over threads in order.
//!
//! Long computations report what they are doing through a [`Log`]: a
//! function taking one line of text, callable from any thread. Where the
//! lines go is the caller's business. A binding that must hand them to its
//! interpreter on the thread that called it (R allows nothing else, and
//! Python needs its lock) runs the work through [`relay`]: the work goes to
//! a scoped thread, and each line comes back over a channel to the calling
//! thread, which delivers it while the work carries on.
//!
//! [`ordered_map`] runs a function over a list on a given number of
//! threads, handing out items in order as a thread pool with a queue would,
//! and passes each result on in list order as soon as it and every result
//! before it are done. The results never depend on the number of threads.
//! Those threads are plain scoped threads, not rayon workers: the work they
//! run is parallel inside (on rayon's pool) and may wait on work shared
//! between items (an ICP target's pyramid, built once by whichever item
//! needs it first), and a rayon worker that waits on a lock can steal, and
//! block on, the very job it is waiting for.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Mutex};

/// Where progress messages go: one line of text at a time, from any thread.
pub type Log<'a> = &'a (dyn Fn(&str) + Sync);

/// A [`Log`] that drops everything.
pub fn quiet(_: &str) {}

/// Run `work` on a scoped thread and deliver its messages on this one.
///
/// `work` is given a [`Log`] that forwards each line over a channel; this
/// thread calls `deliver` with every line in the order they were logged,
/// until `work` returns, and then returns its result.
pub fn relay<T: Send>(work: impl FnOnce(Log) -> T + Send, mut deliver: impl FnMut(&str)) -> T {
    let (tx, rx) = mpsc::channel::<String>();
    std::thread::scope(|s| {
        let handle = s.spawn(move || {
            let tx = Mutex::new(tx);
            let log = move |msg: &str| {
                if let Ok(tx) = tx.lock() {
                    let _ = tx.send(msg.to_string());
                }
            };
            work(&log)
        });
        for msg in rx {
            deliver(&msg);
        }
        match handle.join() {
            Ok(v) => v,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    })
}

/// `f` over `0..n` on `workers` threads, results in index order.
///
/// Items are handed out in order, one at a time, to whichever of the
/// `workers` threads is free (at most `workers` items are in progress at
/// once, which bounds the memory they take). `done` is called with each index and its result in index order, as
/// soon as that result and all before it exist (so a log of results reads
/// the same however many threads ran). With `workers <= 1` everything runs
/// on the calling thread.
pub fn ordered_map<T: Send>(n: usize, workers: usize, f: impl Fn(usize) -> T + Sync, done: impl Fn(usize, &T) + Sync) -> Vec<T> {
    if workers <= 1 || n <= 1 {
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let v = f(k);
            done(k, &v);
            out.push(v);
        }
        return out;
    }
    // One slot per item, each empty until its result lands in it. `Option` is
    // "a value or nothing", so `None` here means "not computed yet".
    let slots: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
    // The queue, in one number: a thread claims the next item by adding one to
    // this atomically, so no two threads can claim the same index.
    let next = AtomicUsize::new(0);
    // Index of the first result not yet passed to `done`, under the lock that
    // serialises the calls to `done`.
    let emitted = Mutex::new(0usize);
    // What each worker runs: claim an item, compute it, then pass on whatever
    // prefix of the results is now complete.
    let run = || loop {
        let k = next.fetch_add(1, Ordering::SeqCst);
        if k >= n {
            break;
        }
        let v = f(k);
        *slots[k].lock().expect("slot") = Some(v);
        // Holding `emitted` is what keeps `done` single-file and in order: the
        // thread that finished item 3 may find 0, 1 and 2 already waiting and
        // deliver all four, while a thread that finished item 9 early delivers
        // nothing and goes back for more work.
        let mut first = emitted.lock().expect("emitter");
        while *first < n {
            let slot = slots[*first].lock().expect("slot");
            match slot.as_ref() {
                Some(v) => done(*first, v),
                None => break,
            }
            // Release this slot before moving on, rather than waiting for the
            // end of the loop body.
            drop(slot);
            *first += 1;
        }
    };
    // Scoped threads may borrow what is around them (`slots`, `f`, `done`)
    // because the scope cannot end until they have all finished, which is what
    // lets this function hand out references rather than copies.
    std::thread::scope(|s| {
        for _ in 0..workers.min(n) {
            s.spawn(run);
        }
    });
    // The threads are done, so the locks can be dropped and the values taken
    // out: `into_inner` consumes each `Mutex`, and every slot is filled because
    // every index was claimed exactly once.
    slots.into_iter().map(|m| m.into_inner().expect("slot").expect("every item ran")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_whatever_the_threads() {
        for workers in [1, 2, 5] {
            let seen = Mutex::new(Vec::new());
            let out = ordered_map(20, workers, |k| {
                std::thread::sleep(std::time::Duration::from_millis(((k * 7) % 5) as u64));
                k * k
            }, |k, v| seen.lock().unwrap().push((k, *v)));
            assert_eq!(out, (0..20).map(|k| k * k).collect::<Vec<_>>());
            assert_eq!(seen.into_inner().unwrap(), (0..20).map(|k| (k, k * k)).collect::<Vec<_>>());
        }
    }

    #[test]
    fn relayed_messages_arrive_in_order_on_the_caller() {
        let caller = std::thread::current().id();
        let mut got = Vec::new();
        let r = relay(|log| {
            for k in 0..5 {
                log(&format!("line {k}"));
            }
            42
        }, |m| {
            assert_eq!(std::thread::current().id(), caller);
            got.push(m.to_string());
        });
        assert_eq!(r, 42);
        assert_eq!(got, (0..5).map(|k| format!("line {k}")).collect::<Vec<_>>());
    }
}
