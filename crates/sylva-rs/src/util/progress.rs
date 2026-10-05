// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Progress reporting for the long-running work.
//!
//! A function that takes a while opens a [`Task`] with a label and how many
//! steps it expects, then counts up as it goes. Counting is a single atomic
//! add, so a worker thread can do it inside a parallel loop; nothing is
//! printed and nothing is locked on that path. A caller that wants to show
//! progress reads [`state`] from another thread as often as it likes, which
//! is how the Python bindings drive a progress bar without ever touching the
//! threads doing the work.
//!
//! Tasks nest: a stage that runs inside another appears after it in `state`,
//! innermost last. Nothing is reported unless someone asks, and a task that
//! is dropped without finishing simply disappears.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// One running stage: what it is, how far it has got, and how far it goes.
#[derive(Debug)]
pub struct Frame {
    pub label: String,
    pub done: AtomicU64,
    pub total: AtomicU64,
}

fn stack() -> &'static Mutex<Vec<Arc<Frame>>> {
    static STACK: OnceLock<Mutex<Vec<Arc<Frame>>>> = OnceLock::new();
    STACK.get_or_init(|| Mutex::new(Vec::new()))
}

/// A stage in progress; it leaves the list when dropped.
#[derive(Debug)]
pub struct Task {
    frame: Arc<Frame>,
}

/// Start a stage of `total` steps (0 if the count is not known yet).
pub fn start(label: impl Into<String>, total: u64) -> Task {
    let frame = Arc::new(Frame { label: label.into(), done: AtomicU64::new(0), total: AtomicU64::new(total) });
    if let Ok(mut s) = stack().lock() {
        s.push(Arc::clone(&frame));
    }
    Task { frame }
}

impl Task {
    /// Count `n` more steps done.
    pub fn inc(&self, n: u64) {
        self.frame.done.fetch_add(n, Ordering::Relaxed);
    }

    /// Set how many steps are done.
    pub fn set(&self, done: u64) {
        self.frame.done.store(done, Ordering::Relaxed);
    }

    /// Set how many steps there are, once that is known.
    pub fn set_total(&self, total: u64) {
        self.frame.total.store(total, Ordering::Relaxed);
    }
}

impl Drop for Task {
    fn drop(&mut self) {
        if let Ok(mut s) = stack().lock() {
            s.retain(|f| !Arc::ptr_eq(f, &self.frame));
        }
    }
}

/// The stages running now, outermost first, as `(label, done, total)`.
pub fn state() -> Vec<(String, u64, u64)> {
    match stack().lock() {
        Ok(s) => s
            .iter()
            .map(|f| (f.label.clone(), f.done.load(Ordering::Relaxed), f.total.load(Ordering::Relaxed)))
            .collect(),
        Err(_) => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only this test's own frames: the registry is global, and the other
    /// tests of the crate run at the same time.
    fn mine(prefix: &str) -> Vec<(String, u64, u64)> {
        state().into_iter().filter(|(l, _, _)| l.starts_with(prefix)).collect()
    }

    #[test]
    fn tasks_nest_and_leave_when_dropped() {
        let outer = start("test-nest outer", 10);
        outer.inc(3);
        {
            let inner = start("test-nest inner", 4);
            inner.inc(1);
            inner.inc(1);
            let s = mine("test-nest");
            assert_eq!(s.len(), 2);
            assert_eq!(s[0], ("test-nest outer".to_string(), 3, 10));
            assert_eq!(s[1], ("test-nest inner".to_string(), 2, 4));
        }
        let s = mine("test-nest");
        assert_eq!(s.len(), 1, "the inner task left when it was dropped");
        outer.set(10);
        outer.set_total(12);
        assert_eq!(mine("test-nest")[0], ("test-nest outer".to_string(), 10, 12));
        drop(outer);
        assert!(mine("test-nest").is_empty());
    }

    #[test]
    fn counting_works_from_several_threads() {
        let t = start("test-threads", 400);
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| (0..100).for_each(|_| t.inc(1)));
            }
        });
        assert_eq!(mine("test-threads")[0].1, 400);
    }
}
