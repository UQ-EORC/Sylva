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
//!
//! The same counting calls are where long work can be stopped. A caller that
//! wants the work to end (a binding whose user pressed Ctrl-C) calls
//! [`cancel`]; the next [`Task::inc`] or [`Task::set`] on any thread then
//! unwinds out of the computation with a [`Cancelled`] payload, which the
//! caller catches (`std::panic::catch_unwind`) before calling
//! [`clear_cancel`]. Nothing is unwound unless someone asked, so a program
//! that never calls [`cancel`] is unaffected. Work is only as interruptible
//! as it counts: a stage that reports three steps can stop at three places.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// One running stage: what it is, how far it has got, and how far it goes.
#[derive(Debug)]
pub struct Frame {
    /// What the stage is doing, for display.
    pub label: String,
    /// Steps completed so far (count).
    pub done: AtomicU64,
    /// Steps expected in all (count); 0 while the total is not known.
    pub total: AtomicU64,
}

/// Set by [`cancel`], read by every checked count.
static CANCEL: AtomicBool = AtomicBool::new(false);

/// What a cancelled computation unwinds with: the payload of the unwinding
/// started by [`check`], for the caller to recognise after catching it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

/// Ask the running work to stop at its next checked count.
///
/// The flag is process-wide and stays set until [`clear_cancel`]: whoever
/// sets it must catch the unwinding and clear it.
pub fn cancel() {
    CANCEL.store(true, Ordering::SeqCst);
}

/// Whether [`cancel`] has been called and not yet cleared.
pub fn cancelled() -> bool {
    CANCEL.load(Ordering::Relaxed)
}

/// Withdraw a [`cancel`], once the work it stopped has unwound.
pub fn clear_cancel() {
    CANCEL.store(false, Ordering::SeqCst);
}

/// Unwind with [`Cancelled`] if [`cancel`] was called; otherwise do nothing.
///
/// The unwinding does not go through the panic hook, so nothing is printed.
pub fn check() {
    if cancelled() {
        std::panic::resume_unwind(Box::new(Cancelled));
    }
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
    /// Count `n` more steps done; unwinds with [`Cancelled`] after a [`cancel`].
    pub fn inc(&self, n: u64) {
        check();
        self.inc_unchecked(n);
    }

    /// Set how many steps are done; unwinds with [`Cancelled`] after a [`cancel`].
    pub fn set(&self, done: u64) {
        check();
        self.set_unchecked(done);
    }

    /// [`Task::inc`] without the cancellation check, for a caller that
    /// cannot be unwound through (a stage opened by an interpreter).
    pub fn inc_unchecked(&self, n: u64) {
        self.frame.done.fetch_add(n, Ordering::Relaxed);
    }

    /// [`Task::set`] without the cancellation check.
    pub fn set_unchecked(&self, done: u64) {
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

    /// The flag itself is not set here: it is process-wide, and the other
    /// tests of the crate count on their own tasks at the same time.
    #[test]
    fn an_uncancelled_check_does_nothing_and_the_payload_is_recognisable() {
        assert!(!cancelled());
        check();
        let caught = std::panic::catch_unwind(|| std::panic::resume_unwind(Box::new(Cancelled))).unwrap_err();
        assert!(caught.is::<Cancelled>());
        let t = start("test-unchecked", 5);
        t.inc_unchecked(2);
        t.set_unchecked(4);
        assert_eq!(mine("test-unchecked")[0].1, 4);
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
