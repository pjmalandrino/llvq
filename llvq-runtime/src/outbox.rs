//! A request's events on their way to its client, and what a failed engine
//! tells its queue.
//!
//! No engine thread waits on a client inside `blocking_send`. [`Outbox`] keeps
//! the events a full channel refused, in order, and gives up on a client that
//! leaves its channel full for its whole send timeout (`LLVQ_SEND_TIMEOUT`).
//! [`refuse_queue`] and [`fail_after_panic`] answer the queue of an engine
//! that failed, without waiting on a client either. Both engines use this
//! file: the Qwen3 one ([`crate::engine`]) and the Inkling one
//! ([`crate::inkling`]).

use crate::engine::{EngineInfo, Event, Job, State};
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// How often a send that found its client's channel full looks again.
pub(crate) const POLL: Duration = Duration::from_millis(1);

/// What [`Outbox::flush`] left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sent {
    /// Every event is in the channel.
    Clear,
    /// The channel is full: the client has not read for a while, less than
    /// its send timeout.
    Blocked,
    /// The client closed the channel, or left it full past the send timeout.
    /// Final: nothing more is sent.
    Gone,
}

/// A request's events on their way to its client, in order.
///
/// The engine thread never waits on a client inside `blocking_send`. An event
/// goes to the back of `queue`, and a flush moves the queue's front into the
/// channel until the queue is empty or the channel full; an event leaves the
/// queue only once `try_send` took it, and only from the front. So a client
/// that reads late gets every event, in the order the engine made them,
/// whatever the delay. A client that leaves the channel full for its whole
/// send timeout (`Job::patience`, counted from the last event it made room
/// for) is gone, exactly as one that closed it: the request stops and the
/// cache keeps what a departed client leaves.
pub(crate) struct Outbox {
    tx: tokio::sync::mpsc::Sender<Event>,
    queue: VecDeque<Event>,
    /// Since when the channel has been full with nothing taken.
    blocked_since: Option<Instant>,
    patience: Option<Duration>,
    gone: bool,
}

impl Outbox {
    pub(crate) fn new(job: &Job) -> Self {
        Outbox { tx: job.events.clone(), queue: VecDeque::new(), blocked_since: None, patience: job.patience, gone: false }
    }

    pub(crate) fn push(&mut self, e: Event) {
        if !self.gone {
            self.queue.push_back(e);
        }
    }

    /// Every event is in the channel.
    pub(crate) fn is_clear(&self) -> bool {
        self.queue.is_empty()
    }

    /// The client closed the channel, or was given up.
    pub(crate) fn is_closed(&self) -> bool {
        self.gone || self.tx.is_closed()
    }

    /// Move the queued events into the channel, oldest first, without waiting.
    pub(crate) fn flush(&mut self) -> Sent {
        let mut took = false;
        while !self.gone {
            let Some(e) = self.queue.pop_front() else {
                self.blocked_since = None;
                return Sent::Clear;
            };
            match self.tx.try_send(e) {
                Ok(()) => took = true,
                Err(tokio::sync::mpsc::error::TrySendError::Full(e)) => {
                    self.queue.push_front(e);
                    let now = Instant::now();
                    let since = match (took, self.blocked_since) {
                        (false, Some(t)) => t,
                        _ => now,
                    };
                    self.blocked_since = Some(since);
                    match self.patience {
                        Some(p) if now.duration_since(since) >= p => break,
                        _ => return Sent::Blocked,
                    }
                }
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => break,
            }
        }
        self.gone = true;
        self.queue.clear();
        Sent::Gone
    }

    /// Queue `e` and wait until the channel took everything: the batch-1
    /// path, where nothing else runs meanwhile. False when the client is gone.
    pub(crate) fn send(&mut self, e: Event) -> bool {
        self.push(e);
        loop {
            match self.flush() {
                Sent::Clear => return true,
                Sent::Gone => return false,
                Sent::Blocked => std::thread::sleep(POLL),
            }
        }
    }
}

/// Answer every job `rx` brings with `msg`, until the engine leaves the list:
/// what a failed engine does. A queued job's channel is empty, so `try_send`
/// always finds room, and the thread never waits on a client.
pub(crate) fn refuse_queue(rx: &std::sync::mpsc::Receiver<Job>, inf: &EngineInfo, msg: &str) {
    while let Ok(job) = rx.recv() {
        inf.queue.fetch_sub(1, Ordering::SeqCst);
        let _ = job.events.try_send(Event::Error { status: 500, message: msg.to_string() });
    }
}

/// After a panic on the batch-1 path, whose weights the caller dropped: the
/// engine fails, the running `job` gets the reason if its channel has room,
/// and the queue is answered. A client whose channel is full has stopped
/// reading; waiting on it would park the thread with every queued request.
/// Dropping `job` closes its channel, and the routes end that stream with an
/// error line (`api::events`).
pub(crate) fn fail_after_panic(job: Job, rx: &std::sync::mpsc::Receiver<Job>, inf: &EngineInfo, msg: String) {
    *inf.state.lock().expect("state lock") = State::Failed(msg.clone());
    let _ = job.events.try_send(Event::Error { status: 500, message: msg.clone() });
    drop(job);
    inf.busy.store(false, Ordering::SeqCst);
    refuse_queue(rx, inf, &msg);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Events, KeepAlive, Options, Prompt, Request};

    /// A job whose client never reads, on a channel of `cap`, given up after
    /// `patience`.
    fn held_job(cap: usize, patience: Option<Duration>) -> (Job, Events) {
        let (tx, rx) = tokio::sync::mpsc::channel(cap);
        let request = Request { prompt: Prompt::Raw(String::new()), think: false, options: Options::default() };
        (Job { request, events: tx, keep_alive: KeepAlive::Forever, queued_at: Instant::now(), patience }, rx)
    }

    fn delta(s: &str) -> Event {
        Event::Delta { content: s.into(), thinking: String::new() }
    }

    /// An outbox keeps what the channel refused, at its front, and the client
    /// that reads again gets it before anything queued after it.
    #[test]
    fn an_outbox_keeps_order_through_a_full_channel() {
        let (job, mut rx) = held_job(1, None);
        let mut out = Outbox::new(&job);
        for s in ["a", "b", "c"] {
            out.push(delta(s));
        }
        assert_eq!(out.flush(), Sent::Blocked);
        assert_eq!(rx.try_recv().unwrap(), delta("a"));
        assert_eq!(out.flush(), Sent::Blocked);
        out.push(delta("d"));
        assert_eq!(rx.try_recv().unwrap(), delta("b"));
        assert_eq!(out.flush(), Sent::Blocked);
        assert_eq!(rx.try_recv().unwrap(), delta("c"));
        assert_eq!(out.flush(), Sent::Clear);
        assert!(out.is_clear());
        assert_eq!(rx.try_recv().unwrap(), delta("d"));
        assert!(rx.try_recv().is_err());
        // Without a send timeout a client that never reads is never given up.
        let (job, _rx) = held_job(1, None);
        let mut out = Outbox::new(&job);
        out.push(delta("a"));
        out.push(delta("b"));
        for _ in 0..3 {
            assert_eq!(out.flush(), Sent::Blocked);
        }
    }

    /// Past its send timeout the client is gone for good: nothing more is
    /// queued, even once the channel has room. A timeout of zero gives up at
    /// the first full channel.
    #[test]
    fn an_outbox_past_its_timeout_is_gone_for_good() {
        let (job, mut rx) = held_job(1, Some(Duration::ZERO));
        let mut out = Outbox::new(&job);
        out.push(delta("a"));
        out.push(delta("b"));
        assert_eq!(out.flush(), Sent::Gone);
        assert!(out.is_closed());
        assert_eq!(rx.try_recv().unwrap(), delta("a"));
        out.push(delta("c"));
        assert_eq!(out.flush(), Sent::Gone, "a client given up came back");
        assert!(!out.send(delta("d")));
        assert!(rx.try_recv().is_err());
    }
}
