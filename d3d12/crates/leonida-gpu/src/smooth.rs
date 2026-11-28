//! Leonida: `LEONIDA_PRESENT_SMOOTH=<fraction>`: each host present is held until at least that
//! fraction of the recent mean interval between presents has passed since the one before, so a
//! quick frame after a slow one waits a little instead of being shown at once (the title's
//! simulation step is unchanged; only when its frame is shown moves). Unset: 0.9; `0`: none.
//!
//! The mean is a running one (the newest interval weighs 0.1), each interval taken into it at
//! most 100 ms (a load's stall would hold presents for long after). Called on the presenting
//! thread right before the host's present ([`hold`]) and right after it ([`presented`]).
//!
//! After kylepelham/El1xr (`pacing.rs`, `VICTORY_PRESENT_SMOOTH`, GPL-3.0), whose numbers on the
//! same title at 19-21 frames a second: the change from one present interval to the next 12.7-15.3
//! ms to 6.2 ms, their spread 11.3 to 7.2 ms, at the same rate. El1xr leaves it off while frame
//! generation's proxy swap chain presents; here it holds the present handed to that proxy too.
//!
//! Measured here with frame generation 2x (Series X, 1080 lines, RTX 4060 Ti, Ryzen 5 7600X,
//! standing, 2026-11-28; `LEONIDA_AB` switching it in phases of 10 s, four starts): the change
//! from one host present interval to the next 13.5 -> 7.6 ms with Reflex's limit at 28 shown a
//! second, 11.1 -> 7.7 without one, 11.5 -> 7.8 and 10.1 -> 7.2 with `LEONIDA_FG_MARKERS=1`;
//! their spread 12.3 -> 8.4, 10.4 -> 8.2, 11.5 -> 8.7 and 13.0 -> 7.6 ms; the median interval
//! within 2 ms. Not measured here without frame generation (El1xr's numbers above are of that).

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The fraction when the switch is unset, `1`, not a number, or switched on by `LEONIDA_AB`.
const DEFAULT: f64 = 0.9;
/// The weight of the newest interval in the running mean.
const WEIGHT: f64 = 0.1;
/// The longest interval taken into the mean, microseconds.
const MAX_US: f64 = 100_000.0;

/// `LEONIDA_PRESENT_SMOOTH` as given: [`DEFAULT`] for unset, empty, not a number or `1` (as a
/// switch: a fraction of 1 would hold every present to the mean); none for `0` or less; others
/// clamped to 0.1 ..= 1.
fn parse(value: Option<&str>) -> Option<f64> {
    let Some(f) = value.and_then(|v| v.trim().parse::<f64>().ok()).filter(|f| f.is_finite()) else {
        return Some(DEFAULT);
    };
    (f > 0.0).then(|| if f == 1.0 { DEFAULT } else { f.clamp(0.1, 1.0) })
}

fn asked() -> Option<f64> {
    static ASKED: OnceLock<Option<f64>> = OnceLock::new();
    *ASKED.get_or_init(|| {
        let asked = parse(std::env::var("LEONIDA_PRESENT_SMOOTH").ok().as_deref());
        tracing::info!(fraction = ?asked, "presents held to a fraction of the recent mean interval (LEONIDA_PRESENT_SMOOTH; 0: none)");
        asked
    })
}

/// The fraction now: as asked, or switched by `LEONIDA_AB` (on: as asked or [`DEFAULT`]).
fn fraction() -> Option<f64> {
    #[cfg(all(windows, feature = "diagnostics"))]
    if let Some(on) = crate::ab::pick("LEONIDA_PRESENT_SMOOTH") {
        return on.then(|| asked().unwrap_or(DEFAULT));
    }
    asked()
}

struct Pace {
    /// When the last present returned.
    last: Option<Instant>,
    /// The running mean interval, microseconds (0: none yet).
    mean_us: f64,
}

static PACE: Mutex<Pace> = Mutex::new(Pace { last: None, mean_us: 0.0 });

/// The running mean after an interval of `us` microseconds.
fn next_mean(mean_us: f64, us: f64) -> f64 {
    let us = us.min(MAX_US);
    if mean_us == 0.0 { us } else { mean_us + WEIGHT * (us - mean_us) }
}

/// How long to hold a present `since` after the last one returned, with the mean `mean_us`.
fn hold_for(fraction: f64, mean_us: f64, since: Duration) -> Duration {
    Duration::from_micros((mean_us * fraction) as u64).saturating_sub(since)
}

/// Right before the host's present: held until the fraction of the mean has passed since the
/// last present returned. Returns how long it held.
pub(crate) fn hold() -> Duration {
    let Some(fraction) = fraction() else { return Duration::ZERO };
    let (last, mean_us) = {
        let p = PACE.lock().unwrap_or_else(|e| e.into_inner());
        (p.last, p.mean_us)
    };
    let Some(last) = last else { return Duration::ZERO };
    let wait = hold_for(fraction, mean_us, last.elapsed());
    if !wait.is_zero() {
        std::thread::sleep(wait);
    }
    wait
}

/// Right after the host's present returned: the interval since the last one goes into the mean
/// (whether or not presents are held, so that a switch turned on finds it ready).
pub(crate) fn presented() {
    let now = Instant::now();
    let mut p = PACE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(last) = p.last {
        p.mean_us = next_mean(p.mean_us, now.duration_since(last).as_micros() as f64);
    }
    p.last = Some(now);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presents_are_held_to_a_fraction_of_the_running_mean() {
        assert_eq!(parse(None), Some(DEFAULT));
        assert_eq!(parse(Some("")), Some(DEFAULT));
        assert_eq!(parse(Some("x")), Some(DEFAULT));
        assert_eq!(parse(Some("0")), None);
        assert_eq!(parse(Some("-1")), None);
        assert_eq!(parse(Some("1")), Some(DEFAULT));
        assert_eq!(parse(Some("0.8")), Some(0.8));
        assert_eq!(parse(Some("3")), Some(1.0));
        assert_eq!(next_mean(0.0, 70_000.0), 70_000.0);
        assert_eq!(next_mean(70_000.0, 80_000.0), 71_000.0);
        // A stall counts as 100 ms.
        assert_eq!(next_mean(70_000.0, 900_000.0), 73_000.0);
        // 0.9 of 70 ms is 63: a present 50 ms after the last waits 13, one 70 ms after none.
        assert_eq!(hold_for(0.9, 70_000.0, Duration::from_millis(50)), Duration::from_millis(13));
        assert_eq!(hold_for(0.9, 70_000.0, Duration::from_millis(70)), Duration::ZERO);
    }
}
