//! drift-watch: data-drift detection with PSI, KL-divergence, and severity
//! classification.
//!
//! The logic is pure numeric code (no external crates) so it ports identically to
//! the Python, C#, Java, Go, and TypeScript siblings. Given a reference
//! ("expected") sample and a live ("actual") sample of a numeric feature, both are
//! binned over the reference range and the amount the distribution moved is
//! quantified.

pub mod drift;

pub use drift::{classify, detect_drift, kl_divergence, psi, DriftResult, Severity};
