//! Agent telemetry module.
//!
//! Telemetry and session tracing. This module provides the `TelemetrySink`
//! trait for recording telemetry events and the `SessionTracer` struct
//! for correlating events to a session with a sequential counter.
//!
//! This module is feature-gated behind `feature = "telemetry"`.
//!

use std::time::{SystemTime, UNIX_EPOCH};

/// Telemetry event types recorded by the tracer.
#[derive(Debug, Clone)]
pub enum TelemetryEvent {
    /// Marked when a span begins.
    SpanStart {
        span_id: String,
        name: String,
        start_time: u64,
    },
    /// Marked when a span ends.
    SpanEnd {
        span_id: String,
        end_time: u64,
    },
    /// A metric value recorded at a point in time.
    Metric {
        name: String,
        value: f64,
        unit: Option<String>,
        timestamp: u64,
    },
    /// A log message emitted during execution.
    Log {
        level: String,
        message: String,
        timestamp: u64,
    },
}

impl TelemetryEvent {
    /// Returns the timestamp associated with this event.
    pub fn timestamp(&self) -> u64 {
        match self {
            TelemetryEvent::SpanStart { start_time, .. } => *start_time,
            TelemetryEvent::SpanEnd { end_time, .. } => *end_time,
            TelemetryEvent::Metric { timestamp, .. } => *timestamp,
            TelemetryEvent::Log { timestamp, .. } => *timestamp,
        }
    }
}

/// A span with start and end timestamps.
#[derive(Debug, Clone)]
pub struct TelemetrySpan {
    /// Unique identifier for this span.
    pub span_id: String,
    /// Human-readable name for the span.
    pub name: String,
    /// When the span started.
    pub start_timestamp: SystemTime,
    /// When the span ended, if it has ended.
    pub end_timestamp: Option<SystemTime>,
}

impl TelemetrySpan {
    /// Creates a new span with the given id and name, starting at the current time.
    pub fn new(span_id: String, name: String) -> Self {
        Self {
            span_id,
            name,
            start_timestamp: SystemTime::now(),
            end_timestamp: None,
        }
    }

    /// Returns the span duration in milliseconds, if it has ended.
    pub fn duration_ms(&self) -> Option<u128> {
        self.end_timestamp
            .and_then(|end| end.duration_since(self.start_timestamp).ok())
            .map(|d| d.as_millis())
    }
}

/// Trait for recording telemetry events.
pub trait TelemetryRecorder: Send + Sync {
    /// Records a single telemetry event.
    fn record(&self, event: TelemetryEvent);

    /// Records multiple events at once.
    fn record_batch(&self, events: Vec<TelemetryEvent>) {
        for event in events {
            self.record(event);
        }
    }
}

/// No-op telemetry recorder that discards all events.
#[derive(Debug, Clone, Default)]
pub struct NoopTelemetryRecorder;

impl TelemetryRecorder for NoopTelemetryRecorder {
    fn record(&self, _event: TelemetryEvent) {}
}

/// Returns the current timestamp as seconds since the Unix epoch.
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
