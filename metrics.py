# This file was creating with GPT6.1 sol in response to the following query

# Could you modify the project such that a prometheus web server is started at port 8000, and captures relevant metrics for the application? 
# The metrics included in the attached file should also appear in this project, along with additional relevant metrics. 
# The way in which the web_server is started should also be similar.

"""Prometheus instrumentation for scoring requests and model attempts."""

from contextlib import contextmanager
from functools import wraps
from time import perf_counter

from prometheus_client import Counter, Gauge, Histogram, Summary


# Preserve the names and types used by the reference application.
REQUEST_COUNTER = Counter("app_requests_total", "Total number of scoring requests")
SUCCESSFUL_REQUESTS = Counter(
    "app_successful_requests_total", "Total number of successful scoring requests"
)
FAILED_REQUESTS = Counter(
    "app_failed_requests_total", "Total number of failed or rejected scoring requests"
)
REQUEST_DURATION = Summary(
    "app_request_duration_seconds", "Time spent processing scoring requests"
)
IN_PROGRESS = Gauge("app_requests_in_progress", "Scoring requests currently executing")
REJECTED_REQUESTS = Counter(
    "app_rejected_requests_total", "Scoring requests rejected before inference", ["reason"]
)
MODEL_ATTEMPTS = Counter(
    "app_model_attempts_total", "Model attempts including fallback attempts", ["model", "outcome"]
)
MODEL_DURATION = Histogram(
    "app_model_duration_seconds", "Time spent in a model attempt", ["model"],
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
)
RESPONSE_LENGTH = Histogram(
    "app_response_length_characters", "Length of successful critique Markdown in Unicode characters",
    ["model"], buckets=(50, 100, 250, 500, 1000, 2000, 5000, 10000, 20000),
)
GENERATION_DURATION = Histogram(
    "app_response_generation_duration_seconds", "Wall time of an individual generation call",
    ["model", "stage", "outcome"],
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
)
FAILOVERS = Counter(
    "app_model_failovers_total", "Fallback attempts after a model failure", ["source", "destination"]
)
CPU_PERCENT = Gauge("app_resource_cpu_percent", "CPU utilization visible to psutil")
MEMORY_PERCENT = Gauge("app_resource_memory_percent", "System memory utilization visible to psutil")
OVERLOADED = Gauge("app_resource_overloaded", "Resource monitor overload state (1 overloaded, 0 normal)")
LAST_SAMPLE = Gauge("app_resource_last_sample_timestamp_seconds", "Unix time of last successful resource sample")
SAMPLE_ERRORS = Counter("app_resource_sample_errors_total", "Failed resource monitoring samples")


def track_request(function):
    """Count each callback once, including validation failures and failed fallbacks."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        REQUEST_COUNTER.inc()
        started = perf_counter()
        IN_PROGRESS.inc()
        try:
            result = function(*args, **kwargs)
        except Exception:
            FAILED_REQUESTS.inc()
            raise
        else:
            SUCCESSFUL_REQUESTS.inc()
            return result
        finally:
            REQUEST_DURATION.observe(perf_counter() - started)
            IN_PROGRESS.dec()
    return wrapped


@contextmanager
def model_attempt(model):
    started = perf_counter()
    try:
        yield
    except Exception:
        MODEL_ATTEMPTS.labels(model, "failure").inc()
        raise
    else:
        MODEL_ATTEMPTS.labels(model, "success").inc()
    finally:
        MODEL_DURATION.labels(model).observe(perf_counter() - started)


@contextmanager
def generation_attempt(model, stage):
    """Time generation after preparation/loading; include failed calls separately."""
    started = perf_counter()
    outcome = "failure"
    try:
        yield
        outcome = "success"
    finally:
        GENERATION_DURATION.labels(model, stage, outcome).observe(perf_counter() - started)
