# InfluxDB schema for real-time telemetry (InfluxDB 2.7+ line protocol).

# Hardware metrics — one point per sample per run.
# measurement: hardware_metrics
#   tags:     device_id, model_id, run_id
#   fields:   power_watts, gpu_temp_c, cpu_temp_c, memory_used_mb,
#             gpu_utilization_pct, memory_bandwidth_gbps, clock_mhz

# Inference metrics — per run / batch size / precision.
# measurement: inference_metrics
#   tags:     run_id, batch_size, precision
#   fields:   latency_ms, tokens_per_second, samples_per_second

# System metrics — host-level.
# measurement: system_metrics
#   tags:     device_id
#   fields:   cpu_utilization_pct, disk_io_mbps, network_io_mbps

# The platform writes these via the influxdb-client extra
# (pip install thor-core[timeseries]) once the time-series writer
# lands; see docs/architecture.md for the integration point.
