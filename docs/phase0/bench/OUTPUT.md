# Benchmark and pilot output (measured on this cloud container, 2026-10-20)

Machine: 4 vCPU Intel Xeon @ 2.10 GHz, 15 GB RAM. Python 3, numpy 2.5.3, scipy 1.18.1, numba 0.68.0.
The scripts are our own code (not copies). They integrate the same equations as the RnD switching law
(rotating frame, fixed-step RK4, steps per lap from the rule in RnD `global/pass8/gpu/envelope_gpu_sweep.py:110-113`).
The pilot outcome tallies are ROUGH (crude thresholds, no accuracy check). They are only for planning, not labels.

## bench.py (parameters as RnD stress_test.py:76-89, kappa = r_max(0), k = 1, 60 laps)

Single thread (`python bench.py 40 60 1`):

    N=1 B=40 laps=60 threads=1: 3.36s total, 84.0 ms/run,  mean steps/lap 5945 | lost 0 conv 34 other 6
    N=2 B=40 laps=60 threads=1: 5.12s total, 128.1 ms/run, mean steps/lap 7110 | lost 0 conv 36 other 4
    N=3 B=40 laps=60 threads=1: 5.33s total, 133.3 ms/run, mean steps/lap 6243 | lost 0 conv 37 other 3
    N=4 B=40 laps=60 threads=1: 4.46s total, 111.4 ms/run, mean steps/lap 4964 | lost 0 conv 39 other 1
    N=5 B=40 laps=60 threads=1: 6.22s total, 155.5 ms/run, mean steps/lap 5464 | lost 0 conv 34 other 6
    N=6 B=40 laps=60 threads=1: 3.83s total, 95.7 ms/run,  mean steps/lap 2913 | lost 0 conv 34 other 6

Four threads (`python bench.py 200 60 4`), wall time per run:

    N=1: 23.5 ms   N=2: 30.4 ms   N=3: 29.0 ms   N=4: 28.9 ms   N=5: 38.8 ms   N=6: 48.9 ms
    lost: 0 of 1200 (all N)

## pilot2.py (kappa free in [0, 3], r_max(0) in [0.05, 3] independent, k log-uniform [0.1, 5], R in [0.5, 4], 60 laps)

    N=1 n=300: zoneA 100 (lost 0, conv 35) | not A 200 (lost 10, conv 162, other 28)
    N=3 n=299: zoneA 110 (lost 0, conv 41) | not A 189 (lost 7,  conv 151, other 31)
    N=6 n=295: zoneA 92  (lost 0, conv 20) | not A 203 (lost 8,  conv 156, other 39)
    wall time 21 s for 894 runs on 4 threads

## pilot3.py (zone A only, N = 3, 300 runs, 300 laps; "settled" = r_max within 1 % of r* from some lap on)

    time 13.8 s (4 threads)
    settled by 300 laps: 254 / 300; never: 46
    settling lap, quantiles 50 / 90 / 99 / max: 24, 150, 264, 278
    beta = k k1^3 / w of the 46 never-settled runs: 29 of them < 0.05, all but one < 2.3, one at 35
    median beta of all 300 runs: 0.19

## Circle law (CDC 2024, k1 = 0), added after the student's decision

### pilot_circle.py (R in [0.5, 4], k log-uniform [0.1, 5], w log-uniform [0.3, 16]; starts in the disk of radius R; 100 laps)

    N=1: k/(wR^3)<0.385: 251 -> conv 85, lost 40, other 126 | >=0.385: 49 -> lost 49, other 0
    N=3: k/(wR^3)<0.385: 244 -> conv 63, lost 58, other 123 | >=0.385: 56 -> lost 56, other 0
    N=6: k/(wR^3)<0.385: 250 -> conv 34, lost 83, other 133 | >=0.385: 50 -> lost 50, other 0

### pilot_circle2.py (dimensionless R = w = 1, k = gamma log-uniform [1e-3, 0.385); N = 1; 400 laps; 400 runs)

    lost 55 settled 224 other 121
    settling lap quantiles 50/90/99/max: 67.5, 308.7, 381.2, 387
    gamma of all 121 'other' runs: between 0.0011 and 0.0071 (slow approach to a tiny ring; r still decreasing)
    wall time: 108 s on 4 threads (includes about 5 s of compilation) -> about 1 core-second per run
