# References used by step 1

## Emrich et al. 2024 (the `prominence` delineator)

J. Emrich, A. Gargano, T. Koka, M. Muma, "Physiology-Informed ECG Delineation Based on Peak
Prominence," *32nd European Signal Processing Conference (EUSIPCO 2024)*, Lyon, pp. 1402–1406.
DOI [10.23919/EUSIPCO63174.2024.10715353](https://doi.org/10.23919/EUSIPCO63174.2024.10715353).

- Open-access PDF (EURASIP): <https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0001402.pdf>
- Author copy: <https://jonasemrich.de/publications/physiology-informed-ecg-delineation-based-on-peak-prominence/EUSIPCO24_ECG_delineation_print.pdf>
- Local copy: [refs/Emrich2024_EUSIPCO_prominence_delineation.pdf](refs/Emrich2024_EUSIPCO_prominence_delineation.pdf)
- Code: <https://github.com/JonasEmrich/prominence-delineator> (GPL-3.0). Ported into NeuroKit2 by the
  first author in [PR #1053](https://github.com/neuropsychology/NeuroKit/pull/1053) (merged
  2024-11-27), first released in NeuroKit2 0.2.11 (2025-05-13); logic unchanged through 0.2.13.
- Companion R-peak detector (FastNVG): J. Emrich, T. Koka, S. Wirth, M. Muma, "Accelerated
  Sample-Accurate R-Peak Detectors Based on Visibility Graphs," EUSIPCO 2023, pp. 1090–1094,
  DOI 10.23919/EUSIPCO58844.2023.10290007. In NeuroKit2: `ecg_peaks(method="emrich2023")`
  (needs `ts2vg`).

### Evaluation protocol on LUDB (Section III of the paper)

- All 200 LUDB records, 12 leads, 10 s; no exclusions stated. Sampling rate not stated (LUDB is 500 Hz).
- Two settings in Table II: all 12 leads **with** a multi-lead correction (detections found in fewer
  than a third of the leads are discarded; those in at least two thirds are set for all leads at
  their average position), and **single lead, the first LUDB lead = lead I**. NeuroKit2's
  `prominence` has no multi-lead step, so its comparable rows are the single-lead ones.
- Cleaning: NeuroKit2's default `ecg_clean` (0.5 Hz high-pass, powerline filter). R-peaks: FastNVG,
  not NeuroKit2's default detector.
- Tolerance: "a 150 ms long tolerance window around an annotation" (ANSI/AAMI EC57). TP if a
  detection lies in the window, FN otherwise, FP for a detection with no annotation. One-to-one
  matching is not stated. Se, PPV and F1 are defined, but only F1 is tabulated.
- Error per TP = smallest interval between detection and annotation, pooled over leads. **m** = mean
  over all detections; **σ** = average of the per-recording standard deviations (not pooled). Sign
  convention not stated.
- First/last beats and incomplete annotations: not stated.
- CSE criterion: σ must be below 2σ_CSE = 10.2 / 12.7 / 6.5 / 11.6 / 30.6 ms for P onset / P offset /
  QRS onset / QRS offset / T offset.

### Table II, LUDB, 150 ms tolerance (F1 in %, m ± σ in ms)

| Point | Single lead (lead I) F1 | Single lead m ± σ | 12-lead + correction F1 | 12-lead m ± σ |
|---|---|---|---|---|
| P onset | 99.75 | 10.6 ± 11.6 | 99.58 | 1.2 ± 4.8 |
| P peak | 99.82 | 2.6 ± 5.8 | 99.54 | 0.6 ± 3.8 |
| P offset | 99.82 | 6.8 ± 9.0 | 99.56 | 0.6 ± 4.0 |
| QRS onset | 99.81 | 5.8 ± 6.2 | 99.92 | −0.2 ± 2.0 |
| R peak | 99.81 | 1.6 ± 2.0 | 99.97 | −0.2 ± 1.8 |
| QRS offset | 99.81 | −9.2 ± 7.2 | 99.94 | −6.0 ± 4.0 |
| T onset | 98.30 | 5.6 ± 16.6 | 99.19 | 0.6 ± 6.6 |
| T peak | 98.33 | 2.6 ± 8.6 | 99.15 | 0.2 ± 4.0 |
| T offset | 98.54 | 14.2 ± 16.0 | 99.22 | 2.4 ± 7.6 |

Other single-lead (lead I) rows in the same table: NeuroKit2 `dwt` F1 99.25 / 98.82 / 98.46 / 98.30 /
99.67 / 96.11 / 96.61 / 91.27 / 87.26 with σ 17.4 / 13.6 / 18.6 / 20.4 / 1.8 / 13.2 / 27.8 / 13.6 /
15.4 ms; NeuroKit2 `peak` F1 99.10 (P onset), 99.53 (P peak), 99.67 (R), 95.84 (T peak), 94.45
(T offset). `cwt` was not evaluated. 12-lead rows quoted from other papers: Kalyakulina et al.
(F1 97.42–99.74, σ 6.2–15.5 ms) and Chen et al. (F1 97.42–99.93, σ 3.4–14.6 ms).

### What this means for our sanity check (decision 43 and 46)

- Compare our **lead I** run of NeuroKit2 `prominence` against the single-lead rows, using the
  **within-recording SD** and the pooled mean, F1 at ±150 ms.
- Our baseline uses NeuroKit2's default R-peak detector (story 23). For the check only, a second
  run uses FastNVG R-peaks via `ecg_peaks(method="emrich2023")`, which is what the paper used.
- Known issue: NeuroKit2 0.2.13 can raise `IndexError`/`ValueError` in `prominence` when an R-peak
  sits at the signal boundary ([PR #1181](https://github.com/neuropsychology/NeuroKit/pull/1181),
  fixed on `dev` only). Decision 42 counts such a lead as having no predictions and reports the count.
- The paper's authors describe the method's QRS onset/offset as "exploratory"; the method is tuned
  for P and T waves.
- The 12-lead rows (σ 2–8 ms) are the published reference for what multi-lead information buys; our
  12-lead trained models can be read against them, with the caveat that those rows include a
  multi-lead voting step.

**Outcome (2026-10-10, `results/neurokit_prominence.json`):** our lead-I means and within-recording
SDs match the single-lead rows within 0.3 ms for all nine points; F1 matches within 1 point for the
QRS and T points. P-wave F1 is 91.8% against 99.8% because the paper evidently did not count P
detections on beats without a P annotation as false positives (AF/flutter people and sinus beats with
no P marked in lead I); our scorer does. See decision 48. The FastNVG variant
(`results/neurokit_prominence_emrich2023.json`) came out far worse (lead-I QRS F1 93.3%, V1–V3
43–64%): NeuroKit2 0.2.13's `emrich2023` detector misses many beats, so it does not reproduce the
paper's pipeline and is not used further (decision 49).
