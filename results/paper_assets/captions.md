# Suggested manuscript captions and placement notes

## Figure 1
`figures/figure01_fidelity_by_calibration.pdf`

Benchmark exact fidelity across the three calibration snapshots and the pooled overall benchmark. Bars show mean fidelity across 10 seed-level benchmark records per calibration (30 records total); points show the individual record means. The agent outperforms both SABRE-best20 and the historical target-aware SABRE fallback on every calibration.

## Figure 2
`figures/figure02_family_breakdown.pdf`

Per-circuit-family exact fidelity and agent-minus-baseline fidelity deltas for all benchmark families. The figure highlights the 5q/8q gains and the narrower or reversed gains at 10q.

## Figure 3 (backup / appendix)
`figures/figure03_pareto_by_family.pdf`

Mean exact fidelity versus mean two-qubit gate count, stratified by circuit family. This figure supports the tradeoff discussion: the agent usually buys fidelity with additional entangling-gate count.

## Figure 4 (backup / appendix)
`figures/figure04_record_fidelity_gains.pdf`

Distribution of record-level fidelity gains (agent minus baseline) over the 30 run-by-calibration benchmark records. Every record shows a positive fidelity gain against both baselines, with coincident points reflecting repeated run-level means.

## Table 1
`tables/table01_overall_summary.tex`

Main summary table for the workshop paper. Keep this in the core results section.

## Table 2
`tables/table02_paired_effects.tex`

Run-by-calibration-cell paired effects. This table is compact enough for the main paper if space permits; otherwise move it to a short appendix / supplementary block.

## Table 3
`tables/table03_per_calibration.tex`

Useful if we want a calibration-specific paragraph, but probably not necessary in a 4-page workshop version.

## Table 4
`tables/table04_per_class.tex`

Good appendix material. If the main paper gets tight, Figure 2 carries the same story more efficiently.

## Table 6
`data/table06_run_cell_uncertainty.csv`

Use this for manuscript intervals and any reviewer response about the statistical unit. The intervals are descriptive bootstrap intervals over the 30 run-by-calibration cell means.

