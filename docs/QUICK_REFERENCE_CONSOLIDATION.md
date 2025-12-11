# Quick Reference: Documentation Consolidation

**One-page guide for immediate action**

---

## What's Happening

Your repo has **25 root .md files** with 60-70% redundancy.
**Goal:** Consolidate to **5-6 core files** for thesis submission.

---

## Quick Stats

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Root .md files | 25 | 5-6 | -76% |
| SLURM scripts in root | 16 | 0 | Organized |
| Loose data files | 5 | 0 | Organized |
| Time needed | - | 6-8 hrs | - |

---

## Three-Step Process

### STEP 1: Backup (5 min)
```bash
cd /home/btd8/llm-paper-analyze
git add -A
git commit -m "Backup before consolidation"
```

### STEP 2: Migrate Files (10 min)
```bash
# Run the migration commands from CONSOLIDATION_PLAN.md
# This moves files to archive/ and creates directory structure
```

### STEP 3: Create Consolidated Docs (6-7 hrs)
Create these 3 new files by merging existing docs:
- **IMPLEMENTATION.md** (merge 5 files)
- **RESULTS.md** (merge results)
- **GUIDE.md** (merge 3 setup files)

Update these 2 files:
- **README.md** (Oct 27 → Dec 11 state)
- **CURRENT_STATUS.md** (Nov 10 → Dec 11 state)

---

## File Consolidation Map

### CREATE: IMPLEMENTATION.md
Merge these 5 files:
```
✓ CELLXGENE_IMPLEMENTATION_SUMMARY.md (base)
✓ FINAL_LLM_SUMMARY.md (LLM section)
✓ LLM_GENERATION_RESULTS.md (LLM details)
✓ Parts of IMPLEMENTATION_COMPARISON.md
✓ Parts of IMPLEMENTATION_COMPLETE.md
```

### CREATE: RESULTS.md
Merge these:
```
✓ TEST_RESULTS_100_DATASETS.md (base)
✓ Benchmark logs and metrics
✓ Classification results
```

### CREATE: GUIDE.md
Merge these 3 files:
```
✓ MANYLATENTS_SETUP.md (base)
✓ SUBSAMPLING_STRATEGY.md (technical)
✓ SETUP_COMPLETE.md (verification)
```

### ARCHIVE: 9 files to archive/docs/
```
→ CLEANUP_STATUS.md
→ PRESENTATION_PREP_PLAN.md
→ PRESENTATION_README.md
→ CELLXGENE_PAPER_PIPELINE_PLAN.md
→ IMPLEMENTATION_COMPARISON.md
→ IMPLEMENTATION_COMPLETE.md
→ GEOMANCER_DASHBOARD_PLAN.md
→ QUICK_STATUS.md
→ SUMMARY.md
```

### UPDATE: 3 core files
```
↻ README.md (update to Dec 2025)
↻ CURRENT_STATUS.md (update to Dec 2025)
↻ ARCHITECTURE.md (verify accuracy)
```

---

## Directory Changes

### SLURM Scripts
```bash
# FROM: /home/btd8/llm-paper-analyze/*.slurm
# TO: /home/btd8/llm-paper-analyze/slurm_jobs/active/*.slurm
```

### Data Outputs
```bash
# FROM: /home/btd8/llm-paper-analyze/data/*.csv
# TO: /home/btd8/llm-paper-analyze/data/analysis_results/phate/*.csv
```

### Docs
```bash
# FROM: /home/btd8/llm-paper-analyze/METRICS_EXPLAINED.md
# TO: /home/btd8/llm-paper-analyze/docs/Metrics_Explained.md
```

---

## Final Structure

```
llm-paper-analyze/
├── README.md                    ← Updated
├── ARCHITECTURE.md              ← Verified
├── IMPLEMENTATION.md            ← NEW (5 files merged)
├── RESULTS.md                   ← NEW (results merged)
├── GUIDE.md                     ← NEW (3 files merged)
├── CURRENT_STATUS.md            ← Updated
│
├── docs/                        ← Supplementary docs
│   ├── SLURM_Guide.md
│   ├── Metrics_Explained.md
│   ├── INDEX.md
│   └── DOCUMENTATION_AUDIT_REPORT.md
│
├── slurm_jobs/                  ← All SLURM scripts
│   ├── active/
│   ├── tests/
│   └── archived/
│
├── data/                        ← Organized outputs
│   ├── analysis_results/
│   │   ├── phate/
│   │   └── classification/
│   ├── papers/
│   ├── geo/
│   └── ...
│
├── archive/                     ← Historical docs
│   └── docs/
│       ├── 2025-10-27/
│       ├── 2025-11-03/
│       ├── 2025-11-04/
│       ├── 2025-11-10/
│       └── 2025-11-28/
│
├── scripts/                     ← Python scripts (unchanged)
├── logs/                        ← Logs (unchanged)
└── manylatents/                 ← Submodule (unchanged)
```

---

## Time Breakdown

| Task | Time |
|------|------|
| Backup & migration | 15 min |
| Create IMPLEMENTATION.md | 2 hr |
| Create RESULTS.md | 1.5 hr |
| Create GUIDE.md | 1.5 hr |
| Update README.md | 45 min |
| Update CURRENT_STATUS.md | 30 min |
| Update cross-references | 1 hr |
| Verify & commit | 30 min |
| **TOTAL** | **8 hr** |

---

## Commands Cheat Sheet

### Backup
```bash
cd /home/btd8/llm-paper-analyze
git add -A && git commit -m "Backup before consolidation"
```

### View Current Structure
```bash
# Count root .md files
find . -maxdepth 1 -name "*.md" | wc -l

# List them
ls -1 *.md
```

### After Migration - Verify
```bash
# Should be 5-6 files
find . -maxdepth 1 -name "*.md" | wc -l

# Check archive
ls archive/docs/

# Check new directories
ls slurm_jobs/
ls data/analysis_results/
ls docs/
```

### Rollback if Needed
```bash
git reset --hard HEAD~1
```

---

## Checklist

Before:
- [ ] Read CONSOLIDATION_SUMMARY.md
- [ ] Review CONSOLIDATION_PLAN.md
- [ ] Create backup

During:
- [ ] Run migration script
- [ ] Create IMPLEMENTATION.md
- [ ] Create RESULTS.md
- [ ] Create GUIDE.md
- [ ] Update README.md
- [ ] Update CURRENT_STATUS.md

After:
- [ ] Verify root has 5-6 .md files
- [ ] Verify slurm_jobs/ exists
- [ ] Verify data/analysis_results/ exists
- [ ] Check no broken links
- [ ] Git commit with descriptive message

---

## Get Help

1. **Detailed audit:** `docs/DOCUMENTATION_AUDIT_REPORT.md`
2. **Step-by-step plan:** `docs/CONSOLIDATION_PLAN.md`
3. **Summary:** `docs/CONSOLIDATION_SUMMARY.md`
4. **This guide:** `docs/QUICK_REFERENCE_CONSOLIDATION.md`

---

## Decision Matrix

| File | Action |
|------|--------|
| CELLXGENE_IMPLEMENTATION_SUMMARY.md | Merge → IMPLEMENTATION.md |
| FINAL_LLM_SUMMARY.md | Merge → IMPLEMENTATION.md |
| LLM_GENERATION_RESULTS.md | Merge → IMPLEMENTATION.md |
| MANYLATENTS_SETUP.md | Merge → GUIDE.md |
| SUBSAMPLING_STRATEGY.md | Merge → GUIDE.md |
| SETUP_COMPLETE.md | Merge → GUIDE.md |
| TEST_RESULTS_100_DATASETS.md | Merge → RESULTS.md |
| README.md | Update (Oct → Dec) |
| CURRENT_STATUS.md | Update (Nov → Dec) |
| ARCHITECTURE.md | Verify & update if needed |
| CLEANUP_STATUS.md | Archive |
| PRESENTATION_*.md | Archive |
| GEOMANCER_DASHBOARD_PLAN.md | Archive |
| QUICK_STATUS.md | Archive |
| SUMMARY.md | Archive |
| METRICS_EXPLAINED.md | Move → docs/ |
| SLURM_GUIDE.md | Move → docs/ |

---

**Ready? Start with: CONSOLIDATION_PLAN.md**
