# Documentation Consolidation Plan
## Thesis Repository Organization

**Created:** December 11, 2025
**Purpose:** Transform scattered documentation into thesis-ready structure

---

## Quick Summary

Your repository has 25 markdown files in the root directory with 60-70% redundancy. This plan consolidates them into 5 core documents and organizes outputs into logical subdirectories.

**Time Required:** 6-8 hours
**Files Affected:** 25 documentation files, 16 SLURM scripts, 5 data files

---

## What You'll Achieve

### Before:
```
llm-paper-analyze/
├── [25 scattered .md files]
├── [16 .slurm files in root]
├── data/[5 loose CSV/JSON files]
└── ... confusion ...
```

### After:
```
llm-paper-analyze/
├── README.md (updated)
├── ARCHITECTURE.md
├── IMPLEMENTATION.md (NEW - consolidates 5 files)
├── RESULTS.md (NEW - consolidates results)
├── GUIDE.md (NEW - consolidates 3 setup files)
├── CURRENT_STATUS.md (updated)
├── docs/ (supplementary documentation)
├── slurm_jobs/ (organized SLURM scripts)
├── data/ (organized outputs)
└── archive/ (historical files preserved)
```

---

## Step-by-Step Instructions

### STEP 1: Backup Everything (5 minutes)

```bash
cd /home/btd8/llm-paper-analyze

# Create backup
git add -A
git commit -m "Backup before documentation consolidation"

# Or create manual backup
tar -czf ~/llm-paper-analyze-backup-$(date +%Y%m%d).tar.gz .
```

### STEP 2: Run Migration Script (10 minutes)

I've created the directory structure and will move files to archive:

```bash
cd /home/btd8/llm-paper-analyze

# Create directory structure
mkdir -p archive/docs/2025-{10-27,11-{03,04,10,28}}
mkdir -p docs
mkdir -p slurm_jobs/{active,archived,tests}
mkdir -p data/analysis_results/{phate,classification}

# Move obsolete documentation to archive
mv CLEANUP_STATUS.md archive/docs/2025-11-03/
mv PRESENTATION_PREP_PLAN.md archive/docs/2025-11-03/
mv PRESENTATION_README.md archive/docs/2025-11-03/
mv CELLXGENE_PAPER_PIPELINE_PLAN.md archive/docs/2025-11-04/
mv IMPLEMENTATION_COMPARISON.md archive/docs/2025-11-04/
mv IMPLEMENTATION_COMPLETE.md archive/docs/2025-11-04/
mv TEST_RESULTS_100_DATASETS.md archive/docs/2025-11-04/
mv GEOMANCER_DASHBOARD_PLAN.md archive/docs/2025-11-28/
mv QUICK_STATUS.md archive/docs/2025-10-27/
mv SUMMARY.md archive/docs/2025-10-27/

# Move supplementary docs
mv METRICS_EXPLAINED.md docs/
mv SLURM_GUIDE.md docs/

# Organize SLURM scripts
mv run_manylatents_array.slurm slurm_jobs/active/
mv run_subsampling*.slurm slurm_jobs/active/
mv run_classification.slurm slurm_jobs/active/
mv run_enhanced_*.slurm slurm_jobs/active/
mv run_wandb_*.slurm slurm_jobs/active/
mv retrain_*.slurm slurm_jobs/active/
mv *_test*.slurm slurm_jobs/tests/ 2>/dev/null || true
mv monitor_*.sh slurm_jobs/active/
mv run_*.sh slurm_jobs/active/ 2>/dev/null || true

# Organize data outputs
mv data/phate_*.csv data/analysis_results/phate/ 2>/dev/null || true
mv data/enhanced_phate_metrics.csv data/analysis_results/phate/ 2>/dev/null || true
mv data/confidence_analysis.csv data/analysis_results/classification/ 2>/dev/null || true
mv data/label_analysis_summary.json data/analysis_results/classification/ 2>/dev/null || true

echo "Migration complete!"
```

### STEP 3: Create Consolidated Documentation (3-4 hours)

#### 3A. Create IMPLEMENTATION.md

Consolidate these files:
- CELLXGENE_IMPLEMENTATION_SUMMARY.md (primary source)
- FINAL_LLM_SUMMARY.md (LLM section)
- LLM_GENERATION_RESULTS.md (LLM details)

**Structure:**
```markdown
# Implementation Documentation

## Overview
[From CELLXGENE_IMPLEMENTATION_SUMMARY intro]

## Database Schema Evolution
[From CELLXGENE_IMPLEMENTATION_SUMMARY sections]

## CELLxGENE Integration Pipeline
[From CELLXGENE_IMPLEMENTATION_SUMMARY]

## LLM Description Generation
[From FINAL_LLM_SUMMARY]

## Testing and Validation
[From CELLXGENE_IMPLEMENTATION_SUMMARY]

## Code Quality and Standards
[From CELLXGENE_IMPLEMENTATION_SUMMARY]
```

#### 3B. Create RESULTS.md

Consolidate these files:
- TEST_RESULTS_100_DATASETS.md
- Data from logs and benchmarks

**Structure:**
```markdown
# Results and Analysis

## Dataset Processing Results
[From various sources]

## Benchmarking Results
[From TEST_RESULTS_100_DATASETS and logs]

## Classification Results
[From ML results]

## Performance Metrics
[From logs and metrics files]

## Quality Assessment
[From validation results]
```

#### 3C. Create GUIDE.md

Consolidate these files:
- MANYLATENTS_SETUP.md (primary)
- SUBSAMPLING_STRATEGY.md (technical details)
- SETUP_COMPLETE.md (verification steps)

**Structure:**
```markdown
# Setup and Reproduction Guide

## Prerequisites
[From MANYLATENTS_SETUP]

## Environment Setup
[From MANYLATENTS_SETUP]

## Data Processing Pipeline
[From SUBSAMPLING_STRATEGY]

## Running Experiments
[From SETUP_COMPLETE]

## Monitoring and Verification
[From all sources]

## Troubleshooting
[From MANYLATENTS_SETUP]
```

### STEP 4: Update Core Documentation (1-2 hours)

#### 4A. Update README.md

Current README is from October 27. Update to:
- Reflect December 2025 state
- Reference new consolidated docs
- Simplify to high-level overview
- Add thesis context

#### 4B. Update CURRENT_STATUS.md

Merge information from:
- CURRENT_STATUS.md (November 10 - base)
- SETUP_COMPLETE.md (verification status)
- Recent work

Focus on: What's the current state as of December 2025?

#### 4C. Review ARCHITECTURE.md

Check if it needs updates for current three-repo state.

### STEP 5: Create Index Documentation (30 minutes)

Create `/home/btd8/llm-paper-analyze/docs/INDEX.md`:

```markdown
# Documentation Index

## Core Documentation
- [README.md](../README.md) - Project overview
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System design
- [IMPLEMENTATION.md](../IMPLEMENTATION.md) - Implementation details
- [RESULTS.md](../RESULTS.md) - Results and analysis
- [GUIDE.md](../GUIDE.md) - Setup and reproduction
- [CURRENT_STATUS.md](../CURRENT_STATUS.md) - Current state

## Supplementary Documentation
- [docs/SLURM_Guide.md](SLURM_Guide.md) - HPC cluster usage
- [docs/Metrics_Explained.md](Metrics_Explained.md) - Metrics definitions
- [scripts/README.md](../scripts/README.md) - Script documentation
- [scripts/QUICK_REFERENCE.md](../scripts/QUICK_REFERENCE.md) - Quick commands

## Historical Documentation
- [archive/docs/](../archive/docs/) - Archived documentation by date

## Directory Structure
See: [DOCUMENTATION_AUDIT_REPORT.md](DOCUMENTATION_AUDIT_REPORT.md)
```

### STEP 6: Update Cross-References (1 hour)

Search and update references in:
- All markdown files
- README files in subdirectories
- Scripts that reference documentation

```bash
# Find references to moved files
grep -r "METRICS_EXPLAINED.md" . --include="*.md" --include="*.py"
grep -r "SLURM_GUIDE.md" . --include="*.md" --include="*.py"

# Update to new locations:
# METRICS_EXPLAINED.md -> docs/Metrics_Explained.md
# SLURM_GUIDE.md -> docs/SLURM_Guide.md
```

### STEP 7: Verify and Commit (30 minutes)

```bash
# Verify structure
ls -la
ls -la docs/
ls -la slurm_jobs/
ls -la data/analysis_results/

# Check git status
git status

# Stage changes
git add -A

# Commit with descriptive message
git commit -m "Consolidate documentation: 25 files → 5 core docs + organized structure

- Created IMPLEMENTATION.md (consolidates 5 files)
- Created RESULTS.md (consolidates test results)
- Created GUIDE.md (consolidates 3 setup files)
- Moved 9 files to archive with dates
- Organized SLURM scripts into slurm_jobs/
- Organized data outputs into subdirectories
- Updated README.md to December 2025 state
- Updated CURRENT_STATUS.md with latest info
- Created docs/ for supplementary documentation

Thesis-ready structure with clear organization."
```

---

## Files to Create

### Priority 1: Must Create (Thesis Essential)

1. **IMPLEMENTATION.md**
   - Source: CELLXGENE_IMPLEMENTATION_SUMMARY.md + FINAL_LLM_SUMMARY.md
   - ~500-800 lines
   - Time: 1.5-2 hours

2. **RESULTS.md**
   - Source: TEST_RESULTS_100_DATASETS.md + logs + metrics
   - ~300-500 lines
   - Time: 1-1.5 hours

3. **GUIDE.md**
   - Source: MANYLATENTS_SETUP.md + SUBSAMPLING_STRATEGY.md + SETUP_COMPLETE.md
   - ~400-600 lines
   - Time: 1-1.5 hours

### Priority 2: Should Update

4. **README.md** (UPDATE)
   - Current: October 27, 2025
   - Target: December 11, 2025
   - Time: 30-45 minutes

5. **CURRENT_STATUS.md** (UPDATE)
   - Current: November 10, 2025
   - Target: December 11, 2025
   - Time: 20-30 minutes

### Priority 3: Nice to Have

6. **docs/INDEX.md** (NEW)
   - Navigation aid
   - Time: 15-20 minutes

7. **docs/API_Reference.md** (NEW)
   - Code usage examples
   - Time: 1-2 hours (optional)

---

## Verification Checklist

After consolidation, verify:

- [ ] Root directory has ~5-6 core .md files (down from 25)
- [ ] All SLURM scripts in `slurm_jobs/` directory
- [ ] All loose data files organized in subdirectories
- [ ] Supplementary docs in `docs/` directory
- [ ] All obsolete files in `archive/` with dates
- [ ] README.md is up-to-date entry point
- [ ] No broken links in documentation
- [ ] Git history preserved
- [ ] Scripts still work with new structure

---

## Files That Can Be Deleted

After merging content, these files can be safely deleted (or archived):

1. CELLXGENE_IMPLEMENTATION_SUMMARY.md → merged into IMPLEMENTATION.md
2. FINAL_LLM_SUMMARY.md → merged into IMPLEMENTATION.md
3. LLM_GENERATION_RESULTS.md → merged into IMPLEMENTATION.md
4. MANYLATENTS_SETUP.md → merged into GUIDE.md
5. SUBSAMPLING_STRATEGY.md → merged into GUIDE.md
6. SETUP_COMPLETE.md → merged into GUIDE.md

**Recommendation:** Move to archive first, delete after verification.

---

## Quick Commands Reference

### View Current Documentation Structure
```bash
cd /home/btd8/llm-paper-analyze
find . -maxdepth 1 -name "*.md" -type f | sort
```

### Count Documentation Files
```bash
# Before consolidation
find . -maxdepth 1 -name "*.md" -type f | wc -l

# After consolidation (target: 5-6)
find . -maxdepth 1 -name "*.md" -type f | wc -l
```

### View Archive
```bash
ls -la archive/docs/
```

### Check SLURM Scripts
```bash
ls slurm_jobs/active/
ls slurm_jobs/tests/
```

### View Data Organization
```bash
tree data/analysis_results/
```

---

## Rollback Plan

If something goes wrong:

```bash
# Option 1: Git reset (if committed backup)
git reset --hard HEAD~1

# Option 2: Restore from tar backup
cd ~
tar -xzf llm-paper-analyze-backup-YYYYMMDD.tar.gz -C /tmp/
# Review and selectively restore files

# Option 3: Manual restoration
# Files are in archive/, just move back to root
```

---

## Timeline

| Task | Time | Cumulative |
|------|------|------------|
| Backup | 5 min | 5 min |
| Run migration script | 10 min | 15 min |
| Create IMPLEMENTATION.md | 1.5-2 hr | 2.25 hr |
| Create RESULTS.md | 1-1.5 hr | 3.5 hr |
| Create GUIDE.md | 1-1.5 hr | 4.75 hr |
| Update README.md | 30-45 min | 5.5 hr |
| Update CURRENT_STATUS.md | 20-30 min | 6 hr |
| Create INDEX.md | 15-20 min | 6.25 hr |
| Update cross-references | 1 hr | 7.25 hr |
| Verify and commit | 30 min | 7.75 hr |
| **TOTAL** | **~8 hours** | |

---

## Next Steps

1. **Review this plan** - Understand what will change
2. **Create backup** - Git commit or tar archive
3. **Run migration script** - Reorganize files (10 min)
4. **Create consolidated docs** - Main work (4-5 hours)
5. **Update core docs** - README, STATUS (1 hour)
6. **Verify and commit** - Final checks (30 min)

---

## Questions?

**Before starting:**
- Review: `/home/btd8/llm-paper-analyze/docs/DOCUMENTATION_AUDIT_REPORT.md`
- This plan: `/home/btd8/llm-paper-analyze/docs/CONSOLIDATION_PLAN.md`

**During consolidation:**
- Check: File decision matrix in DOCUMENTATION_AUDIT_REPORT.md Appendix A
- Reference: Cross-reference map in DOCUMENTATION_AUDIT_REPORT.md Appendix B

**After completion:**
- Verify: Checklist above
- Commit: With descriptive message

---

**Ready to begin? Start with STEP 1: Backup Everything**
