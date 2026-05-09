# Blind Earth Experiment - Current Status

## Running Jobs

### Job 219409: Combined (Student Models + OLMo3) at 2° Resolution
- **Status**: Running on dws-10
- **Walltime Used**: ~11 minutes / 48 hours max
- **Resolution**: 2° (16,200 queries per model = ~2-3 min per model)
- **Models**: 4 Qwen2.5 student models + 14 OLMo3 checkpoints = 18 total

#### Completed (2° resolution):
✅ Qwen/Qwen2.5-0.5B-Instruct
✅ Qwen/Qwen2.5-1.5B-Instruct  
✅ Qwen/Qwen2.5-3B-Instruct
✅ Qwen/Qwen2.5-7B-Instruct
✅ allenai/OLMo-3-1025-7B@stage1-step2000
✅ allenai/OLMo-3-1025-7B@stage1-step5000
✅ allenai/OLMo-3-1025-7B@stage1-step10000

#### Pending (will complete in order):
- [ ] allenai/OLMo-3-1025-7B@stage1-step25000
- [ ] allenai/OLMo-3-1025-7B@stage1-step50000
- [ ] allenai/OLMo-3-1025-7B@stage1-step100000
- [ ] allenai/OLMo-3-1025-7B@stage1-step200000
- [ ] allenai/OLMo-3-1025-7B@stage1-step500000
- [ ] allenai/OLMo-3-1025-7B@stage1-step1000000
- [ ] allenai/OLMo-3-1025-7B@stage1-step1413000
- [ ] allenai/OLMo-3-1025-7B@stage3-step1000
- [ ] allenai/OLMo-3-1025-7B@stage3-step6000
- [ ] allenai/OLMo-3-1025-7B@stage3-step11921
- [ ] allenai/OLMo-3-1025-7B@main

---

### Job 219406: OLMo3 Only at 4° Resolution  
- **Status**: Running on dws-09
- **Walltime Used**: ~29 minutes / 12 hours max
- **Resolution**: 4° (4,050 queries per model = ~35 sec per model)
- **Models**: 14 OLMo3 checkpoints

#### Completed (4° resolution):
✅ allenai/OLMo-3-1025-7B@stage1-step2000
✅ allenai/OLMo-3-1025-7B@stage1-step5000
✅ allenai/OLMo-3-1025-7B@stage1-step10000
✅ allenai/OLMo-3-1025-7B@stage1-step25000
✅ allenai/OLMo-3-1025-7B@stage1-step50000
✅ allenai/OLMo-3-1025-7B@stage1-step100000

#### Pending:
- [ ] allenai/OLMo-3-1025-7B@stage1-step200000 (and rest)

---

## Key Findings

### Geographic Knowledge Emergence (OLMo3 Checkpoints)
- **Step 2,000-10,000**: No geographic knowledge - outputs ~0.5 (uniform random)
- **Step 25,000+**: Expected to show first signs of continental recognition
- **Step 1M+**: Expected to show clear land/water distinction

### Model Size Comparison (2° resolution)
- **Qwen2.5-0.5B**: All green (predicts "Land" everywhere)
- **Qwen2.5-7B**: Clear continental recognition visible

---

## Output Locations
- Maps: `/ceph/adinchev/experiment/Generated models/*.png`
- Data: `/ceph/adinchev/experiment/Generated models/*_data.json`
- SLURM logs: `/ceph/adinchev/experiment/logs/blind_earth_*.log`

## Next Steps
Monitor jobs and compare results when complete:
```bash
squeue -u adinchev
ls -lht "Generated models/"
```
