from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
MODELS=ROOT/'models'
TABULAR_MODELS=MODELS/'tabular'
IMAGING_MODELS=MODELS/'imaging'
HYBRID_MODELS=MODELS/'hybrid'
[ p.mkdir(parents=True, exist_ok=True) for p in [MODELS,TABULAR_MODELS,IMAGING_MODELS,HYBRID_MODELS] ]
