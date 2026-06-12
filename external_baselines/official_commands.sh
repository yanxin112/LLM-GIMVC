export MICA_OFFICIAL_COMMAND='python run_official_mica_adapter.py --data "{data}" --output "{output}" --dataset "{dataset}" --missing-rate "{missing_rate}" --seed "{seed}" --device "{device}"'
export FREECSL_OFFICIAL_COMMAND='python run_official_freecsl_adapter.py --data "{data}" --output "{output}" --dataset "{dataset}" --missing-rate "{missing_rate}" --seed "{seed}" --device "{device}"'

# JGA_IMVC_OFFICIAL_COMMAND is intentionally unset: no official public
# implementation or executable entrypoint was found as of 2026-06-12.
