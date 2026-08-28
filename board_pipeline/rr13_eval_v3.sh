#!/bin/bash
# rr13 INT8 matrix eval v3 (hardened):
# main30 -> method18 -> amp3; DC/C2 alternating; one directory per model.

set -Eeuo pipefail
umask 022
export LC_ALL=C

E=/mnt/ssd/eval_output/rr13
H=/mnt/ssd/hbm_rr13
IDX=/mnt/ssd/eval_index.txt
DATA=/mnt/ssd/nuscenes/v1.0-trainval
EVAL_BIN=/root/eval/eval_batch_inst
SCORE_PY=/root/eval/score_nuscenes.py
PY=/root/venv_eval/bin/python
LOCK_FILE="$E/.rr13_eval_v3.lock"
SUMMARY="$E/summary.txt"

EXPECTED_FRAMES=""
SAMPLER_PID=""
CURRENT_RUN=""
CURRENT_DIR=""
QUEUE_OK=0
QUEUE_WAITING=0
QUEUE_FAILED=0
QUEUE_COLLAPSED=0
QUEUE_SKIPPED=0

declare -A PHASE_EPOCH=()
declare -A PHASE_ISO=()

log_summary() {
  printf '%s %s\n' "$(date -Is)" "$*" >> "$SUMMARY"
}

vfe_of() {
  case "$1" in
    dc_*)  printf '%s\n' /root/eval/vfe_weights_dc_rr13 ;;
    c2_*)  printf '%s\n' /root/eval/vfe_weights_conv2d_rr13 ;;
    amp_*) printf '%s\n' /root/eval/vfe_weights_amp ;;
    *) return 64 ;;
  esac
}

queue_script_running() {
  local path="$1"
  ps -eo comm=,args= | awk -v path="$path" '
    $1 == "bash" && $2 == "bash" && $3 == path { found=1 }
    END { exit(found ? 0 : 1) }
  '
}

validate_vfe() {
  local dir="$1" f
  local files=(
    vfe_pfn_layers_0_linear_weight.bin
    vfe_pfn_layers_0_norm_bias.bin
    vfe_pfn_layers_0_norm_running_mean.bin
    vfe_pfn_layers_0_norm_running_var.bin
    vfe_pfn_layers_0_norm_weight.bin
  )
  [[ -d "$dir" ]] || return 1
  for f in "${files[@]}"; do
    [[ -s "$dir/$f" ]] || return 1
  done
}

write_status() {
  local dir="$1" run="$2" state="$3" reason="$4"
  local tmp="$dir/.status.json.tmp.$$"
  printf '{"run":"%s","status":"%s","reason":"%s","updated_at":"%s"}\n' \
    "$run" "$state" "$reason" "$(date -Is)" > "$tmp"
  mv -f -- "$tmp" "$dir/status.json"
}

write_phases() {
  local dir="$1" run="$2" key tmp
  tmp="$dir/.phases.json.tmp.$$"
  local order=(run_start smoke_end full_eval_start full_eval_end score_start score_end run_end aborted)
  {
    printf '{\n  "run": "%s"' "$run"
    for key in "${order[@]}"; do
      if [[ -n "${PHASE_EPOCH[$key]-}" ]]; then
        printf ',\n  "%s": [%s, "%s"]' "$key" "${PHASE_EPOCH[$key]}" "${PHASE_ISO[$key]}"
      fi
    done
    printf '\n}\n'
  } > "$tmp"
  mv -f -- "$tmp" "$dir/phases.json"
}

phase() {
  local dir="$1" run="$2" key="$3" stamp
  stamp="$(date '+%s.%N %Y-%m-%dT%H:%M:%S.%N%:z')"
  PHASE_EPOCH[$key]="${stamp%% *}"
  PHASE_ISO[$key]="${stamp#* }"
  write_phases "$dir" "$run"
}

sampler() {
  local csv="$1" ts load mem z
  local zones=(/sys/class/thermal/thermal_zone*/temp)
  local values=()

  while :; do
    ts="$(date +%s.%N)"
    values=()
    for z in "${zones[@]}"; do
      if ! read -r value < "$z"; then value=0; fi
      values+=("$value")
    done
    load="$(awk '{print $1}' /proc/loadavg)"
    mem="$(awk '/MemAvailable/{print int($2/1024); exit}' /proc/meminfo)"
    {
      printf '%s' "$ts"
      for value in "${values[@]}"; do printf ',%s' "$value"; done
      printf ',%s,%s\n' "$load" "$mem"
    } >> "$csv"
    sleep 1
  done
}

start_sampler() {
  local csv="$1" z
  local zones=(/sys/class/thermal/thermal_zone*/temp)
  {
    printf 'epoch'
    for z in "${zones[@]}"; do
      printf ',%s_mC' "$(basename "$(dirname "$z")")"
    done
    printf ',load1,mem_available_MB\n'
  } > "$csv"
  sampler "$csv" &
  SAMPLER_PID=$!
}

stop_sampler() {
  if [[ -n "${SAMPLER_PID:-}" ]]; then
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    SAMPLER_PID=""
  fi
}

hw_summ() {
  local dir="$1" run="$2" tmp
  tmp="$dir/.hw.json.tmp.$$"
  "$PY" - "$dir/hw.csv" "$tmp" "$run" <<'PY'
import csv, json, math, sys

src, dst, run = sys.argv[1:]
with open(src, newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
if not rows:
    raise SystemExit('no hardware samples')

thermal = [x for x in reader.fieldnames if x.startswith('thermal_zone') and x.endswith('_mC')]
if not thermal:
    raise SystemExit('no thermal columns')

def nums(name):
    values = [float(r[name]) for r in rows]
    if not values or not all(math.isfinite(v) for v in values):
        raise SystemExit(f'invalid values in {name}')
    return values

epoch = nums('epoch')
load = nums('load1')
mem = nums('mem_available_MB')
temp_values = [nums(name) for name in thermal]
out = {
    'run': run,
    'sampling_period_s': 1,
    'samples': len(rows),
    'epoch_start': epoch[0],
    'epoch_end': epoch[-1],
    'thermal_zones': thermal,
    'thermal_mC_mean': [round(sum(v) / len(v), 1) for v in temp_values],
    'thermal_mC_max': [max(v) for v in temp_values],
    'load1_mean': round(sum(load) / len(load), 3),
    'load1_max': max(load),
    'mem_available_MB_mean': round(sum(mem) / len(mem), 1),
    'mem_available_MB_min': min(mem),
}
with open(dst, 'w') as f:
    json.dump(out, f, indent=2, sort_keys=True)
    f.write('\n')
PY
  mv -f -- "$tmp" "$dir/hw.json"
}

validate_result() {
  local json_file="$1" expected="$2"
  "$PY" -c '
import json, sys
p, expected = sys.argv[1], int(sys.argv[2])
with open(p) as f:
    d = json.load(f)
r = d.get("results")
if not isinstance(r, dict) or len(r) != expected:
    raise SystemExit(2)
if not all(isinstance(v, list) for v in r.values()):
    raise SystemExit(3)
print(sum(len(v) for v in r.values()))
' "$json_file" "$expected"
}

validate_timing() {
  local csv_file="$1" expected="$2"
  "$PY" -c '
import csv, math, sys
p, expected = sys.argv[1], int(sys.argv[2])
with open(p, newline="") as f:
    rows = list(csv.DictReader(f))
required = {"frame", "decode_nms_ms", "load_vfe_ms", "bpu_submit_done_ms", "wall_ms"}
if len(rows) != expected or not rows or not required.issubset(rows[0]):
    raise SystemExit(2)
if [int(r["frame"]) for r in rows] != list(range(expected)):
    raise SystemExit(3)
for r in rows:
    if not all(math.isfinite(float(r[k])) and float(r[k]) >= 0 for k in required - {"frame"}):
        raise SystemExit(4)
' "$csv_file" "$expected"
}

validate_metrics() {
  local metrics="$1"
  "$PY" -c '
import json, math, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
m, n = d.get("mean_ap"), d.get("nd_score")
if not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in (m, n)):
    raise SystemExit(2)
print(f"mAP: {m:.4f} NDS: {n:.4f}")
' "$metrics"
}

write_provenance() {
  local dir="$1" hbm="$2" vfe="$3" tmp
  tmp="$dir/.provenance.sha256.tmp.$$"
  {
    sha256sum "$EVAL_BIN" "$SCORE_PY" "$IDX" "$0"
    md5sum "$hbm"
    sha256sum "$vfe"/*.bin
    "$PY" -c 'import importlib.metadata as m; print("nuscenes-devkit==" + m.version("nuscenes-devkit"))'
    date -Is
  } > "$tmp"
  mv -f -- "$tmp" "$dir/provenance.sha256"
}

is_complete() {
  local dir="$1"
  [[ -s "$dir/status.json" && -s "$dir/full.json" && -s "$dir/full.json.timing.csv" \
     && -s "$dir/score/metrics_summary.json" && -s "$dir/hw.json" && -s "$dir/phases.json" ]] || return 1
  "$PY" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "OK" else 1)' "$dir/status.json" \
    && validate_result "$dir/full.json" "$EXPECTED_FRAMES" >/dev/null \
    && validate_timing "$dir/full.json.timing.csv" "$EXPECTED_FRAMES" \
    && validate_metrics "$dir/score/metrics_summary.json" >/dev/null \
    && "$PY" -c 'import json,sys; json.load(open(sys.argv[1]))' "$dir/hw.json" \
    && "$PY" -c 'import json,sys; json.load(open(sys.argv[1]))' "$dir/phases.json"
}

fail_run() {
  local dir="$1" run="$2" reason="$3"
  stop_sampler
  phase "$dir" "$run" run_end || true
  hw_summ "$dir" "$run" || true
  write_status "$dir" "$run" FAILED "$reason"
  log_summary "$run FAILED reason=$reason"
  QUEUE_FAILED=$((QUEUE_FAILED + 1))
  CURRENT_RUN=""
  CURRENT_DIR=""
}

eval_one() {
  local run="$1" hbm="$H/$1.hbm" vfe dir smoke_tmp full_tmp dets rc score_text
  local score_tmp score_attempt old_score

  if [[ ! -s "$hbm" ]]; then
    log_summary "$run WAITING missing_hbm"
    QUEUE_WAITING=$((QUEUE_WAITING + 1))
    return 0
  fi

  vfe="$(vfe_of "$run")" || {
    log_summary "$run FAILED unknown_vfe_prefix"
    QUEUE_FAILED=$((QUEUE_FAILED + 1))
    return 0
  }
  if ! validate_vfe "$vfe"; then
    log_summary "$run FAILED invalid_vfe_dir=$vfe"
    QUEUE_FAILED=$((QUEUE_FAILED + 1))
    return 0
  fi

  dir="$E/$run"
  mkdir -p "$dir"
  if is_complete "$dir"; then
    printf 'SKIP %s\n' "$run"
    log_summary "$run SKIP verified_complete"
    QUEUE_SKIPPED=$((QUEUE_SKIPPED + 1))
    return 0
  fi

  CURRENT_RUN="$run"
  CURRENT_DIR="$dir"
  PHASE_EPOCH=()
  PHASE_ISO=()
  write_status "$dir" "$run" RUNNING started
  write_provenance "$dir" "$hbm" "$vfe"
  phase "$dir" "$run" run_start
  start_sampler "$dir/hw.csv"

  smoke_tmp="$dir/.smoke.$$.json"
  if "$EVAL_BIN" "$hbm" "$vfe/" "$IDX" "$DATA/" "$smoke_tmp" 10 0 > "$dir/smoke.log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  phase "$dir" "$run" smoke_end
  if [[ "$rc" -ne 0 ]]; then
    fail_run "$dir" "$run" "smoke_eval_exit_$rc"
    return 0
  fi
  if ! dets="$(validate_result "$smoke_tmp" 10)"; then
    fail_run "$dir" "$run" smoke_json_invalid
    return 0
  fi
  mv -f -- "$smoke_tmp" "$dir/smoke.json"
  if [[ -f "$smoke_tmp.timing.csv" ]]; then
    mv -f -- "$smoke_tmp.timing.csv" "$dir/smoke.json.timing.csv"
  fi
  log_summary "$run SMOKE dets=$dets"

  if [[ "$dets" -lt 500 ]]; then
    stop_sampler
    phase "$dir" "$run" run_end
    if ! hw_summ "$dir" "$run"; then
      write_status "$dir" "$run" FAILED collapsed_hw_summary_failed
      log_summary "$run FAILED reason=collapsed_hw_summary_failed"
      QUEUE_FAILED=$((QUEUE_FAILED + 1))
    else
      write_status "$dir" "$run" COLLAPSED "smoke_dets_$dets"
      log_summary "$run COLLAPSED smoke_dets=$dets"
      QUEUE_COLLAPSED=$((QUEUE_COLLAPSED + 1))
    fi
    CURRENT_RUN=""
    CURRENT_DIR=""
    return 0
  fi

  full_tmp="$dir/.full.$$.json"
  phase "$dir" "$run" full_eval_start
  if "$EVAL_BIN" "$hbm" "$vfe/" "$IDX" "$DATA/" "$full_tmp" 0 0 > "$dir/full.log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  phase "$dir" "$run" full_eval_end
  if [[ "$rc" -ne 0 ]]; then
    fail_run "$dir" "$run" "full_eval_exit_$rc"
    return 0
  fi
  if ! validate_result "$full_tmp" "$EXPECTED_FRAMES" >/dev/null; then
    fail_run "$dir" "$run" full_json_invalid_or_incomplete
    return 0
  fi
  if [[ ! -s "$full_tmp.timing.csv" ]] || ! validate_timing "$full_tmp.timing.csv" "$EXPECTED_FRAMES"; then
    fail_run "$dir" "$run" timing_csv_invalid_or_incomplete
    return 0
  fi
  mv -f -- "$full_tmp" "$dir/full.json"
  mv -f -- "$full_tmp.timing.csv" "$dir/full.json.timing.csv"

  phase "$dir" "$run" score_start
  score_tmp="$dir/.score.$$"
  score_text=""
  rc=1
  for score_attempt in 1 2; do
    mkdir -p "$score_tmp"
    if MPLBACKEND=Agg "$PY" "$SCORE_PY" "$dir/full.json" --output_dir "$score_tmp" \
         > "$dir/score.log" 2>&1; then
      if score_text="$(validate_metrics "$score_tmp/metrics_summary.json")"; then
        rc=0
        break
      fi
    fi
    printf '\nscore attempt %d failed at %s\n' "$score_attempt" "$(date -Is)" >> "$dir/score.log"
    sleep 2
  done
  phase "$dir" "$run" score_end
  if [[ "$rc" -ne 0 ]]; then
    fail_run "$dir" "$run" score_failed_or_metrics_invalid
    return 0
  fi

  if [[ -e "$dir/score" ]]; then
    old_score="$dir/score.previous.$(date +%s)"
    mv -- "$dir/score" "$old_score"
  fi
  mv -- "$score_tmp" "$dir/score"
  stop_sampler
  phase "$dir" "$run" run_end
  if ! hw_summ "$dir" "$run"; then
    write_status "$dir" "$run" FAILED hw_summary_failed
    log_summary "$run FAILED reason=hw_summary_failed"
    QUEUE_FAILED=$((QUEUE_FAILED + 1))
    CURRENT_RUN=""
    CURRENT_DIR=""
    return 0
  fi

  write_status "$dir" "$run" OK verified
  log_summary "$run OK $score_text"
  QUEUE_OK=$((QUEUE_OK + 1))
  CURRENT_RUN=""
  CURRENT_DIR=""
}

cleanup() {
  local rc=$?
  trap - EXIT
  stop_sampler
  if [[ -n "${CURRENT_DIR:-}" && -n "${CURRENT_RUN:-}" ]]; then
    phase "$CURRENT_DIR" "$CURRENT_RUN" aborted || true
    hw_summ "$CURRENT_DIR" "$CURRENT_RUN" || true
    write_status "$CURRENT_DIR" "$CURRENT_RUN" ABORTED "queue_exit_$rc" || true
    log_summary "$CURRENT_RUN ABORTED queue_exit=$rc" || true
  fi
  exit "$rc"
}

selftest() {
  local td lock score
  td="$(mktemp -d /tmp/rr13_eval_v3_selftest.XXXXXX)"
  trap 'rm -rf -- "$td"' RETURN

  printf 'epoch,thermal_zone0_mC,thermal_zone1_mC,load1,mem_available_MB\n' > "$td/hw.csv"
  printf '1,10,20,0.5,900\n2,20,40,1.5,800\n' >> "$td/hw.csv"
  hw_summ "$td" demo
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["mem_available_MB_min"] == 800; assert d["thermal_mC_mean"] == [15.0,30.0]' "$td/hw.json"

  PHASE_EPOCH=()
  PHASE_ISO=()
  phase "$td" demo run_start
  phase "$td" demo run_end
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert "run_start" in d and "run_end" in d' "$td/phases.json"

  "$PY" -c 'import json,sys; json.dump({"meta":{},"results":{str(i):[] for i in range(10)}},open(sys.argv[1],"w"))' "$td/result.json"
  [[ "$(validate_result "$td/result.json" 10)" == 0 ]]
  if validate_result "$td/result.json" 9 >/dev/null 2>&1; then return 1; fi

  "$PY" -c 'import json,sys; json.dump({"mean_ap":0.25,"nd_score":0.35},open(sys.argv[1],"w"))' "$td/metrics.json"
  score="$(validate_metrics "$td/metrics.json")"
  [[ "$score" == 'mAP: 0.2500 NDS: 0.3500' ]]

  lock="$td/test.lock"
  (
    exec 8> "$lock"
    flock -n 8
    if (exec 7> "$lock"; flock -n 7); then exit 1; fi
  )
  printf 'SELFTEST_OK\n'
}

main() {
  local script_copy tmp_copy total
  mkdir -p "$E"
  exec 9> "$LOCK_FILE"
  if ! flock -n 9; then
    printf 'Another rr13 v3 queue owns %s; refusing to start.\n' "$LOCK_FILE" >&2
    exit 75
  fi

  if pgrep -x eval_batch_inst >/dev/null; then
    printf 'eval_batch_inst is already running; refusing to start.\n' >&2
    exit 76
  fi
  if queue_script_running /root/eval/rr13_eval_v2.sh; then
    printf 'rr13_eval_v2.sh is already running; refusing to start.\n' >&2
    exit 77
  fi

  [[ -x "$EVAL_BIN" && -s "$SCORE_PY" && -x "$PY" && -s "$IDX" && -d "$DATA" ]] || {
    printf 'Required eval dependency is missing.\n' >&2
    exit 78
  }
  EXPECTED_FRAMES="$(head -n 1 "$IDX")"
  [[ "$EXPECTED_FRAMES" =~ ^[0-9]+$ && "$EXPECTED_FRAMES" -eq 6019 ]] || {
    printf 'Unexpected eval frame count: %s\n' "$EXPECTED_FRAMES" >&2
    exit 79
  }
  validate_vfe /root/eval/vfe_weights_dc_rr13
  validate_vfe /root/eval/vfe_weights_conv2d_rr13
  validate_vfe /root/eval/vfe_weights_amp

  if [[ "${RR13_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    printf 'PREFLIGHT_OK expected_frames=%s hbm_present=%s thermal_zones=%s\n' \
      "$EXPECTED_FRAMES" \
      "$(find "$H" -maxdepth 1 -type f -name '*.hbm' | wc -l)" \
      "$(find /sys/class/thermal -maxdepth 1 -type l -name 'thermal_zone*' | wc -l)"
    return 0
  fi

  script_copy="$E/rr13_eval_v3.sh"
  tmp_copy="$E/.rr13_eval_v3.sh.tmp.$$"
  cp -- "$0" "$tmp_copy"
  chmod 0755 "$tmp_copy"
  mv -f -- "$tmp_copy" "$script_copy"

  trap cleanup EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM

  log_summary "QUEUE_START script_sha256=$(sha256sum "$0" | awk '{print $1}') expected_frames=$EXPECTED_FRAMES"

  if [[ -n "${RR13_ONLY_RUN:-}" ]]; then
    if [[ ! "$RR13_ONLY_RUN" =~ ^(dc|c2)_rr13_maxpct_pc_asym_s(42|1|2|7|59)_(256|512|1024)$ \
       && ! "$RR13_ONLY_RUN" =~ ^(dc|c2)_rr13_(max_pc_asym|maxpct_pt_asym|kl)_s(42|1|2)_256$ \
       && ! "$RR13_ONLY_RUN" =~ ^amp_ampv1_maxpct_pc_asym_(42|1|2)_256$ \
       && ! "$RR13_ONLY_RUN" =~ ^dc_kl_maskfix$ \
       && ! "$RR13_ONLY_RUN" =~ ^dc_kl_masktrunk$ ]]; then
      printf 'RR13_ONLY_RUN is not a planned run: %s\n' "$RR13_ONLY_RUN" >&2
      return 80
    fi
    log_summary "QUEUE_MODE only_run=$RR13_ONLY_RUN"
    eval_one "$RR13_ONLY_RUN"
    total=$((QUEUE_OK + QUEUE_WAITING + QUEUE_FAILED + QUEUE_COLLAPSED + QUEUE_SKIPPED))
    log_summary "QUEUE_FINISH total=$total ok=$QUEUE_OK skipped=$QUEUE_SKIPPED collapsed=$QUEUE_COLLAPSED failed=$QUEUE_FAILED waiting=$QUEUE_WAITING"
    [[ "$QUEUE_FAILED" -eq 0 ]]
    return
  fi

  for z in 256 512 1024; do
    for s in 42 1 2 7 59; do
      eval_one "dc_rr13_maxpct_pc_asym_s${s}_${z}"
      eval_one "c2_rr13_maxpct_pc_asym_s${s}_${z}"
    done
  done
  for cfg in max_pc_asym maxpct_pt_asym kl; do
    for s in 42 1 2; do
      eval_one "dc_rr13_${cfg}_s${s}_256"
      eval_one "c2_rr13_${cfg}_s${s}_256"
    done
  done
  for s in 42 1 2; do
    eval_one "amp_ampv1_maxpct_pc_asym_${s}_256"
  done

  total=$((QUEUE_OK + QUEUE_WAITING + QUEUE_FAILED + QUEUE_COLLAPSED + QUEUE_SKIPPED))
  log_summary "QUEUE_FINISH total=$total ok=$QUEUE_OK skipped=$QUEUE_SKIPPED collapsed=$QUEUE_COLLAPSED failed=$QUEUE_FAILED waiting=$QUEUE_WAITING"
  if [[ "$QUEUE_FAILED" -ne 0 ]]; then
    return 1
  fi
}

if [[ "${RR13_SELFTEST:-0}" == 1 ]]; then
  selftest
elif [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
