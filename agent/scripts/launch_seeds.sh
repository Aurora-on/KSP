#!/usr/bin/env bash
# Launch independent training runs on resources already available on this host.
set -eo pipefail

usage() {
    cat <<'EOF'
Usage: agent/scripts/launch_seeds.sh [--seeds 31,32,33] [--device cuda|cpu]
       [--gpus 0,1] [--jobs-per-device 1] [--out-root agent/runs/s3-l40]
       [--config FILE] [--data DIR] [--world-model FILE] [--validation FILE] [--resume]

GPU IDs are CUDA_VISIBLE_DEVICES identifiers. Without --gpus, inherit the
current CUDA_VISIBLE_DEVICES mask (default 0 if unset). CPU uses one device.
Run directories are OUT_ROOT/seed-N; logs append to OUT_ROOT/seed-N.log.
--resume continues existing runs and starts missing seeds.
EOF
}

fail() { echo "$*" >&2; exit 2; }
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo_root"
seeds=31,32,33
device=cuda
gpu_list=${CUDA_VISIBLE_DEVICES-0}
jobs_per_device=1
out_root=agent/runs/s3-l40
config=agent/configs/s3_nominal.json
data=agent/data/s3-nominal-prefill
world_model=agent/runs/s1-contact-seed31/best.pt
validation=agent/data/s3-validation.json
resume=0

while (($#)); do
    case "$1" in
        --resume) resume=1; shift; continue ;;
        --help|-h) usage; exit 0 ;;
    esac
    (($# >= 2)) && [[ -n $2 ]] || fail "missing argument for $1"
    case "$1" in
        --seeds) seeds=$2 ;;
        --device) device=$2 ;;
        --gpus) gpu_list=$2 ;;
        --jobs-per-device) jobs_per_device=$2 ;;
        --out-root) out_root=$2 ;;
        --config) config=$2 ;;
        --data) data=$2 ;;
        --world-model) world_model=$2 ;;
        --validation) validation=$2 ;;
        *) fail "unknown argument: $1" ;;
    esac
    shift 2
done

[[ $jobs_per_device =~ ^[1-9][0-9]*$ ]] || fail "jobs per device must be a positive integer"
[[ $seeds =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || fail "seeds must be comma-separated nonnegative integers"
IFS=, read -r -a seed_ids <<< "$seeds"
if [[ $device == cpu ]]; then
    devices=(cpu)
elif [[ $device == cuda ]]; then
    [[ $gpu_list =~ ^[[:alnum:]/-]+(,[[:alnum:]/-]+)*$ ]] || fail "provide a nonempty comma-separated CUDA device list"
    IFS=, read -r -a devices <<< "$gpu_list"
else
    fail "device must be cuda or cpu"
fi

unique() {
    local seen=, item
    for item in "$@"; do
        [[ $seen != *",$item,"* ]] || fail "duplicate identifier: $item"
        seen+="$item,"
    done
}
unique "${seed_ids[@]}"
unique "${devices[@]}"
for path in "$config" "$world_model" "$validation" "$data/collection.json" "$data/replay.npz"; do
    [[ -r $path ]] || fail "missing or unreadable input: $path"
done
command -v mun-agent >/dev/null || fail "activate the environment containing mun-agent"
for seed in "${seed_ids[@]}"; do
    [[ $resume == 1 || ! -e $out_root/seed-$seed ]] || fail "run exists: $out_root/seed-$seed (use --resume to continue)"
done
mkdir -p "$out_root"
capacity=$(( jobs_per_device * ${#devices[@]} ))
pids=()
batch_seeds=()

interrupt() {
    local pid
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait || true
    exit "$1"
}
trap 'interrupt 130' INT
trap 'interrupt 143' TERM
wait_batch() {
    local result=0 i code
    for i in "${!pids[@]}"; do
        if wait "${pids[i]}"; then
            echo "Finished seed ${batch_seeds[i]}"
        else
            code=$?
            echo "Seed ${batch_seeds[i]} failed with status $code" >&2
            if (( result == 0 )); then result=$code; fi
        fi
    done
    pids=()
    batch_seeds=()
    return "$result"
}

for i in "${!seed_ids[@]}"; do
    seed=${seed_ids[i]}
    gpu=${devices[i % ${#devices[@]}]}
    run_dir=$out_root/seed-$seed
    log=$out_root/seed-$seed.log
    args=(train-policy --config "$config" --seed "$seed" --device "$device" --data "$data"
          --world-model "$world_model" --validation "$validation" --out "$run_dir")
    if [[ $resume == 1 && -d $run_dir ]]; then args+=(--resume); fi
    echo "Starting seed $seed on $device ($gpu), log $log"
    if [[ $device == cuda ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" mun-agent "${args[@]}" >>"$log" 2>&1 &
    else
        mun-agent "${args[@]}" >>"$log" 2>&1 &
    fi
    pids+=("$!")
    batch_seeds+=("$seed")
    if (( ${#pids[@]} == capacity )); then wait_batch; fi
done
wait_batch
