#!/bin/bash
# ========================================================================= #
# Filename:                                                                 #
#    run.bash                                                               #
#                                                                           #
# Description:                                                              # 
#    Convenient running script                                              #
# ========================================================================= #
export PYTHONPATH=${PWD}:$PYTHONPATH

print_usage()
{
    echo "Convenience script to load parameters and run an agent"
    echo "Usage (baseline model): ./run.bash -b <baseline>"
    echo "Usage (other model): ./run.bash -s <script> -c <config> [OPTIONS]"
    echo ""
    echo " Arguments:"
    echo "  -h, --help      |  shows this message"
    echo "  -b, --baseline  |  run a baseline model ['random', 'sac', 'mpc']"
    echo "  -c, --config    |  configuration file, passed as -c to script"
    echo "  -i, --ip        |  ip address to pass to script"
    echo "  -p, --port      |  port to pass to script"
    echo "  -s, --script    |  python script to run"
    echo "  -t, --type      |  type of node, passed as -t to script"
    echo "  -m, --margin    |  safety margin for SafePO"
    echo "  -d, --dirhash   |  directory hash; used for logging on distributed clusters"
    echo "  -r, --runtime   |  runtime environment -- local | phoebe"
}

while [[ $# -gt 0 ]]; do
  opt="$1"
  shift;
  current_arg="$1"
  case "$opt" in
    "-h"|"--help"       ) print_usage; exit 1;;
    "-b"|"--baseline"   ) BASELINE="$1"; shift;;
    "-c"|"--config"     ) CONFIG="$1"; shift;;
    "-i"|"--ip"         ) IP="$1"; shift;;
    "-p"|"--port"       ) PORT="$1"; shift;;
    "-s"|"--script"     ) SCRIPT="$1"; shift;;
    "-t"|"--type"       ) TYPE="$1"; shift;;
    "-m"|"--margin"     ) MARGIN="$1"; shift;;
    "-d"|"--dirhash"    ) DIRHASH="$1"; shift;;
    "-r"|"--runtime"    ) RUNTIME="$1"; shift;;
    *                   ) echo "Invalid option: \""$opt"\"" >&2
                        exit 1;;
  esac
done

if [ "$DIRHASH" == "" ]; then DIRHASH=None; fi

if [ "$BASELINE" == "" ] && [ "$SCRIPT" = "" ]; then
    echo "Invalid arguments. Please specify a baseline (with -b) or script (with -s)"
    exit 1;
fi

if [ "$BASELINE" != "" ]; then
    if [ "$BASELINE" == "random" ]; then
        python3 scripts/runner_random.py configs/params_random.yaml
        exit 0;
    elif [ "$BASELINE" == "sac" ]; then
        python3 scripts/runner_sac.py --yaml configs/params-sac.yaml --dirhash $DIRHASH --runtime $RUNTIME
        exit 0;
    elif [ "$BASELINE" = "safesac" ]; then
        python3 scripts/runner_safesac.py --yaml configs/params-safesac.yaml --safety_margin $MARGIN --dirhash $DIRHASH --runtime $RUNTIME
        exit 0;
    elif [ "$BASELINE" = "spar" ]; then
        python3 scripts/runner_spar.py --yaml configs/params-spar.yaml --safety_margin $MARGIN --dirhash $DIRHASH --runtime $RUNTIME
        exit 0;
    elif [ "$BASELINE" = "safesac_random" ]; then
        python3 scripts/runner_safesac.py --yaml configs/params-safesac-random.yaml --safety_margin $MARGIN --dirhash $DIRHASH --runtime $RUNTIME
        exit 0;
    elif [ "$BASELINE" == "mpc" ]; then
        python3 scripts/runner_mpc.py configs/params_mpc.yaml
        exit 0;
    else
        echo "Expecting baseline to be in ['random', 'sac', 'mpc', 'safesac', 'safesac_dp', 'safesac_random']"
        exit 1;
    fi
fi

if [ "$CONFIG" == "" ]; then
    echo "Please include a configuration file with -c or --config"
fi

COMMAND="${SCRIPT} -c ${CONFIG}"

if [ "$IP" != "" ]; then COMMAND="${COMMAND} -i ${IP}"; fi
if [ "$PORT" != "" ]; then COMMAND="${COMMAND} -p ${PORT}"; fi
if [ "$TYPE" != "" ]; then COMMAND="${COMMAND} -t ${TYPE}"; fi

python3 ${COMMAND}
