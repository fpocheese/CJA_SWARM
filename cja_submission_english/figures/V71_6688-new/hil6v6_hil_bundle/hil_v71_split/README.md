# V71 split HIL prototype

This directory separates the V71 closed loop into:

- `hil_env_server.py`: server-side environment, defenders, aircraft dynamics,
  hit/kill logic, observation masking, and terminal PN action wrapper.
- `hil_policy_node.py`: NX-side offensive policy inference for one aircraft.
- `hil_protocol.py`: JSON-lines TCP protocol helpers.

The split preserves the original V71 evaluation chain:

```text
FOVPenetrationEnv -> PhaseMaskedFOVWrapper(v65_strict_los)
                  -> TerminalPNActionWrapper(gain=3.0, max_action=0.8)
```

Each policy node receives one masked 30-D observation and returns one raw
3-D actor action. The environment server applies terminal PN guidance before
calling `env.step`, matching the original reproduction scripts.

## Local smoke test on the server

Start the environment:

```bash
cd ~/000000GSY_mutiUAV/swarm_attack_v2
~/miniconda3/envs/rlgpu/bin/python hil_v71_split/hil_env_server.py \
  --case 4v4 --host 127.0.0.1 --port 5500 --seed 90000 \
  --out /tmp/v71_hil_4v4_summary.json
```

Start four policy nodes in another shell:

```bash
cd ~/000000GSY_mutiUAV/swarm_attack_v2
for i in 0 1 2 3; do
  ~/miniconda3/envs/rlgpu/bin/python hil_v71_split/hil_policy_node.py \
    --agent-id "$i" --source-agent "$i" \
    --server-host 127.0.0.1 --server-port 5500 &
done
wait
```

For `6v6` or `8v8`, start six or eight policy nodes and use
`--source-agent $((i % 4))` because V71 has four trained actor checkpoints
that are cloned cyclically in the existing evaluation scripts.
