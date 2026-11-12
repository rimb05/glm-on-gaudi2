#!/bin/bash
# The batch of twin benchmarks behind the layout findings in the README (per-layer slope, sliced experts against
# expert parallelism, the router lever, 4 stages x 4 cards, one server alone). Run from the workstation; needs
# the twins twin26 and twin26-nomtp under /workspace/glm53/twins on both servers (convert/make_twin.py) and all
# cards free. Every run appends a line to /workspace/glm53/results/bench.tsv on the server that ran it.
. "$(dirname "$0")/lib.sh"
cd "$REPO"
T=/workspace/glm53/twins
node_ssh 1 'cd /workspace/glm53 && sudo -n docker run --rm --memory 40g --cpus 16 --entrypoint bash -v /workspace/glm53:/workspace/glm53 -w /workspace/glm53 glm53-serve:dev1 -c "python3 code/convert/make_twin.py --small models/GLM-5.3-3.5B-A3.3B-small --out twins/twin14-nomtp --layers 14 --experts 256 --link-from twins/twin26-nomtp --threads 16 2>&1 | grep -v ^Calling | tail -1; chown -R 1000:1000 twins/twin14-nomtp"'
F="GLM_ENGINE_SKIP_EMPTY=1 GLM_PP_FASTLINK=1"
EP="--enable-expert-parallel --enable-ep-weight-filter"
echo "### a twin14 PP2xTP8 fast";      bin/bench2.sh twin14-pp2tp8-fast "$F" $T/twin14-nomtp 2 8 6,8 $EP 2>&1 | tail -1
echo "### c twin26 PP2xTP8 fast, experts sliced (no EP)"; bin/bench2.sh twin26-pp2tp8-fast-noep "$F" $T/twin26 2 8 14,12 2>&1 | tail -1
echo "### d twin26 PP2xTP8 fast + router fast"; bin/bench2.sh twin26-pp2tp8-fast-rf "$F GLM_ROUTER_FAST=1" $T/twin26 2 8 14,12 $EP 2>&1 | tail -1
echo "### e twin26 PP4xTP4 fast";      bin/bench2.sh twin26-pp4tp4-fast "$F" $T/twin26 4 4 6,8,8,4 $EP 2>&1 | tail -1
echo "### b twin26 one server TP8 EP8 (async scheduling)"
node_ssh 1 'for c in $(sudo -n docker ps -a --format "{{.Names}}" | grep "^glm53-" | grep -v twin-gen); do sudo -n docker rm -f $c >/dev/null 2>&1; done'
node_ssh 0 "/workspace/glm53/serving/bench1.sh twin26-ep8 '' $T/twin26 $EP 2>&1 | tail -1"
echo "### b2 twin14 one server TP8 EP8 (async scheduling)"
node_ssh 0 "/workspace/glm53/serving/bench1.sh twin14-ep8 '' $T/twin14-nomtp $EP 2>&1 | tail -1"
echo "### done"
