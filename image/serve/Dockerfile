# GLM-5.3 serving image: the pinned stock v0.30.0 base + gaudi-cookbook fixes + the glm-5-mxfp4 port's patches
# (DSA sparse attention, MTP/speculative-decoding fixes, decode levers; its MXFP4 switches stay off for FP8 experts).
ARG BASE=glm53-base:v0.30.0-058f107
FROM ${BASE}
ARG COOKBOOK_COMMIT=1540263772481343d92ce34a2734a34b44dbbda7
ENV COOKBOOK=/opt/gaudi-cookbook
RUN git clone https://github.com/thediymaker/gaudi-cookbook.git $COOKBOOK && \
    git -C $COOKBOOK checkout ${COOKBOOK_COMMIT} && \
    test "$(git -C $COOKBOOK rev-parse HEAD)" = "${COOKBOOK_COMMIT}"
# 1. cookbook fixes: the release profile, the two opt-ins the GLM recipe uses, and pipeline parallelism
RUN python3 $COOKBOOK/fixes/apply_all.py --profile v0.30.0 vllm-long-prefill-share glm47-streaming-text-markers pipeline-parallel
# 2. the port's runtime package and its 14 patches (applied on top of the fixed sources, in the port's order)
RUN BUILD=$(mktemp -d) && cp -r $COOKBOOK/ports/glm-5-mxfp4/glm5_hpu $BUILD/ && \
    PIP_NO_INDEX=1 pip install --no-deps --no-build-isolation --no-cache-dir $BUILD/glm5_hpu && rm -rf $BUILD
RUN python3 $COOKBOOK/ports/glm-5-mxfp4/patches/apply_all.py
# 3. everything must report "applied"
RUN python3 $COOKBOOK/fixes/apply_all.py --check --profile v0.30.0 vllm-long-prefill-share glm47-streaming-text-markers pipeline-parallel && \
    python3 $COOKBOOK/ports/glm-5-mxfp4/patches/apply_all.py --check && \
    python3 -c "import glm5_hpu.dsa, glm5_hpu.decode_lean; print('glm5_hpu ok')"
# This repository's own code is not part of the image: the containers mount /workspace/glm53 and
# serving/entry.sh applies the hooks at start, so a code change needs a sync and a restart, not a build.
WORKDIR /root
