# Correctness-only provider overlay; the runtime/builder image must already
# contain the current SDK and installed persistent Transolver runner fixture.
ARG NATIVE_QA_IMAGE
FROM ${NATIVE_QA_IMAGE}

ARG SOURCE_SHA
COPY qa/native_inference/cfd-overlay-requirements.lock /tmp/cfd-overlay.lock
COPY qa/native_inference/cfd_overlay_guard.py /opt/physicsnemo/image/cfd_overlay_guard.py
RUN python3 -m pip install --no-deps --require-hashes -r /tmp/cfd-overlay.lock \
    && python3 -m pip install --no-deps --no-build-isolation \
        'nvidia-physicsnemo-cfd @ git+https://github.com/NVIDIA/physicsnemo-cfd.git@921f14dc2ac14c04aabffaba3290deb792379dd8' \
    && apt-get update \
    && apt-get install -y --no-install-recommends time \
    && rm -rf /var/lib/apt/lists/* \
    && python3 /opt/physicsnemo/image/cfd_overlay_guard.py \
        --contract /opt/physicsnemo/image/geotransolver-environment.json \
        --verify /opt/physicsnemo/image/base-environment.json \
        --output /opt/physicsnemo/image/cfd-qa-environment.json \
    && python3 -c 'import pyvista; from physicsnemo.cfd.evaluation.benchmarks.engine import run_benchmark; from physicsnemo.cfd.evaluation.models.wrappers.transolver.wrapper import TransolverWrapper'

COPY qa/native_inference /opt/native-cfd-qa/qa/native_inference
COPY qa/scripts/cfd_parity_contract.py qa/scripts/run_native_cfd_reference.py \
     qa/scripts/run_native_cfd_job.py /opt/native-cfd-qa/qa/scripts/
COPY qa/inference/cfd_parity_surface_run1_transolver.json /opt/native-cfd-qa/qa/inference/
COPY plugins/physicsnemo-cfd-surface-benchmark/examples/public_run_1_transolver_request.json \
     /opt/native-cfd-qa/plugins/physicsnemo-cfd-surface-benchmark/examples/

ENV PYTHONPATH="/opt/native-cfd-qa"
ENV PATH="/opt/physicsnemo-native-runner/bin:${PATH}"
RUN python3 -c 'from pathlib import Path; import json,re,sys; root=Path("/opt/native-cfd-qa"); sha=sys.argv[1]; assert re.fullmatch(r"[0-9a-f]{40}",sha); (root/"source.json").write_text(json.dumps({"source_sha":sha})); from qa.native_inference.cfd_reference import verify_provider; from qa.scripts.cfd_parity_contract import read_json_object; verify_provider(read_json_object(root/"qa/inference/cfd_parity_surface_run1_transolver.json"))' "${SOURCE_SHA}" \
    && test -x /opt/physicsnemo-native-runner/bin/physicsnemo-transolver-runner

LABEL org.opencontainers.image.revision="${SOURCE_SHA}"
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["python3", "-m", "qa.scripts.run_native_cfd_job", "--help"]
