# Inference Service Onboarding

Use this page as the shortest entry point for the current plugin-based PhysicsNeMo Serve workflow.

## Service Consumer

1. List workflows.

   ```bash
   curl http://HOST:8080/v1/infer/workflows
   ```

2. Inspect one workflow.

   ```bash
   curl http://HOST:8080/v1/infer/<workflow_id>/schema
   curl http://HOST:8080/v1/infer/<workflow_id>/readiness
   ```

3. Submit a run.

   ```bash
   curl -X POST http://HOST:8080/v1/infer/<workflow_id>/run ...
   ```

4. Poll and fetch outputs.

   ```bash
   curl http://HOST:8080/v1/infer/<workflow_id>/<run_id>/status
   curl http://HOST:8080/v1/infer/<workflow_id>/<run_id>/results
   curl -OJ "http://HOST:8080/v1/infer/<workflow_id>/<run_id>/results?artifact=primary"
   ```

For details, start with [inference-service-user-guide.md](./inference-service-user-guide.md).

## Plugin Author

1. Scaffold a plugin.

   ```bash
   python scripts/plugin_dev.py init plugins/my-plugin
   python scripts/plugin_dev.py init plugins/my-default-plugin --pipeline default
   python scripts/plugin_dev.py init plugins/my-multipart-plugin --content-type multipart/form-data
   python scripts/plugin_dev.py init plugins/biology-demo --pipeline postprocess --runtime custom --executor-class python.gpu.biology --phase-executor prepare=python.cpu.biology --phase-executor postprocess=python.cpu.biology --phase-executor readiness=python.cpu.biology
   ```

2. Edit:
   - `plugin.yaml`
   - `workflow.py`
3. Run local checks.

   ```bash
   python scripts/plugin_dev.py check plugins/my-plugin
   python scripts/plugin_dev.py check-env plugins/my-plugin
   python scripts/plugin_dev.py run-example plugins/my-plugin
   ```

4. Bring up a local stack.

   ```bash
   python scripts/plugin_dev.py run-local plugins/my-plugin --dry-run
   python scripts/plugin_dev.py run-local plugins/my-plugin
   ```

For the full contract, start with [plugin-authoring-guide.md](./plugin-authoring-guide.md).
