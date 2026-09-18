// Exercise the actual viewer script with a large result, without a GPU or browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

async function check(count, fullBody = false) {
  const fields = Array.from({length: count}, (_, i) => [i - count / 2, 1, 2, 3]);
  const ids = fields.map((_, i) => i);
  const payload = {
    metadata: {}, summary: {point_count: count, source_cell_count: fullBody ? count : count * 2, backend: 'aoti'},
    prediction_mesh: {vertices: [[0,0,0],[1,0,0],[0,1,0]], triangles: [[0,1,2]], triangle_cell_ids: [0], source_cell_ids: ids},
    context_mesh: {vertices: [], triangles: []}, centers: fields.map((_, i) => [i, 0, 0]),
    fields, validation: {status: 'not_compared'}, hashes: {},
  };
  if (fullBody) payload.mesh_provenance = {label: 'Coarsened demonstration mesh', source_cell_count: 8828095, output_cell_count: count, method: 'Quadric decimation'};
  const nodes = new Map();
  function node(id) {
    if (!nodes.has(id)) nodes.set(id, {id, textContent: '', style: {}, classList: {add() {}, toggle() {}}, append() {}, replaceChildren() {}, addEventListener() {}, on() {}});
    return nodes.get(id);
  }
  node('result-data').textContent = JSON.stringify(payload);
  const plots = new Map(), layouts = new Map(), errors = [];
  const window = {};
  const document = {getElementById: node, createElement: () => node(Symbol())};
  const Plotly = {newPlot: async (target, traces, layout) => {plots.set(target.id, traces); layouts.set(target.id, layout)}, restyle() {}, relayout() {}};
  const html = fs.readFileSync(path.join(__dirname, '../demo/viewer.html'), 'utf8');
  const script = html.match(/<script>\s*'use strict';([\s\S]*?)<\/script>/)[1];
  await vm.runInNewContext(script, {document, Plotly, window, console: {error: e => errors.push(e.message)}});
  assert.deepEqual(errors, [], `Viewer must render ${count} cells without a JavaScript exception`);
  assert.equal(window.viewerReady, true);
  const surface = plots.get('patch-view')[0];
  assert.equal(surface.cmin, -count / 2);
  assert.equal(surface.cmax, count / 2 - 1);
  assert.equal(surface.intensitymode, 'cell');
  const series = plots.get('series-view')[0];
  assert.equal(series.y.length, count, 'Retain every field value');
  assert.equal(series.type, count > 2000 ? 'scattergl' : 'scatter');
  assert.equal(series.mode, count > 2000 ? 'lines' : 'lines+markers');
  if (fullBody) {
    assert.equal(node('surface-title').textContent, 'Full-car surface');
    assert.equal(node('coverage').textContent, `All ${count.toLocaleString()} cells · coarsened demonstration mesh`);
    assert.equal(node('scope').textContent, 'Complete coarsened car surface · numerical agreement, not full-resolution CFD accuracy');
    assert.equal(layouts.get('patch-view').scene.camera.up.z, 1, 'The complete car must use its vertical axis for the camera');
    assert.equal(layouts.get('patch-view').scene.camera.eye.x, .7, 'A closed surface must not derive the camera from canceling face normals');
    assert.equal(plots.get('patch-view')[1].marker.opacity, .015, 'Full-body hover markers must not wash out the field colors');
    assert.equal(node('context-heading').textContent, 'Complete car geometry');
    assert.equal(plots.get('context-view').length, 1, 'All-source views must not mark a single sample centroid');
  } else {
    assert.equal(node('surface-title').textContent, 'Predicted surface patch');
    assert.equal(node('coverage').textContent, `First ${count.toLocaleString()} of ${(count * 2).toLocaleString()} source cells`);
    assert.equal(plots.get('context-view').length, 2, 'Prefix views retain their sample location marker');
  }
}
(async () => {await check(65536, true); await check(131072); await check(75); console.log('Full-body, large and small viewer scripts passed.');})().catch(error => {console.error(error); process.exitCode = 1;});
