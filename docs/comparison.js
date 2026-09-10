'use strict';
const byId = id => document.getElementById(id);
const element = (tag, text = '', className = '') => {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
};
const number = (value, decimals = 2) => Number.isFinite(value) ? value.toFixed(decimals) : '未测';
const statusText = {passed: '抵达终点', failed: '未完成', pending: '待测'};
function link(text, url) {
  const node = element('a', text);
  const parsed = new URL(url, location.href);
  if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('Unsupported source URL');
  node.href = parsed.href;
  return node;
}
function paragraphs(parent, items, className = '') {
  for (const item of items || []) parent.append(element('p', item, className));
}
async function main() {
  const response = await fetch('data/task_a_comparison.json');
  if (!response.ok) throw new Error('Data unavailable');
  const data = await response.json();
  if (data.schema_version !== 1 || !Array.isArray(data.task_a_runs)) throw new Error('Unexpected schema');
  const runs = data.task_a_runs;
  if (new Set(runs.map(row => row.id)).size !== runs.length) throw new Error('Duplicate runs');
  for (const row of runs) {
    if (!Object.hasOwn(statusText, row.status)) throw new Error('Unknown result');
    if (row.status !== 'pending' && (!Number.isFinite(row.max_forward_m) || !Number.isFinite(row.sim_seconds))) throw new Error('Missing recorded metric');
  }
  let metric = 'distance';
  let selected = runs.find(row => row.id === 'base_only_seed42_01')?.id || runs[0].id;
  function detail(row) {
    selected = row.id;
    const panel = byId('detail');
    panel.replaceChildren(element('h3', row.label));
    panel.append(element('p', `结果：${statusText[row.status]} · seed ${row.seed ?? '未记录'}`));
    panel.append(element('p', `最远进展 ${number(row.max_forward_m, 3)} m · ${row.completed ? '完成' : '运行至退出'} ${number(row.sim_seconds)} s · 控制步数 ${row.steps ?? '未记录'}`, 'facts'));
    panel.append(element('p', `本地原始进度分 ${number(row.score, 4)}；不是官方线上成绩。`, 'muted'));
    paragraphs(panel, row.notes);
    if (row.source_url) panel.append(link('查看原始结果', row.source_url));
    if (row.video_url) { panel.append(document.createTextNode(' · ')); panel.append(link('查看本次录像', row.video_url)); }
    for (const button of byId('bars').querySelectorAll('button')) button.setAttribute('aria-pressed', String(button.dataset.runId === selected));
  }
  function chart() {
    const shown = runs.filter(row => metric === 'time' ? row.status === 'passed' && row.completed === true : Number.isFinite(row.max_forward_m));
    const value = row => metric === 'time' ? row.sim_seconds : row.max_forward_m;
    const maximum = Math.max(1, ...shown.map(value));
    const unit = metric === 'time' ? 's' : 'm';
    byId('chart-note').textContent = metric === 'time'
      ? `仅显示 ${shown.length} 条抵达终点的运行，用时越短越好。其余 ${runs.length-shown.length} 条记录仍保留在下方完整表格中。`
      : `显示 ${shown.length} 条已有结果。进展越远越好；绿色为抵达终点，橙色为未完成。点击一行查看条件。`;
    byId('bars').replaceChildren();
    for (const row of shown) {
      const button = element('button', '', `bar-row ${row.status}`);
      button.type = 'button'; button.dataset.runId = row.id;
      button.setAttribute('aria-pressed', String(row.id === selected));
      button.setAttribute('aria-label', `${row.label}，${statusText[row.status]}，${number(value(row))} ${unit}，查看条件`);
      button.append(element('span', row.label, 'bar-name'));
      const track = element('span', '', 'bar-track'); track.setAttribute('aria-hidden', 'true');
      const fill = element('span', '', 'bar-fill'); fill.style.display = 'block'; fill.style.width = `${100*value(row)/maximum}%`;
      track.append(fill); button.append(track, element('span', `${number(value(row))} ${unit}`, 'bar-value'));
      button.addEventListener('click', () => detail(row)); byId('bars').append(button);
    }
    if (!shown.some(row => row.id === selected) && shown.length) detail(shown[shown.length-1]);
  }
  for (const button of document.querySelectorAll('[data-metric]')) button.addEventListener('click', () => {
    metric = button.dataset.metric;
    for (const other of document.querySelectorAll('[data-metric]')) other.setAttribute('aria-pressed', String(other === button));
    chart();
  });
  for (const row of runs) {
    const tr = element('tr'); tr.append(element('th', row.label)); tr.firstChild.scope = 'row';
    const state = element('td'); state.append(element('span', statusText[row.status], `tag ${row.status}`));
    tr.append(state, element('td', `${number(row.max_forward_m, 3)} m`), element('td', `${number(row.sim_seconds)} s${row.status === 'failed' ? '（退出）' : ''}`));
    const source = element('td'); if (row.source_url) source.append(link('结果', row.source_url)); else source.textContent = '尚无结果';
    tr.append(source); byId('runs-table').append(tr);
  }
  for (const pair of data.comparisons || []) {
    const card = element('article', '', 'card'); card.append(element('h3', pair.label));
    if (Number.isFinite(pair.time_reduction_percent)) card.append(element('p', `两次成功运行的用时减少 ${number(pair.saved_sim_seconds)} s（${number(pair.time_reduction_percent)}%）。`));
    else if (Number.isFinite(pair.max_forward_delta_m)) card.append(element('p', `两次记录的最远进展相差 ${number(pair.max_forward_delta_m)} m。`));
    paragraphs(card, pair.notes, 'muted'); byId('pairs').append(card);
  }
  const ref = data.external_reference;
  if (ref?.rows) {
    const scroll = element('div', '', 'table-wrap'); scroll.tabIndex = 0; scroll.setAttribute('role', 'region'); scroll.setAttribute('aria-label', 'MuJoCo 独立测试指标，窄屏可横向滚动');
    const table = element('table'); const head = element('thead'); const labels = element('tr');
    for (const title of ['控制方法', '速度 RMSE (m/s)', '高度 RMSE (mm)', '机械活动量 (W)', '质量达标']) { const th = element('th', title); th.scope = 'col'; labels.append(th); }
    head.append(labels); table.append(head); const body = element('tbody');
    for (const row of ref.rows) {
      const tr = element('tr'); tr.append(element('th', row.label)); tr.firstChild.scope = 'row';
      for (const cell of [number(row.velocity_rmse_mps, 5), number(row.height_rmse_mm), number(row.mechanical_activity_w), `${row.quality_pass_n}/${row.quality_total_n} ${row.quality_unit}`]) tr.append(element('td', cell));
      body.append(tr);
    }
    table.append(body); scroll.append(table); byId('external').append(scroll);
    const protocol = ref.protocol;
    if (protocol) paragraphs(byId('external'), [protocol.task, `${protocol.robot}；${protocol.simulator}；${protocol.observation}。`, protocol.mechanical_activity_definition, protocol.aggregation].filter(Boolean), 'muted');
    paragraphs(byId('external'), ref.notes, 'muted');
    byId('external').append(link('原仓库完整报告', ref.source_url), document.createTextNode(' · '), link('独立算术审计', ref.audit_url));
  }
  for (const paper of data.papers || []) {
    const card = element('article', '', 'card'); const title = element('h3', '', 'paper-title'); title.append(link(paper.title, paper.source_url)); card.append(title);
    paragraphs(card, [`${paper.authors} · ${paper.year}`, paper.idea, `原平台：${paper.original_platform}`, paper.transfer_status]);
    if (paper.reported_results) card.append(element('p', paper.reported_results, 'muted'));
    card.append(element('p', paper.limits, 'muted')); byId('papers').append(card);
  }
  byId('loading').hidden = true; byId('content').hidden = false;
  chart(); detail(runs.find(row => row.id === selected));
}
main().catch(error => {
  byId('loading').hidden = true; byId('load-error').hidden = false;
  console.error('Comparison data could not be rendered:', error);
});
