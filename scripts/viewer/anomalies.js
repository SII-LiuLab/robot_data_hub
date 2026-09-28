// This view consumes records only. Reason strings have no behavioral meaning.
export const STREAMS = ['left_eef', 'right_eef', 'left_gripper', 'right_gripper'];
export const containsTime = (record, ns) => record.start_ns === record.end_ns
  ? ns === record.start_ns : record.start_ns <= ns && ns < record.end_ns;
export function filterRecords(records, {episode = '', stream = '', reason = ''}) {
  return records.filter(record => (!episode || record.episode_id === episode) &&
    (!stream || record.stream === stream) && (!reason || record.reason === reason));
}
const seconds = ns => (ns / 1e9).toFixed(6) + ' s';
const interval = record => record.start_ns === record.end_ns ? seconds(record.start_ns)
  : `[${seconds(record.start_ns)}, ${seconds(record.end_ns)})`;
const label = record => `#${record.index} · ${record.episode_id} · ${record.stream} · ${record.reason} · ${interval(record)}`;

export class AnomalyReview {
  constructor(root, marks, onJump) {
    this.root = root;
    this.marks = marks;
    this.onJump = onJump;
    this.records = [];
    this.filtered = [];
    this.selected = null;
    this.episode = null;
    this.markButtons = [];
    this.get = id => root.querySelector('#' + id);
    for (const name of ['episode', 'stream', 'reason']) {
      this.get('anomaly-' + name).addEventListener('change', () => this.refresh());
    }
    this.get('anomaly-record').addEventListener('change', event => {
      if (event.target.value !== '') this.select(Number(event.target.value));
    });
    this.get('anomaly-prev').addEventListener('click', () => this.step(-1));
    this.get('anomaly-next').addEventListener('click', () => this.step(1));
  }
  setReport(records) {
    this.root.hidden = records === null;
    this.records = (records || []).map((record, id) => ({...record, id}));
    for (const [name, field] of [['episode', 'episode_id'], ['stream', 'stream'], ['reason', 'reason']]) {
      const select = this.get('anomaly-' + name);
      select.replaceChildren(new Option('All ' + name + 's', ''));
      for (const value of new Set(this.records.map(record => record[field]))) {
        select.add(new Option(value, value));
      }
      select.disabled = !this.records.length;
    }
    this.refresh();
  }
  refresh() {
    this.filtered = filterRecords(this.records, Object.fromEntries(
      ['episode', 'stream', 'reason'].map(name => [name, this.get('anomaly-' + name).value])));
    if (!this.filtered.some(record => record.id === this.selected)) this.selected = null;
    const select = this.get('anomaly-record');
    select.replaceChildren(new Option(this.filtered.length ? 'Choose an anomaly…' : 'No anomalies', ''));
    for (const record of this.filtered) select.add(new Option(label(record), String(record.id)));
    select.disabled = !this.filtered.length;
    this.get('anomaly-count').textContent = `${this.filtered.length} / ${this.records.length} records`;
    this.updateSelection();
    this.drawMarks();
  }
  updateSelection() {
    const index = this.filtered.findIndex(record => record.id === this.selected);
    this.get('anomaly-record').value = index < 0 ? '' : String(this.selected);
    this.get('anomaly-prev').disabled = index <= 0;
    this.get('anomaly-next').disabled = !this.filtered.length || index === this.filtered.length - 1;
    const record = this.records[this.selected];
    this.get('anomaly-detail').textContent = record
      ? `${record.stream} · ${record.reason} · ${record.start_ns === record.end_ns ? 'Point' : 'Interval'} ${interval(record)} · ${record.start_ns}–${record.end_ns} ns`
      : this.records.length ? 'Choose a record to jump to its start time.' : 'The report contains no anomalies.';
    for (const [button, item] of this.markButtons) {
      button.setAttribute('aria-pressed', String(item.id === this.selected));
    }
  }
  select(id) {
    const record = this.records[id];
    if (!record) return;
    this.selected = id;
    this.updateSelection();
    this.onJump(record);
  }
  step(direction) {
    const index = this.filtered.findIndex(record => record.id === this.selected);
    const record = this.filtered[index + direction];
    if (record) this.select(record.id);
  }
  setEpisode(episode) {
    this.episode = episode;
    if (this.selected !== null && this.records[this.selected].episode_id !== episode.episode_id) {
      this.selected = null;
    }
    this.drawMarks();
    this.updateSelection();
  }
  drawMarks() {
    this.marks.replaceChildren();
    this.markButtons = [];
    if (!this.episode) return;
    const records = this.filtered.filter(record => record.episode_id === this.episode.episode_id);
    for (const stream of STREAMS) {
      const items = records.filter(record => record.stream === stream);
      if (!items.length) continue;
      const row = document.createElement('div'); row.className = 'anomaly-lane';
      row.setAttribute('aria-label', stream + ' anomalies');
      const name = document.createElement('span'); name.textContent = stream;
      const track = document.createElement('div'); track.className = 'anomaly-track';
      for (const record of items) {
        const button = document.createElement('button'); button.type = 'button';
        button.className = 'anomaly-mark';
        const end = Math.max(1, this.episode.end_ns);
        button.style.left = Math.min(100, record.start_ns / end * 100) + '%';
        button.style.width = Math.max(0, (record.end_ns - record.start_ns) / end * 100) + '%';
        button.classList.toggle('point', record.start_ns === record.end_ns);
        button.title = label(record); button.setAttribute('aria-label', label(record));
        button.setAttribute('aria-pressed', String(record.id === this.selected));
        button.addEventListener('click', () => this.select(record.id));
        track.append(button); this.markButtons.push([button, record]);
      }
      row.append(name, track); this.marks.append(row);
    }
  }
  activeStreams(ns) {
    return new Set(this.records.filter(record => record.episode_id === this.episode?.episode_id &&
      containsTime(record, ns)).map(record => record.stream));
  }
}
