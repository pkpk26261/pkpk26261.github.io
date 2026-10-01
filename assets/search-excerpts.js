'use strict';
// Share matching and display offsets so NFKC matches retain original spelling.
((root) => {
  const normalize = value => value.normalize('NFKC').toLocaleLowerCase();
  function ranges(text, terms) {
    let folded = ''; const offsets = [];
    const segments = new Intl.Segmenter('zh-Hant', {granularity:'grapheme'}).segment(text);
    for (const {segment, index} of segments) {
      const value = normalize(segment);
      folded += value;
      for (let i = 0; i < value.length; i++) offsets.push([index, index + segment.length]);
    }
    const found = [];
    for (const word of terms) {
      const term = normalize(word); if (!term) continue;
      let index = folded.indexOf(term);
      while (index >= 0) {
        found.push([offsets[index][0], offsets[index + term.length - 1][1]]);
        index = folded.indexOf(term, index + term.length);
      }
    }
    found.sort((a,b) => a[0] - b[0]);
    return found.reduce((merged, item) => {
      const last = merged[merged.length - 1];
      if (last && item[0] <= last[1]) last[1] = Math.max(last[1], item[1]);
      else merged.push(item.slice());
      return merged;
    }, []);
  }
  function snippet(text, terms, fallback = '') {
    const clean = text.replace(/\s+/g, ' ').trim();
    const matches = ranges(clean, terms);
    if (!matches.length) return fallback || clean.slice(0, 140);
    const start = Math.max(0, matches[0][0] - 40);
    const end = Math.min(clean.length, Math.max(start + 140, matches[0][1] + 40));
    return (start ? '…' : '') + clean.slice(start, end) + (end < clean.length ? '…' : '');
  }
  function appendHighlighted(node, text, terms) {
    let cursor = 0;
    for (const [start, end] of ranges(text, terms)) {
      node.append(document.createTextNode(text.slice(cursor, start)));
      const mark = document.createElement('mark'); mark.textContent = text.slice(start, end);
      node.append(mark); cursor = end;
    }
    node.append(document.createTextNode(text.slice(cursor)));
  }
  function compareDates(a, b, mode) {
    const key = mode === 'updated' ? 'updated' : 'published';
    return (b[key] || b.published).localeCompare(a[key] || a.published) || b.published.localeCompare(a.published);
  }
  const api = {normalize, ranges, snippet, appendHighlighted, compareDates};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ReaderSearch = api;
})(typeof window === 'undefined' ? {} : window);
