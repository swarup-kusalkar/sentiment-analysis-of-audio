// Call Analysis Results UI — Phase 9 placeholder

document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');
    const statusEl = document.getElementById('status');
    const resultsEl = document.getElementById('results');

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = '#4a90d9'; });
    dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = '#ccc'; });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#ccc';
        const file = e.dataTransfer.files[0];
        if (file) handleUpload(file);
    });

    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) handleUpload(file);
    });

    function handleUpload(file) {
        statusEl.textContent = `Processing "${file.name}"...`;
        resultsEl.innerHTML = '';

        const formData = new FormData();
        formData.append('file', file);

        fetch('/analyse/', { method: 'POST', body: formData })
            .then(res => res.json())
            .then(data => renderResults(data))
            .catch(err => { statusEl.textContent = `Error: ${err.message}`; });
    }

    function renderResults(data) {
        statusEl.textContent = data.status === 'ok'
            ? `Analysis complete — ${data.speaker_count} speaker(s) detected`
            : data.status;

        if (!data.results) return;

        resultsEl.innerHTML = data.results.map(speaker => `
            <div class="speaker-card">
                <div class="speaker-card-header">
                    <strong>${speaker.speaker_id}</strong>
                    <span>${formatTime(speaker.segment_time_range)}</span>
                    ${speaker.confidence?.needs_human_review ? '<span class="review-badge">Needs Review</span>' : ''}
                </div>
                <div class="speaker-card-body">
                    ${renderField('Transcript', speaker.transcript)}
                    ${renderField('Emotion', `${speaker.emotion?.primary}${speaker.emotion?.secondary?.length ? ' / ' + speaker.emotion.secondary.join(', ') : ''}`)}
                    ${renderField('Tone', speaker.tone?.label)}
                    ${speaker.abuse?.flagged ? `<div class="abuse-flag">⚠ Abuse Detected — ${speaker.abuse.category} (severity ${speaker.abuse.severity_1_to_5}/5)</div>` : renderField('Abuse', 'None detected')}
                    ${renderField('Energy / Loudness', `${speaker.energy_loudness?.level}: ${speaker.energy_loudness?.description}`)}
                    ${renderField('Summary', speaker.summary)}
                </div>
            </div>
        `).join('');
    }

    function renderField(label, value) {
        return `<div class="field"><div class="field-label">${label}</div><div class="field-value">${value || '—'}</div></div>`;
    }

    function formatTime(range) {
        if (!range) return '';
        const s = range.start_s || 0;
        const e = range.end_s || 0;
        return `${fmtSec(s)} – ${fmtSec(e)}`;
    }

    function fmtSec(s) {
        const m = Math.floor(s / 60);
        const sec = Math.floor(s % 60);
        return `${m}:${String(sec).padStart(2, '0')}`;
    }
});