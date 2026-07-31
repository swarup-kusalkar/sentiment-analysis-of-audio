// Call Analysis Results UI — Phase 9
// Matches designdoc.md §11 specification exactly

(() => {
    // DOM Elements
    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');
    const uploadStatus = document.getElementById('upload-status');
    const progressBar = document.getElementById('progress-bar');
    const progressFill = progressBar.querySelector('.progress-fill');
    const progressText = progressBar.querySelector('.progress-text');
    const resultsSection = document.getElementById('results-section');
    const emptyState = document.getElementById('empty-state');
    const speakersContainer = document.getElementById('speakers-container');
    const exportBtn = document.getElementById('export-json');
    const template = document.getElementById('speaker-card-template');

    // Result header elements
    const resultFileName = document.getElementById('result-file-name');
    const resultDuration = document.getElementById('result-duration');
    const resultSpeakerCount = document.getElementById('result-speaker-count');
    const overallConfidence = document.getElementById('overall-confidence');
    const reviewBadge = document.getElementById('review-badge');

    // State
    let currentAnalysis = null;
    let analysisStartTime = 0;

    // Utility functions
    const formatDuration = (seconds) => {
        if (seconds < 60) return `${seconds.toFixed(1)} s`;
        const mins = Math.floor(seconds / 60);
        const secs = Math.round(seconds % 60);
        return `${mins} min ${secs} s`;
    };

    const formatConfidence = (conf) => {
        const pct = Math.round(conf * 100);
        let cls = 'high';
        if (pct < 50) cls = 'low';
        else if (pct < 75) cls = 'medium';
        return `<span class="confidence ${cls}">${pct}%</span>`;
    };

    const getEnergyIcon = (level) => {
        const icons = {
            whisper: '🤫',
            quiet: '🔈',
            normal: '🔉',
            loud: '🔊',
            shouting: '📢'
        };
        return icons[level.toLowerCase()] || '🔉';
    };

    const capitalize = (str) => str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();

    // Upload handling
    function handleUpload(file) {
        resetUI();
        uploadStatus.textContent = `Processing "${file.name}"...`;
        uploadStatus.className = 'upload-status';
        progressBar.hidden = false;
        progressFill.style.width = '0%';
        progressText.textContent = 'Uploading...';

        const formData = new FormData();
        formData.append('file', file);

        analysisStartTime = Date.now();
        simulateProgress();

        fetch('/analyse/', { method: 'POST', body: formData })
            .then(async (res) => {
                if (!res.ok) {
                    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
                    throw new Error(err.detail || `HTTP ${res.status}`);
                }
                return res.json();
            })
            .then((data) => {
                stopProgress();
                renderResults(data);
            })
            .catch((err) => {
                stopProgress();
                uploadStatus.textContent = `Error: ${err.message}`;
                uploadStatus.className = 'upload-status error';
                progressBar.hidden = true;
            });
    }

    // Simulated progress (since analysis is synchronous)
    let progressInterval = null;
    function simulateProgress() {
        let progress = 0;
        progressInterval = setInterval(() => {
            progress = Math.min(progress + Math.random() * 8, 90);
            progressFill.style.width = `${progress}%`;
            const elapsed = Math.round((Date.now() - analysisStartTime) / 1000);
            progressText.textContent = `Analyzing... ${elapsed}s elapsed`;
        }, 500);
    }

    function stopProgress() {
        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        progressText.textContent = 'Complete!';
        setTimeout(() => {
            progressBar.hidden = true;
        }, 500);
    }

    // Render results
    function renderResults(data) {
        currentAnalysis = data;
        uploadStatus.textContent = data.status === 'ok' 
            ? `Analysis complete — ${data.speaker_count} speaker(s) detected` 
            : data.status;
        uploadStatus.className = data.status === 'ok' ? 'upload-status success' : 'upload-status';

        // Header
        resultFileName.textContent = data.audio_hash ? `Audio: ${data.audio_hash.slice(0, 16)}...` : 'Analysis Result';
        resultDuration.textContent = `Duration: ${formatDuration(data.duration_seconds)}`;
        resultSpeakerCount.textContent = `${data.speaker_count} speaker(s) detected`;

        overallConfidence.innerHTML = formatConfidence(data.overall_confidence);
        reviewBadge.hidden = !data.needs_human_review;

        // Clear and render speaker cards
        speakersContainer.innerHTML = '';
        
        if (data.results && data.results.length > 0) {
            data.results.forEach((speaker, idx) => {
                const card = createSpeakerCard(speaker, idx);
                speakersContainer.appendChild(card);
            });
        }

        // Show results section
        resultsSection.hidden = false;
        emptyState.hidden = true;
        exportBtn.disabled = false;
    }

    function createSpeakerCard(speaker, index) {
        const clone = template.content.cloneNode(true);
        const card = clone.querySelector('.speaker-card');

        // Card header
        card.querySelector('[data-speaker-id]').textContent = `Speaker ${index + 1}`;
        card.querySelector('[data-time-range]').textContent = 
            `${formatTime(speaker.segment_time_range?.start_s)} – ${formatTime(speaker.segment_time_range?.end_s)}`;
        
        const confidenceEl = card.querySelector('[data-confidence]');
        const confPct = Math.round((speaker.confidence?.overall || 0) * 100);
        confidenceEl.innerHTML = formatConfidence(speaker.confidence?.overall || 0);
        
        const reviewBadgeEl = card.querySelector('[data-review]');
        reviewBadgeEl.hidden = !speaker.confidence?.needs_human_review;

        if (speaker.confidence?.needs_human_review) {
            card.classList.add('review-needed');
        }

        // Play button (placeholder - no audio playback in PoC)
        const playBtn = card.querySelector('[data-play]');
        playBtn.hidden = true; // Hide for PoC since we don't have audio segments served

        // Transcript
        const transcript = speaker.transcript || '—';
        clone.querySelector('[data-transcript]').textContent = transcript;

        // Energy / Loudness
        const energy = speaker.energy_loudness || { level: 'normal', description: '', dsp_agrees: true };
        const energyLevelEl = clone.querySelector('[data-energy-level]');
        energyLevelEl.innerHTML = `${getEnergyIcon(energy.level)} ${capitalize(energy.level.toLowerCase())}`;
        energyLevelEl.className = `energy-level ${energy.level.toLowerCase()}`;
        
        const dspDisagree = clone.querySelector('[data-dsp-disagree]');
        dspDisagree.hidden = energy.dsp_agrees !== false;
        
        clone.querySelector('[data-energy-description]').textContent = energy.description || '—';

        // Emotion
        const emotion = speaker.emotion || { primary: 'unknown', secondary: [], reasoning: '' };
        const primaryEl = clone.querySelector('[data-emotion-primary]');
        primaryEl.textContent = capitalize(emotion.primary);
        primaryEl.className = `emotion-primary ${emotion.primary.toLowerCase()}`;

        const secondaryEl = clone.querySelector('[data-emotion-secondary]');
        secondaryEl.innerHTML = '';
        if (emotion.secondary && emotion.secondary.length > 0) {
            emotion.secondary.forEach(s => {
                const tag = document.createElement('span');
                tag.className = 'tag';
                tag.textContent = capitalize(s);
                secondaryEl.appendChild(tag);
            });
        } else {
            secondaryEl.innerHTML = '<span class="tag" style="opacity:0.5">—</span>';
        }
        clone.querySelector('[data-emotion-reasoning]').textContent = emotion.reasoning || '—';

        // Tone
        const tone = speaker.tone || { label: 'unknown', reasoning: '' };
        const toneLabelEl = clone.querySelector('[data-tone-label]');
        toneLabelEl.textContent = capitalize(tone.label);
        toneLabelEl.className = `tone-label ${tone.label.toLowerCase()}`;
        clone.querySelector('[data-tone-reasoning]').textContent = tone.reasoning || '—';

        // Abuse
        const abuse = speaker.abuse || { flagged: false, category: 'none', severity_1_to_5: 0, evidence_span: null, reasoning: '' };
        const abuseContent = clone.querySelector('[data-abuse-content]');
        abuseContent.innerHTML = '';

        if (abuse.flagged) {
            const div = document.createElement('div');
            div.className = 'abuse-flagged';
            div.innerHTML = `
                <div class="abuse-header">
                    <span class="abuse-badge">🚨 Abuse Detected</span>
                    <div class="abuse-meta">
                        <span class="abuse-category">Category: ${capitalize(abuse.category)}</span>
                        <span class="abuse-severity">Severity: ${abuse.severity_1_to_5}/5</span>
                    </div>
                </div>
                ${abuse.evidence_span ? `<div class="evidence-box">${escapeHtml(abuse.evidence_span)}</div>` : ''}
                <p class="abuse-reasoning">${escapeHtml(abuse.reasoning || '—')}</p>
            `;
            abuseContent.appendChild(div);
        } else {
            const div = document.createElement('div');
            div.className = 'abuse-not-flagged';
            div.innerHTML = `
                <span class="badge">✓ No Abuse Detected</span>
                <p class="reasoning">${escapeHtml(abuse.reasoning || 'No profanity, threats, or personal attacks detected.')}</p>
            `;
            abuseContent.appendChild(div);
        }

        // Summary
        const summary = speaker.summary || '—';
        clone.querySelector('[data-summary]').innerHTML = `<p>${escapeHtml(summary)}</p>`;

        return card;
    }

    // Helpers
    function formatTime(seconds) {
        if (seconds === undefined || seconds === null) return '0:00';
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${String(s).padStart(2, '0')}`;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function resetUI() {
        uploadStatus.textContent = '';
        uploadStatus.className = 'upload-status';
        progressBar.hidden = true;
        resultsSection.hidden = true;
        emptyState.hidden = false;
        speakersContainer.innerHTML = '';
        exportBtn.disabled = true;
        currentAnalysis = null;
    }

    // Export JSON
    exportBtn.addEventListener('click', () => {
        if (!currentAnalysis) return;
        const blob = new Blob([JSON.stringify(currentAnalysis, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `analysis_${currentAnalysis.audio_hash?.slice(0,12) || Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    });

    // Drag & drop
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('audio/')) {
            handleUpload(file);
        } else {
            uploadStatus.textContent = 'Please drop an audio file';
            uploadStatus.className = 'upload-status error';
        }
    });

    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) {
            handleUpload(file);
            fileInput.value = ''; // Allow re-upload of same file
        }
    });

    // Initialize
    console.log('Call Analysis UI ready');
})();