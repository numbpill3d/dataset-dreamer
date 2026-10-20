/**
 * dataset-dreamer UI Application
 * 90s/00s Aero mono Y2K brutalist minimalist
 */

class DatasetDreamerApp {
    constructor() {
        this.ws = null;
        this.signals = {};
        this.signalHistory = new Map();
        this.latentAnimationId = null;
        this.latentParticles = [];
        this.latent3D = false;
        this.latentTrail = false;
        this.latentSpeed = 1.0;
        this.clockInterval = null;

        this.initElements();
        this.bindEvents();
        this.connectWebSocket();
        this.startClock();
        this.loadSignals();
        this.checkModelStatus();
    }

    initElements() {
        // toolbars
        this.btnLoad = document.getElementById('tool-load');
        this.btnTrain = document.getElementById('tool-train');
        this.btnGenerate = document.getElementById('tool-generate');
        this.btnInterpolate = document.getElementById('tool-interpolate');
        this.modelType = document.getElementById('model-type');
        this.epochs = document.getElementById('epochs');
        this.batchSize = document.getElementById('batch-size');
        this.genRows = document.getElementById('gen-rows');
        this.temperature = document.getElementById('temperature');

        // status
        this.statusText = document.getElementById('status-text');
        this.dataInfo = document.getElementById('data-info');
        this.modelInfo = document.getElementById('model-info');
        this.signalIndicator = document.getElementById('signal-indicator');
        this.signalText = document.getElementById('signal-text');

        // panes
        this.signalTable = document.getElementById('signal-table').querySelector('tbody');
        this.signalGraph = document.getElementById('signal-graph');
        this.dataGrid = document.getElementById('data-grid');
        this.previewRows = document.getElementById('preview-rows');
        this.latentCanvas = document.getElementById('latent-canvas');
        this.latent3DCheckbox = document.getElementById('latent-3d');
        this.latentTrailCheckbox = document.getElementById('latent-trail');
        this.latentSpeedInput = document.getElementById('latent-speed');

        // modals
        this.modalFile = document.getElementById('modal-file');
        this.modalTrain = document.getElementById('modal-train');
        this.modalGenerate = document.getElementById('modal-generate');
        this.console = document.getElementById('console');
        this.consoleBody = document.getElementById('console-body');

        // modal controls
        this.filePath = document.getElementById('file-path');
        this.trainModelType = document.getElementById('train-model-type');
        this.trainEpochs = document.getElementById('train-epochs');
        this.trainBatch = document.getElementById('train-batch');
        this.trainLr = document.getElementById('train-lr');
        this.trainLatent = document.getElementById('train-latent');
        this.trainSignals = document.getElementById('train-signals');
        this.signalOverrides = document.getElementById('signal-overrides');
        this.genRowsModal = document.getElementById('gen-rows-modal');
        this.genTempModal = document.getElementById('gen-temp-modal');
        this.trainProgress = document.getElementById('train-progress');
        this.progressBar = document.getElementById('progress-bar');
        this.progressText = document.getElementById('progress-text');

        // canvas setup
        this.ctx = this.latentCanvas.getContext('2d');
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    bindEvents() {
        // toolbar
        this.btnLoad.addEventListener('click', () => this.openModal('file'));
        this.btnTrain.addEventListener('click', () => this.openModal('train'));
        this.btnGenerate.addEventListener('click', () => this.openModal('generate'));
        this.btnInterpolate.addEventListener('click', () => this.interpolate());

        // toolbar inputs
        this.modelType.addEventListener('change', (e) => this.log(`model type: ${e.target.value}`));
        this.epochs.addEventListener('change', (e) => this.log(`epochs: ${e.target.value}`));
        this.batchSize.addEventListener('change', (e) => this.log(`batch size: ${e.target.value}`));
        this.genRows.addEventListener('change', (e) => this.log(`gen rows: ${e.target.value}`));
        this.temperature.addEventListener('change', (e) => this.log(`temperature: ${e.target.value}`));

        // pane controls
        document.getElementById('refresh-signals').addEventListener('click', () => this.loadSignals());
        this.previewRows.addEventListener('change', () => this.loadDataPreview());

        // latent controls
        this.latent3DCheckbox.addEventListener('change', (e) => { this.latent3D = e.target.checked; });
        this.latentTrailCheckbox.addEventListener('change', (e) => { this.latentTrail = e.target.checked; });
        this.latentSpeedInput.addEventListener('input', (e) => { this.latentSpeed = parseFloat(e.target.value); });

        // modal buttons
        document.querySelectorAll('.modal-close, #cancel-load, #cancel-train, #cancel-generate, #console-close').forEach(btn => {
            btn.addEventListener('click', () => this.closeAllModals());
        });
        document.querySelectorAll('.modal-backdrop').forEach(bg => {
            bg.addEventListener('click', () => this.closeAllModals());
        });

        document.getElementById('confirm-load').addEventListener('click', () => this.loadData());
        document.getElementById('confirm-train').addEventListener('click', () => this.trainModel());
        document.getElementById('confirm-generate').addEventListener('click', () => this.generateData());
        document.getElementById('console-clear').addEventListener('click', () => this.consoleBody.innerHTML = '');

        // keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'o') { e.preventDefault(); this.openModal('file'); }
            if (e.key === 'F5') { e.preventDefault(); this.openModal('train'); }
            if (e.key === 'F9') { e.preventDefault(); this.openModal('generate'); }
            if (e.key === 'F10') { e.preventDefault(); this.interpolate(); }
            if (e.ctrlKey && e.key === 'r') { e.preventDefault(); this.loadSignals(); }
            if (e.key === 'Escape') { this.closeAllModals(); }
            if (e.ctrlKey && e.key === '`') { e.preventDefault(); this.toggleConsole(); }
        });

        // window controls
        document.getElementById('btn-min').addEventListener('click', () => this.log('minimize not implemented'));
        document.getElementById('btn-max').addEventListener('click', () => this.log('maximize not implemented'));
        document.getElementById('btn-close').addEventListener('click', () => this.log('close not implemented'));
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

        this.ws.onopen = () => {
            this.log('websocket connected');
            this.ws.send(JSON.stringify({ type: 'signals' }));
        };

        this.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            this.handleWebSocketMessage(msg);
        };

        this.ws.onclose = () => {
            this.log('websocket disconnected, reconnecting...');
            setTimeout(() => this.connectWebSocket(), 3000);
        };

        this.ws.onerror = (err) => {
            this.log(`websocket error: ${err}`, 'error');
        };
    }

    handleWebSocketMessage(msg) {
        switch (msg.type) {
            case 'pong':
                break;
            case 'signals':
                this.updateSignals(msg);
                break;
            default:
                this.log(`ws: ${JSON.stringify(msg)}`, 'info');
        }
    }

    startClock() {
        this.clockInterval = setInterval(() => {
            const now = new Date();
            document.getElementById('clock').textContent =
                now.toLocaleTimeString('en-US', { hour12: false });
        }, 1000);
    }

    async loadSignals() {
        this.setStatus('fetching signals...');
        try {
            const resp = await fetch('/api/signals');
            const data = await resp.json();
            this.updateSignals(data);
            this.setStatus('ready');
        } catch (err) {
            this.log(`signal fetch failed: ${err}`, 'error');
            this.setStatus('signal fetch failed');
        }
    }

    updateSignals(data) {
        this.signals = data.signals || {};
        this.renderSignalTable();
        this.renderSignalGraph();
        this.updateSignalOverrides();
        this.signalIndicator.classList.add('active');
        this.signalText.textContent = `SIGNALS: ${Object.keys(this.signals).length} ACTIVE`;
    }

    renderSignalTable() {
        const sources = [
            { key: 'moon_phase', label: 'MOON PHASE' },
            { key: 'solar_wind', label: 'SOLAR WIND' },
            { key: 'git_activity', label: 'GIT ACTIVITY' },
            { key: 'crypto_volatility', label: 'CRYPTO VOL' },
        ];

        this.signalTable.innerHTML = sources.map(s => {
            const sig = this.signals[s.key];
            if (!sig) return '';

            const val = sig.value !== undefined ? sig.value.toFixed(4) : '--';
            const status = sig.metadata?.error ? 'err' : (sig.metadata?.fallback ? 'warn' : 'ok');
            const weight = sig.metadata?.weight ?? 1.0;

            return `
                <tr>
                    <td>${s.label}</td>
                    <td class="signal-value">${val}</td>
                    <td class="signal-weight">${weight.toFixed(2)}</td>
                    <td><span class="signal-status ${status}"></span></td>
                </tr>
            `;
        }).join('');

        // update history for graph
        sources.forEach(s => {
            if (this.signals[s.key]?.value !== undefined) {
                if (!this.signalHistory.has(s.key)) this.signalHistory.set(s.key, []);
                const hist = this.signalHistory.get(s.key);
                hist.push(this.signals[s.key].value);
                if (hist.length > 100) hist.shift();
            }
        });
    }

    renderSignalGraph() {
        const ctx = this.signalGraph.getContext('2d');
        const w = this.signalGraph.width = this.signalGraph.clientWidth;
        const h = this.signalGraph.height = this.signalGraph.clientHeight;

        ctx.clearRect(0, 0, w, h);
        ctx.strokeStyle = '#1e1e24';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i <= 4; i++) {
            const y = (h / 4) * i;
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
        }
        ctx.stroke();

        const colors = {
            moon_phase: '#7b5cff',
            solar_wind: '#00ccaa',
            git_activity: '#ffaa00',
            crypto_volatility: '#ff3366',
        };

        Object.entries(this.signalHistory).forEach(([key, hist]) => {
            if (hist.length < 2) return;
            ctx.strokeStyle = colors[key] || '#7b5cff';
            ctx.lineWidth = 1.5;
            ctx.globalAlpha = 0.8;
            ctx.beginPath();
            hist.forEach((val, i) => {
                const x = (w / (hist.length - 1)) * i;
                const y = h - (val + 1) / 2 * h;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
            ctx.globalAlpha = 1;
        });
    }

    updateSignalOverrides() {
        const sources = ['moon_phase', 'solar_wind', 'git_activity', 'crypto_volatility'];
        this.signalOverrides.innerHTML = sources.map(key => {
            const sig = this.signals[key];
            const val = sig?.value !== undefined ? sig.value.toFixed(4) : '0.0000';
            const label = key.replace('_', ' ').toUpperCase();
            return `
                <label>
                    <span>${label}</span>
                    <input type="number" data-signal="${key}" value="${val}" step="0.01" min="-1" max="1">
                </label>
            `;
        }).join('');
    }

    async loadDataPreview() {
        try {
            const rows = parseInt(this.previewRows.value);
            const resp = await fetch(`/api/data/preview?rows=${rows}`);
            const data = await resp.json();
            this.renderDataGrid(data);
        } catch (err) {
            this.log(`preview failed: ${err}`, 'error');
        }
    }

    renderDataGrid(data) {
        if (data.error) {
            this.dataGrid.innerHTML = `<div class="empty">${data.error}</div>`;
            return;
        }

        const { columns, sample } = data;
        if (!sample || sample.length === 0) {
            this.dataGrid.innerHTML = '<div class="empty">NO DATA</div>';
            return;
        }

        let html = '<table class="data-table"><thead><tr>';
        columns.forEach(col => {
            html += `<th>${col}</th>`;
        });
        html += '</tr></thead><tbody>';

        sample.forEach(row => {
            html += '<tr>';
            columns.forEach(col => {
                const val = row[col];
                const isNum = typeof val === 'number';
                html += `<td class="${isNum ? 'numeric' : ''}">${val !== null && val !== undefined ? val : ''}</td>`;
            });
            html += '</tr>';
        });

        html += '</tbody></table>';
        this.dataGrid.innerHTML = html;
    }

    async checkModelStatus() {
        try {
            const resp = await fetch('/api/model/list');
            const data = await resp.json();
            if (data.models && data.models.length > 0) {
                this.modelInfo.textContent = `MODEL: ${data.models[0].type.toUpperCase()}`;
            }
        } catch (err) {
            // ignore
        }
    }

    async loadData() {
        const path = this.filePath.value.trim();
        if (!path) return this.log('path required', 'error');

        this.setStatus('loading data...');
        this.closeAllModals();

        try {
            const resp = await fetch('/api/data/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path })
            });
            const data = await resp.json();
            if (data.success) {
                this.dataInfo.textContent = `DATA: ${data.shape[0]} ROWS × ${data.shape[1]} COLS`;
                this.log(`loaded ${data.shape[0]} rows, ${data.shape[1]} columns`, 'success');
                await this.loadDataPreview();
            } else {
                this.log(`load failed: ${data.error}`, 'error');
            }
        } catch (err) {
            this.log(`load error: ${err}`, 'error');
        }
        this.setStatus('ready');
    }

    openModal(type) {
        this.closeAllModals();
        if (type === 'file') {
            this.modalFile.hidden = false;
            this.filePath.focus();
        } else if (type === 'train') {
            this.populateTrainModal();
            this.modalTrain.hidden = false;
        } else if (type === 'generate') {
            this.populateGenerateModal();
            this.modalGenerate.hidden = false;
        }
    }

    closeAllModals() {
        [this.modalFile, this.modalTrain, this.modalGenerate].forEach(m => m.hidden = true);
    }

    populateTrainModal() {
        this.trainModelType.value = this.modelType.value;
        this.trainEpochs.value = this.epochs.value;
        this.trainBatch.value = this.batchSize.value;
        this.trainLr.value = '0.0002';
        this.trainLatent.value = '16';

        const sources = ['moon_phase', 'solar_wind', 'git_activity', 'crypto_volatility'];
        this.trainSignals.innerHTML = sources.map(key => {
            const label = key.replace('_', ' ').toUpperCase();
            return `
                <label>
                    <input type="checkbox" value="${key}" checked>
                    <span>${label}</span>
                </label>
            `;
        }).join('');
    }

    populateGenerateModal() {
        this.genRowsModal.value = this.genRows.value;
        this.genTempModal.value = this.temperature.value;
        this.updateSignalOverrides();
    }

    async trainModel() {
        const signals = Array.from(this.trainSignals.querySelectorAll('input:checked')).map(cb => cb.value);

        this.setStatus('training...');
        this.trainProgress.hidden = false;
        this.progressBar.style.width = '0%';
        this.progressText.textContent = '0%';
        this.closeAllModals();

        try {
            const resp = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_type: this.trainModelType.value,
                    epochs: parseInt(this.trainEpochs.value),
                    batch_size: parseInt(this.trainBatch.value),
                    signals: signals,
                })
            });

            // simulate progress since we can't get real progress from sdv
            let progress = 0;
            const interval = setInterval(() => {
                progress = Math.min(95, progress + Math.random() * 5);
                this.progressBar.style.width = `${progress}%`;
                this.progressText.textContent = `${Math.round(progress)}%`;
            }, 500);

            const data = await resp.json();
            clearInterval(interval);
            this.progressBar.style.width = '100%';
            this.progressText.textContent = '100%';

            if (data.success) {
                this.modelInfo.textContent = `MODEL: ${this.trainModelType.value.toUpperCase()}`;
                this.log(`training complete: ${data.model_path}`, 'success');
            } else {
                this.log(`training failed: ${data.error}`, 'error');
            }
        } catch (err) {
            this.log(`train error: ${err}`, 'error');
        }

        setTimeout(() => { this.trainProgress.hidden = true; }, 1000);
        this.setStatus('ready');
    }

    async generateData() {
        const overrides = {};
        this.signalOverrides.querySelectorAll('input[data-signal]').forEach(input => {
            overrides[input.dataset.signal] = parseFloat(input.value);
        });

        this.setStatus('generating...');
        this.closeAllModals();

        try {
            const resp = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rows: parseInt(this.genRowsModal.value),
                    temperature: parseFloat(this.genTempModal.value),
                    signal_overrides: overrides,
                })
            });

            const data = await resp.json();
            if (data.success) {
                this.log(`generated ${data.rows} rows → ${data.output}`, 'success');
                this.dataInfo.textContent = `DATA: ${data.rows} ROWS (GENERATED)`;
                await this.loadDataPreview();
            } else {
                this.log(`generate failed: ${data.error}`, 'error');
            }
        } catch (err) {
            this.log(`generate error: ${err}`, 'error');
        }
        this.setStatus('ready');
    }

    async interpolate() {
        this.log('interpolation requires two data points - not implemented in UI yet', 'warn');
    }

    setStatus(text) {
        this.statusText.textContent = text.toUpperCase();
    }

    log(message, level = 'info') {
        const line = document.createElement('div');
        line.className = `log-line ${level}`;
        const time = new Date().toLocaleTimeString('en-US', { hour12: false });
        line.textContent = `[${time}] ${message}`;
        this.consoleBody.appendChild(line);
        this.consoleBody.scrollTop = this.consoleBody.scrollHeight;
    }

    toggleConsole() {
        this.console.hidden = !this.console.hidden;
    }

    resizeCanvas() {
        const rect = this.latentCanvas.getBoundingClientRect();
        this.latentCanvas.width = rect.width * window.devicePixelRatio;
        this.latentCanvas.height = rect.height * window.devicePixelRatio;
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        this.drawLatent();
    }

    drawLatent() {
        const w = this.latentCanvas.clientWidth;
        const h = this.latentCanvas.clientHeight;

        if (!this.latentTrail) {
            this.ctx.fillStyle = '#0a0a0c';
            this.ctx.fillRect(0, 0, w, h);
        } else {
            this.ctx.fillStyle = 'rgba(10, 10, 12, 0.1)';
            this.ctx.fillRect(0, 0, w, h);
        }

        // grid
        this.ctx.strokeStyle = '#1e1e24';
        this.ctx.lineWidth = 0.5;
        for (let i = 0; i <= 10; i++) {
            const x = (w / 10) * i;
            const y = (h / 10) * i;
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0); this.ctx.lineTo(x, h);
            this.ctx.moveTo(0, y); this.ctx.lineTo(w, y);
            this.ctx.stroke();
        }

        // particles from latent space
        const time = Date.now() * 0.001 * this.latentSpeed;

        // generate particles if empty
        if (this.latentParticles.length === 0) {
            for (let i = 0; i < 80; i++) {
                this.latentParticles.push({
                    x: Math.random() * w,
                    y: Math.random() * h,
                    vx: (Math.random() - 0.5) * 0.5,
                    vy: (Math.random() - 0.5) * 0.5,
                    phase: Math.random() * Math.PI * 2,
                    hue: 260 + Math.random() * 40,
                    size: 1 + Math.random() * 2,
                });
            }
        }

        this.latentParticles.forEach(p => {
            // flow field based on simplex-like noise
            const n = Math.sin(p.x * 0.01 + time) * Math.cos(p.y * 0.01 + time * 0.7);
            p.vx += Math.sin(n * Math.PI * 2) * 0.02;
            p.vy += Math.cos(n * Math.PI * 2) * 0.02;
            p.vx *= 0.98;
            p.vy *= 0.98;
            p.x += p.vx;
            p.y += p.vy;
            p.phase += 0.02;

            // wrap
            if (p.x < 0) p.x = w;
            if (p.x > w) p.x = 0;
            if (p.y < 0) p.y = h;
            if (p.y > h) p.y = 0;

            // draw
            const alpha = 0.3 + 0.4 * Math.sin(p.phase);
            this.ctx.fillStyle = `hsla(${p.hue}, 60%, 60%, ${alpha})`;
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            this.ctx.fill();
        });

        // 13 sigil anchors (Organization XIII geometry)
        this.drawSigils(time);

        // connections
        if (this.latentParticles.length > 1) {
            this.ctx.strokeStyle = 'rgba(123, 92, 255, 0.05)';
            this.ctx.lineWidth = 0.5;
            for (let i = 0; i < this.latentParticles.length; i += 3) {
                for (let j = i + 1; j < this.latentParticles.length; j += 4) {
                    const dx = this.latentParticles[i].x - this.latentParticles[j].x;
                    const dy = this.latentParticles[i].y - this.latentParticles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 80) {
                        this.ctx.beginPath();
                        this.ctx.moveTo(this.latentParticles[i].x, this.latentParticles[i].y);
                        this.ctx.lineTo(this.latentParticles[j].x, this.latentParticles[j].y);
                        this.ctx.stroke();
                    }
                }
            }
        }

        requestAnimationFrame(() => this.drawLatent());
    }

    drawSigils(time) {
        const w = this.latentCanvas.clientWidth;
        const h = this.latentCanvas.clientHeight;
        const cx = w / 2;
        const cy = h / 2;
        const maxRadius = Math.min(cx, cy) * 0.7;

        for (let seat = 0; seat < 13; seat++) {
            const baseAngle = seat * Math.PI * 2 / 13;
            const rotation = time * 0.05 + baseAngle;

            this.ctx.strokeStyle = `hsla(${260 + seat * 12}, 50%, 50%, 0.25)`;
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();

            for (let i = 0; i <= 13; i++) {
                const angle = i * Math.PI * 2 / 13 + rotation;
                const r = maxRadius * (0.8 + 0.4 * Math.sin(time * 0.3 + i * 0.5));
                const x = cx + r * Math.cos(angle);
                const y = cy + r * Math.sin(angle);
                if (i === 0) this.ctx.moveTo(x, y);
                else this.ctx.lineTo(x, y);
            }
            this.ctx.closePath();
            this.ctx.stroke();
        }
    }
}

// initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new DatasetDreamerApp();
});

// expose globally for debugging
window.DatasetDreamerApp = DatasetDreamerApp;