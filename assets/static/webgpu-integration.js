// integration of WebGPU renderer into main app

// Add to app.js after LatentRenderer import
// import { LatentRenderer } from './latent-renderer.js';

// In DatasetDreamerApp class, add:

/*
    async initWebGPU() {
        try {
            this.latentRenderer = new LatentRenderer(this.latentCanvas);
            await this.latentRenderer.init();
            this.latentRenderer.start();
            this.log('WebGPU renderer initialized', 'success');
        } catch (err) {
            this.log(`WebGPU init failed: ${err.message}`, 'warn');
            this.fallbackToCanvas2D();
        }
    }

    fallbackToCanvas2D() {
        // The existing drawLatent() method serves as fallback
        this.drawLatent();
    }

    // Update the render loop to use WebGPU when available
    drawLatent() {
        if (this.latentRenderer) {
            // WebGPU handles its own render loop
            return;
        }
        // ... existing canvas 2d code
    }

    // Update signals to also update WebGPU renderer
    updateSignals(data) {
        // ... existing code
        if (this.latentRenderer) {
            this.latentRenderer.updateSignals(this.signals);
        }
    }

    // Add parameter controls for WebGPU
    bindWebGPUControls() {
        // Connect latent controls to renderer params
        this.latentSpeedInput.addEventListener('input', (e) => {
            this.latentSpeed = parseFloat(e.target.value);
            if (this.latentRenderer) {
                this.latentRenderer.setParam('signalInfluence', this.latentSpeed);
            }
        });
    }
*/

// Modified drawLatent to integrate with WebGPU
// Replace the drawLatent method in app.js with this version:

/*
    drawLatent() {
        if (this.latentRenderer) {
            // WebGPU renderer runs independently
            return;
        }

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
            const n = Math.sin(p.x * 0.01 + time) * Math.cos(p.y * 0.01 + time * 0.7);
            p.vx += Math.sin(n * Math.PI * 2) * 0.02;
            p.vy += Math.cos(n * Math.PI * 2) * 0.02;
            p.vx *= 0.98;
            p.vy *= 0.98;
            p.x += p.vx;
            p.y += p.vy;
            p.phase += 0.02;

            if (p.x < 0) p.x = w;
            if (p.x > w) p.x = 0;
            if (p.y < 0) p.y = h;
            if (p.y > h) p.y = 0;

            const alpha = 0.3 + 0.4 * Math.sin(p.phase);
            this.ctx.fillStyle = `hsla(${p.hue}, 60%, 60%, ${alpha})`;
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            this.ctx.fill();
        });

        // sigil anchors
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
        // Draw 13 sigil anchors (Organization XIII)
        for (let seat = 0; seat < 13; seat++) {
            const baseAngle = seat * Math.PI * 2 / 13;
            const rotation = time * 0.05 + baseAngle;
            const cx = this.latentCanvas.clientWidth / 2;
            const cy = this.latentCanvas.clientHeight / 2;
            const radius = Math.min(cx, cy) * 0.7;

            this.ctx.strokeStyle = `hsla(${260 + seat * 10}, 50%, 50%, 0.3)`;
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();

            for (let i = 0; i <= 13; i++) {
                const angle = i * Math.PI * 2 / 13 + rotation;
                const r = radius * (0.8 + 0.4 * Math.sin(time * 0.3 + i * 0.5));
                const x = cx + r * Math.cos(angle);
                const y = cy + r * Math.sin(angle);
                if (i === 0) this.ctx.moveTo(x, y);
                else this.ctx.lineTo(x, y);
            }
            this.ctx.closePath();
            this.ctx.stroke();
        }
    }
*/

// Export this as a module to be imported
export const webgpuIntegration = `
// This file documents the WebGPU integration pattern
// Copy the relevant sections into app.js
`;