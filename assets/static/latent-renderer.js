/**
 * WebGPU Latent Drift Renderer
 * Renders latent space particles with Y2K/phosphor aesthetic
 */

class LatentRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.device = null;
        this.context = null;
        this.format = 'bgra8unorm';
        this.pipelines = {};
        this.buffers = {};
        this.bindGroups = {};
        this.uniforms = {};
        this.particles = [];
        this.sigils = [];
        this.time = 0;
        this.animationId = null;
        this.params = {
            pointSize: 4.0,
            signalInfluence: 1.0,
            scanlineIntensity: 0.3,
            phosphorDecay: 0.15,
            wireframe: true,
            showSigils: true,
        };
        this.camera = {
            position: [0, 0, 5],
            target: [0, 0, 0],
            up: [0, 1, 0],
            fov: 45 * Math.PI / 180,
        };
    }

    async init() {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }

        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance',
        });
        if (!adapter) throw new Error('No WebGPU adapter');

        this.device = await adapter.requestDevice({
            requiredFeatures: [],
            requiredLimits: {},
        });

        this.context = this.canvas.getContext('webgpu');
        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'premultiplied',
        });

        await this.createShaders();
        await this.createBuffers();
        await this.createBindGroups();
        this.generateParticles(2000);
        this.generateSigils();

        this.device.lost.then(info => {
            console.warn('WebGPU device lost:', info.reason);
        });

        return true;
    }

    async createShaders() {
        // Import shader sources
        const shaders = await import('./latent.wgsl');

        // Point sprite pipeline
        this.pipelines.points = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({
                    code: shaders.getShader('point_sprite_vertex'),
                }),
                entryPoint: 'vs_main',
                buffers: [
                    {
                        arrayStride: 48, // position(12) + latent(12) + signal(16) + idx(4) + padding(4)
                        attributes: [
                            { shaderLocation: 0, offset: 0, format: 'float32x3' }, // position
                            { shaderLocation: 1, offset: 12, format: 'float32x3' }, // latent
                            { shaderLocation: 2, offset: 24, format: 'float32x4' }, // signal
                            { shaderLocation: 3, offset: 40, format: 'uint32' }, // idx
                        ],
                    },
                ],
            },
            fragment: {
                module: this.device.createShaderModule({
                    code: shaders.getShader('point_sprite_fragment'),
                }),
                entryPoint: 'fs_main',
                targets: [{
                    format: this.format,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add',
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add',
                        },
                    },
                }],
            },
            primitive: {
                topology: 'point-list',
            },
            depthStencil: {
                depthWriteEnabled: false,
                depthCompare: 'less',
                format: 'depth24plus',
            },
        });

        // Wireframe pipeline
        this.pipelines.wireframe = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({
                    code: shaders.getShader('wireframe_vertex'),
                }),
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 48,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x3' },
                        { shaderLocation: 1, offset: 12, format: 'float32x3' },
                        { shaderLocation: 2, offset: 24, format: 'float32x4' },
                        { shaderLocation: 3, offset: 40, format: 'uint32' },
                    ],
                }],
            },
            fragment: {
                module: this.device.createShaderModule({
                    code: shaders.getShader('wireframe_fragment'),
                }),
                entryPoint: 'fs_main',
                targets: [{
                    format: this.format,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one',
                            operation: 'add',
                        },
                        alpha: {},
                    },
                }],
            },
            primitive: {
                topology: 'line-strip',
            },
            depthStencil: {
                depthWriteEnabled: false,
                depthCompare: 'less',
                format: 'depth24plus',
            },
        });

        // Post-process pipeline
        this.pipelines.post = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({
                    code: `
                        @vertex
                        fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                            var pos = array<vec2<f32>, 4>(
                                vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0),
                                vec2<f32>(-1.0, 3.0), vec2<f32>(3.0, 3.0)
                            );
                            return vec4<f32>(pos[idx], 0.0, 1.0);
                        }
                    `,
                }),
                entryPoint: 'vs_main',
            },
            fragment: {
                module: this.device.createShaderModule({
                    code: shaders.getShader('post_process'),
                }),
                entryPoint: 'fs_main',
                targets: [{ format: this.format }],
            },
            primitive: { topology: 'triangle-strip' },
        });
    }

    async createBuffers() {
        // Uniform buffers
        this.buffers.camera = this.device.createBuffer({
            size: 64 * 2, // 2 mat4x4
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.buffers.time = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.buffers.params = this.device.createBuffer({
            size: 32, // Params struct
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.buffers.frameParams = this.device.createBuffer({
            size: 48, // FrameParams struct
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Depth texture
        this.depthTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height, 1],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        // Offscreen render target for post-process
        this.offscreenTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height, 1],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.offscreenView = this.offscreenTexture.createView();

        // Post-process sampler
        this.sampler = this.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });
    }

    async createBindGroups() {
        // Render bind group
        this.bindGroups.render = this.device.createBindGroup({
            layout: this.pipelines.points.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.camera } },
                { binding: 1, resource: { buffer: this.buffers.camera, offset: 64 } },
                { binding: 2, resource: { buffer: this.buffers.time } },
                { binding: 3, resource: { buffer: this.buffers.params } },
                { binding: 4, resource: { buffer: this.buffers.frameParams } },
            ],
        });

        // Post-process bind group
        this.bindGroups.post = this.device.createBindGroup({
            layout: this.pipelines.post.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.offscreenView },
                { binding: 1, resource: this.sampler },
                { binding: 2, resource: { buffer: this.buffers.frameParams } },
            ],
        });
    }

    generateParticles(count) {
        this.particles = [];
        for (let i = 0; i < count; i++) {
            // distribute in latent space cube
            const latent = [
                (Math.random() - 0.5) * 4,
                (Math.random() - 0.5) * 4,
                (Math.random() - 0.5) * 2,
            ];
            // map to world space
            const position = [
                latent[0] + (Math.random() - 0.5) * 0.5,
                latent[1] + (Math.random() - 0.5) * 0.5,
                latent[2] + (Math.random() - 0.5) * 0.3,
            ];
            // signal values (will be updated from actual signals)
            const signal = [0, 0, 0, 0];

            this.particles.push({ position, latent, signal, idx: i });
        }
        this.uploadParticles();
    }

    uploadParticles() {
        const vertexData = new Float32Array(this.particles.length * 12); // 12 floats per vertex
        this.particles.forEach((p, i) => {
            const offset = i * 12;
            vertexData[offset] = p.position[0];
            vertexData[offset + 1] = p.position[1];
            vertexData[offset + 2] = p.position[2];
            vertexData[offset + 3] = p.latent[0];
            vertexData[offset + 4] = p.latent[1];
            vertexData[offset + 5] = p.latent[2];
            vertexData[offset + 6] = p.signal[0];
            vertexData[offset + 7] = p.signal[1];
            vertexData[offset + 8] = p.signal[2];
            vertexData[offset + 9] = p.signal[3];
        });

        if (this.buffers.particles) {
            this.buffers.particles.destroy();
        }
        this.buffers.particles = this.device.createBuffer({
            size: vertexData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.buffers.particles.getMappedRange()).set(vertexData);
        this.buffers.particles.unmap();
    }

    generateSigils() {
        this.sigils = [];
        for (let seat = 0; seat < 13; seat++) {
            const vertices = [];
            const baseAngle = seat * Math.PI * 2 / 13;
            for (let i = 0; i < 13; i++) {
                const angle = i * Math.PI * 2 / 13 + baseAngle;
                const r = 3.5 * (0.8 + 0.4 * Math.sin(i * 0.5));
                vertices.push([
                    r * Math.cos(angle),
                    r * Math.sin(angle),
                    Math.sin(i) * 0.2,
                ]);
            }
            // close the loop
            vertices.push([...vertices[0]]);
            this.sigils.push({ vertices, seat });
        }
        this.uploadSigils();
    }

    uploadSigils() {
        const totalVertices = this.sigils.reduce((sum, s) => sum + s.vertices.length, 0);
        const vertexData = new Float32Array(totalVertices * 12);
        let offset = 0;

        this.sigils.forEach((sigil, si) => {
            sigil.vertices.forEach((v, vi) => {
                const idx = (offset + vi) * 12;
                vertexData[idx] = v[0];
                vertexData[idx + 1] = v[1];
                vertexData[idx + 2] = v[2];
                // latent coords
                vertexData[idx + 3] = v[0] * 0.1;
                vertexData[idx + 4] = v[1] * 0.1;
                vertexData[idx + 5] = v[2] * 0.1;
                // signal
                vertexData[idx + 6] = 0.5;
                vertexData[idx + 7] = 0.3;
                vertexData[idx + 8] = 0.7;
                vertexData[idx + 9] = 0.2;
            });
            offset += sigil.vertices.length;
        });

        if (this.buffers.sigils) {
            this.buffers.sigils.destroy();
        }
        this.buffers.sigils = this.device.createBuffer({
            size: vertexData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.buffers.sigils.getMappedRange()).set(vertexData);
        this.buffers.sigils.unmap();
    }

    updateSignals(signals) {
        // Update particle signal values based on external signals
        const signalArray = [
            signals.moon_phase?.value || 0,
            signals.solar_wind?.value || 0,
            signals.git_activity?.value || 0,
            signals.crypto_volatility?.value || 0,
        ];

        this.particles.forEach((p, i) => {
            // each particle gets slightly different signal mix based on its latent position
            const mix = Math.abs(Math.sin(p.latent[0] + p.latent[1] + i * 0.1));
            p.signal = signalArray.map(s => s * (0.5 + mix * 0.5));
        });
        this.uploadParticles();
    }

    updateCamera() {
        const { position, target, up, fov } = this.camera;
        const aspect = this.canvas.width / this.canvas.height;

        // view matrix
        const z = this.normalize(this.sub(position, target));
        const x = this.normalize(this.cross(up, z));
        const y = this.cross(z, x);

        const view = new Float32Array([
            x[0], y[0], z[0], 0,
            x[1], y[1], z[1], 0,
            x[2], y[2], z[2], 0,
            -this.dot(x, position), -this.dot(y, position), -this.dot(z, position), 1,
        ]);

        // projection matrix
        const f = 1.0 / Math.tan(fov / 2);
        const proj = new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, -100.1 / 99.9, -1,
            0, 0, -200.1 / 99.9, 0,
        ]);

        this.device.queue.writeBuffer(this.buffers.camera, 0, view);
        this.device.queue.writeBuffer(this.buffers.camera, 64, proj);
    }

    updateUniforms(deltaTime) {
        this.time += deltaTime;

        this.device.queue.writeBuffer(this.buffers.time, 0, new Float32Array([this.time]));

        const params = new Float32Array([
            this.params.pointSize,
            this.params.signalInfluence,
            this.params.scanlineIntensity,
            this.params.phosphorDecay,
            this.params.wireframe ? 1 : 0,
            0, 0, 0, // padding
        ]);
        this.device.queue.writeBuffer(this.buffers.params, 0, params);

        const frameParams = new Float32Array([
            this.canvas.width, this.canvas.height,
            this.time,
            this.params.scanlineIntensity,
            this.params.phosphorDecay,
            this.params.wireframe ? 1 : 0,
            this.params.pointSize,
            0, // padding
            this.camera.position[0], this.camera.position[1], this.camera.position[2], 0,
        ]);
        this.device.queue.writeBuffer(this.buffers.frameParams, 0, frameParams);
    }

    render() {
        if (!this.device) return;

        // Resize handling
        if (this.canvas.width !== this.canvas.clientWidth ||
            this.canvas.height !== this.canvas.clientHeight) {
            this.resize();
        }

        this.updateCamera();

        const commandEncoder = this.device.createCommandEncoder();

        // First pass: render to offscreen texture
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.offscreenView,
                clearValue: { r: 0.04, g: 0.04, b: 0.05, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
            depthStencilAttachment: {
                view: this.depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        });

        // Render wireframe first (behind particles)
        if (this.params.wireframe && this.buffers.sigils) {
            renderPass.setPipeline(this.pipelines.wireframe);
            renderPass.setBindGroup(0, this.bindGroups.render);
            renderPass.setVertexBuffer(0, this.buffers.sigils);
            this.sigils.forEach(sigil => {
                renderPass.draw(sigil.vertices.length);
            });
        }

        // Render particles
        if (this.buffers.particles) {
            renderPass.setPipeline(this.pipelines.points);
            renderPass.setBindGroup(0, this.bindGroups.render);
            renderPass.setVertexBuffer(0, this.buffers.particles);
            renderPass.draw(this.particles.length);
        }

        renderPass.end();

        // Second pass: post-process to screen
        const postPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });

        postPass.setPipeline(this.pipelines.post);
        postPass.setBindGroup(0, this.bindGroups.post);
        postPass.draw(4);
        postPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    resize() {
        this.canvas.width = this.canvas.clientWidth * window.devicePixelRatio;
        this.canvas.height = this.canvas.clientHeight * window.devicePixelRatio;

        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'premultiplied',
        });

        // Recreate depth texture
        this.depthTexture.destroy();
        this.depthTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height, 1],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        // Recreate offscreen texture
        this.offscreenTexture.destroy();
        this.offscreenTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height, 1],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.offscreenView = this.offscreenTexture.createView();

        // Update post bind group
        this.bindGroups.post = this.device.createBindGroup({
            layout: this.pipelines.post.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.offscreenView },
                { binding: 1, resource: this.sampler },
                { binding: 2, resource: { buffer: this.buffers.frameParams } },
            ],
        });
    }

    start() {
        let lastTime = performance.now();
        const frame = (now) => {
            const deltaTime = (now - lastTime) / 1000;
            lastTime = now;

            this.updateUniforms(deltaTime);
            this.render();

            this.animationId = requestAnimationFrame(frame);
        };
        this.animationId = requestAnimationFrame(frame);
    }

    stop() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    destroy() {
        this.stop();
        Object.values(this.buffers).forEach(b => b?.destroy());
        this.depthTexture?.destroy();
        this.offscreenTexture?.destroy();
        this.device?.destroy();
    }

    // Vector math helpers
    sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
    cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
    dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
    normalize(v) {
        const len = Math.sqrt(this.dot(v, v));
        return len > 0 ? [v[0]/len, v[1]/len, v[2]/len] : [0, 0, 1];
    }

    setParam(key, value) {
        if (key in this.params) this.params[key] = value;
    }
}

// Export for use in app.js
export { LatentRenderer };