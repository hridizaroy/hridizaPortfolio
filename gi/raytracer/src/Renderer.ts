// Starter code guided from
// https://giftmugweni.hashnode.dev/how-to-set-up-webgpu-with-typescript-and-vite-a-simplified-guide

export class Renderer
{
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private canvasFormat!: GPUTextureFormat;

    private vertexBuffer!: GPUBuffer;
    private indexBuffer!: GPUBuffer;
    private uniformBuffer!: GPUBuffer;

    private uniforms!: Float32Array;

    private shaderModule!: GPUShaderModule;
    private bindGroupLayout!: GPUBindGroupLayout;
    private bindGroup!: GPUBindGroup
    private pipeline!: GPURenderPipeline;
    private renderPassDescriptor!: GPURenderPassDescriptor;
    
    private readonly MAX_SIZE : number = 100;

    constructor(private canvas: HTMLCanvasElement) {}

    public async init()
    {
        // Setup
        await this.getGPU();
        this.connectCanvas();

        // Pipeline
        this.loadShaders();
        this.createBuffers();
        this.createPipeline();

        // Render
        this.createRenderPassDescriptor();
        this.render();
    }

    // Setup
    private async getGPU()
    {
        // Check if webgpu is supported
        if (!navigator.gpu)
        {
            this.onError("WebGPU not supported on this browser.");
            return;
        }

        // Get GPU adapter with a preference for high performance/discrete GPUs
        const adapter: GPUAdapter | null = await navigator.gpu.requestAdapter(
        {
            powerPreference: "high-performance"
        });

        if (!adapter)
        {
            this.onError("No GPU Adapter found.");
            return;
        }

        // Get logical interface
        this.device = await adapter.requestDevice();
    }

    private connectCanvas()
    {
        // // Connect canvas with GPU interface
        const context = this.canvas.getContext("webgpu");

        if (!context)
        {
            this.onError("Failed to get canvas context :/");
            return;
        }

        this.context = context;

        this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure(
        {
            device: this.device,
            format: this.canvasFormat // texture format
        });
    }

    // Pipeline
    private loadShaders()
    {
        // Vertex and Fragment shaders
        this.shaderModule = this.device.createShaderModule(
            {
                label: "Ray tracing shader",
                code:
                /* wgsl */ `
                @group(0) @binding(0) var<uniform> scene: Scene;

                // TODO: Is using a TS var here okay?
                const MAX_SIZE = ${this.MAX_SIZE};

                struct Scene
                {
                    numObjects : u32,
                    sphereOffset: u32,
                    planeOffset: u32,
                    coneOffset: u32,

                    data: array<vec4<f32>, MAX_SIZE / 4>
                };

                struct Sphere
                {
                    center: f32,
                    radius: f32
                };

                struct Plane
                {
                    center: f32
                };

                struct Cone
                {
                    center: f32
                };

                @vertex
                fn vertexMain(@location(0) pos: vec2f)
                    -> @builtin(position) vec4f
                {
                    return vec4f(pos, 0.0f, 1.0f);
                }

                @fragment
                fn fragmentMain() -> @location(0) vec4f
                {
                    return vec4f(1.0f, 0.0f, 0.0f, 1.0f);
                }
                `
            });
    }

    private createBuffers()
    {
        // Vertex Buffer
        const vertices = new Float32Array(
        [
            // 2 Triangles
            // X, Y
            -1.0, -1.0,
            +1.0, -1.0,
            -1.0, +1.0,
            +1.0, +1.0
        ]);

        this.vertexBuffer = this.device.createBuffer(
        {
            label: "Vertex Buffer",
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });

        const indices = new Uint16Array([0, 1, 2, 1, 3, 2]);

        // Index Buffer
        this.indexBuffer = this.device.createBuffer(
        {
            label: "Index Buffer",
            size: indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
        });

        // Uniform buffer
        this.uniforms = new Float32Array(this.MAX_SIZE);
        this.uniformBuffer = this.device.createBuffer(
        {
            label: "Uniform buffer",
            size: this.uniforms.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Write buffers
        this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices);
        this.device.queue.writeBuffer(this.indexBuffer, 0, indices);
    }

    private createPipeline()
    {
        // Vertex Buffer Layout
        const vertexBufferLayout : GPUVertexBufferLayout =
        {
            // 2 values per vertex (x, y)
            arrayStride: 8,
            attributes:
            [{
                format: "float32x2",
                offset: 0,
                shaderLocation: 0
            }]
        };

        this.bindGroupLayout = this.device.createBindGroupLayout(
        {
            label: "Raytracer bind group layout",
            entries: 
            [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: {} // Uniform buffer
            }]
        });

        // TODO: Should this be done here or elsewhere?
        this.bindGroup = this.device.createBindGroup(
        {
            label: "Vertex Bind group",
            layout: this.bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.uniformBuffer }
                }
            ]
        });

        // Pipeline Layout
        const pipelineLayout = this.device.createPipelineLayout(
        {
            label: "Raytracer Pipeline Layout",
            bindGroupLayouts: [ this.bindGroupLayout ]
        });

        // Pipeline
        this.pipeline = this.device.createRenderPipeline(
        {
            label: "Raytracing pipeline",
            layout: pipelineLayout,
            vertex:
            {
                module: this.shaderModule,
                entryPoint: "vertexMain",
                buffers: [vertexBufferLayout]
            },
            fragment:
            {
                module: this.shaderModule,
                entryPoint: "fragmentMain",
                targets: 
                [{
                    format: this.canvasFormat
                }]
            },
            primitive:
            {
                topology: 'triangle-list',
            }
        });
    }

    // Rendering
    private createRenderPassDescriptor()
    {
        this.renderPassDescriptor =
        {
            label: "Render Pass Description",
            colorAttachments:
            [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: [0.2, 0.2, 0.2, 1],
                loadOp: "clear",
                storeOp: "store",
            }]
        };
    }

    private render()
    {
        // update view
        (this.renderPassDescriptor.colorAttachments as any)[0].view = 
            this.context.getCurrentTexture().createView();
        
        // create command buffer
        const encoder = this.device.createCommandEncoder();

        // renderpass
        const pass = encoder.beginRenderPass(this.renderPassDescriptor);

        pass.setIndexBuffer(this.indexBuffer, "uint16");
        pass.setPipeline(this.pipeline);
        pass.setVertexBuffer(0, this.vertexBuffer);            
        pass.setBindGroup(0, this.bindGroup);
        
        // Hard-coded since we're only using the vertex buffer for drawing
        // 2 triangles to cover the screen
        pass.drawIndexed(6);

        pass.end();

        // Finish command buffer and immediately submit it
        this.device.queue.submit([encoder.finish()]);

        // Loop every frame
        requestAnimationFrame(() => this.render());
    }


    // Misc
    private onError(msg: string)
    {
        document.body.innerHTML = `<p>${msg}</p>`;
        console.error(msg);
    }
}
