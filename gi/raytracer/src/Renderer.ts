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

                struct Ray
                {
                    origin: vec3f,
                    dir: vec3f
                }

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
                    center: vec3f,
                    radius: f32
                };

                // struct Plane
                // {
                //     center: vec3f
                // };

                struct Cone
                {
                    center: vec3f
                };

                struct Camera
                {
                    focalLength: f32,

                    // TODO: Make these vectors
                    imageHeight: f32,
                    imageWidth: f32,
                    filmPlaneHeight: f32,
                    filmPlaneWidth: f32
                };

                @vertex
                fn vertexMain(@location(0) pos: vec2f)
                    -> @builtin(position) vec4f
                {
                    return vec4f(pos, 0.0f, 1.0f);
                }

                fn viewTransformMatrix(eye: vec3f, lookAt: vec3f, down: vec3f) -> mat4x4<f32>
                {
                    var forward = normalize(lookAt - eye);
                    var right = normalize(cross(down, forward));
                    var d = normalize(cross(right, -forward));

                    return mat4x4<f32>(
                        right.x, d.x, forward.x, 0.0f,
                        right.y, d.y, forward.y, 0.0f,
                        right.z, d.z, forward.z, 0.0f,
                        -dot(eye, right), -dot(eye, d), -dot(eye, forward), 1.0f
                    );
                }

                fn sphereRayIntersectDist(ray: Ray, sphere: Sphere) -> f32
                {
                    var raySphereToCam: vec3f = ray.origin - sphere.center;

                    var a : f32 = dot(ray.dir, ray.dir);
                    var b : f32 = 2.0f * dot(ray.dir, raySphereToCam);
                    var c : f32 = dot(raySphereToCam, raySphereToCam) - sphere.radius * sphere.radius;
                    var discriminant = b * b - 4.0f * a * c;

                    if (discriminant < 0.0f)
                    {
                        return -1.0f;
                    }

                    var closestT: f32 = (-b - sqrt(discriminant)) / (2.0f * a);

                    // Make sure that the hit is in front of the camera
                    return select(-1.0f, closestT, closestT > 0.0f);
                }

                @fragment
                fn fragmentMain(@builtin(position) fragCoord: vec4f)
                    -> @location(0) vec4f
                {
                    // NOTE: +Y is towards the bottom of the screen

                    // TODO: Making everything square for now, but will need to deal with aspect ratios later
                    let resolution = vec2f(500.0f, 500.0f);
                    let aspect = resolution.x / resolution.y;
                    let uv : vec2f = (fragCoord.xy / resolution.y) * 2.0f - 1.0f;


                    var cam: Camera;
                    cam.imageHeight = resolution.y;
                    cam.imageWidth = resolution.x;
                    cam.focalLength = 1.0f;
                    cam.filmPlaneHeight = 25.0f;
                    cam.filmPlaneWidth = 25.0f;


                    let w : f32 = cam.filmPlaneWidth/cam.imageWidth;
                    let h : f32 = cam.filmPlaneHeight/cam.imageHeight;
                    let pixelVal = vec2f((fragCoord.x - resolution.x * 0.5f) * w,
                                    (fragCoord.y - resolution.y * 0.5f) * h)
                                    + vec2f(0.5f * w, 0.5f * h);

                    var eye = vec3f(100.0f, -80.0f, -10.0f);

                    // Temp
                    var sphere: Sphere;
                    sphere.radius = 2.8f;
                    sphere.center = vec3f(100.0f, -80.0f, -2.0f);

                    var sphere2: Sphere;
                    sphere2.radius = 10.0f;
                    sphere2.center = vec3f(70.0f, -80.0f, 10.0f);

                    var view = viewTransformMatrix(
                        eye,
                        sphere2.center,
                        vec3f(0.0f, 1.0f, 0.0f)
                    );

                    var rayDir : vec3f = normalize(vec3f(pixelVal, cam.focalLength));
                    // var rayDir : vec3f = normalize(vec3f(uv, cam.focalLength));

                    // return vec4f(rayDir, 1.0f);

                    var ray: Ray;
                    ray.origin = vec3f(0.0f);
                    ray.dir = rayDir;

                    sphere.center = (view * vec4f(sphere.center, 1.0f)).xyz;
                    sphere2.center = (view * vec4f(sphere2.center, 1.0f)).xyz;

                    // return vec4f((sphere.center.z) * 0.85, 0.0f, 0.0f, 1.0f);

                    var hitDist1 = sphereRayIntersectDist(ray, sphere);
                    var hitDist2 = sphereRayIntersectDist(ray, sphere2);

                    // if (hitDist2 > 0.0f)
                    // {
                    //     return vec4f(0.0f, 0.8f, 0.0f, 1.0f);
                    // }

                    if (hitDist1 > 0.0f && hitDist2 > 0.0f)
                    {
                        if (hitDist1 < hitDist2)
                        {
                            return vec4f(0.0f, 0.8f, 0.0f, 1.0f);
                        }

                        return vec4f(0.8f, 0.0f, 0.0f, 1.0f);
                    }
                    else if (hitDist1 > 0.0f)
                    {
                        return vec4f(0.0f, 0.8f, 0.0f, 1.0f);
                    }
                    else if (hitDist2 > 0.0f)
                    {
                        return vec4f(0.8f, 0.0f, 0.0f, 1.0f);
                    }

                    return vec4f(0.0f, 0.0f, 0.0f, 1.0f);
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
