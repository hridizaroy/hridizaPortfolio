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
                    radius: f32,
                    color: vec3f
                };

                struct Triangle
                {
                    v0: vec3f,
                    v1: vec3f,
                    v2: vec3f,
                    color: vec3f
                }

                // struct Quad
                // {
                //     v0: vec3f,
                //     v1: vec3f,
                //     v2: vec3f,
                //     v3: vec3f,
                //     color: vec3f
                // };

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


                struct Quad
                {
                    v0: vec3<f32>, // First vertex (corner of the quad)
                    v1: vec3<f32>, // Second vertex
                    v2: vec3<f32>, // Third vertex
                    v3: vec3<f32>, // Fourth vertex
                    normal: vec3<f32>, // Normal to the plane,
                    color: vec3<f32>
                };

                fn intersect_ray_plane(ray: Ray, plane_point: vec3<f32>, plane_normal: vec3<f32>) -> f32 {
                    let denom = dot(plane_normal, ray.dir);
                    
                    if abs(denom) < 1e-6 {
                        return -1.0; // Ray is parallel to the plane, no intersection
                    }
                    
                    let t = dot(plane_normal, plane_point - ray.origin) / denom;
                    
                    if t < 0.0 {
                        return -1.0; // Intersection is behind the ray origin, invalid
                    }
                    
                    return t; // Return the distance along the ray to the intersection point
                }


                fn is_point_inside_quad(point: vec3<f32>, quad: Quad) -> bool {
                    // Barycentric coordinates or point-in-polygon test (for a quadrilateral)
                    
                    // Calculate vectors for the quadrilateral edges
                    let v0v1 = quad.v1 - quad.v0;
                    let v0v3 = quad.v3 - quad.v0;
                    let v0p = point - quad.v0;
                    let v1v2 = quad.v2 - quad.v1;
                    let v1p = point - quad.v1;
                    
                    // Use dot products to check if the point is inside the quad
                    let d00 = dot(v0v1, v0v1);
                    let d01 = dot(v0v1, v0v3);
                    let d11 = dot(v0v3, v0v3);
                    let d20 = dot(v0p, v0v1);
                    let d21 = dot(v0p, v0v3);
                    let d02 = dot(v1v2, v1v2);
                    let d12 = dot(v1v2, v1p);
                    
                    // Calculate barycentric coordinates
                    let denom = d00 * d11 - d01 * d01;
                    let alpha = (d11 * d20 - d01 * d21) / denom;
                    let beta = (d00 * d21 - d01 * d20) / denom;
                
                    // The point is inside the quad if alpha and beta are between 0 and 1
                    return alpha >= 0.0 && beta >= 0.0 && (alpha + beta) <= 1.0;
                }
                
                fn intersect_ray_quad(ray: Ray, quad: Quad) -> f32 {
                    // First, find the intersection with the plane of the quad
                    let hitDist = intersect_ray_plane(ray, quad.v0, quad.normal);
                    
                    if hitDist < 0.0 {
                        return -1.0; // No intersection, or intersection behind the ray origin
                    }
                    
                    // Now calculate the intersection point
                    let intersection_point = ray.origin + hitDist * ray.dir;
                    
                    // Check if the intersection point lies inside the quad
                    if is_point_inside_quad(intersection_point, quad) {
                        return hitDist; // Return the distance to the intersection point
                    }
                    
                    return -1.0; // The point is outside the quad, no valid intersection
                }


                fn triangleRayIntersectDist(ray: Ray, triangle: Triangle) -> f32
                {
                    // Compute the plane's normal
                    // Note: We reverse the cross product order to account for +Y being downwards
                    var v0v1 : vec3f = triangle.v1 - triangle.v0;
                    var v0v2 : vec3f = triangle.v2 - triangle.v0;
                    var N : vec3f = cross(v0v2, v0v1); // Reversed order
                 
                    // Check if the ray and plane are parallel
                    var NdotRayDirection : f32 = dot(N, ray.dir);
                    if (abs(NdotRayDirection) < 0.0005) // Almost 0
                    {
                        return -1.0; // They are parallel, so they don't intersect!
                    }
                
                    // Compute d parameter
                    var d : f32 = -dot(N, triangle.v0);
                    
                    // Compute t
                    var t = -(dot(N, ray.origin) + d) / NdotRayDirection;
                    
                    // Check if the triangle is behind the ray
                    if (t < 0.0)
                    {
                        return -1.0; // The triangle is behind
                    }
                 
                    // Compute the intersection point
                    var P: vec3f = ray.origin + t * ray.dir;
                 
                    // Inside-Outside Test
                    var edge0 = triangle.v1 - triangle.v0;
                    var edge1 = triangle.v2 - triangle.v1;
                    var edge2 = triangle.v0 - triangle.v2;
                    
                    var C0 = P - triangle.v0;
                    var C1 = P - triangle.v1;
                    var C2 = P - triangle.v2;
                    
                    // Note: We reverse the cross product order here as well
                    if (dot(N, cross(edge0, C0)) > 0.0 &&
                        dot(N, cross(edge1, C1)) > 0.0 &&
                        dot(N, cross(edge2, C2)) > 0.0)
                    {
                        return t; // The ray hits the triangle
                    }
                
                    return -1.0; // The ray doesn't hit the triangle
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

                    // pixels
                    cam.imageHeight = resolution.y;
                    cam.imageWidth = resolution.x;

                    // TODO: Is the focal length okay?
                    // Look into projection matrix/perspective logic?
                    // Should I be checking for intersections "only past the focal plane"?
                    cam.filmPlaneHeight = 25.0f;
                    cam.filmPlaneWidth = 25.0f;

                    let fov = 5.0f;
                    cam.focalLength = (cam.filmPlaneHeight / 2.0f)/tan(radians(fov/2));


                    let w : f32 = cam.filmPlaneWidth/cam.imageWidth;
                    let h : f32 = cam.filmPlaneHeight/cam.imageHeight;
                    let pixelVal = vec2f((fragCoord.x - resolution.x * 0.5f) * w,
                                    (fragCoord.y - resolution.y * 0.5f) * h)
                                    + vec2f(0.5f * w, 0.5f * h);

                    var eye = vec3f(389.486, -1.855, 0.573);
                    // var eye = vec3f(0.0, 0.0, -500.0f);

                    // Temp
                    var sphere: Sphere;
                    sphere.radius = 6.0f;
                    sphere.center = vec3f(6.11, 5.0, 6.0);
                    sphere.color = vec3f(0.0f, 0.8f, 0.0f);

                    var sphere2: Sphere;
                    sphere2.radius = 8.0f;
                    sphere2.center = vec3f(11.361, -2.813, 0.124);
                    sphere2.color =  vec3f(0.8f, 0.0f, 0.0f);

                    let quad = Quad(
                        vec3<f32>(110.0, 20.0f, 6000.0), // First vertex of the quad
                        vec3<f32>( 0.0, 20.0f, -1500.0), // Second vertex
                        vec3<f32>( -110.0, 20.0f,  -2000.0), // Third vertex
                        vec3<f32>(-110.0, 20.0f,  8000.0), // Fourth vertex
                        vec3<f32>(0.0, -1.0, 0.0),    // Normal vector of the plane (pointing up,
                        vec3<f32>(0.1f, 0.1f, 0.8f) // color
                    );

                    var triangle: Triangle;
                    triangle.v0 = vec3f(-100.0, 100.0, 100.0);
                    triangle.v1 = vec3f(100.0, 100.0, 100.0);
                    triangle.v2 = vec3f(0.0, -100.0, 100.0);
                    triangle.color = vec3f(0.0f, 0.0f, 1.0f);

                    var view = viewTransformMatrix(
                        eye,
                        sphere2.center,
                        vec3f(0.0f, 1.0f, 0.0f)
                    );

                    var rayDir : vec3f = normalize(vec3f(pixelVal, cam.focalLength));

                    var ray: Ray;
                    ray.origin = vec3f(0.0f);
                    ray.dir = rayDir;

                    sphere.center = (view * vec4f(sphere.center, 1.0f)).xyz;
                    sphere2.center = (view * vec4f(sphere2.center, 1.0f)).xyz;

                    var hitDist1 = max(0.0f, sphereRayIntersectDist(ray, sphere));
                    var hitDist2 = max(0.0f, sphereRayIntersectDist(ray, sphere2));
                    var hitDist3 = max(0.0f, intersect_ray_quad(ray, quad));

                    var dist: array<f32, 3> = array<f32, 3>(hitDist1, hitDist2, hitDist3);

                    var minDist: f32 = 10000.0f;
                    var minIdx: i32 = -1;

                    for (var ii: i32; ii < 3; ii++)
                    {
                        if (dist[ii] < minDist && dist[ii] > cam.focalLength)
                        {
                            minDist = dist[ii];
                            minIdx = ii;
                        }
                    }

                    var returnColor: vec3f;

                    if (minIdx == -1)
                    {
                        returnColor = vec3f(0.0f, 0.0f, 0.0f);
                    }
                    else if (minIdx == 0)
                    {
                        returnColor = sphere.color;
                    }
                    else if (minIdx == 1)
                    {
                        returnColor = sphere2.color;
                    }
                    else
                    {
                        returnColor = quad.color;
                    }

                    return vec4f(returnColor, 1.0f);
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
