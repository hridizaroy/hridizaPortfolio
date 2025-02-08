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

                fn viewTransformMatrix(eye: vec3f, lookAt: vec3f, up: vec3f) -> mat4x4<f32>
                {
                    var n = normalize(lookAt - eye);
                    var u = normalize(cross(up, -n));
                    var v = normalize(cross(n, u));

                    return mat4x4<f32>(
                        u.x, v.x, n.x, 0.0f,
                        u.y, v.y, n.y, 0.0f,
                        u.z, v.z, n.z, 0.0f,
                        -dot(eye, u), -dot(eye, v), -dot(eye, n), 1.0f
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


                // TODO: Redo
                struct Plane {
                    point: vec3<f32>, // A point on the plane
                    normal: vec3<f32>, // The normal to the plane
                };

                struct Quad {
                    v0: vec3<f32>, // First vertex (corner of the quad)
                    v1: vec3<f32>, // Second vertex
                    v2: vec3<f32>, // Third vertex
                    v3: vec3<f32>, // Fourth vertex
                    normal: vec3<f32>, // Normal to the plane
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

                struct Cylinder {
                    radius: f32,    // Radius of the cylinder
                    height: f32,    // Height of the cylinder (the cylinder extends from -height/2 to +height/2 along the Y-axis)
                    center: vec3<f32>, // Center of the cylinder (in this case, it will be aligned with Y-axis)
                };

                fn intersect_ray_cylinder(ray: Ray, cylinder: Cylinder) -> f32 {
                    // Compute the direction and origin of the ray with respect to the cylinder's position
                    let ray_origin = ray.origin - cylinder.center;  // Translate the ray origin to the cylinder's local space
                
                    // Ray direction components
                    let a = ray.dir.x * ray.dir.x + ray.dir.z * ray.dir.z;
                    let b = 2.0 * (ray_origin.x * ray.dir.x + ray_origin.z * ray.dir.z);
                    let c = ray_origin.x * ray_origin.x + ray_origin.z * ray_origin.z - cylinder.radius * cylinder.radius;
                
                    // Solve the quadratic equation: a*t^2 + b*t + c = 0
                    let discriminant = b * b - 4.0 * a * c;
                
                    if discriminant < 0.0 {
                        return -1.0; // No intersection
                    }
                
                    // Compute the two possible intersection distances (t1, t2)
                    let sqrt_discriminant = sqrt(discriminant);
                    let t1 = (-b - sqrt_discriminant) / (2.0 * a);
                    let t2 = (-b + sqrt_discriminant) / (2.0 * a);
                
                    // Check if the intersection points are within the cylinder's height
                    // The cylinder's height extends from -height/2 to +height/2 along the Y-axis
                    let y1 = ray.origin.y + t1 * ray.dir.y;
                    let y2 = ray.origin.y + t2 * ray.dir.y;
                
                    if (y1 >= -cylinder.height / 2.0 && y1 <= cylinder.height / 2.0) {
                        return t1; // First intersection is valid
                    }
                
                    if (y2 >= -cylinder.height / 2.0 && y2 <= cylinder.height / 2.0) {
                        return t2; // Second intersection is valid
                    }
                
                    return -1.0; // No intersection within the cylinder's height
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

                    var eye = vec3f(-0.2f, 0.1f, -1.0f);

                    // Temp
                    var sphere: Sphere;
                    sphere.radius = 1.0f;
                    sphere.center = vec3f(-0.15f, -0.25f, 1.5f);

                    var sphere2: Sphere;
                    sphere2.radius = 0.93f;
                    sphere2.center = vec3f(0.6f, 0.6f, 2.0f);

                    var plane: Plane;
                    plane.normal = vec3f(1.0f, -1.0f, 0.0f);
                    plane.point = vec3f(-0.15f, 1.75f, 1.5f);

                    var view = viewTransformMatrix(
                        eye,
                        vec3f(12.687f, -2.639f, 3.0f),
                        vec3f(0.0f, -1.0f, 0.0f)
                    );

                    // var rayDir : vec3f = normalize(vec3f(pixelVal, cam.focalLength));
                    var rayDir : vec3f = normalize(vec3f(uv, cam.focalLength));

                    // return vec4f(rayDir, 1.0f);

                    var ray: Ray;
                    ray.origin = eye;
                    ray.dir = rayDir;

                    // sphere.center = (view * vec4f(sphere.center, 1.0f)).xyz;
                    // sphere.center.y *= -1.0f;
                    // sphere.center.z *= -1.0f;

                    // return vec4f((sphere.center.z) * 0.85, 0.0f, 0.0f, 1.0f);

                    let quad = Quad(
                        vec3<f32>(0.0, 0.2, -2.0), // First vertex of the quad
                        vec3<f32>( 10.5, 0.2, 25.5), // Second vertex
                        vec3<f32>( 1.0, 0.2,  1.0), // Third vertex
                        vec3<f32>(-1.0, 0.2,  1.2), // Fourth vertex
                        vec3<f32>(0.0, -1.0, 0.0)    // Normal vector of the plane (pointing up)
                    );

                    let cylinder = Cylinder(0.5, 1.0, vec3<f32>(0.2, 0.7, 1.0f));

                    let t = intersect_ray_cylinder(ray, cylinder);
                
                    let validHitDist3 = max(0.0f, intersect_ray_quad(ray, quad));

                    var validHitDist1 = max(0.0f, sphereRayIntersectDist(ray, sphere));
                    var validHitDist2 = max(0.0f, sphereRayIntersectDist(ray, sphere2));
                    // var validHitDist3 = max(0.0f, intersect_ray_plane(ray, plane));

                    var minHitDist = 1e6;  // Some large number

                    // Now find the least positive value
                    // if (validHitDist1 > 0.0f && validHitDist1 < minHitDist) {
                    //     minHitDist = validHitDist1;
                    // }
                    // if (validHitDist2 > 0.0f && validHitDist2 < minHitDist) {
                    //     minHitDist = validHitDist2;
                    // }
                    // if (validHitDist3 > 0.0f && validHitDist3 < minHitDist) {
                    //     minHitDist = validHitDist3;
                    // }

                    

                    // If minHitDist is still large, there was no intersection
                    // if (minHitDist == validHitDist1)
                    // {
                    //     return vec4f(0.8f, 0.0f, 0.0f, 1.0f); // Return gray color if there's a valid intersection
                    // }
                    // else if minHitDist == validHitDist2
                    // {
                    //     return vec4f(0.0f, 0.8f, 0.0f, 1.0f);
                    // }
                    // else if minHitDist == validHitDist3
                    // {
                    //     return vec4f(0.5f, 0.5f, 0.5f, 0.1f);
                    // }

                    if (t > 0.0f)
                    {
                        return vec4f(0.0f, 0.0f, 1.0f, 1.0f);
                    }

                    if (validHitDist1 > 0.0f && validHitDist2 > 0.0f)
                    {
                        if (validHitDist1 < validHitDist2)
                        {
                            return vec4f(1.0f, 0.0f, 0.0f, 1.0f);
                        }

                        return vec4f(0.0f, 1.0f, 0.0f, 1.0f);
                    }
                    else if (validHitDist1 > 0.0f)
                    {
                        return vec4f(1.0f, 0.0f, 0.0f, 1.0f);
                    }
                    else if (validHitDist2 > 0.0f)
                    {
                        return vec4f(0.0f, 1.0f, 0.0f, 1.0f);
                    }
                    else if (validHitDist3 > 0.0f)
                    {
                        return vec4f(0.5f, 0.5f, 0.5f, 1.0f);
                    }

                    return vec4f(0.0f, 0.0f, 0.0f, 1.0f);
                }

                // @fragment
                // fn fragmentMain(@builtin(position) fragCoord: vec4f)
                //     -> @location(0) vec4f
                // {
                //     // NOTE: +Y is towards the bottom of the screen

                //     // TODO: Making everything square for now, but will need to deal with aspect ratios later
                //     let resolution = vec2f(500.0f, 500.0f);
                //     let aspect = resolution.x / resolution.y;
                //     let uv : vec2f = (fragCoord.xy / resolution.y) * 2.0f - 1.0f;


                //     var cam: Camera;
                //     cam.imageHeight = resolution.y;
                //     cam.imageWidth = resolution.x;
                //     cam.focalLength = 1.0f;
                //     cam.filmPlaneHeight = 25.0f;
                //     cam.filmPlaneWidth = 25.0f;


                //     let w : f32 = cam.filmPlaneWidth/cam.imageWidth;
                //     let h : f32 = cam.filmPlaneHeight/cam.imageHeight;
                //     let pixelVal = vec2f((fragCoord.x - resolution.x * 0.5f) * w,
                //                     (fragCoord.y - resolution.y * 0.5f) * h)
                //                     + vec2f(0.5f * w, 0.5f * h);

                //     var eye = vec3f(16.687f, -2.639f, 0.2f);

                //     // Temp
                //     var sphere: Sphere;
                //     sphere.radius = 3.5f;
                //     sphere.center = vec3f(9.11, -1.63, 1.352);

                //     var view = viewTransformMatrix(
                //         eye,
                //         vec3f(12.687f, -2.639f, 3.0f),
                //         vec3f(0.0f, -1.0f, 0.0f)
                //     );

                //     var rayDir : vec3f = normalize(vec3f(pixelVal, cam.focalLength));

                //     // return vec4f(rayDir, 1.0f);

                //     var ray: Ray;
                //     ray.origin = vec3f(0.0f);
                //     ray.dir = rayDir;

                //     sphere.center = (view * vec4f(sphere.center, 1.0f)).xyz;
                //     // sphere.center.y *= -1.0f;
                //     // sphere.center.z *= -1.0f;

                //     // return vec4f((sphere.center.z) * 0.85, 0.0f, 0.0f, 1.0f);

                //     var hitDist = sphereRayIntersectDist(ray, sphere);

                //     if (hitDist > 0.0f)
                //     {
                //         return vec4f(1.0f, 0.0f, 0.0f, 1.0f);
                //     }

                //     return vec4f(0.0f, (hitDist) * 0.1, 0.0f, 1.0f);
                // }
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
