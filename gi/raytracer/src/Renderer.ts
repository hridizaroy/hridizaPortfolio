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
        await this.createPipeline();

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
        // Connect canvas with GPU interface
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
                @group(0) @binding(0) var<uniform> spheres: Spheres;

                @group(0) @binding(1) var mySampler: sampler;
                @group(0) @binding(2) var myTexture: texture_2d<f32>;

                // TODO: Is using a TS var here okay?
                const MAX_SIZE = ${this.MAX_SIZE};

                struct Ray
                {
                    origin: vec3f,
                    dir: vec3f,
                    exists: bool,
                    k: f32
                }

                struct Sphere
                {
                    center: vec3f,
                    radius: f32,
                    color: vec3f,
                    kr: f32,
                    kt: f32,
                    refractionIdx: f32,
                };

                struct Triangle
                {
                    v0: vec3f,
                    v1: vec3f,
                    v2: vec3f,
                    color: vec3f
                }

                struct Camera
                {
                    focalLength: f32,

                    // TODO: Make these vectors
                    imageHeight: f32,
                    imageWidth: f32,
                    filmPlaneHeight: f32,
                    filmPlaneWidth: f32
                };

                
                struct Spheres
                {
                    numObjects : f32,
                    data: array<Sphere, MAX_SIZE>
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
                    color: vec3<f32>,
                    kr: f32,
                    kt: f32,
                    refractionIdx: f32
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
                    // Split the quad into two triangles
                    let v0 = quad.v0;
                    let v1 = quad.v1;
                    let v2 = quad.v2;
                    let v3 = quad.v3;
                
                    // First triangle (v0, v1, v2)
                    let triangle1 = is_point_inside_triangle(point, v0, v1, v2);
                
                    // Second triangle (v0, v2, v3)
                    let triangle2 = is_point_inside_triangle(point, v0, v2, v3);
                
                    // Point is inside the quad if it is inside either of the two triangles
                    return triangle1 || triangle2;
                }
                
                fn is_point_inside_triangle(point: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> bool {
                    // Calculate vectors for the triangle edges
                    let v0v1 = v1 - v0;
                    let v0v2 = v2 - v0;
                    let v0p = point - v0;
                    let v1v2 = v2 - v1;
                    let v1p = point - v1;
                
                    // Use dot products to calculate barycentric coordinates
                    let d00 = dot(v0v1, v0v1);
                    let d01 = dot(v0v1, v0v2);
                    let d11 = dot(v0v2, v0v2);
                    let d20 = dot(v0p, v0v1);
                    let d21 = dot(v0p, v0v2);
                    let d02 = dot(v1v2, v1v2);
                    let d12 = dot(v1v2, v1p);
                
                    // Calculate the barycentric coordinates
                    let denom = d00 * d11 - d01 * d01;
                    let alpha = (d11 * d20 - d01 * d21) / denom;
                    let beta = (d00 * d21 - d01 * d20) / denom;
                
                    // The point is inside the triangle if alpha, beta, and (alpha + beta) are within [0, 1]
                    return alpha >= 0.0 && beta >= 0.0 && (alpha + beta) <= 1.0;
                }
                
                
                struct QuadData
                {
                    hitDist: f32,
                    intersectPoint: vec3f
                }
                fn intersect_ray_quad(ray: Ray, quad: Quad) -> QuadData {
                    // First, find the intersection with the plane of the quad
                    let hitDist = intersect_ray_plane(ray, quad.v0, quad.normal);
                    
                    var returnData: QuadData;
                    returnData.hitDist = -1.0;
                    if hitDist < 0.0 {
                        return returnData; // No intersection, or intersection behind the ray origin
                    }
                    
                    // Now calculate the intersection point
                    let intersection_point = ray.origin + hitDist * ray.dir;
                    returnData.intersectPoint = intersection_point;
                    
                    // Check if the intersection point lies inside the quad
                    if is_point_inside_quad(intersection_point, quad) {
                        returnData.hitDist = hitDist;
                        return returnData; // Return the distance to the intersection point
                    }
                    
                    return returnData; // The point is outside the quad, no valid intersection
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

                // TODO: Return both hit distances
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
                    var t2: f32 = (-b + sqrt(discriminant)) / (2.0f * a);

                    // Make sure that the hit is in front of the camera
                    if (closestT <= 0.09 && t2 <= 0.09)
                    {
                        return -1.0f;
                    }

                    return select(t2, closestT, closestT > 0.05f);
                }

                fn reflect(I: vec3<f32>, N: vec3<f32>) -> vec3<f32>
                {
                    return I - 2.0 * dot(I, N) * N;
                }

                struct Light
                {
                    position: vec3f,
                    intensity: f32,
                    color: vec3f,
                    ka: f32,
                    kd: f32,
                    ks: f32
                }

                struct IllumData
                {
                    normal: vec3f,
                    S: vec3f,
                    R: vec3f,
                    H: vec3f,
                    V: vec3f
                }

                struct Payload
                {
                    intersection: bool,
                    color: vec4f,
                    reflectionRay: Ray,
                    nextKr: f32,
                    transmissionRay: Ray,
                    nextKt: f32
                }

                fn calculateTransmissionRay(ir1: f32, ir2: f32, iRay: vec3f, n: vec3f) -> vec3f
                {
                    var indexRefRatio = ir1/ir2;
                    var cosTi = dot(-iRay, n);
                    var sin2Tt = pow(indexRefRatio, 2.0f) * (1.0f - pow(cosTi, 2.0f));

                    var transmissionRay = (indexRefRatio) * iRay + 
                                            (indexRefRatio * cosTi - sqrt(1.0f - sin2Tt)) * n;

                    return transmissionRay;
                }  
                
                fn calculateTransmissionRay2(ir1: f32, ir2: f32, iRay: vec3f, n: vec3f) -> vec3f
                {
                    // Assume all vectors are normalized
                    var theta1 = acos(dot(-iRay, n));
                    var sinTheta2 = ir1 * sin(theta1)/ir2;
                    
                    if (sinTheta2 > 1.0f)
                    {
                        return vec3(0.0f);
                    }
                    
                    var theta2 = asin(sinTheta2);

                    var rightDir = normalize(cross(n, cross(iRay, n)));
                    var transmissionRay: vec3f = cos(theta2) * (-n) + sin(theta2) * rightDir;

                    return normalize(transmissionRay);
                }  
                
                const quadRefVertex: vec3f = vec3<f32>(-30.0, 20.0f, -3000.0);
                const quadWidth = 60.0f;
                const quadLength = 6000.0f;

                const MAX_DEPTH: u32 = 7;
                fn illuminate(ray: Ray, focalLength: f32,
                    sphere: Sphere, sphere2: Sphere, quad: Quad
                ) -> Payload
                {
                    var hitDist1 = max(0.0f, sphereRayIntersectDist(ray, sphere));
                    var hitDist2 = max(0.0f, sphereRayIntersectDist(ray, sphere2));

                    var quadData: QuadData = intersect_ray_quad(ray, quad);
                    var hitDist3 = max(0.0f, quadData.hitDist);
                    var quadIntersectPoint = quadData.intersectPoint;

                    var dist: array<f32, 3> = array<f32, 3>(hitDist1, hitDist2, hitDist3);

                    var minDist: f32 = 10000.0f;
                    var minIdx: i32 = -1;

                    for (var ii: i32; ii < 3; ii++)
                    {
                        if (dist[ii] < minDist && dist[ii] > focalLength)
                        {
                            minDist = dist[ii];
                            minIdx = ii;
                        }
                    }

                    var returnColor: vec3f;

                    var skyColor = vec3f(0.78f, 0.91f, 1.0f);
                    
                    var ambientColor = vec3f(0.24f, 0.27f, 0.3f);

                    var specularColor = vec3f(1.0f, 1.0f, 1.0f);

                    var payload : Payload;

                    if (minIdx == -1)
                    {
                        payload.color = vec4f(skyColor, 1.0f);
                        payload.intersection = false;

                        return payload;
                    }
                    
                    payload.intersection = true;

                    /** TODO
                     * Create a struct for payload with intersection point, 
                     * intersection distance, normal etc.
                     * 
                     * Create a struct for representing scene objects
                     * 
                     * Handle the object hit identification more cleanly
                     * 
                     * Create a function for a shadow ray
                     * 
                     * Send in scene data from CPU side code
                     * 
                     * Reduce if statements to optimize shader
                     * 
                     * Rename color to irradiance where applicable
                     * 
                     * Store illum model constants in object
                     */

                    var light: Light;
                    light.position = vec3f(0.0f, -30.0, -20.0);
                    light.intensity = 8.0f;
                    light.color = vec3f(1.0f, 1.0f, 1.0f);
                    light.ka = 0.1f;
                    light.kd = 0.5f;
                    light.ks = 0.15f;

                    var light2: Light;
                    light2.position = vec3f(0.0f, -50.0, -30.0);
                    light2.intensity = 10.0f;
                    light2.color = vec3f(1.0f, 1.0f, 1.0f);

                    var shadowRay: Ray;
                    shadowRay.origin = ray.origin + minDist * ray.dir;
                    shadowRay.dir = normalize(light.position - shadowRay.origin);
                    shadowRay.exists = true;

                    if (minIdx == 0)
                    {
                        returnColor = sphere.color * light.intensity;
                        shadowRay.origin += normalize(shadowRay.origin - sphere.center) * 0.001;
                        payload.nextKr = sphere.kr;
                        payload.nextKt = sphere.kt;
                    }
                    else if (minIdx == 1)
                    {
                        returnColor = sphere2.color * light.intensity;
                        shadowRay.origin += normalize(shadowRay.origin - sphere2.center) * 0.001;
                        payload.nextKr = sphere2.kr;
                        payload.nextKt = sphere2.kt;
                    }
                    else
                    {
                        returnColor = quad.color * light.intensity;
                        shadowRay.origin += vec3f(0.0f, -0.1, 0.0f);
                        payload.nextKr = quad.kr;
                        payload.nextKt = quad.kt;
                    }

                    var lightHitDist = max(0.0f, length(light.position - shadowRay.origin));
                    hitDist1 = max(0.0f, sphereRayIntersectDist(shadowRay, sphere));
                    hitDist2 = max(0.0f, sphereRayIntersectDist(shadowRay, sphere2));

                    var quadData2: QuadData = intersect_ray_quad(shadowRay, quad);
                    hitDist3 = max(0.0f, quadData2.hitDist);

                    var dist2: array<f32, 4> = array<f32, 4>(hitDist1, hitDist2, hitDist3, lightHitDist);

                    var minDist2 = 10000.0f;
                    var minIdx2 = -1;

                    for (var ii: i32; ii < 4; ii++)
                    {
                        if (dist2[ii] < minDist2 && dist2[ii] > 0.0f)
                        {
                            minDist2 = dist2[ii];
                            minIdx2 = ii;
                        }
                    }

                    var localIllum: vec4f;

                    // if the shadow ray reaches the light
                    if (minIdx2 == 3)
                    {
                        var illumData: IllumData;
                        if (minIdx == 0)
                        {
                            illumData.normal = normalize(shadowRay.origin - sphere.center);
                        }
                        else if (minIdx == 1)
                        {
                            illumData.normal = normalize(shadowRay.origin - sphere2.center);
                        }
                        else if (minIdx == 2) @diagnostic(off, derivative_uniformity)
                        {
                            illumData.normal = vec3f(0.0f, -1.0f, 0.0f);

                            // quad uv coords
                            var uv: vec3f = quadData.intersectPoint - quadRefVertex;
                            uv.x = uv.x / quadWidth;
                            uv.y = uv.z / quadLength;
                            uv.z = 0.0f;

                            // returnColor = textureSample(myTexture, mySampler, uv.xy).xyz * light.intensity;

                            var uvScale = 10.0f;
                            var scaledUV = uv * uvScale;
                            var uvFloor = floor(scaledUV);

                            // var minDist: f32 = 100.0f;
                            // for (var dx: f32 = -1.0; dx <= 1.0; dx+=1.0f)
                            // {
                            //     for (var dy: f32 = -1.0; dy <= 1.0; dy+= 1.0f)
                            //     {
                            //         var neighbor = uvFloor.xy + vec2f(dx, dy);

                            //         var uvNoiseX: f32 = random2D(neighbor);
                            //         var uvNoiseY: f32 = random(uvNoiseX);
        
                            //         var voronoiPoint = neighbor + vec2f(uvNoiseX, uvNoiseY);
                                    
                            //         var currDist = length(voronoiPoint - scaledUV.xy);

                            //         minDist = min(minDist, currDist);
                            //     }
                            // }

                            // returnColor = vec3f(1.0 - minDist) * light.intensity;

                            if (u32(scaledUV.x) % 2 == u32(scaledUV.y) % 2)
                            {
                                returnColor = vec3f(1.0, 0.0, 0.0) * light.intensity;
                            }
                            else
                            {
                                returnColor = vec3f(1.0, 1.0, 0.0) * light.intensity;
                            }

                            var uvNoise = random2D(uvFloor.xy);

                            if (uvNoise < 0.3 )
                            {
                                returnColor *= 0.3;
                            }
                        }

                        illumData.S = normalize(light.position - shadowRay.origin);
                        illumData.V = -ray.dir;
                        illumData.R = reflect(illumData.S, illumData.normal);
                        illumData.H = (illumData.S + illumData.normal)/2.0f;

                        var ke: f32 = 40.0f;
                        var irradiance: vec3f;
                        irradiance = light.ka * returnColor * ambientColor
                                        + light.kd * light.color * returnColor * dot(illumData.S, illumData.normal)
                                        + light.ks * light.color * specularColor * pow(max(dot(illumData.H, illumData.normal), 0.0f), ke);

                        irradiance += light.kd * light2.color * returnColor * dot(illumData.S, illumData.normal)
                        + light.ks * light2.color * specularColor * pow(max(dot(illumData.H, illumData.normal), 0.0f), ke);
                        
                        localIllum = vec4f(irradiance, 1.0f);
                    }
                    // if shadow ray hits another object
                    else
                    {
                        localIllum = vec4f(light.ka * returnColor * ambientColor, 1.0f);
                    }

                    
                    payload.color = (1.0f - payload.nextKt) * localIllum;

                    var reflectionRay: Ray;
                    reflectionRay.origin = ray.origin + minDist * ray.dir;
                    reflectionRay.exists = true;

                    var normal: vec3f;
                    
                    if (minIdx == 0)
                    {
                        normal = normalize(reflectionRay.origin - sphere.center);
                    }
                    else if (minIdx == 1)
                    {
                        normal = normalize(reflectionRay.origin - sphere2.center);
                    }
                    else
                    {
                        normal = vec3f(0.0f, -1.0f, 0.0f);
                    }

                    reflectionRay.dir = normalize(reflect(ray.dir, normal));
                    reflectionRay.origin += normal * 1e-4;

                    payload.reflectionRay = reflectionRay;

                    // transmission ray
                    var transmissionRay: Ray;

                    var ir1 = 1.0;
                    var ir2 = 0.95;
                    
                    transmissionRay.origin = ray.origin + minDist * ray.dir;
                    transmissionRay.exists = true;

                    // If we are currently inside
                    if (dot(normal, ray.dir) > 0.0f)
                    {
                        normal = -normal;
                        ir1 = 0.95;
                        ir2 = 1.0;

                        transmissionRay.origin += -normal * 0.001;
                    }
                    else
                    {
                        transmissionRay.origin += normal * 0.001;
                    }

                    transmissionRay.dir = calculateTransmissionRay2(ir1, ir2, ray.dir, normal);

                    if (all(transmissionRay.dir == vec3f(0.0f)))
                    {
                        transmissionRay = reflectionRay;
                    }

                    payload.transmissionRay = transmissionRay;
                    
                    return payload;
                }

                fn getReturnColor(pixelVal: vec2f, cam: Camera) -> vec4f
                {
                    var eye = vec3f(389.486, -1.855, 0.573);

                    var sphere: Sphere;
                    sphere.radius = 6.0f;
                    sphere.center = vec3f(-10.0, 5.0, 8.0);
                    sphere.color = vec3f(0.0f, 0.0f, 0.0f);
                    sphere.kr = 0.9;
                    sphere.kt = 0.0;
                    sphere.refractionIdx = 1.0f;

                    var sphere2: Sphere;
                    sphere2.radius = 8.0f;
                    sphere2.center = vec3f(11.361, -2.813, 0.124);
                    sphere2.color =  vec3f(0.0f, 0.0f, 0.2f);
                    sphere2.kr = 0.0;
                    sphere2.kt = 0.8;
                    sphere2.refractionIdx = 1.0f;

                    var quad: Quad;
                    quad = Quad(
                        vec3<f32>(30.0, 20.0f, 3000.0), // First vertex of the quad
                        vec3<f32>( 30.0, 20.0f, -3000.0), // Second vertex
                        quadRefVertex, // Third vertex
                        vec3<f32>(-30.0, 20.0f,  3000.0), // Fourth vertex
                        vec3<f32>(0.0, -1.0, 0.0),    // Normal vector of the plane (pointing up)
                        vec3<f32>(1.0f, 1.0f, 1.0f), // color
                        0.0, // kr
                        0.0,  // kt,
                        1.0f // index of refraction
                    );

                    var view = viewTransformMatrix(
                        eye,
                        sphere2.center,
                        vec3f(0.0f, 1.0f, 0.0f)
                    );

                    var rayDir : vec3f = normalize(vec3f(pixelVal, cam.focalLength));

                    var ray: Ray;
                    ray.origin = vec3f(0.0f);
                    ray.dir = rayDir;
                    ray.exists = true;
                    ray.k = 1.0;

                    var transmissionRay: Ray;

                    sphere.center = (view * vec4f(sphere.center, 1.0f)).xyz;
                    sphere2.center = (view * vec4f(sphere2.center, 1.0f)).xyz;
                    // TODO: Why are why not converting quad to view space

                    var finalColor = vec4f(0.0);
                    var focalLength = cam.focalLength;
                    
                    const maxRays = u32(pow(2.0, f32(MAX_DEPTH)));
                    var prevOutputRays: array<Ray, maxRays>;
                    var nextOutputRays: array<Ray, maxRays>;

                    prevOutputRays[0] = ray;

                    var numRays: u32 = 1;
                    for (var depth: u32 = 1; depth < MAX_DEPTH; depth++)
                    {
                        var nextNumRays: u32 = 0;

                        for (var ii: u32 = 0; ii < numRays; ii++)
                        {
                            var inputRay = prevOutputRays[ii];

                            if (!inputRay.exists)
                            {
                                continue;
                            }

                            var payload = illuminate(inputRay, focalLength, sphere, sphere2, quad);
                            finalColor += inputRay.k * payload.color;

                            if (!payload.intersection || 
                                (payload.nextKr == 0.0 && payload.nextKt == 0.0))
                            {
                                break;
                            }

                            if (payload.nextKr > 0.0)
                            {
                                nextOutputRays[nextNumRays] = payload.reflectionRay;
                                nextOutputRays[nextNumRays].k = inputRay.k * payload.nextKr;
                                nextNumRays++;
                            }
                            
                            if (payload.nextKt > 0.0)
                            {
                                nextOutputRays[nextNumRays] = payload.transmissionRay;
                                nextOutputRays[nextNumRays].k = inputRay.k * payload.nextKt;
                                nextNumRays++;
                            }                            
                        }

                        numRays = nextNumRays;

                        for (var ii: u32 = 0; ii < nextNumRays; ii++)
                        {
                            prevOutputRays[ii] = nextOutputRays[ii];
                        }

                        focalLength = 0.0; // only use focalLength for first ray (originating from camera)
                    }

                    finalColor.a = 1.0;

                    return finalColor;
                }

                fn random(seed: f32) -> f32
                {
                    return fract(sin(seed) * 43758.5453123);  // Simple pseudo-random number
                }

                fn random2D(seed: vec2f) -> f32
                {
                    return fract(sin(dot(seed, vec2f(12.9898,78.233))) * 43758.5453123);
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

                    // var color: vec4f = vec4f(0.0f, 0.0f, 0.0f, 0.0f);

                    // var numIters: f32 = 20.0f;
                    // for (var ii: f32 = 0.0f; ii < numIters; ii += 1.0f)
                    // {
                    //     var xVal: f32 = random(ii * 2.0f);
                    //     var yVal: f32 = random(ii * 2.0f + 1.0f);

                    //     let pixelVal = vec2f((fragCoord.x - resolution.x * 0.5f) * w,
                    //                 (fragCoord.y - resolution.y * 0.5f) * h)
                    //                 + vec2f(xVal * w, yVal * h);
                        
                    //     color += getReturnColor(pixelVal, cam);
                    // }

                    // color = color/numIters;

                    let pixelVal = vec2f((fragCoord.x - resolution.x * 0.5f) * w,
                                    (fragCoord.y - resolution.y * 0.5f) * h)
                                    + vec2f(0.5 * w, 0.5 * h);
                    
                    var color: vec4f = getReturnColor(pixelVal, cam);

                    return color;

                    var logLumSum = 0.0;
                    var delta = 0.00001;

                    var irradiance: vec4f = vec4f(0.0f, 0.0f, 0.0f, 0.0f);

                    var ldmax = delta;

                    for (var x = 0.0; x < resolution.x; x += 1.0)
                    {
                        for (var y = 0.0; y < resolution.y; y += 1.0)
                        {
                            let pixelVal = vec2f((x - resolution.x * 0.5f) * w,
                                    (y - resolution.y * 0.5f) * h)
                                    + vec2f(0.5 * w, 0.5 * h);
                        
                            var color: vec4f = getReturnColor(pixelVal, cam);
                            var luminance = 0.27 * color.r + 0.67 * color.g + 0.06 * color.b;

                            ldmax = max(luminance, ldmax);

                            logLumSum += log(luminance + delta);

                            if (x == fragCoord.x && y == fragCoord.y)
                            {
                                irradiance = color;
                            }
                        }
                    }
                    
                    var targetRGB: vec4f;
                    var wardTR: bool = true;

                    var lwa = exp(1.0/(resolution.x * resolution.y) * logLumSum);

                    if (wardTR)
                    {
                        // Ward TR
                        var sf = pow((1.219 + pow(ldmax/2.0, 0.4))/(1.219 + pow(lwa, 0.4)), 2.5);
                        targetRGB = sf * irradiance;  
                    }
                    else
                    {
                        // Reinhard TR
                        var a = 0.18;
                        var scaledLum = (a / lwa) * irradiance;
                        var reflectance = scaledLum/(1.0 + scaledLum);
                        targetRGB = reflectance * ldmax;
                    }

                    // TODO: Check if the division is correct
                    return targetRGB/ldmax;
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

        // TODO: Write to unformbuffer with scene data

        // Write buffers
        this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices);
        this.device.queue.writeBuffer(this.indexBuffer, 0, indices);
    }

    private async createPipeline()
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
            },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                sampler:
                {
                    type: "filtering"
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.FRAGMENT,
                texture:
                {
                    sampleType: 'float',
                    viewDimension: '2d',
                    multisampled: false,
                }
            }]
        });

        const sampler = this.device.createSampler();

        const response = await fetch('./flowers.jpg');
        const blob = await response.blob();
        const imageBitmap = await createImageBitmap(blob);

        const texture = this.device.createTexture(
            {
                label: 'floor image',
                size: [imageBitmap.width, imageBitmap.height, 1],
                format: 'rgba8unorm',
                usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT
            });
        
        this.device.queue.copyExternalImageToTexture
        (
            { source: imageBitmap },
            { texture: texture },
            [imageBitmap.width, imageBitmap.height]
        );
        
        const textureView = texture.createView();

        // TODO: Should this be done here or elsewhere?
        this.bindGroup = this.device.createBindGroup(
        {
            label: "Vertex Bind group",
            layout: this.bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.uniformBuffer }
                },
                {
                    binding: 1,
                    resource: sampler
                },
                {
                    binding: 2,
                    resource: textureView
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
