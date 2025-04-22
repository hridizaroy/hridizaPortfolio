var I=Object.defineProperty;var _=(c,e,i)=>e in c?I(c,e,{enumerable:!0,configurable:!0,writable:!0,value:i}):c[e]=i;var t=(c,e,i)=>_(c,typeof e!="symbol"?e+"":e,i);(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))a(r);new MutationObserver(r=>{for(const s of r)if(s.type==="childList")for(const u of s.addedNodes)u.tagName==="LINK"&&u.rel==="modulepreload"&&a(u)}).observe(document,{childList:!0,subtree:!0});function i(r){const s={};return r.integrity&&(s.integrity=r.integrity),r.referrerPolicy&&(s.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?s.credentials="include":r.crossOrigin==="anonymous"?s.credentials="omit":s.credentials="same-origin",s}function a(r){if(r.ep)return;r.ep=!0;const s=i(r);fetch(r.href,s)}})();class G{constructor(e){t(this,"device");t(this,"context");t(this,"canvasFormat");t(this,"vertexBuffer");t(this,"indexBuffer");t(this,"uniformBuffer");t(this,"uniforms");t(this,"hairStateStorage");t(this,"shaderModule");t(this,"bindGroupLayout");t(this,"bindGroup");t(this,"pipeline");t(this,"renderPassDescriptor");t(this,"compute_shaderModule");t(this,"compute_bindGroupLayout");t(this,"compute_bindGroup");t(this,"compute_pipeline");t(this,"bins_shaderModule");t(this,"bins_bindGroupLayout");t(this,"bins_bindGroup");t(this,"bins_pipeline");t(this,"step",!1);t(this,"numHairStrands",2*2);t(this,"numBins",0);t(this,"strandVertices",new Float32Array([0,0,2.8,0,-.1,2.8,0,-.2,2.8,0,-.3,2.8]));t(this,"indices",new Uint16Array(Array.from({length:this.strandVertices.length/3},(e,i)=>i)));this.canvas=e}async init(){await this.getGPU(),this.connectCanvas(),this.loadShaders(),this.loadComputeShader(),this.loadBinsShader(),this.createBuffers(),this.createPipeline(),this.compute_createPipeline(),this.bins_createPipeline(),this.createRenderPassDescriptor(),this.render()}async getGPU(){if(!navigator.gpu){this.onError("WebGPU not supported on this browser.");return}const e=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!e){this.onError("No GPU Adapter found.");return}this.device=await e.requestDevice(),this.device.addEventListener("uncapturederror",i=>console.log(i.error.message))}connectCanvas(){const e=this.canvas.getContext("webgpu");if(!e){this.onError("Failed to get canvas context :/");return}this.context=e,this.canvasFormat=navigator.gpu.getPreferredCanvasFormat(),this.context.configure({device:this.device,format:this.canvasFormat})}loadShaders(){this.shaderModule=this.device.createShaderModule({label:"Hair shader",code:`
            @group(0) @binding(0) var<uniform> sceneData: SceneData;
            @group(0) @binding(1) var<storage> positions: array<f32>;

            struct Ray
            {
                origin: vec3f,
                dir: vec3f
            }

            struct Camera
            {
                location: vec3f,
                imageDimensions: vec2f,
                filmPlaneDimensions: vec2f
            };

            struct SceneData
            {
                resolution: vec2f,
                numStrandVertices: f32
            };

            // TODO: Re-check coord system
            fn viewTransformMatrix(eye: vec3f, lookAt: vec3f, down: vec3f) 
                                    -> mat4x4<f32>
            {
                var forward = normalize(lookAt - eye);
                var right = normalize(cross(down, forward));
                var d = normalize(cross(forward, right));

                var translationMat : mat4x4<f32> = mat4x4<f32>(
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    -eye.x, -eye.y, -eye.z, 1.0
                );

                var rotationMat : mat4x4<f32> = mat4x4<f32>(
                    right.x, d.x, forward.x, 0.0,
                    right.y, d.y, forward.y, 0.0,
                    right.z, d.z, forward.z, 0.0,
                    0.0, 0.0, 0.0, 1.0
                );

                return rotationMat * translationMat;

                // return mat4x4<f32>(
                //     right.x, u.x, forward.x, 0.0f,
                //     right.y, u.y, forward.y, 0.0f,
                //     right.z, u.z, forward.z, 0.0f,
                //     -dot(eye, right), -dot(eye, d), -dot(eye, forward), 1.0f
                // );
            }

            fn projectionMatrix(angle: f32, aspect_ratio: f32, near: f32, far: f32) -> mat4x4<f32>
            {
                let a: f32 = 1.0 / tan(radians(angle/2));
                let m1 = far/(far - near);
                let m2 = -near * far/(far -near);

                return mat4x4<f32>(
                    vec4<f32>(a * aspect_ratio, 0.0, 0.0, 0.0),
                    vec4<f32>(0.0, a, 0.0, 0.0),
                    vec4<f32>(0.0f, 0.0f, m1, 1.0),
                    vec4<f32>(0.0, 0.0, m2, 0.0)
                );
            }

            struct VertReturn
            {
                @builtin(position) pos : vec4f,
                @location(0) dir: vec3f
            }

            @vertex
            fn vertexMain(@builtin(instance_index) instance: u32,
                            @builtin(vertex_index) vert_idx: u32)
                -> VertReturn
            {
                // Get pos from storage buffer                    
                let i: f32 = f32(instance);
                let numStrandVertices: f32 = sceneData.numStrandVertices;

                // vertex index is indicative of position of particle within a hair strand
                // TODO: Is there a better way to get this data? Uniform buffer or something?
                var pos: vec4f = vec4f(
                            positions[u32(i * numStrandVertices) + vert_idx * 3],
                            positions[u32(i * numStrandVertices) + vert_idx * 3 + 1],
                            positions[u32(i * numStrandVertices) + vert_idx * 3 + 2],
                            1.0f);

                var cam: Camera;
                cam.imageDimensions = sceneData.resolution;

                // meters
                cam.filmPlaneDimensions = vec2f(25.0f, 25.0f);

                cam.location = vec3f(0.8f, 0.6f, 0.0f);

                let lookAt: vec3f = vec3f(0.0f, 0.3f, 2.8f);

                var view: mat4x4<f32> = viewTransformMatrix(
                    cam.location,
                    lookAt,
                    vec3f(0.0f, 1.0f, 0.0f)
                );

                let angle = 30.0f;

                // arbitrary far clip plane for now
                var projection : mat4x4<f32> = projectionMatrix(angle, sceneData.resolution.y/sceneData.resolution.x, 1.0f, 100.0f);
                var result: vec4f = projection * view * pos;
            
                // TODO: Clean code and var names
                var pos2: vec3f;
                
                if ( vert_idx < u32(numStrandVertices - 1.0) )
                {
                    pos2 = vec3f(
                        positions[u32(i * numStrandVertices) + (vert_idx + 1) * 3],
                        positions[u32(i * numStrandVertices) + (vert_idx + 1) * 3 + 1],
                        positions[u32(i * numStrandVertices) + (vert_idx + 1) * 3 + 2]);
                }
                else
                {
                    pos2 = pos.xyz;
                    pos = vec4f(
                        positions[u32(i * numStrandVertices) + (vert_idx - 1) * 3],
                        positions[u32(i * numStrandVertices) + (vert_idx - 1) * 3 + 1],
                        positions[u32(i * numStrandVertices) + (vert_idx - 1) * 3 + 2],
                        1.0f);
                }

                
                var returnVal: VertReturn;
                returnVal.pos = result;
                returnVal.dir = normalize(pos.xyz - pos2);

                return returnVal;
            }

            @fragment
            fn fragmentMain(input: VertReturn)
                -> @location(0) vec4f
            {
                let lightPos = vec3(5.0, -10.0, 5.0);
                let lightVec = normalize(lightPos - input.pos.xyz);

                let cosT: f32 = sqrt(1.0 - pow(dot(lightVec, input.dir), 2));

                let lightColor = vec3f(1.0, 1.0, 1.0);

                let hairColor = vec3f(0.4, 0.1, 0.04);

                return vec4f(lightColor * hairColor * cosT, 1.0f);
            }
            `})}loadComputeShader(){this.compute_shaderModule=this.device.createShaderModule({label:"Hair simulation shader",code:`
                @group(0) @binding(0) var<uniform> sceneData: SceneData;
                @group(0) @binding(1) var<storage> positionsIn: array<f32>; // TODO: Read as array of vec3f
                @group(0) @binding(2) var<storage> velocitiesIn: array<f32>;
                @group(0) @binding(3) var<storage, read_write> positionsOut: array<f32>;
                @group(0) @binding(4) var<storage, read_write> velocitiesOut: array<f32>;

                @group(0) @binding(5) var<storage> prevPosIn: array<f32>;
                @group(0) @binding(6) var<storage, read_write> prevPosOut: array<f32>;

                @group(0) @binding(7) var<storage, read_write> bins: array<array<atomic<u32>, 5>>; // TODO: Don't hardcode

                struct SceneData
                {
                    resolution: vec2f,
                    numStrandVertices: f32,
                    radius: f32,
                    scalpCenter: vec3<f32>,
                    rest_length: f32,
                    boundingBoxSide: f32,
                    maxStrands: f32,
                    binSideLength: f32,
                    numBinsPerDim: f32
                };


                const mass = 0.05f;
                const gravity : f32 = -9.8f;
                const deltaTime : f32 = 1.0f/600.0f;

                const damping = 0.1f;
                const k = 50.0f;

                // TODO: Why is the force reducing over time even when particles are in the same position?
                fn calculateForces(idx: u32, last_vertex: bool) -> vec3<f32>
                {
                    let rest_length = sceneData.rest_length;
                    let vi : vec3<f32> = vec3(velocitiesIn[idx], velocitiesIn[idx + 1],
                                            velocitiesIn[idx + 2]);

                    let curr_pos : vec3<f32> = vec3(positionsIn[idx], positionsIn[idx + 1],
                                                 positionsIn[idx + 2]);
                    let prev_pos : vec3<f32> = vec3(positionsIn[idx - 3], positionsIn[idx - 2],
                                                positionsIn[idx - 1]);

                    let length1 : f32 = length(curr_pos - prev_pos);
                    let dir1 : vec3<f32> = normalize(prev_pos - curr_pos);
                    
                    // Spring force towards previous strand
                    var force : vec3<f32> = dir1 * (length1 - rest_length) * k;

                    force += (-damping * vi);

                    force.y += mass * gravity;

                    if (!last_vertex)
                    {
                        let next_pos : vec3<f32> = vec3(positionsIn[idx + 3], positionsIn[idx + 4],
                                                        positionsIn[idx + 5]
                                                    );

                        let length2: f32 = length(curr_pos - next_pos);
                        let dir2 : vec3<f32> = normalize(next_pos - curr_pos);
                        
                        // Spring force towards next strand
                        force += dir2 * (length2 - rest_length) * k;
                    }

                    // Add wind force
                    force.x += 2.0;
                    force.y += -0.4;

                    if (idx / u32(sceneData.numStrandVertices) == 0)
                    {
                        force.x -= 5.0;
                    }
                    
                    return force;
                }

                @compute
                @workgroup_size(8) // TODO: Don't hard code workgroup size
                fn computeMain(@builtin(global_invocation_id) id: vec3<u32>)
                {                   
                    let idx = id.x;

                    let numStrandVertices = sceneData.numStrandVertices;

                    let vert_idx = f32(idx % u32(numStrandVertices));

                    if ( vert_idx > 2.0 && idx % 3 == 0 )
                    {
                        let force: vec3<f32> = calculateForces(idx, vert_idx >= numStrandVertices - 3.0f);
                        let acceleration: vec3<f32> = force / mass;

                        // TODO: Do we even need to store velocities?
                        // Maybe for some force/damping?
                        // velocitiesOut[idx] = velocitiesIn[idx] + acceleration.x * deltaTime;
                        // velocitiesOut[idx + 1] = velocitiesIn[idx + 1] + acceleration.y * deltaTime;
                        // velocitiesOut[idx + 2] = velocitiesIn[idx + 2] + acceleration.z * deltaTime;

                        var finalPos: vec3<f32>;
                        // finalPos.x = positionsIn[idx] + velocitiesOut[idx] * deltaTime;
                        // finalPos.y = positionsIn[idx + 1] + velocitiesOut[idx + 1] * deltaTime;
                        // finalPos.z = positionsIn[idx + 2] + velocitiesOut[idx + 2] * deltaTime;

                        finalPos.x = 2.0 * positionsIn[idx] - prevPosIn[idx] + acceleration.x * deltaTime * deltaTime;
                        finalPos.y = 2.0 * positionsIn[idx + 1] - prevPosIn[idx + 1] + acceleration.y * deltaTime * deltaTime;
                        finalPos.z = 2.0 * positionsIn[idx + 2] - prevPosIn[idx + 2] + acceleration.z * deltaTime * deltaTime;

                        prevPosOut[idx] = positionsIn[idx];
                        prevPosOut[idx + 1] = positionsIn[idx + 1];
                        prevPosOut[idx + 2] = positionsIn[idx + 2];

                        // Constrain position on head surface
                        let distFromCenter: f32 = length(finalPos - sceneData.scalpCenter);
                        if (distFromCenter < sceneData.radius)
                        {
                            finalPos += (sceneData.radius - distFromCenter) * normalize(finalPos - sceneData.scalpCenter);
                            velocitiesOut[idx] = 0.0;
                            velocitiesOut[idx + 1] = 0.0;
                            velocitiesOut[idx + 2] = 0.0;
                        }

                        positionsOut[idx] = finalPos.x;
                        positionsOut[idx + 1] = finalPos.y;
                        positionsOut[idx + 2] = finalPos.z;

                        velocitiesOut[idx] = (finalPos.x - prevPosIn[idx])/deltaTime;
                        velocitiesOut[idx + 1] = (finalPos.y - prevPosIn[idx + 1])/deltaTime;
                        velocitiesOut[idx + 2] = (finalPos.z - prevPosIn[idx + 2])/deltaTime;

                        // Calculate Grid Index
                        // StartPos = ScalpCenter - floor(BoundingBoxSide / 2)
                        // Pos - StartPos (3D Vectors)
                        // Divide by binSideLength
                        // Take floor
                        // Index = Pos.z * (binSideLength ^ 2) + Pos.y * (binSideLength) + Pos.x
                        var maxStrands = u32(sceneData.maxStrands);
                        var numBinsPerDim = u32(sceneData.numBinsPerDim);

                        var startPos: vec3f = sceneData.scalpCenter - floor(sceneData.boundingBoxSide / 2.0);
                        var cornerIdx3D: vec3u = vec3u(floor((finalPos - startPos)/sceneData.binSideLength));
                        var gridIdx: u32 = cornerIdx3D.y * numBinsPerDim * numBinsPerDim
                                            + cornerIdx3D.z * numBinsPerDim + cornerIdx3D.x;

                        
                        // var currStrands = atomicLoad(&bins[gridIdx][0]);
                        
                        if (atomicLoad(&bins[gridIdx][0]) < maxStrands)
                        {
                            atomicAdd(&bins[gridIdx][0], 1);
                            atomicStore(&bins[gridIdx][min(atomicLoad(&bins[gridIdx][0]), maxStrands)], idx);
                        }

                        // TODO for intersections
                        // Define grid side length and start and end points
                        // Define max number of strands within grid
                        // Get grid index from position mid point
                        // Get index for grid buffer
                        // Store idx in grid buffer
                        // Run another compute shader for checking intersections 
                            // Make sure to 0 out counters (idx 0) at the end of calculations for each bin
                    }
                }
            `})}loadBinsShader(){this.bins_shaderModule=this.device.createShaderModule({label:"Bins shader",code:`
                @group(0) @binding(0) var<uniform> sceneData: SceneData;
                @group(0) @binding(1) var<storage, read_write> positionsOut: array<f32>;
                @group(0) @binding(2) var<storage, read_write> velocitiesOut: array<f32>;
                @group(0) @binding(3) var<storage, read_write> prevPosOut: array<f32>;

                @group(0) @binding(4) var<storage, read_write> bins: array<array<atomic<u32>, 4>>;  // TODO: Don't hardcode

                struct SceneData
                {
                    resolution: vec2f,
                    numStrandVertices: f32,
                    radius: f32,
                    scalpCenter: vec3<f32>,
                    rest_length: f32
                };

                const deltaTime : f32 = 1.0f/600.0f;

                fn areIntersecting(p1: vec3f, p2: vec3f, q1: vec3f, q2: vec3f) -> bool
                {
                    let u = p2 - p1;
                    let v = q2 - q1;
                    let w = p1 - q1;

                    var a = dot(u, u);
                    var b = dot(u, v);
                    var c = dot(v, v);
                    var d = dot(u, w);
                    var e = dot(v, w);

                    var denominator = a * c - b * b;

                    if (denominator < 1e-6)
                    {
                        return false;
                    }

                    var s = (b * e - c * d) / denominator;
                    var t = (a * e - b * d) / denominator;

                    if (s < 0.0 || s > 1.0 || t < 0.0 || t > 1.0)
                    {
                        return false;
                    }

                    var p = p1 + s * u;
                    var q = q1 + t * v;

                    return length(p - q) < 1e-6;
                }

                @compute
                @workgroup_size(64) // TODO: Don't hard code workgroup size
                fn computeMain(@builtin(global_invocation_id) id: vec3<u32>)
                {
                    let idx = id.x;
                    
                    var currStrands: u32 = atomicLoad(&bins[idx][0]);

                    if (currStrands > 1)
                    {
                        // TODO: Do this in a loop for each pair of strands
                        var strandIdx1 = atomicLoad(&bins[idx][1]);
                        var strandIdx2 = atomicLoad(&bins[idx][2]);

                        let numStrandVertices = u32(sceneData.numStrandVertices);

                        let strandNum1: u32 = strandIdx1 / numStrandVertices;
                        let strandNum2: u32 = strandIdx2 / numStrandVertices;

                        let vertIdx1 = strandIdx1 % numStrandVertices;
                        let vertIdx2 = strandIdx2 % numStrandVertices;

                        if (vertIdx1 > 2 && vertIdx2 > 2 && strandNum1 != strandNum2)
                        {
                            var p1 = vec3f(positionsOut[strandIdx1],
                                                positionsOut[strandIdx1 + 1],
                                                positionsOut[strandIdx1 + 2]
                                            );

                            var p2 = vec3f(positionsOut[strandIdx1 - 3],
                                                positionsOut[strandIdx1 - 2],
                                                positionsOut[strandIdx1 - 1]
                                            );

                            var q1 = vec3f(positionsOut[strandIdx2],
                                                positionsOut[strandIdx2 + 1],
                                                positionsOut[strandIdx2 + 2]
                                            );

                            var q2 = vec3f(positionsOut[strandIdx2 - 3],
                                                positionsOut[strandIdx2 - 2],
                                                positionsOut[strandIdx2 - 1]
                                            );

                            
                            
                            // velocitiesOut[strandIdx1 + 1] = 0.0;                            
                            // velocitiesOut[strandIdx2 + 1] = 0.0;
                            if areIntersecting(p1, p2, q1, q2)
                            {
                                // TODO: Replace with forces
                                // velocitiesOut[strandIdx1 - 1] = 0.0;
                                // velocitiesOut[strandIdx1 - 2] = 0.0;
                                // velocitiesOut[strandIdx1 - 3] = 0.0;

                                // velocitiesOut[strandIdx1] = 0.0;
                                // velocitiesOut[strandIdx1 + 1] = 0.0;
                                // velocitiesOut[strandIdx1 + 2] = 0.0;

                                // velocitiesOut[strandIdx2 - 1] = 0.0;
                                // velocitiesOut[strandIdx2 - 2] = 0.0;
                                // velocitiesOut[strandIdx2 - 3] = 0.0;

                                // velocitiesOut[strandIdx2] = 0.0;
                                // velocitiesOut[strandIdx2 + 1] = 0.0;
                                // velocitiesOut[strandIdx2 + 2] = 0.0;

                    
                                // // positionsOut[strandIdx1] = 0.0;
                                // positionsOut[strandIdx1 + 1] += 5.0;
                                // // positionsOut[strandIdx1 + 2] = 0.0;

                                // // positionsOut[strandIdx2] = 0.0;
                                // positionsOut[strandIdx2] += 5.0;
                                // positionsOut[strandIdx2 + 2] = 0.0;
                            }
                        }
                    }

                    // Reset current num strands to 0
                    atomicStore(&bins[idx][0], 0);
                }
            `})}createBuffers(){this.vertexBuffer=this.device.createBuffer({label:"Strand vertices",size:this.strandVertices.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),this.indexBuffer=this.device.createBuffer({label:"Strand indices",size:this.indices.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST});const e=.5,i=0,a=.5,r=2.8,s=.2;var u=e/5,o=4*e;o=Math.ceil(o/u)*u;const x=4;this.numBins=Math.pow(o/u,3);const P=(x+1)*this.numBins,S=new Int32Array(P);this.uniforms=new Float32Array(12),this.uniforms[0]=this.canvas.width,this.uniforms[1]=this.canvas.height,this.uniforms[2]=this.strandVertices.length,this.uniforms[3]=e,this.uniforms[4]=i,this.uniforms[5]=a,this.uniforms[6]=r,this.uniforms[7]=s,this.uniforms[8]=o,this.uniforms[9]=x,this.uniforms[10]=u,this.uniforms[11]=o/u,this.uniformBuffer=this.device.createBuffer({label:"Uniform buffer",size:Math.ceil(this.uniforms.byteLength/16)*16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.device.queue.writeBuffer(this.vertexBuffer,0,this.strandVertices),this.device.queue.writeBuffer(this.indexBuffer,0,this.indices),this.device.queue.writeBuffer(this.uniformBuffer,0,this.uniforms);const n=new Float32Array(this.numHairStrands*this.strandVertices.length),h=new Float32Array(this.numHairStrands*this.strandVertices.length);this.hairStateStorage=[this.device.createBuffer({label:"Positions",size:n.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.createBuffer({label:"Velocities",size:h.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.createBuffer({label:"PositionsCopy",size:n.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.createBuffer({label:"VelocitiesCopy",size:h.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.createBuffer({label:"prevPos",size:h.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.createBuffer({label:"prevPosCopy",size:h.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.createBuffer({label:"bins",size:S.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})],this.device.queue.writeBuffer(this.hairStateStorage[1],0,h),this.device.queue.writeBuffer(this.hairStateStorage[3],0,h),this.device.queue.writeBuffer(this.hairStateStorage[6],0,S);let p=Math.sqrt(this.numHairStrands);function O(f,g,v){return Math.sqrt(f*f+g*g+v*v)}for(let f=0;f<p;f++){let g=f*Math.PI/(p-1),v=e*Math.sin(g),B=i+e*Math.cos(g);for(let b=0;b<p;b++){let y=b*Math.PI/(p-1);for(let m=0;m<this.strandVertices.length;m+=3){let d=(f*p+b)*this.strandVertices.length+m;n[d]=B,n[d+1]=a+v*Math.sin(y)+this.strandVertices[m+1],n[d+2]=r+v*Math.cos(y);let l=O(n[d]-i,n[d+1]-a,n[d+2]-r);l<e&&(n[d]+=(e-l)*(n[d]-i)/l,n[d+1]+=(e-l)*(n[d+1]-a)/l,n[d+2]+=(e-l)*(n[d+2]-r)/l)}}}this.device.queue.writeBuffer(this.hairStateStorage[0],0,n),this.device.queue.writeBuffer(this.hairStateStorage[2],0,n),this.device.queue.writeBuffer(this.hairStateStorage[4],0,n),this.device.queue.writeBuffer(this.hairStateStorage[5],0,n)}createPipeline(){const e={arrayStride:12,attributes:[{format:"float32x3",offset:0,shaderLocation:0}]};this.bindGroupLayout=this.device.createBindGroupLayout({label:"Hair Group Layout Vertex",entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]}),this.bindGroup=[this.device.createBindGroup({label:"Vertex Bind group A",layout:this.bindGroupLayout,entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:{buffer:this.hairStateStorage[0]}}]}),this.device.createBindGroup({label:"Vertex Bind group B",layout:this.bindGroupLayout,entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:{buffer:this.hairStateStorage[2]}}]})];const i=this.device.createPipelineLayout({label:"Hair Pipeline Layout Vertex",bindGroupLayouts:[this.bindGroupLayout]});this.pipeline=this.device.createRenderPipeline({label:"Hair pipeline",layout:i,vertex:{module:this.shaderModule,entryPoint:"vertexMain",buffers:[e]},fragment:{module:this.shaderModule,entryPoint:"fragmentMain",targets:[{format:this.canvasFormat}]},primitive:{topology:"line-strip",stripIndexFormat:"uint16"}})}compute_createPipeline(){this.compute_bindGroupLayout=this.device.createBindGroupLayout({label:"Hair Bind Group Layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),this.compute_bindGroup=[this.device.createBindGroup({label:"Simulation bind group A",layout:this.compute_bindGroupLayout,entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:{buffer:this.hairStateStorage[0]}},{binding:2,resource:{buffer:this.hairStateStorage[1]}},{binding:3,resource:{buffer:this.hairStateStorage[2]}},{binding:4,resource:{buffer:this.hairStateStorage[3]}},{binding:5,resource:{buffer:this.hairStateStorage[4]}},{binding:6,resource:{buffer:this.hairStateStorage[5]}},{binding:7,resource:{buffer:this.hairStateStorage[6]}}]}),this.device.createBindGroup({label:"Simulation bind group B",layout:this.compute_bindGroupLayout,entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:{buffer:this.hairStateStorage[2]}},{binding:2,resource:{buffer:this.hairStateStorage[3]}},{binding:3,resource:{buffer:this.hairStateStorage[0]}},{binding:4,resource:{buffer:this.hairStateStorage[1]}},{binding:5,resource:{buffer:this.hairStateStorage[5]}},{binding:6,resource:{buffer:this.hairStateStorage[4]}},{binding:7,resource:{buffer:this.hairStateStorage[6]}}]})];const e=this.device.createPipelineLayout({label:"Hair Pipeline Layout Compute",bindGroupLayouts:[this.compute_bindGroupLayout]});this.compute_pipeline=this.device.createComputePipeline({label:"Simulation pipeline",layout:e,compute:{module:this.compute_shaderModule,entryPoint:"computeMain"}})}bins_createPipeline(){this.bins_bindGroupLayout=this.device.createBindGroupLayout({label:"Bins Bind Group Layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),this.bins_bindGroup=[this.device.createBindGroup({label:"Bins bind group A",layout:this.bins_bindGroupLayout,entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:{buffer:this.hairStateStorage[1]}},{binding:2,resource:{buffer:this.hairStateStorage[3]}},{binding:3,resource:{buffer:this.hairStateStorage[5]}},{binding:4,resource:{buffer:this.hairStateStorage[6]}}]}),this.device.createBindGroup({label:"Bins bind group B",layout:this.bins_bindGroupLayout,entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:{buffer:this.hairStateStorage[0]}},{binding:2,resource:{buffer:this.hairStateStorage[2]}},{binding:3,resource:{buffer:this.hairStateStorage[4]}},{binding:4,resource:{buffer:this.hairStateStorage[6]}}]})];const e=this.device.createPipelineLayout({label:"Hair Pipeline Layout Bins",bindGroupLayouts:[this.bins_bindGroupLayout]});this.bins_pipeline=this.device.createComputePipeline({label:"Bins Simulation pipeline",layout:e,compute:{module:this.bins_shaderModule,entryPoint:"computeMain"}})}createRenderPassDescriptor(){this.renderPassDescriptor={label:"Render Pass Description",colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:[1,1,1,1],loadOp:"clear",storeOp:"store"}]}}render(){const e=Number(this.step);this.renderPassDescriptor.colorAttachments[0].view=this.context.getCurrentTexture().createView();const i=this.device.createCommandEncoder(),a=i.beginComputePass();a.setPipeline(this.compute_pipeline),a.setBindGroup(0,this.compute_bindGroup[e]);const r=Math.ceil(this.numHairStrands*this.strandVertices.length/8);a.dispatchWorkgroups(r),a.end();const s=i.beginComputePass();s.setPipeline(this.bins_pipeline),s.setBindGroup(0,this.bins_bindGroup[e]);const u=Math.ceil(this.numBins/64);s.dispatchWorkgroups(u),s.end();const o=i.beginRenderPass(this.renderPassDescriptor);o.setIndexBuffer(this.indexBuffer,"uint16"),o.setPipeline(this.pipeline),o.setVertexBuffer(0,this.vertexBuffer),o.setBindGroup(0,this.bindGroup[e]),o.drawIndexed(this.indices.length,this.numHairStrands),o.end(),this.device.queue.submit([i.finish()]),this.step=!this.step,requestAnimationFrame(()=>this.render())}onError(e){document.body.innerHTML=`<p>${e}</p>`,console.error(e)}}const D=document.getElementById("webgpu-canvas"),w=new G(D);w.init();
