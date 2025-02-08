import { Renderer } from "./Renderer";

// Get access to canvas
const canvas = document.getElementById("raytracer_canvas") as HTMLCanvasElement;

// Start rendering
const renderer = new Renderer(canvas);
renderer.init();
