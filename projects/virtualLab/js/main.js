
// Object variables
const Wall = document.querySelector(".wall");
const Wall_width = Wall.offsetWidth;
const Container = document.querySelector(".container");
const OuterContainer = document.querySelector(".container_outer");
const BackWall = document.querySelector(".back");
const Notes_obj = document.querySelector(".notes");
const Teacher = document.querySelector(".teacher");
const LabStuff = document.querySelector(".lab_stuff");

// Setting initial distance from front wall
OuterContainer.style.perspective = Wall_width / 2 + "px";

// Setting intial anchor point
Container.style.transformOrigin = "50% 50% " + Wall_width / 2 + "px";

// Setting distance from front wall to back wall = wall width
BackWall.style.transform = "translateZ(" + Wall_width + "px)";

// Setting lab_stuff container height to wall height
LabStuff.style.height = Wall_width + "px";

Notes_obj.style.display = "none";


//Movement

// Constants for restraints
const minZ = -0.45 * Wall_width;
const maxZ = 0.2 * Wall_width;

// 0.3 and 0.6 because the walls are placed at bottom = 0, left = 0 initially
// So 1/3rd of the wall is to the left, and 2/3rds are to the right of the initial transform origin
const maxX = 0.3 * Wall_width;
const minX = -0.6 * Wall_width;

// Position variables
var z = 0;
var x = 0;
var y = 0;

var newX;
var newZ;

// How big each step/turn is
const move_step = 5;
const rotate_step = 1;

//Look Around
var rotateY = 0;
var rotateX = 0;

// controlling teacher orientation
let x_distance = x + 0.65 * Wall_width - innerWidth/2;
let z_distance = Wall_width / 4 - z;
var rotateTeacher = -1* Math.atan(x_distance/z_distance) * 180 / Math.PI;
Teacher.style.transform = "translateZ(" + Wall_width / 4 + "px) rotateY(" + (rotateTeacher) + "deg)";

document.body.addEventListener('keydown', move);
document.body.addEventListener('keyup', jump);
document.body.addEventListener('mouseup', notes_toggle);

//Movement and looking around

function move(e) {
    Container.style.transition = "0s"; // resetting the transition, in case it is changed by the jump()

    var rotate_val;

    if (Container.style.transform == "") { // initial case
        rotate_val = 0; // no rotation
    }
    else {
        // otherwise, set rotation
        rotate_val = parseInt(Container.style.transform.split("rotateY(")[1].split("deg")[0]);
    }

    //w
    if (e.keyCode == 119 || e.keyCode == 87 ) {
        // use vector math to set movement according to direction
        newX = x - move_step * Math.sin(rotate_val * Math.PI / 180);
        newZ = z + move_step * Math.cos(rotate_val * Math.PI / 180);

        if (maxX >= newX && newX >= minX && minZ <= newZ && newZ <= maxZ) {
            x = newX;
            z = newZ;
        }
    }
    //s
    else if (e.keyCode == 115 || e.keyCode == 83) {
        newX = x + move_step * Math.sin(rotate_val * Math.PI / 180);
        newZ = z - move_step * Math.cos(rotate_val * Math.PI / 180);

        if (maxX >= newX && newX >= minX && minZ <= newZ && newZ <= maxZ) {
            x = newX;
            z = newZ;
        }
    }
    //a
    else if (e.keyCode == 97 || e.keyCode == 65) {
        newX = x + move_step * Math.cos(rotate_val * Math.PI / 180);
        newZ = z + move_step * Math.sin(rotate_val * Math.PI / 180);

        if (maxX >= newX && newX >= minX && minZ <= newZ && newZ <= maxZ) {
            x = newX;
            z = newZ;
        }
    }
    //d
    else if (e.keyCode == 100 || e.keyCode == 68) {
        newX = x - move_step * Math.cos(rotate_val * Math.PI / 180);
        newZ = z - move_step * Math.sin(rotate_val * Math.PI / 180);

        if (maxX >= newX && newX >= minX && minZ <= newZ && newZ <= maxZ) {
            x = newX;
            z = newZ;
        }
    }

    // up arrow
    else if (e.keyCode == 38) {
        rotateX += rotate_step;
    }
    // down arrow
    else if (e.keyCode == 40) {
        rotateX -= rotate_step;
    }
    // right arrow
    else if (e.keyCode == 39) {
        rotateY += rotate_step;
    }
    // left arrow
    else if (e.keyCode == 37) {
        rotateY -= rotate_step;
    }
    
    // Restraints

    if (rotateX > 90) {
        rotateX = 90;
    }
    else if (rotateX < -90) {
        rotateX = -90;
    }

    // Make teacher rotate to follow you
    // Use vectors to calculate rotation
    let x_distance = x + 0.65 * Wall_width - innerWidth/2;
    let z_distance = Wall_width / 4 - z;
    rotateTeacher = -1 * Math.atan(x_distance/z_distance) * 180 / Math.PI;
    Container.style.transformOrigin = innerWidth / 2 - x + "px " + "50% " + (Wall_width/2 - z) + "px"; // changing anchor point to current position
    Container.style.transform = "translateZ(" + z + "px) translateX(" + x + "px)" + "rotateX(" + rotateX + "deg) " + "rotateY(" + rotateY + "deg)";

    Teacher.style.transform = "translateZ(" + Wall_width / 4 + "px) rotateY(" + (rotateTeacher) + "deg)";
}

function jump(e) {
    //spacebar              
    if (e.keyCode == 32) {
        //Jump
        y = 10;
        Container.style.transition = "0.6s";
        Container.style.transform = "translateZ(" + z + "px) translateX(" + x + "px)" + "rotateX(" + rotateX + "deg) " + "rotateY(" + rotateY + "deg)" + "translateY(" + y + "%)";
        setTimeout(() => {
            y = 0;
            Container.style.transform = "translateZ(" + z + "px) translateX(" + x + "px)" + "rotateX(" + rotateX + "deg) " + "rotateY(" + rotateY + "deg)" + "translateY(" + y + "%)";
        }, 600);
    }
}

// Turn notes off and on
function notes_toggle(e){
    // show/hide on right click
    if (e.button == 2) {
        if (Notes_obj.style.display == "none") {
            Notes_obj.style.display = "block";
            document.body.removeEventListener('keydown', move);
            document.body.removeEventListener('keyup', jump);
        }
        else {
            Notes_obj.style.display = "none";
            document.body.addEventListener('keydown', move);
            document.body.addEventListener('keyup', jump);
        }
    }
}