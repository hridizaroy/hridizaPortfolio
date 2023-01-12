var table = document.querySelector(".table");
var beakers = document.getElementsByClassName("beakers");

// Beaker click behavior
for (var i = 0; i < beakers.length; i++) {
    beakers[i].addEventListener('click', holder_single);
}

//Setting test tube images
document.querySelector(".tt_1").style.content = 'url("media/Lab_equip/Test tube - Clear Blue-01.svg")';
document.querySelector(".tt_2").style.content = 'url("media/Lab_equip/Test Tube - Clear Orange-01.svg")';
document.querySelector(".tt_3").style.content = 'url("media/Lab_equip/Test tube - Clear Green-01.svg")';


//Test tube and holder animations
document.querySelector(".tt_1").addEventListener('click', tt1);
document.querySelector(".tt_2").addEventListener('click', tt2);
document.querySelector(".tt_3").addEventListener('click', tt3);

function holder_single() {
    var holder = document.querySelector(".holder_single_empty");
    holder.style.animation = "holder_single 0.7s normal forwards";
    setTimeout(() => {
        holder.style.content = 'url("media/Lab_equip/Holder- single-01.svg")';
        holder.style.animation = "none";
    }, 700);
}

function tt1() {
    if (this.style.content == 'url("media/Lab_equip/Test tube - Clear Blue-01.svg")') {
        this.style.animation = "tt1 0.7s normal forwards";
        setTimeout(()=> {
            this.style.content = 'url("media/Lab_equip/Test tubes - Blue-01.svg")';
            this.style.animation = "none";
            }, 700);
    }
    else {
        this.style.animation = "tt1 0.7s reverse forwards";
        setTimeout(() => {
            this.style.content = 'url("media/Lab_equip/Test tube - Clear Blue-01.svg")';
            this.style.animation = "none";
        }, 700);
    }
}

function tt2() {
    if (this.style.content == 'url("media/Lab_equip/Test Tube - Clear Orange-01.svg")') {
        this.style.animation = "tt2 0.7s normal forwards";
        setTimeout(() => {
            this.style.content = 'url("media/Lab_equip/Test tubes - Orange-01.svg")';
            this.style.animation = "none";
        }, 700);
    }
    else {
        this.style.animation = "tt2 0.7s reverse forwards";
        setTimeout(() => {
            this.style.content = 'url("media/Lab_equip/Test Tube - Clear Orange-01.svg")';
            this.style.animation = "none";
        }, 700);
    }
}

function tt3() {
    if (this.style.content == 'url("media/Lab_equip/Test tube - Clear Green-01.svg")') {
        this.style.animation = "tt3 0.7s normal forwards";
        setTimeout(() => {
            this.style.content = 'url("media/Lab_equip/Test Tubes-Green-01.svg")';
            this.style.animation = "none";
        }, 700);
    }
    else {
        this.style.animation = "tt3 0.7s reverse forwards";
        setTimeout(() => {
            this.style.content = 'url("media/Lab_equip/Test tube - Clear Green-01.svg")';
            this.style.animation = "none";
        }, 700);
    }
}