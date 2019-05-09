const spinner = document.querySelector(".spinner p");
const spinnerContainer = document.querySelector(".spinner");
let rotateCount = 0;
let startTime = null;
let rAF;
const btn = document.querySelector("button");
const result = document.querySelector(".result");
const p1bar = document.querySelector(".p1-bar");
const p2bar = document.querySelector(".p2-bar");
const p1barContent = p1bar.cloneNode(true);
const p2barContent = p2bar.cloneNode(true);

function random(min,max){
  var num = Math.floor(Math.random()*(max-min)) + min;
  return num;
}

function draw(timestamp){
  if(!startTime){
   startTime = timestamp;
  }

  let rotateCount = (timestamp - startTime) / 3;
  spinner.style.transform = "rotate(" + rotateCount + "deg)";

  if(rotateCount > 359){
    rotateCount -= 360;
  }

  rAF = requestAnimationFrame(draw);
}

function reset(){
  btn.style.display = "block";
  result.textContent = "";
  result.style.display = "none";
  p1bar.innerHTML = p1barContent.innerHTML;
  p2bar.innerHTML = p2barContent.innerHTML;
}

function addKey(bar, key){
  var p = document.createElement("p");
  p.innerHTML = "Press <strong>\'" + key + "\'</strong>!!!!!!";

  var space = document.createElement("p");
  space.classList.add("space");

  bar.appendChild(space);
  bar.appendChild(p);
}

function setEndgame(){
  cancelAnimationFrame(rAF);
  spinnerContainer.style.display = "none";
  result.style.display = "block";
  result.textContent = "PLAYERS GO!!";

  const p1keys = [ "w", "a", "s", "d" ];
  const p2keys = [ "i", "j", "k", "l" ];

  // Choose the first random key for each player
  var p1key = p1keys[random(0, 4)];
  var p2key = p2keys[random(0, 4)];

  addKey(p1bar, p1key);
  addKey(p2bar, p2key);

  var p1count = 0;
  var p2count = 0;

  document.addEventListener("keydown", function keyHandler(e){
    console.log(e.key);

    // Check if key increases someone's score
    if(e.key === p1key){
      p1count += 1;
    } else if(e.key === p2key){
      p2count += 1;
    }

    // Check if any player won
    if(p1count == 5){
      result.textContent = "Player 1 won!!";
      document.removeEventListener("keydown", keyHandler);
      setTimeout(reset, 5000);
      return;
    } else if(p2count == 5){
      result.textContent = "Player 2 won!!";
      document.removeEventListener("keydown", keyHandler);
      setTimeout(reset, 5000);
      return;
    }

    // If nobody won, add new key for the payer who got it right
    if(e.key === p1key){
      p1key = p1keys[random(0, 4)];
      addKey(p1bar, p1key);
    } else if(e.key === p2key){
      p2key = p2keys[random(0, 4)];
      addKey(p2bar, p2key);
    }
  });
}

function start(){
  draw();
  spinnerContainer.style.display = "block";
  btn.style.display = "none";
  setTimeout(setEndgame, random(50,100));
}

result.style.display = "none";
spinnerContainer.style.display = "none";

btn.addEventListener("click", start);

