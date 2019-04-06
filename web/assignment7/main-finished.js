// setup canvas
var canvas = document.querySelector('canvas');
var ctx = canvas.getContext('2d');
var paragraph = document.querySelector('p');

var width = canvas.width = window.innerWidth;
var height = canvas.height = window.innerHeight;

// Variaveis para ver as teclas pressionadas
var pressedA = false;
var pressedW = false;
var pressedD = false;
var pressedS = false;

const TIME_SENSITIVITY = 40; // Milisegundo
const PIXELS_PER_SECOND = 500; // Velocidade do EvilCircle em pixels por segundo
var timestamp = -1;
/**
 * Queremos que o EvilCircle se mova a 650 pixels por segundo (em uma dimensão).
 * Essa função retorna a velocidade correta, levando em consideração TIME_SENSITIVITY.
 */
function sensitive_velocity(){
  var updatesPerSecond = 1000 / TIME_SENSITIVITY;
  return PIXELS_PER_SECOND / updatesPerSecond;
}

// define array to store balls
var balls = [];

var score = 0;

paragraph.textContent = "Score: 0";


// function to generate random number
function random(min,max) {
  var num = Math.floor(Math.random()*(max-min)) + min;
  return num;
}


function Shape(x, y, velX, velY, exists) {
  this.x = x;
  this.y = y;
  this.velX = velX;
  this.velY = velY;
  this.exists = exists;
}

function Ball(x, y, velX, velY, color, size, exists){
  Shape.call(this, x, y, velX, velY, exists);
  this.color = color;
  this.size = size;
}

// Ball herda de Shape
Ball.prototype = Object.create(Shape.prototype);
Object.defineProperty(Ball.prototype, 'constructor', { 
  value: Ball,
  enumerable: false,
  writable: true
});

// define ball draw method
Ball.prototype.draw = function() {
  if(this.exists === false) return;

  ctx.beginPath();
  ctx.fillStyle = this.color;
  ctx.arc(this.x, this.y, this.size, 0, 2 * Math.PI);
  ctx.fill();
};

// define ball update method
Ball.prototype.update = function() {
  if((this.x + this.size) >= width) {
    this.velX = -(this.velX);
  }

  if((this.x - this.size) <= 0) {
    this.velX = -(this.velX);
  }

  if((this.y + this.size) >= height) {
    this.velY = -(this.velY);
  }

  if((this.y - this.size) <= 0) {
    this.velY = -(this.velY);
  }

  this.x += this.velX;
  this.y += this.velY;
};

// define ball collision detection
Ball.prototype.collisionDetect = function() {
  for(var j = 0; j < balls.length; j++) {
    if(!(this === balls[j])) {
      var dx = this.x - balls[j].x;
      var dy = this.y - balls[j].y;
      var distance = Math.sqrt(dx * dx + dy * dy);

      if (distance < this.size + balls[j].size) {
        balls[j].color = this.color = 'rgb(' + random(0,255) + ',' + random(0,255) + ',' + random(0,255) +')';
      }
    }
  }
};

function EvilCircle(x, y, exists){
  var vel = sensitive_velocity();
  Shape.call(this, x, y, vel, vel, exists);
  this.color = 'white';
  this.size = 10;
}

// Herda de Shape
EvilCircle.prototype = Object.create(Shape.prototype);
Object.defineProperty(EvilCircle.prototype, 'constructor', { 
  value: EvilCircle,
  enumerable: false,
  writable: true
});

// Pinta o circulo
EvilCircle.prototype.draw = function() {
  ctx.beginPath();
  ctx.strokeStyle = this.color;
  ctx.arc(this.x, this.y, this.size, 0, 2 * Math.PI);
  ctx.lineWidth = 3;
  ctx.stroke();
};

// Verifica se saiu da tela
EvilCircle.prototype.checkBounds = function() {
  if((this.x + this.size) >= width) {
    this.x = width - this.size;
  }

  if((this.x - this.size) <= 0) {
    this.x = this.size;
  }

  if((this.y + this.size) >= height) {
    this.y = height - this.size;
  }

  if((this.y - this.size) <= 0) {
    this.y = this.size;
  }
};

EvilCircle.prototype.update = function(){
  if (pressedA) { // a
    this.x -= this.velX;
  }
  
  if (pressedD) { // d
    this.x += this.velX;
  }
  
  if (pressedW) { // w
    this.y -= this.velY;
  }
  
  if (pressedS) { // s
    this.y += this.velY;
  }
}

// Controla o circulo
EvilCircle.prototype.setControls = function(){
  // As teclas WASD 
  window.onkeydown = function(e) {
    if (e.keyCode === 65) { // a
      pressedA = true;
    } else if (e.keyCode === 68) { // d
      pressedD = true;
    } else if (e.keyCode === 87) { // w
      pressedW = true;
    } else if (e.keyCode === 83) { // s
      pressedS = true;
    }
  }

  window.onkeyup = function(e){
    if (e.keyCode === 65) { // a
      pressedA = false;
    } else if (e.keyCode === 68) { // d
      pressedD = false;
    } else if (e.keyCode === 87) { // w
      pressedW = false;
    } else if (e.keyCode === 83) { // s
      pressedS = false;
    }
  }
}

// define ball collision detection
EvilCircle.prototype.collisionDetect = function() {
  for(var j = 0; j < balls.length; j++) {
    if(balls[j].exists === false)
      continue;

    var dx = this.x - balls[j].x;
    var dy = this.y - balls[j].y;
    var distance = Math.sqrt(dx * dx + dy * dy);

    if (distance < this.size + balls[j].size) {
      // Apaga a bola
      balls[j].exists = false;
      score += 1;
      paragraph.textContent = "Score: " + score;
    }
  }
};


// define loop that keeps drawing the scene constantly
function loop() {
  ctx.fillStyle = 'rgba(0,0,0,0.25)';
  ctx.fillRect(0,0,width,height);

  while(balls.length < 25) {
    var size = random(10,20);
    var ball = new Ball(
      // ball position always drawn at least one ball width
      // away from the adge of the canvas, to avoid drawing errors
      random(0 + size,width - size),
      random(0 + size,height - size),
      random(-7,7),
      random(-7,7),
      'rgb(' + random(0,255) + ',' + random(0,255) + ',' + random(0,255) +')',
      size,
      true
    );
    balls.push(ball);
  }

  if(performance.now() - timestamp > TIME_SENSITIVITY){
    evil.update();
    timestamp = performance.now();
  }

  for(var i = 0; i < balls.length; i++) {
    balls[i].draw();
    balls[i].update();
    balls[i].collisionDetect();
    evil.draw();
    evil.checkBounds();
    evil.collisionDetect();
  }

  requestAnimationFrame(loop);
}

var evil = new EvilCircle(width/2, height/2, true);
evil.setControls();

loop();