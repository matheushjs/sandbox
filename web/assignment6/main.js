var displayedImage = document.querySelector('.displayed-img');
var thumbBar = document.querySelector('.thumb-bar');

var btn = document.querySelector('button');
var overlay = document.querySelector('.overlay');

var index = 0;

/* Looping through images */
for(var i = 1; i <= 5; i++){
  var newImage = document.createElement('img');

  newImage.setAttribute('src', "./images/pic" + String(index + 1) + ".jpg");
  newImage.onclick = function(){
    displayedImage.setAttribute('src', this.getAttribute('src'));
  };
  
  thumbBar.appendChild(newImage);
  index = (index+1)%5;
}

/* Wiring up the Darken/Lighten button */
btn.onclick = function(){
  var cls = btn.getAttribute('class');

  if(cls === 'dark'){
    btn.setAttribute('class', 'light');
    btn.textContent = "Lighten";
    overlay.setAttribute('style', 'background: rgba(0,0,0,0.5);');
  } else {
    btn.setAttribute('class', 'dark');
    btn.textContent = "Darken";
    overlay.setAttribute('style', '');
  }
}
