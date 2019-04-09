function randomValueFromArray(array){
  return array[Math.floor(Math.random()*array.length)];
}

var storyText = "It was 94 fahrenheit outside, so :insertx: went for a walk. When they got to :inserty:, they stared in horror for a few moments, then :insertz:. Bob saw the whole thing, but was not surprised — :insertx: weighs 300 pounds, and it was a hot day.";
var insertX = [
  "Willy the Goblin",
  "Big Daddy",
  "Father Christmas"
];
var insertY = [
  "the soup kitchen",
  "Disneyland",
  "the White House"
];
var insertZ = [
  "spontaneously combusted",
  "melted into a puddle on the sidewalk",
  "turned into a slug and crawled away"
];

function result(customName, story) {
  var newStory = storyText;
  var xItem = randomValueFromArray(insertX);
  var yItem = randomValueFromArray(insertY);
  var zItem = randomValueFromArray(insertZ);

  newStory = newStory
    .replace(":insertx", xItem)
    .replace(":insertx", xItem)
    .replace(":inserty", yItem)
    .replace(":insertz", zItem);  

  if(customName.value !== '') {
    newStory = newStory.replace("Bob", customName.value);
  }

  if(document.getElementById("uk").checked) {
    newStory = newStory
      .replace("94 fahrenheit", "34.4 celsius")
      .replace("300 pounds", "21.4 stones");
  }

  story.textContent = newStory;
  story.style.visibility = 'visible';
}

window.onload = function(){  
  var customName = document.getElementById('customname');
  var randomize = document.querySelector('.randomize');
  var story = document.querySelector('.story');

  randomize.addEventListener('click', function(){
    result(customName, story);
  });
}