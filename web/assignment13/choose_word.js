function goButton(tabs) {
  var activeButs = document.querySelectorAll(".btn-outline-primary.active");
  var words = [];

  // Get the words within the active buttons
  for(var i = 0; i < activeButs.length; i++){
    words.push(activeButs[i].textContent);
  }

  // Send list of words to the main.js
  browser.tabs.sendMessage(tabs[0].id, {
    command: "go",
    words: words
  });
}



async function placeWords(){
  var json = await fetch("words.json")
    .then(response => { return response.json() })
    .catch(e => console.error("ERROR 1"));

  var words = [];

  // Then we select 10 random words
  for(var i = 0; i < 10; i++){
    var rand = Math.floor(Math.random() * json.length);
    var word = json.splice(rand, 1);
    console.log(word);
    words.push(word);
  }

  var box = document.getElementById("word-box");

  // For each word, we add a button
  for(var i = 0; i < 10; i++){
    var but = document.createElement("button");
    but.classList = "btn btn-outline-primary";
    but.textContent = words[i];
    box.appendChild(but);
  }

  console.log(words);
}

/**
 * Listen for clicks on the buttons, and send the appropriate message to
 * the content script in the page.
 */
function listenForClicks() {
  /* 'Controler' design pattern. Handles button clicks. */
  document.addEventListener("click", (e) => {
    if(e.target.classList.contains("gobutton")){
      browser.tabs.query({active: true, currentWindow: true})
        .then(goButton)
        .catch(() => console.error("Error."));
    } else if(e.target.classList.contains("btn-outline-primary")){
      e.target.classList.toggle("active");
    }
  });
}

browser.tabs.executeScript({file: "/main.js"})
  .then(placeWords)
  .then(listenForClicks)
  .catch((e) => {
    console.error("Fail. " + String(e));
  });
