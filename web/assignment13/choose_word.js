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

  console.log(words);
}

/**
 * Listen for clicks on the buttons, and send the appropriate message to
 * the content script in the page.
 */
function listenForClicks() {
  document.addEventListener("click", (e) => {

    function go(tabs) {
      browser.tabs.sendMessage(tabs[0].id, {
        command: "reset",
      });
    }

    /**
     * Get the active tab,
     * then call "beastify()" or "reset()" as appropriate.
     */
    if(e.target.classList.contains("gobutton")){
      browser.tabs.query({active: true, currentWindow: true})
        .then(go)
        .catch(() => console.error("Error."));
    }
  });
}

/**
 * When the popup loads, inject a content script into the active tab,
 * and add a click handler.
 * If we couldn't inject the script, handle the error.
 */
browser.tabs.executeScript({file: "/main.js"})
.then(placeWords)
.then(listenForClicks)
.catch((e) => {
  console.error("Fail. " + String(e));
});
