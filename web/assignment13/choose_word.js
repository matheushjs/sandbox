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
        .catch(() => console.err("Error."));
    }
  });
}

/**
 * When the popup loads, inject a content script into the active tab,
 * and add a click handler.
 * If we couldn't inject the script, handle the error.
 */
browser.tabs.executeScript({file: "/main.js"})
.then(listenForClicks)
.catch((e) => {
  console.err("Fail. " + String(e));
});
