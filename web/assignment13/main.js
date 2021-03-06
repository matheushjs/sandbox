(function() {
  /**
   * Check and set a global guard variable.
   * If this content script is injected into the same page again,
   * it will do nothing next time.
   */
  if (window.hasRun) {
    return;
  }
  window.hasRun = true;

  /**
   * Listen for messages from the background script.
  */
  browser.runtime.onMessage.addListener((message) => {
    if(message.command === "go"){
      console.log(message.words);
      window.open("https://google.com/search?q=" + message.typed + "+" + message.words.join("+"));
    }
  });

})();
