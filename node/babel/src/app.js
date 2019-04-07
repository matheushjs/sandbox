import mod2 from "./mod2.js";
import mod1 from "./mod1.js";

window.onload = () => {
  mod1();
  alert(mod2.global);
}

Promise.resolve().finally();
