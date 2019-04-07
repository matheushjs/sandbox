var mod2 = require("./mod2.js");

console.log(mod2.global);

mod2.global = 3;

var mod1 = require("./mod1.js");

