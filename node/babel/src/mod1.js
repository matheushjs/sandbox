var mod2 = require("./mod2.js");


module.exports = () => {
	alert(mod2.global);
	mod2.global = 3;
};
