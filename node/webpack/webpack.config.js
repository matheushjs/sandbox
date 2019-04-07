var path = require("path");

module.exports = {
  entry: {
    app: './src/app.js',
    mod1: './src/mod1.js'
  },
  output: {
    path: path.resolve(__dirname, 'build')
  }
};
