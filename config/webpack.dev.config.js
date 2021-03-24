const { merge } = require('webpack-merge');
const common = require('./webpack.common.config');
const path = require('path');

const config ={
    devServer: {
        contentBase:path.join(__dirname, "../dist")
    }
}

const options={
    mode:"development"
}

module.exports = merge(common(options),config)