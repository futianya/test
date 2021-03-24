const { merge } = require('webpack-merge')
const common = require('./webpack.common.config')

const config ={
    plugins: [
        new optimizeCss({
            cssProcessor: require('cssnano'),
            cssProcessorOptions: { discardComments: { removeAll: true } },
            canPrint: true
        }),
    ],
}

const options={
    mode:"production"
}

module.exports = merge(common(options),config)