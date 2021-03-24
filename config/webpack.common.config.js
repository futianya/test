const webpack = require("webpack");
const path = require("path");
const HtmlWebpackPlugin = require('html-webpack-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const  MiniCssExtractPlugin = require("mini-css-extract-plugin");

const BUILD_DIR = path.resolve(__dirname, "../build");
const APP_DIR = path.resolve(__dirname, "../src");

function config(options){
  return{
    mode:options.mode,
    entry: APP_DIR + "/index.js",
    output: {
      path: BUILD_DIR,
      filename: "bundle.js"
    },
    module: {
      rules: [
        {
          test: /\.(js|jsx)$/,
          exclude: /node_modules/,
          use: {
            loader: "babel-loader",
            options:{
              plugins: ["react-hot-loader/babel"]
            }
          }
        },
        {
          test: /\.css$/,
          use: [MiniCssExtractPlugin.loader,
          { loader: "css-loader" }
          ]
        },
        {
          test: /\.less$/,
          use : [
              MiniCssExtractPlugin.loader,
              { loader: "css-loader" },
              { loader: "less-loader" }
          ]
        },
        {
          test: /\.(jpg|png|svg|gif)$/,
          use: [
            {
              loader: "url-loader",
              options: {
                limit: 10240,
                name: "[hash].[ext]",
              }
            }
          ]
        }
      ]
    },
    plugins:[
      new HtmlWebpackPlugin({
        template: "./public/index.html",
        inject: true,
        sourceMap:true,
        chunksSortMode: "auto"
      }),
      new CleanWebpackPlugin(
        {
        cleanOnceBeforeBuildPatterns:[path.resolve(process.cwd(),"build/"), path.resolve(process.cwd(), "dist/")]
        }
      ),
      new MiniCssExtractPlugin({
        filename: "[name][hash].css"
      })
    ]
  }
}

module.exports = config;