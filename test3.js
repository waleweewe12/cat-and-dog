const tf = require('@tensorflow/tfjs-node')
const path = require('path')
const fs = require('fs')
const sharp = require('sharp')

let trainData = []
let tensorFeatures
let AllFeatures = []
let labelArray = []

const loaddata = ()=>{
    /*load cat image */
    let directoryPath = path.join(__dirname, 'mytrain');
    let files = fs.readdirSync(directoryPath)
    files.forEach(function (file) {
        /*push Buffer data */
        let data = fs.readFileSync(directoryPath+"\\"+file)
        trainData.push(data) 
        /*push label data */
        if(file.includes("cat"))
        {
            labelArray.push([0,1])
        }else{
            labelArray.push([1,0])
        }
    });
    
    for(let i=0;i<trainData.length;i++)
    {
        AllFeatures[i] = tf.node.decodeImage(trainData[i]) 
    }

    tensorFeatures = tf.stack(AllFeatures)
}

loaddata()

AllFeatures.forEach(data=>{
    console.log(data)
})

/*for(let i=0;i<AllFeatures.length;i++)
{
    //let insidedata = AllFeatures[i].flatten().arraySync()
    for(let j=0;j<AllFeatures[i].flatten().arraySync().length;j++)
    {
        AllFeatures[i].flatten().arraySync()[j] /= 255
    }
}
AllFeatures.forEach(data=>{
    console.log(data.flatten().arraySync())
})*/

let tensorLabels = tf.tensor2d(labelArray)

const train=async()=>{

        const model = tf.sequential();
        model.add(tf.layers.conv2d({
            inputShape: [256, 256, 3], // numberOfChannels = 3 for colorful images and one otherwise
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
        }))
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
        }))
        model.add(tf.layers.flatten())
        model.add(tf.layers.dense({
            units: 64, 
            activation: 'relu'
        }))
        model.add(tf.layers.dense({
            units: 2, 
            activation: 'softmax'
        }))

        const LEARNING_RATE = 0.005;

        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics:['accuracy']
        });

        await model.fit(tensorFeatures, tensorLabels, {
            epochs: 10
        })
        let modelpath = path.join(__dirname,'models')
        model.save('file://'+modelpath)
        //console.log(model.evaluate(tensorFeatures,tensorLabels))
}

//train()
