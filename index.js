const tf = require('@tensorflow/tfjs-node')
const path = require('path')
const fs = require('fs')

let trainData = []
const tensorFeature_cat = []
const tensorFeature_dog = []
let tensorFeatures
let AllFeatures = []

const loaddata = ()=>{
    /*load cat image */
    let directoryPath = path.join(__dirname, 'cat');
    let files = fs.readdirSync(directoryPath)
    files.forEach(function (file) {
        let data = fs.readFileSync(directoryPath+"\\"+file)
        trainData.push(data) 
    });
    
    for(let i=0;i<trainData.length;i++)
    {
        tensorFeature_cat[i] = tf.node.decodeJpeg(trainData[i])
    }

    trainData = []

    /*load dog image */
    directoryPath = path.join(__dirname, 'dog');
    files = fs.readdirSync(directoryPath)
    files.forEach(function (file) {
        let data = fs.readFileSync(directoryPath+"\\"+file)
        trainData.push(data)
    });
    
    for(let i=0;i<trainData.length;i++)
    {
        tensorFeature_dog[i] = tf.node.decodeJpeg(trainData[i])
    }
    
    const b = tf.scalar(255)

    for(let i=0;i<20;i++)
    {
        if(i<10)
        {
            AllFeatures.push(tensorFeature_cat[i].div(b))
        }
        else{
            AllFeatures.push(tensorFeature_dog[i-10].div(b))
        }
    }

    tensorFeatures = tf.stack(AllFeatures)
}

loaddata()

let labelArray = []
for(let i=0;i<20;i++)
{
    if(i<10)
    {
        labelArray.push([0,1])
    }
    else{
        labelArray.push([1,0])
    }
}

let tensorLabels = tf.tensor2d(labelArray)

const train=async()=>{
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [256, 256, 3], // numberOfChannels = 3 for colorful images and one otherwise
        filters: 64,
        kernelSize: 3,
        activation: 'relu',
        padding:"same" 
    }))
    model.add(tf.layers.maxPooling2d({
        poolSize:2
    }))
    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: 'relu',
        padding:"same" 
    }))
    model.add(tf.layers.maxPooling2d({
        poolSize:2
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
        optimizer: "adam",
        loss: "categoricalCrossentropy",
        metrics:['accuracy']
    })

    await model.fit(tensorFeatures, tensorLabels, {
        epochs: 10,
        batchSize:32,
        validationSplit:0.1
    })
    let modelpath = path.join(__dirname,'testmodels')
    model.save('file://'+modelpath)
    result = await model.predict(tensorFeatures)

    const ypred = result.arraySync();

    ypred.forEach(showresult2)

    function showresult2(item,index){
        console.log(index+1+" "+item)
    }

    let test = model.evaluate(tensorFeatures,tensorLabels)
    test.forEach(data=>{
        data.print()
    })
}

train()









