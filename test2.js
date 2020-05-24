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
    let directoryPath = path.join(__dirname, 'train');
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

/*labelArray.forEach((data)=>{
    console.log(data)
})*/

let tensorLabels = tf.tensor2d(labelArray)

const model = tf.sequential();
model.add(
    tf.layers.dense({
        units: 128,
        inputShape: [128,128,3],
        activation: "relu"
    })
)
model.add(
    tf.layers.dense({
        units: 128,
        activation: "relu"
    })
)
model.add(tf.layers.flatten())
model.add(
    tf.layers.dense({
        units: 2, 
        activation: 'softmax'
    })
)

const LEARNING_RATE = 0.0001;

model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: "meanSquaredError"
});

model.fit(tensorFeatures, tensorLabels, {
    epochs: 10
});
let modelpath = path.join(__dirname,'models')
model.save('file://'+modelpath)

/*let data = fs.readFileSync('./dogtest.jpg')

let testdata = tf.node.decodeImage(data)

let testdata_Array = []
testdata_Array.push(testdata)

const showresult = async()=>{
    let result = await model.predict(tf.stack(testdata_Array))
    const ypred = result.flatten().arraySync();

    ypred.forEach(showresult2)

    function showresult2(item,index){
        console.log(index+1+" "+item)
    }
}

//showresult()*/