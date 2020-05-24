const tf = require('@tensorflow/tfjs-node')

const a = tf.tensor1d([2, 4, 6, 8]);
const b = tf.scalar(255);

a.div(b).print();  // or tf.div(a, b)

let c = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);

c.div(b).print(); 

console.log(a)
